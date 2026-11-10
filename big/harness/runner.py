"""Async workload execution for mock, live agent loop, and teacher-forced replay."""
from __future__ import annotations
import asyncio, json, os, time, uuid
from pathlib import Path
from typing import Any
import httpx
from .tools import run_tool, schema_valid

MODEL=os.getenv('MODEL','unsloth/Qwen3.8-27B-NVFP4')

class Client:
    def __init__(self,base_url:str,api_key:str,model:str=MODEL,timeout:float=3600):
        self.base=base_url.rstrip('/'); self.key=api_key; self.model=model; self.timeout=timeout; self.client=None
    async def __aenter__(self):
        try:
            from openai import AsyncOpenAI
            self.client=AsyncOpenAI(base_url=self.base+'/v1',api_key=self.key,timeout=self.timeout)
        except ImportError: raise RuntimeError('Install project dependencies: pip install -e .')
        return self
    async def __aexit__(self,*_): await self.client.close()
    async def complete(self,messages,tools,effort,max_tokens=4096):
        kwargs={'enable_thinking':False} if effort=='off' else {'reasoning_effort':effort}
        start=time.perf_counter(); first=answer_at=None; last=None; gaps=[]; content=''; reasoning=''; calls={}; usage=None; finish=None
        stream=await self.client.chat.completions.create(model=self.model,messages=messages,tools=tools,tool_choice='auto',stream=True,stream_options={'include_usage':True},temperature=1.0,top_p=0.95,max_tokens=max_tokens,extra_body={'top_k':20,'chat_template_kwargs':kwargs})
        async for ch in stream:
            now=time.perf_counter()
            if ch.usage: usage=ch.usage
            if not ch.choices: continue
            choice=ch.choices[0]; d=choice.delta
            if choice.finish_reason: finish=choice.finish_reason
            think=getattr(d,'reasoning_content',None) or getattr(d,'reasoning',None)
            if think or d.content or d.tool_calls:
                if first is None: first=now-start
                elif last is not None: gaps.append(now-last)
                last=now
            if think: reasoning+=think
            if d.content:
                content+=d.content
                if answer_at is None: answer_at=now-start
            for tc in d.tool_calls or []:
                slot=calls.setdefault(tc.index,{'id':tc.id or '', 'name':'','args':''})
                if tc.id: slot['id']=tc.id
                if tc.function:
                    slot['name']+=tc.function.name or ''; slot['args']+=tc.function.arguments or ''
        e2e=time.perf_counter()-start; prompt_tokens=getattr(usage,'prompt_tokens',None); completion_tokens=getattr(usage,'completion_tokens',None)
        details=getattr(usage,'prompt_tokens_details',None); cached=getattr(details,'cached_tokens',None) if details else None
        rdetails=getattr(usage,'completion_tokens_details',None); reasoning_tokens=getattr(rdetails,'reasoning_tokens',None) if rdetails else None
        parsed=[]; json_ok=True; valid=True
        for c in calls.values():
            try: args=json.loads(c['args'] or '{}')
            except Exception: args={}; json_ok=False
            valid &= schema_valid(c['name'],args,tools)
            parsed.append({**c,'arguments':args})
        return {'prompt_tokens':prompt_tokens,'completion_tokens':completion_tokens,'reasoning_tokens':reasoning_tokens,'cached_prompt_tokens':cached,'ttft_s':first,'time_to_first_answer_s':answer_at,'mean_itl_s':(e2e-first)/max((completion_tokens or 0)-1,1) if first is not None and completion_tokens else None,'chunk_gaps_s':gaps,'e2e_s':e2e,'finish_reason':finish,'tool_calls':parsed,'tool_json_valid':json_ok,'tool_schema_valid':valid,'content':content,'reasoning':reasoning,'usage_raw':usage.model_dump() if usage else None}

class MockClient:
    async def complete(self,messages,tools,effort,max_tokens=4096):
        await asyncio.sleep(0.002); expected=''
        for m in messages:
            if m.get('role') in ('user','system'):
                import re
                x=re.search(r'CASE=\d+',m.get('content') or '')
                if x: expected=x.group(0)
        return {'prompt_tokens':sum(len((m.get('content') or '').split()) for m in messages),'completion_tokens':7,'reasoning_tokens':2 if effort!='off' else 0,'cached_prompt_tokens':None,'ttft_s':0.002,'time_to_first_answer_s':0.002,'mean_itl_s':0.0005,'chunk_gaps_s':[],'e2e_s':0.005,'finish_reason':'tool_calls','tool_calls':[{'id':str(uuid.uuid4()),'name':'search_docs','args':'{}','arguments':{'query':expected},'function':{'name':'search_docs','arguments':json.dumps({'query':expected})}}],'tool_json_valid':True,'tool_schema_valid':True,'content':'','reasoning':'','usage_raw':None}

async def _metrics_poller(base_url:str,out:Path,stop:asyncio.Event):
    async with httpx.AsyncClient(timeout=3) as c:
        while not stop.is_set():
            ts=time.time()
            try:
                r=await c.get(base_url.rstrip('/')+'/metrics'); r.raise_for_status()
                with out.open('a',encoding='utf8') as f:
                    for line in r.text.splitlines():
                        if not line or line.startswith('#'): continue
                        parts=line.rsplit(' ',1)
                        if len(parts)==2: f.write(json.dumps({'timestamp':ts,'metric':parts[0],'value':parts[1]})+'\n')
            except Exception as e:
                with (out.parent/'metrics_errors.log').open('a') as f: f.write(f'{ts} {type(e).__name__}: {e}\n')
            try: await asyncio.wait_for(stop.wait(),timeout=1)
            except asyncio.TimeoutError: pass

async def run_workload(trace:list[dict],out_dir:str|Path,mode='mock',base_url='http://localhost:8000',api_key='EMPTY',model=MODEL,concurrency=8,rate:float|None=None,seed=2026,warmup_s=0,timeout=3600)->dict:
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    if mode not in ('mock','live','replay'): raise ValueError('mode must be mock, live, or replay')
    if mode=='replay' and any(not s.get('steps') for s in trace): raise ValueError('replay trace has no recorded steps')
    req_path=out/'requests.jsonl'; sess_path=out/'sessions.jsonl'
    rf=req_path.open('w',encoding='utf8'); sf=sess_path.open('w',encoding='utf8')
    lock=asyncio.Lock(); sem=asyncio.Semaphore(concurrency); stop=asyncio.Event(); poller=asyncio.create_task(_metrics_poller(base_url,out/'metrics.jsonl',stop)) if mode!='mock' else None
    client=MockClient() if mode=='mock' else None
    async def execute(sess,remote):
        sstart=time.perf_counter(); elapsed0=time.time(); messages=list(sess['prefix']); step_rows=[]; total_in=total_out=0; llm=tooltime=0.0; success=False; errors=0
        try:
            for idx,step in enumerate(sess['steps']):
                async with sem:
                    r=await (client.complete(messages,sess['tools'],step.get('thinking_effort',sess['thinking_effort'])) if mode=='mock' else remote.complete(messages,sess['tools'],step.get('thinking_effort',sess['thinking_effort'])))
                r['session_id']=sess['session_id']; r['step']=idx; r['class']=sess['class']; r['submit_time']=time.time(); r['mode']=mode
                r['request_id']=str(uuid.uuid4()); step_rows.append(r); llm+=r.get('e2e_s') or 0; total_in+=r.get('prompt_tokens') or 0; total_out+=r.get('completion_tokens') or 0
                # Always preserve request evidence before advancing.
                async with lock: rf.write(json.dumps(r,ensure_ascii=False,default=str)+'\n'); rf.flush()
                if mode=='replay':
                    messages.append(step['assistant']); messages.append(step['tool_result'])
                    if step.get('final_answer'): success=sess.get('expected_answer','') in (step['tool_result'].get('content') or '')
                elif mode=='mock':
                    messages.append(step['assistant']); messages.append(step['tool_result'])
                    if step.get('final_answer'): success=sess.get('expected_answer','') in (step['tool_result'].get('content') or '')
                else:
                    if not r['tool_calls']:
                        success=sess.get('expected_answer','') in r.get('content',''); break
                    assistant={'role':'assistant','content':r.get('content') or None,'tool_calls':[]}
                    for tc in r['tool_calls']:
                        fn=tc.get('function') or {'name':tc.get('name'),'arguments':tc.get('args','{}')}
                        assistant['tool_calls'].append({'id':tc.get('id') or str(uuid.uuid4()),'type':'function','function':fn})
                    messages.append(assistant)
                    for tc in r['tool_calls']:
                        name=tc.get('name') or tc.get('function',{}).get('name',''); args=tc.get('arguments')
                        if args is None:
                            try: args=json.loads(tc.get('args','{}'))
                            except Exception: args={}
                        t0=time.perf_counter(); result=await run_tool(name,args,sess,step); tooltime+=time.perf_counter()-t0
                        messages.append({'role':'tool','tool_call_id':tc.get('id') or '', 'content':result})
                        if sess.get('expected_answer','') in result: success=True
            record={'session_id':sess['session_id'],'class':sess['class'],'project_id':sess.get('project_id'),'start_time':elapsed0,'duration_s':time.perf_counter()-sstart,'steps':len(step_rows),'tokens_in':total_in,'tokens_out':total_out,'task_success':success,'llm_time_s':llm,'tool_time_s':tooltime,'queue_attributed_s':None,'errors':errors,'mode':mode}
        except Exception as e:
            errors+=1; record={'session_id':sess['session_id'],'class':sess['class'],'project_id':sess.get('project_id'),'start_time':elapsed0,'duration_s':time.perf_counter()-sstart,'steps':len(step_rows),'tokens_in':total_in,'tokens_out':total_out,'task_success':False,'llm_time_s':llm,'tool_time_s':tooltime,'queue_attributed_s':None,'errors':errors,'error':f'{type(e).__name__}: {e}','mode':mode}
            async with lock: (out/'errors.jsonl').open('a').write(json.dumps(record)+'\n')
        async with lock: sf.write(json.dumps(record,ensure_ascii=False)+'\n'); sf.flush()
    async def driver():
        tasks=[]; previous=asyncio.get_running_loop().time(); rng=__import__('random').Random(seed)
        async with (remote if mode!='mock' else _null_async_ctx()) as active:
            trace_origin=float(trace[0].get('arrival_s',0)) if trace else 0
            wall_origin=asyncio.get_running_loop().time()
            for sess in trace:
                target=wall_origin+(float(sess.get('arrival_s',0))-trace_origin)
                # Trace arrivals are relative offsets; optionally override as fresh Poisson input.
                if rate and rate>0:
                    previous+=rng.expovariate(rate); target=previous
                await asyncio.sleep(max(0,target-asyncio.get_running_loop().time()))
                tasks.append(asyncio.create_task(execute(sess,active if mode!='mock' else None)))
            await asyncio.gather(*tasks)
    remote=Client(base_url,api_key,model,timeout) if mode!='mock' else None
    try: await driver()
    finally:
        if poller: stop.set(); await poller
        rf.close(); sf.close()
    if warmup_s>0: _mark_warmup(req_path,sess_path,warmup_s)
    return {'requests':len(trace),'request_log':str(req_path),'session_log':str(sess_path)}

class _null_async_ctx:
    async def __aenter__(self): return None
    async def __aexit__(self,*_): return False

def _mark_warmup(req_path:Path,sess_path:Path,warmup_s:float):
    # Preserve raw evidence, annotate each row to make downstream exclusion reproducible.
    for path,field in ((req_path,'submit_time'),(sess_path,'start_time')):
        rows=[]
        source=path.read_text().splitlines()
        if not source: continue
        origin=float(json.loads(source[0]).get(field,0))
        for line in source:
            row=json.loads(line); row['warmup']=float(row.get(field,0))-origin<warmup_s; rows.append(row)
        path.write_text(''.join(json.dumps(x,ensure_ascii=False,default=str)+'\n' for x in rows))
