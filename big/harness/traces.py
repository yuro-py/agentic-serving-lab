"""Deterministic synthetic agent trace generation and replay records."""
from __future__ import annotations
import hashlib, json, random
from pathlib import Path
from typing import Any

PRESETS = {
    "A": {"name":"coding", "system_tokens":6000, "project_tokens":[20000,40000], "steps":[12,40], "tool_tokens":[1500,8000], "effort":"medium", "delay":[0.8,5.0]},
    "B": {"name":"research", "system_tokens":4000, "project_tokens":[0,0], "steps":[8,25], "tool_tokens":[4000,15000], "effort":"low", "delay":[1.5,8.0]},
    "C": {"name":"support", "system_tokens":2500, "project_tokens":[0,0], "steps":[4,8], "tool_tokens":[300,1000], "effort":"off", "delay":[0.2,1.0]},
    "D": {"name":"long_doc", "system_tokens":0, "project_tokens":[60000,120000], "steps":[1,1], "tool_tokens":[0,0], "effort":"low", "delay":[0,0]},
}
MIX = ["A"]*4+["B"]*2+["C"]*3+["D"]
TOOLS = [
 {"type":"function","function":{"name":"search_docs","description":"Search the project documents","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
 {"type":"function","function":{"name":"read_file","description":"Read a named file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
 {"type":"function","function":{"name":"run_python","description":"Evaluate a safe arithmetic expression","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}},
 {"type":"function","function":{"name":"sql_query","description":"Query the synthetic table","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
 {"type":"function","function":{"name":"calculator","description":"Calculate a numeric expression","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}},
 {"type":"function","function":{"name":"final_answer","description":"Return the final task answer","parameters":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}}},
]

def _length(rng: random.Random, lo: int, hi: int) -> int:
    if lo == hi: return lo
    # Lognormal-like long tail, bounded, with median near geometric midpoint.
    median=(lo*hi)**0.5
    return max(lo,min(hi,int(rng.lognormvariate(__import__('math').log(median),0.65))))

def _text(seed: str, token_count: int) -> str:
    words=[]; i=0
    while len(words)<token_count:
        digest=hashlib.sha256(f"{seed}:{i}".encode()).hexdigest()
        words.extend([f"record_{digest[j:j+8]}" for j in range(0,64,8)]); i+=1
    return " ".join(words[:token_count])

def build_trace(preset: str, sessions: int, seed: int, small: bool=False, rate: float=0.1) -> list[dict[str,Any]]:
    rng=random.Random(seed); rows=[]; clock=0.0
    for i in range(sessions):
        clock += rng.expovariate(rate) if rate>0 else 0
        code=rng.choice(MIX) if preset.upper()=="M" else preset.upper()
        p=PRESETS[code]; project=f"project-{rng.randrange(8):02d}" if code=="A" else None
        sys_tokens=p["system_tokens"]; proj_tokens=_length(rng,*p["project_tokens"])
        nsteps=_length(rng,*p["steps"])
        if small: sys_tokens=min(sys_tokens,40); proj_tokens=min(proj_tokens,80); nsteps=min(nsteps,3)
        task_value=rng.randint(10,999); task=f"Find the value for case-{task_value} and report it as CASE={task_value}."
        prefix=[]
        if sys_tokens: prefix.append({"role":"system","content":f"You are a careful tool-using agent.\n{_text('shared-system-v1',sys_tokens)}"})
        if proj_tokens: prefix.append({"role":"system","content":f"Shared project context {project or 'long-document'}:\n{_text('project:'+str(project or i//4),proj_tokens)}"})
        prefix.append({"role":"user","content":task})
        steps=[]
        for s in range(nsteps):
            delay=_length(rng,int(p['delay'][0]*1000),int(p['delay'][1]*1000))/1000 if p['delay'][1] else 0
            tool_tokens=_length(rng,*p['tool_tokens']) if p['tool_tokens'][1] else 0
            if small: tool_tokens=min(tool_tokens,40)
            # Recorded turn is tool-call + deterministic tool result; replay prompts are stable.
            tool_name="search_docs" if code in ("A","B") else "calculator"
            call_id=f"call_{i}_{s}"
            assistant={"role":"assistant","content":None,"tool_calls":[{"id":call_id,"type":"function","function":{"name":tool_name,"arguments":json.dumps({"query":f"case-{task_value}"} if tool_name=="search_docs" else {"expression":f"{task_value}"})}}]}
            result=f"Found CASE={task_value}. {_text(f'{tool_name}:{task_value}:{s}',tool_tokens)}"
            steps.append({"index":s,"tool_delay_s":delay,"thinking_effort":p['effort'],"assistant":assistant,"tool_result":{"role":"tool","tool_call_id":call_id,"content":result},"final_answer": s==nsteps-1})
        rows.append({"schema_version":1,"session_id":f"s{i:06d}","arrival_s":round(clock,6),"project_id":project,"preset":code,"class":code,"thinking_effort":p['effort'],"prefix":prefix,"tools":TOOLS,"max_steps":nsteps,"expected_answer":f"CASE={task_value}","steps":steps,"seed":seed})
    return rows

def write_jsonl(rows: list[dict[str,Any]], path: str|Path) -> str:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_jsonl(path: str|Path) -> list[dict[str,Any]]:
    with Path(path).open(encoding='utf8') as f: return [json.loads(line) for line in f if line.strip()]
