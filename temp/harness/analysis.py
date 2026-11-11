"""Aggregate raw run evidence and regenerate plots."""
from __future__ import annotations
import json, math
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _read_jsonl(path):
    if not path.exists(): return pd.DataFrame()
    rows=[]
    for line in path.read_text().splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: continue
    return pd.DataFrame(rows)

def _pct(series,p):
    x=pd.to_numeric(series,errors='coerce').dropna()
    return float(x.quantile(p)) if len(x) else None

def analyze(root:str|Path,out:str|Path,slo_ttft=3.0,slo_itl=.05,instance_usd_hour:float|None=None,ideal_cache_tokens:int|None=None):
    root=Path(root); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    dirs=[root] if (root/'requests.jsonl').exists() else sorted(p.parent for p in root.rglob('requests.jsonl'))
    summaries=[]; req_all=[]; sess_all=[]; metrics_all=[]
    for d in dirs:
        req=_read_jsonl(d/'requests.jsonl'); ses=_read_jsonl(d/'sessions.jsonl'); met=_read_jsonl(d/'metrics.jsonl')
        if len(req):
            if 'warmup' in req: req=req[~req['warmup'].fillna(False).astype(bool)].copy()
            if not len(req): continue
            req['run']=d.name; req_all.append(req)
            valid=req.get('tool_json_valid',pd.Series(False,index=req.index)).fillna(False).astype(bool)
            tt=pd.to_numeric(req.get('ttft_s'),errors='coerce'); itl=pd.to_numeric(req.get('mean_itl_s'),errors='coerce')
            good=(tt<=slo_ttft)&(itl<=slo_itl)&valid
            out_tokens=pd.to_numeric(req.get('completion_tokens'),errors='coerce').sum()
            duration=(pd.to_numeric(req.get('submit_time'),errors='coerce').max()-pd.to_numeric(req.get('submit_time'),errors='coerce').min()) if len(req)>1 else float(pd.to_numeric(req.get('e2e_s'),errors='coerce').sum())
            summary={'run':d.name,'path':str(d),'requests':len(req),'ttft_p50_s':_pct(tt,.5),'ttft_p99_s':_pct(tt,.99),'answer_p50_s':_pct(req.get('time_to_first_answer_s'),.5),'itl_p50_s':_pct(itl,.5),'itl_p95_s':_pct(itl,.95),'itl_p99_s':_pct(itl,.99),'e2e_p50_s':_pct(req.get('e2e_s'),.5),'e2e_p99_s':_pct(req.get('e2e_s'),.99),'tool_json_valid_rate':float(valid.mean()),'tool_schema_valid_rate':float(req.get('tool_schema_valid',pd.Series(False,index=req.index)).fillna(False).astype(bool).mean()),'goodput_requests_s':float(good.sum()/max(duration,1e-9)),'output_tokens_s':float(out_tokens/max(duration,1e-9)),'total_tokens_s':float((out_tokens+pd.to_numeric(req.get('prompt_tokens'),errors='coerce').sum())/max(duration,1e-9)),'cache_tokens':float(pd.to_numeric(req.get('cached_prompt_tokens'),errors='coerce').sum()) if 'cached_prompt_tokens' in req else None,'draft_tokens':float(pd.to_numeric(req.get('draft_tokens'),errors='coerce').sum()) if 'draft_tokens' in req else None,'accepted_draft_tokens':float(pd.to_numeric(req.get('accepted_draft_tokens'),errors='coerce').sum()) if 'accepted_draft_tokens' in req else None}
            if instance_usd_hour is not None and summary['output_tokens_s']>0: summary['cost_per_million_output_tokens_usd']=instance_usd_hour/(summary['output_tokens_s']*3600)*1_000_000
            summaries.append(summary)
        if len(ses):
            if 'warmup' in ses: ses=ses[~ses['warmup'].fillna(False).astype(bool)].copy()
            if len(ses): ses['run']=d.name; sess_all.append(ses)
        if len(met): met['run']=d.name; metrics_all.append(met)
    sdf=pd.DataFrame(summaries); rdf=pd.concat(req_all,ignore_index=True) if req_all else pd.DataFrame(); sesdf=pd.concat(sess_all,ignore_index=True) if sess_all else pd.DataFrame(); mdf=pd.concat(metrics_all,ignore_index=True) if metrics_all else pd.DataFrame()
    sdf.to_csv(out/'summary.csv',index=False)
    rdf.to_csv(out/'requests_flat.csv',index=False) if len(rdf) else None
    sesdf.to_csv(out/'sessions_flat.csv',index=False) if len(sesdf) else None
    mdf.to_csv(out/'metrics_flat.csv',index=False) if len(mdf) else None
    _plots(sdf,rdf,sesdf,mdf,out,ideal_cache_tokens)
    lines=['# Results','',f'Source run root: `{root}`','',f'SLO used for this report: request TTFT p99 target ≤ {slo_ttft:g}s; per-request mean ITL target ≤ {slo_itl*1000:g}ms; tool JSON valid.','', '## Run summary','']
    if len(sdf): lines += [sdf.to_markdown(index=False)]
    else: lines+=['No raw request logs found. Results are pending actual runs.']
    lines+=['','## Plots','', 'Plots are regenerated from the raw request/session/server metrics collected in the source run folders.','']
    for name in ('throughput_latency.png','goodput.png','latency_split.png','ablation_waterfall.png','cache_reuse.png','speculation.png','kv_preemptions.png','cost_quality.png'):
        if (out/name).exists(): lines.append(f'![{name}]({name})')
        else: lines.append(f'- `{name}`: pending required source metrics.')
    lines += ['', '## Interpretation','', 'No recommendation is generated from an empty or mock-only dataset. Label mock results as harness plumbing only. The trace metadata and per-run metadata are authoritative for exact inputs and environment.','']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    return {'runs':len(sdf),'output':str(out),'summary':sdf.to_dict('records')}

def _plots(sdf,rdf,sesdf,mdf,out,ideal):
    if len(sdf):
        fig,ax=plt.subplots(figsize=(8,5)); ax.scatter(sdf['output_tokens_s'],sdf['e2e_p99_s'],s=45); 
        for _,r in sdf.iterrows(): ax.annotate(r['run'],(r['output_tokens_s'],r['e2e_p99_s']))
        ax.set(xlabel='Output tokens / s',ylabel='Request E2E p99 (s)',title='Offered-load proxy vs tail latency'); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/'throughput_latency.png',dpi=160); plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,5)); ax.bar(sdf['run'],sdf['goodput_requests_s']); ax.set(ylabel='Good requests / s',title='SLO goodput'); ax.tick_params(axis='x',rotation=30); fig.tight_layout(); fig.savefig(out/'goodput.png',dpi=160); plt.close(fig)
        if 'cost_per_million_output_tokens_usd' in sdf:
            fig,ax=plt.subplots(figsize=(7,5)); success=sesdf.groupby('run')['task_success'].mean() if len(sesdf) and 'task_success' in sesdf else pd.Series(dtype=float)
            y=sdf['run'].map(success); ax.scatter(sdf['cost_per_million_output_tokens_usd'],y); ax.set(xlabel='USD / 1M output tokens',ylabel='Task success rate',title='Cost and task success'); fig.tight_layout(); fig.savefig(out/'cost_quality.png',dpi=160); plt.close(fig)
        if len(sdf)>1:
            baseline=sdf.iloc[0]['output_tokens_s']; vals=sdf['output_tokens_s']-baseline
            fig,ax=plt.subplots(figsize=(8,5)); ax.bar(sdf['run'],vals); ax.axhline(0,color='black',lw=.8); ax.set(ylabel='Output tokens / s change from first run',title='Ablation waterfall (ordered input runs)'); ax.tick_params(axis='x',rotation=30); fig.tight_layout(); fig.savefig(out/'ablation_waterfall.png',dpi=160); plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,5)); ax.bar(sdf['run'],sdf['ttft_p50_s'],label='TTFT p50'); ax.bar(sdf['run'],sdf['e2e_p50_s'],alpha=.55,label='E2E p50'); ax.set(ylabel='Seconds',title='Latency summary (queue/prefill/decode split needs server histograms)'); ax.legend(); ax.tick_params(axis='x',rotation=30); fig.tight_layout(); fig.savefig(out/'latency_split.png',dpi=160); plt.close(fig)
    if len(rdf):
        if 'cached_prompt_tokens' in rdf and pd.to_numeric(rdf['cached_prompt_tokens'],errors='coerce').notna().any():
            hit=pd.to_numeric(rdf['cached_prompt_tokens'],errors='coerce').sum(); ratio=hit/ideal if ideal else None
            fig,ax=plt.subplots(figsize=(6,4)); ax.bar(['Measured cached','Ideal reusable'],[hit,ideal or 0]); ax.set(ylabel='Prompt tokens',title=f'Prefix cache reuse (efficiency={ratio:.3f})' if ratio is not None else 'Prefix cache reuse'); fig.tight_layout(); fig.savefig(out/'cache_reuse.png',dpi=160); plt.close(fig)
        if 'accepted_draft_tokens' in rdf and 'draft_tokens' in rdf and pd.to_numeric(rdf['draft_tokens'],errors='coerce').notna().any():
            fig,ax=plt.subplots(figsize=(7,4)); ax.scatter(rdf['draft_tokens'],rdf['accepted_draft_tokens']); ax.set(xlabel='Draft tokens',ylabel='Accepted draft tokens',title='Speculative acceptance'); fig.tight_layout(); fig.savefig(out/'speculation.png',dpi=160); plt.close(fig)
    if len(mdf) and 'metric' in mdf:
        kv=mdf[mdf.metric.str.contains('kv_cache_usage|preemption',case=False,regex=True)]
        if len(kv):
            fig,ax=plt.subplots(figsize=(9,4));
            for key,g in kv.groupby('metric'): ax.plot(g['timestamp'],pd.to_numeric(g.value,errors='coerce'),label=key)
            ax.set(title='KV usage and preemption metrics over time',xlabel='Unix timestamp',ylabel='Value'); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out/'kv_preemptions.png',dpi=160); plt.close(fig)
    # Emit every chart slot with a useful explanation so the report always renders completely.
    placeholders={
      'throughput_latency.png':'Run multiple offered-load points to draw this curve.',
      'goodput.png':'No requests met the selected SLO or request data is unavailable.',
      'latency_split.png':'Queue, prefill and decode server histogram metrics are unavailable.',
      'ablation_waterfall.png':'At least two ordered configurations are required.',
      'cache_reuse.png':'Prompt cache counters and ideal reusable-token count are required.',
      'speculation.png':'Draft and accepted-draft counters are required.',
      'kv_preemptions.png':'KV usage or preemption server metrics are required.',
      'cost_quality.png':'Set hourly cost and collect session task-success values.'}
    for name,caption in placeholders.items():
        if not (out/name).exists():
            fig,ax=plt.subplots(figsize=(8,3)); ax.axis('off'); ax.text(.5,.5,caption,ha='center',va='center',wrap=True); fig.tight_layout(); fig.savefig(out/name,dpi=140); plt.close(fig)
