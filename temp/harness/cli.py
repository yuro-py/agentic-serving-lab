"""Command line interface for trace, run, experiment lifecycle, and analysis."""
from __future__ import annotations
import argparse, asyncio, hashlib, json, os, shlex, shutil, signal, subprocess, sys, time
from pathlib import Path
import yaml
from .traces import build_trace, write_jsonl, read_jsonl
from .runner import run_workload
from .analysis import analyze

ROOT=Path(__file__).resolve().parents[1]

def _json_write(path,data): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(data,indent=2,default=str)+'\n')

def cmd_trace(a):
    rows=build_trace(a.preset,a.sessions,a.seed,a.small,a.rate); digest=write_jsonl(rows,a.output)
    meta={'path':str(Path(a.output)),'sha256':digest,'seed':a.seed,'preset':a.preset,'sessions':len(rows),'small':a.small,'token_counting':'whitespace approximation; server tokenizer usage is recorded during serving'}
    _json_write(Path(a.output).with_suffix('.meta.json'),meta); print(json.dumps(meta,indent=2))

def cmd_run(a):
    trace=read_jsonl(a.trace); dest=Path(a.output); dest.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(Path(a.trace).read_bytes()).hexdigest()
    meta={'mode':a.mode,'trace':str(Path(a.trace).resolve()),'trace_sha256':digest,'seed':a.seed,'concurrency':a.concurrency,'rate_override':a.rate,'model':a.model,'base_url':a.base_url,'started_at':time.time(),'warmup_s':a.warmup}
    _json_write(dest/'run.json',meta)
    result=asyncio.run(run_workload(trace,dest,a.mode,a.base_url,a.api_key,a.model,a.concurrency,a.rate,a.seed,a.warmup,a.timeout)); meta.update(result); meta['finished_at']=time.time(); _json_write(dest/'run.json',meta); print(json.dumps(meta,indent=2))

def _validate_flags(server_help:str,argv:list[str]):
    unknown=[]
    for x in argv:
        if x.startswith('--') and x.split('=',1)[0] not in server_help: unknown.append(x.split('=',1)[0])
    if unknown: raise SystemExit('These configured vLLM flags are absent from this installed `vllm serve --help`: '+', '.join(sorted(set(unknown))))

def _expand_args(args:list[str],env:dict[str,str])->list[str]:
    import re
    expanded=[]
    for arg in args:
        expanded.append(re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}',lambda m:env.get(m.group(1),''),str(arg)))
    return expanded

def _gpu_loggers(folder:Path):
    procs=[]
    if shutil.which('nvidia-smi'):
        q='gpu=timestamp,index,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu'
        procs.append(subprocess.Popen(['nvidia-smi',f'--query-gpu={q}','--format=csv','-l','1'],stdout=(folder/'gpu.csv').open('w'),stderr=subprocess.STDOUT,start_new_session=True))
        procs.append(subprocess.Popen(['nvidia-smi','dmon','-s','pucvmet','-d','1'],stdout=(folder/'dmon.csv').open('w'),stderr=subprocess.STDOUT,start_new_session=True))
    return procs

def cmd_config(a):
    cfg=yaml.safe_load(Path(a.config).read_text()); exp=cfg.get('experiment','E0'); name=cfg.get('name',Path(a.config).stem); root=Path(cfg.get('runs_dir','runs')); repeat=a.repeat or 1
    for rep in range(repeat):
        out=root/exp/name/f'repeat-{rep+1:02d}'; out.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy(); env.update({str(k):str(v) for k,v in cfg.get('environment',{}).items()})
        if a.api_key: env['API_KEY']=a.api_key
        key=env.get('API_KEY','EMPTY'); base=cfg.get('base_url','http://127.0.0.1:8000'); model=cfg.get('model','unsloth/Qwen3.8-27B-NVFP4')
        command=cfg.get('serve_command')
        if not command: command=['vllm','serve',model]+_expand_args(list(cfg.get('args',[])),env)
        elif isinstance(command,str): command=shlex.split(command)
        command=_expand_args(command,env)
        # Do not dump API key into the saved command.
        safe_command=[('<API_KEY>' if x==key and key!='EMPTY' else x) for x in command]
        _json_write(out/'config.resolved.json',{'config':cfg,'command':safe_command,'environment':{k:v for k,v in env.items() if k in cfg.get('environment',{}) or k in ('CUDA_VISIBLE_DEVICES','NCCL_DEBUG')},'trace':str(Path(a.trace).resolve()) if a.trace else None,'repeat':rep+1})
        helpres=subprocess.run(['vllm','serve','--help'],capture_output=True,text=True,env=env)
        if helpres.returncode: raise SystemExit(f"Could not inspect this vLLM install's serve flags: {helpres.stderr[-2000:]}")
        _validate_flags(helpres.stdout,command[2:] if len(command)>2 and command[:2]==['vllm','serve'] else command)
        with (out/'serve_help.txt').open('w') as f: f.write(helpres.stdout)
        server_log=(out/'server.log').open('w')
        proc=subprocess.Popen(command,stdout=server_log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        procs=[]
        try:
            deadline=time.time()+float(cfg.get('health_timeout_s',1800))
            while time.time()<deadline:
                if proc.poll() is not None: raise RuntimeError(f'vLLM server exited early with status {proc.returncode}; inspect {out}/server.log')
                try:
                    import urllib.request
                    with urllib.request.urlopen(base+'/health',timeout=3): break
                except Exception: time.sleep(2)
            else: raise TimeoutError('Timed out waiting for server /health')
            with (out/'startup.log').open('w') as f:
                f.write(f"healthy_at={time.time()}\n"); f.flush()
            try:
                import urllib.request
                with urllib.request.urlopen(base+'/metrics',timeout=10) as r: metrics=r.read().decode()
                names=sorted({line.split('{',1)[0] for line in metrics.splitlines() if line.startswith('vllm:')})
                (out/'metric_names.txt').write_text('\n'.join(names)+'\n')
                _json_write(out/'metric_names.meta.json',{'source':'live /metrics','count':len(names)})
            except Exception as e: (out/'metrics_snapshot_error.log').write_text(str(e))
            procs=_gpu_loggers(out)
            if a.trace and a.mode!='mock':
                run_mode=a.mode; run_workload(read_jsonl(a.trace),out,run_mode,base,key,model,a.concurrency,None,a.seed,cfg.get('warmup_s',120),a.timeout)
            elif a.trace:
                run_workload(read_jsonl(a.trace),out,'mock',base,key,model,a.concurrency,None,a.seed,0,a.timeout)
            else: print(f'Server healthy. Logs: {out}; no trace supplied, so workload was skipped.')
        finally:
            for p in procs:
                try: p.terminate(); p.wait(timeout=5)
                except Exception: p.kill()
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try: proc.wait(timeout=45)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            server_log.close()
        _capture_metadata(out,cfg,command,env)

def _capture_metadata(out,cfg,command,env):
    def capture(cmd):
        try: return subprocess.run(cmd,capture_output=True,text=True,timeout=60).stdout
        except Exception as e: return f'Unavailable: {e}\n'
    (out/'host_facts.txt').write_text('\n'.join([f"host={os.uname().nodename}",capture(['nvidia-smi']),capture(['nvidia-smi','topo','-m']),capture(['lscpu']),capture(['free','-h']),capture(['df','-h'])]))
    (out/'versions.txt').write_text(capture([sys.executable,'-m','pip','freeze']))
    _json_write(out/'run_metadata.json',{'started_by':'agentic-serving-lab','finished_at':time.time(),'python':sys.version,'command':[('<API_KEY>' if x.startswith('--api-key=') else x) for x in command],'environment':{k:env.get(k) for k in cfg.get('environment',{})},'host':os.uname().nodename})

def cmd_env(_a):
    dest=ROOT/'env'; dest.mkdir(exist_ok=True)
    for name,cmd in [('nvidia-smi.txt',['nvidia-smi']),('topology.txt',['nvidia-smi','topo','-m']),('cpu.txt',['lscpu']),('memory.txt',['free','-h']),('disk.txt',['df','-h']),('pip-freeze.txt',[sys.executable,'-m','pip','freeze'])]:
        try: r=subprocess.run(cmd,capture_output=True,text=True,timeout=60); (dest/name).write_text(r.stdout+r.stderr)
        except Exception as e: (dest/name).write_text(f'Unavailable: {e}\n')
    (dest/'versions.txt').write_text(f"python={sys.version}\nhost={os.uname().nodename}\n")
    print(f'Captured host facts under {dest}')

def main():
    p=argparse.ArgumentParser(prog='agentic-lab',description='Reproducible agentic serving experiments'); sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('trace',help='Generate a deterministic agent workload trace'); t.add_argument('--preset',default='M',choices=['A','B','C','D','M']); t.add_argument('--sessions',type=int,default=100); t.add_argument('--seed',type=int,default=2026); t.add_argument('--rate',type=float,default=.1); t.add_argument('--small',action='store_true'); t.add_argument('--output',default='traces/mixed.jsonl'); t.set_defaults(func=cmd_trace)
    r=sub.add_parser('run',help='Run a trace against mock, live, or replay backend'); r.add_argument('--trace',required=True); r.add_argument('--mode',default='mock',choices=['mock','live','replay']); r.add_argument('--output',required=True); r.add_argument('--base-url',default=os.getenv('BASE_URL','http://127.0.0.1:8000')); r.add_argument('--api-key',default=os.getenv('API_KEY','EMPTY')); r.add_argument('--model',default=os.getenv('MODEL','unsloth/Qwen3.8-27B-NVFP4')); r.add_argument('--concurrency',type=int,default=8); r.add_argument('--rate',type=float); r.add_argument('--seed',type=int,default=2026); r.add_argument('--warmup',type=float,default=0); r.add_argument('--timeout',type=float,default=3600); r.set_defaults(func=cmd_run)
    c=sub.add_parser('run-config',help='Start a configured vLLM server, run workload and collect logs'); c.add_argument('config'); c.add_argument('--trace'); c.add_argument('--mode',default='replay',choices=['live','replay','mock']); c.add_argument('--repeat',type=int,default=1); c.add_argument('--api-key',default=os.getenv('API_KEY','')); c.add_argument('--concurrency',type=int,default=8); c.add_argument('--seed',type=int,default=2026); c.add_argument('--timeout',type=float,default=3600); c.set_defaults(func=cmd_config)
    a=sub.add_parser('analyze',help='Regenerate summary and plots from raw runs'); a.add_argument('runs'); a.add_argument('--output',default='analysis/results'); a.add_argument('--slo-ttft',type=float,default=3); a.add_argument('--slo-itl-ms',type=float,default=50); a.add_argument('--hourly-cost',type=float); a.add_argument('--ideal-cache-tokens',type=int); a.set_defaults(func=lambda x:print(json.dumps(analyze(x.runs,x.output,x.slo_ttft,x.slo_itl_ms/1000,x.hourly_cost,x.ideal_cache_tokens),indent=2)))
    e=sub.add_parser('env',help='Capture GPU, CPU, memory, disk, and Python package facts'); e.set_defaults(func=cmd_env)
    args=p.parse_args(); args.func(args)

if __name__=='__main__': main()
