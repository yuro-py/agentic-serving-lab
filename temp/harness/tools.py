"""Deterministic mock tool implementations with bounded, safe execution."""
from __future__ import annotations
import ast, hashlib, json, operator, re

OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod,ast.Pow:operator.pow,ast.USub:operator.neg,ast.UAdd:operator.pos}
def safe_calc(expr: str) -> str:
    def visit(n):
        if isinstance(n,ast.Expression): return visit(n.body)
        if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)): return n.value
        if isinstance(n,ast.UnaryOp) and type(n.op) in OPS: return OPS[type(n.op)](visit(n.operand))
        if isinstance(n,ast.BinOp) and type(n.op) in OPS:
            a,b=visit(n.left),visit(n.right)
            if isinstance(n.op,ast.Pow) and abs(b)>10: raise ValueError('exponent too large')
            return OPS[type(n.op)](a,b)
        raise ValueError('unsupported expression')
    value=visit(ast.parse(expr,mode='eval'))
    if abs(value)>10**100: raise ValueError('result too large')
    return str(value)

async def run_tool(name: str,args: dict, session: dict, step: dict) -> str:
    await __import__('asyncio').sleep(max(0,float(step.get('tool_delay_s',0))))
    expected=session.get('expected_answer','')
    if name in ('calculator','run_python'):
        try: return safe_calc(str(args.get('expression','0')))
        except Exception as e: return f"calculation error: {e}"
    if name=='final_answer': return str(args.get('answer',''))
    q=str(args.get('query',args.get('path','')))
    return f"Synthetic {name} result for {q}. The exact value is {expected}."

def schema_valid(name: str,args: dict,tools: list[dict]) -> bool:
    spec=next((t['function'] for t in tools if t.get('function',{}).get('name')==name),None)
    if not spec or not isinstance(args,dict): return False
    params=spec.get('parameters',{}); required=params.get('required',[])
    if not all(k in args for k in required): return False
    props=params.get('properties',{})
    return all(k in props and (props[k].get('type')!='string' or isinstance(v,str)) for k,v in args.items())

def ideal_prefix_tokens(rows: list[dict]) -> int:
    # Exact content tokens that repeat across sessions; heuristic tokenization is documented.
    seen=set(); total=0; reusable=0
    for s in rows:
        for m in s.get('prefix',[]):
            text=m.get('content') or ''; n=len(text.split()); total+=n
            key=hashlib.sha256(text.encode()).hexdigest()
            if key in seen: reusable+=n
            else: seen.add(key)
    return reusable
