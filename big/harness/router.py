"""Simple routing policies for paired replica experiments (library helper)."""
from __future__ import annotations
import itertools, hashlib

class ReplicaRouter:
    def __init__(self,base_urls,policy='round-robin'):
        if len(base_urls)<1: raise ValueError('at least one replica is required')
        if policy not in ('round-robin','least-loaded','session-sticky'): raise ValueError('unknown policy')
        self.urls=list(base_urls); self.policy=policy; self.rr=itertools.cycle(range(len(base_urls))); self.loads=[0]*len(base_urls); self.sticky={}
    def choose(self,session_id):
        if self.policy=='session-sticky':
            if session_id not in self.sticky: self.sticky[session_id]=int(hashlib.sha256(session_id.encode()).hexdigest(),16)%len(self.urls)
            return self.urls[self.sticky[session_id]]
        i=next(self.rr) if self.policy=='round-robin' else min(range(len(self.urls)),key=lambda x:(self.loads[x],x))
        return self.urls[i]
    def started(self,url): self.loads[self.urls.index(url)]+=1
    def finished(self,url): self.loads[self.urls.index(url)]=max(0,self.loads[self.urls.index(url)]-1)
