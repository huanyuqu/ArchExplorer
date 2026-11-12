"""Synthetic directed binary tree with explicit link and endpoint service.

Leaves follow actual container endpoint order; the gateway attaches to node 1.
Transfers reserve all required links as a cut-through fluid flow, followed by
the maximum hop latency tail. Packet/store-forward timing is outside this model.
"""
from arch_explorer.adf.registry import ComponentType,ParamSpec
from arch_explorer.components.base import GATEWAY,Demand,NetworkModel
from arch_explorer.components._support.network import FAIR


class TreeModel(NetworkModel):
    def __init__(self,component,point,context):
        super().__init__(component,point,context)
        self.bandwidth=self.positive('link_bandwidth')
        self.port_bandwidth=self.positive('port_bandwidth')
        self.hop_latency=self.param('hop_latency')
        if self.hop_latency<0:raise ValueError('tree hop latency must be nonnegative')
        self.leaves=1<<max(0,(self.endpoints-1).bit_length())
        self._resources={}

    def _resource(self,key,capacity):
        if key not in self._resources:self._resources[key]=self.resource(key,capacity,'B/ns')
        return self._resources[key].name

    def edges(self,src,dst):
        self._check(src,dst)
        a=1 if src==GATEWAY else self.leaves+src
        b=1 if dst==GATEWAY else self.leaves+dst
        up=[];down=[]
        while a!=b:
            if a>b:up.append((a,a//2));a//=2
            else:down.append((b//2,b));b//=2
        return up+list(reversed(down))

    def route(self,src,dst):return self.route_many(src,[dst])

    def route_many(self,src,destinations):
        targets=tuple(destinations)
        if not targets or len(set(targets))!=len(targets):raise ValueError('multicast needs nonempty unique destinations')
        self._check(src,*targets)
        targets=tuple(d for d in targets if d!=src)
        if not targets:return Demand()
        links={edge for dst in targets for edge in self.edges(src,dst)}
        resources={self._resource(f'edge{a}_to_{b}',self.bandwidth):1.0 for a,b in sorted(links)}
        resources[self._resource('out'+str(src),self.port_bandwidth)]=1.0
        for d in targets:resources[self._resource('in'+str(d),self.port_bandwidth)]=1.0
        return Demand(resources,max(len(self.edges(src,d)) for d in targets)*self.hop_latency)


class MulticastTree(ComponentType):
    name='multicast_tree'
    role='network'
    params={'link_bandwidth':ParamSpec('B/ns'),'port_bandwidth':ParamSpec('B/ns'),'hop_latency':ParamSpec('ns')}
    conventions=FAIR
    def build(self,point,context):return TreeModel(self,point,context)
