"""Independent heap-tree and cut-through conservation oracle for SIM50.

Derives routes from endpoint indices and integer parent chains, never a
production topology builder/resolver. Does not tune against paper timings.
"""
from fractions import Fraction
import math


def endpoint_path(count,source,destination):
    if type(count) is not int or count <= 0:
        raise ValueError('positive endpoint count required')
    if any(type(x) is not int or not -1 <= x < count for x in (source,destination)):
        raise ValueError('endpoint outside actual child expansion')
    leaves = 1 << (count-1).bit_length()
    def ancestors(endpoint):
        node = 1 if endpoint == -1 else leaves+endpoint
        chain = []
        while node:
            chain.append(node);node //= 2
        return chain
    left,right = ancestors(source),ancestors(destination)
    common = next(n for n in left if n in right)
    return left[:left.index(common)+1]+list(reversed(right[:right.index(common)]))


def tree_transfer(count,source,destinations,size,*,path,link_bandwidth,port_bandwidth,
                  hop_latency,multicast):
    if (not destinations or len(set(destinations)) != len(destinations)
            or source in destinations):
        raise ValueError('distinct non-source destinations required')
    if (isinstance(size,bool) or not isinstance(size,(int,float,Fraction))
            or not math.isfinite(size) or size <= 0):
        raise ValueError('positive finite byte amount required')
    for value in (link_bandwidth,port_bandwidth):
        if not math.isfinite(value) or value <= 0:raise ValueError('finite positive bandwidth required')
    if not math.isfinite(hop_latency) or hop_latency < 0:raise ValueError('nonnegative finite latency required')
    routes = [endpoint_path(count,source,dest) for dest in destinations]
    batches = [list(zip(route,route[1:])) for route in routes]
    if multicast:
        batches = [set(edge for edges in batches for edge in edges)]
    demand = {}
    for edges in batches:
        for a,b in edges:
            resource = f'{path}:edge{a}_to_{b}'
            demand[resource] = demand.get(resource,0)+size
    demand[f'{path}:out{source}'] = size if multicast else size*len(destinations)
    for dest in destinations:demand[f'{path}:in{dest}'] = size
    capacities = {name:(port_bandwidth if ':out' in name or ':in' in name else link_bandwidth)
                  for name in demand}
    service = {name:Fraction(amount)/Fraction(capacities[name]) for name,amount in demand.items()}
    return dict(demand_bytes=demand,capacity_bytes_per_ns=capacities,
                isolated_cutthrough_busy_ns=max(service.values(),default=Fraction(0)),
                startup_tail_ns=(max(len(route)-1 for route in routes)*Fraction(hop_latency)),
                note='isolated fluid service plus startup tail; real contending schedule may take longer')


def hierarchical_transfer(adf,source,destinations,size,*,multicast):
    """Derive actual paths from plain ADF JSON, independent of routing models.

    Every direct child, including a nested matrix, occupies one parent endpoint.
    Repeats expand in declaration order. Network entries occupy no endpoints.
    Memory read/write ports are combined resources named <memory>:port.
    """
    if not destinations or len(set(destinations))!=len(destinations) or source in destinations:
        raise ValueError('distinct physical target memories required')
    containers,points = {},{}
    def walk(node,path):
        if node['type']=='point':
            points[path]=node;return
        endpoints=[]
        for child in node.get('elements',[]):
            repeat=child.get('repeat',1)
            if type(repeat) is not int or repeat<=0:raise ValueError('positive repeat required')
            for i in range(repeat):
                name=child['name']+(f'[{i}]' if repeat>1 else '')
                target=path+'/'+name;endpoints.append(target);walk(child,target)
        containers[path]=(endpoints,node.get('networks',[]))
    walk(adf,adf['name'])
    if source not in points or any(dest not in points for dest in destinations):
        raise ValueError('explicit physical memory point required')
    def param(node,name,unit):
        value=node['params'][name]
        if value['unit']!=unit or not value.get('source'):raise ValueError('sourced matching unit required')
        return value['value']
    demand,capacities={},{}
    def charge(name,amount,rate):
        demand[name]=demand.get(name,0)+amount;capacities[name]=rate
    charge(source+':port',size*(1 if multicast else len(destinations)),param(points[source],'bandwidth','B/ns'))
    for dest in destinations:charge(dest+':port',size,param(points[dest],'bandwidth','B/ns'))
    tails={dest:Fraction(param(points[source],'latency','ns')) for dest in destinations}
    for path,(endpoints,networks) in containers.items():
        def endpoint(memory):
            return next((i for i,child in enumerate(endpoints) if memory==child or memory.startswith(child+'/')),-1)
        src=endpoint(source)
        active=[dest for dest in destinations if endpoint(dest)!=src]
        if not active:continue
        if len(networks)!=1 or networks[0]['component']!='multicast_tree':
            raise ValueError('one declared multicast tree per crossed container required')
        network=networks[0];network_path=path+'/'+network['name']
        bandwidth=param(network,'link_bandwidth','B/ns')
        port=param(network,'port_bandwidth','B/ns')
        hop=param(network,'hop_latency','ns')
        batches=[list(dict.fromkeys(endpoint(dest) for dest in active))] if multicast else [[endpoint(dest)] for dest in active]
        for targets in batches:
            piece=tree_transfer(len(endpoints),src,targets,size,path=network_path,
                                link_bandwidth=bandwidth,port_bandwidth=port,
                                hop_latency=hop,multicast=multicast)
            for resource,amount in piece['demand_bytes'].items():
                charge(resource,amount,piece['capacity_bytes_per_ns'][resource])
        for dest in active:tails[dest]+=(len(endpoint_path(len(endpoints),src,endpoint(dest)))-1)*Fraction(hop)
    return dict(demand_bytes=demand,capacity_bytes_per_ns=capacities,
                isolated_cutthrough_busy_ns=max(Fraction(n)/Fraction(capacities[k]) for k,n in demand.items()),
                startup_tail_ns=max(tails.values()))
