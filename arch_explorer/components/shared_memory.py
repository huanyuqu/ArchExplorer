"""Small synthetic grouped/private storage declarations, not timing formulas."""
from arch_explorer.adf import Param,SpacePoint,SpaceMatrix
from arch_explorer.components.api import ImplementationBinding

SOURCE='explicit synthetic grouped storage/tree service; not a paper measurement or device calibration'


def shared_memory_hardware(*,groups=2,tiles_per_group=2,shared=True,capacity=32768,
                           link_bandwidth=16.0,port_bandwidth=16.0,hop_latency=1.0):
    """Return (ADF, topology); topology contains actual paths, never path patterns."""
    if (any(type(n) is not int or not 1<=n<=64 for n in (groups,tiles_per_group))
            or type(shared) is not bool or type(capacity) is not int or capacity<1):
        raise ValueError('synthetic topology needs groups/tiles 1..64, bool shared and positive byte capacity')
    def p(v,u):return Param(v,u,SOURCE)
    def tree():return SpacePoint('tree','multicast_tree',dict(link_bandwidth=p(link_bandwidth,'B/ns'),
                           port_bandwidth=p(port_bandwidth,'B/ns'),hop_latency=p(hop_latency,'ns')))
    def memory(n,cap,dram=False):
        params=dict(capacity=p(cap,'B'),bandwidth=p(port_bandwidth,'B/ns'),latency=p(4.0 if dram else 1.0,'ns'))
        if dram:params['technology']=p('hbm','1')
        return SpacePoint(n,'dram' if dram else 'sram',params)
    def compute():return SpacePoint('sm','sm',dict(precision=p('fp32','1')),implementation=ImplementationBinding('sm.events','1',
        dict(lanes=p(8,'count'),sfu_lanes=p(1,'count'),clock_ghz=p(1.0,'GHz'))))
    core=SpaceMatrix('tile',[compute(),*([] if shared else [memory('sram',capacity)])],[tree()],repeat=tiles_per_group)
    cluster=SpaceMatrix('group',([memory('sram',capacity)] if shared else [])+[core],[tree()],repeat=groups)
    adf=SpaceMatrix('board',[memory('dram',2**26,True),memory('root_sram',capacity),
                            SpaceMatrix('root',[compute()],[tree()]),cluster],[tree()])
    from arch_explorer.adf.dynamic import instances
    actual=dict(instances(adf))
    # Repeat==1 paths have no index. Match actual instances rather than guessing.
    metadata={};compute_storage={}
    group_paths=sorted({path.rsplit('/tile',1)[0] for path,point in actual.items() if point.component=='sm' and '/group' in path})
    for i,path in enumerate(group_paths):
        computes=sorted(n for n,pnt in actual.items() if pnt.component=='sm' and n.startswith(path+'/'))
        stores={c:(path+'/sram' if shared else c.rsplit('/',1)[0]+'/sram') for c in computes}
        compute_storage.update(stores);metadata['group'+str(i)]=dict(computes=computes,storage=stores)
    topology=dict(schema='arch-explorer.shared-memory-topology.v1',groups=metadata,compute_storage=compute_storage,
                  root_compute='board/root/sm',root_memory='board/root_sram',dram='board/dram',
                  source=SOURCE,shared=shared)
    return adf,topology
