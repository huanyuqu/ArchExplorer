"""Explicit per-tile MAC/SRAM declarations for physically blocked workloads.

The default has exactly 64 compute instances and 64 one-MiB SRAM instances.
No root SRAM, root compute or host memory is inserted. External memory exists
only when explicitly requested as a separately named physical candidate.
"""
import math
from arch_explorer.adf import Param, SpacePoint, SpaceMatrix
from arch_explorer.components.api import ImplementationBinding

SOURCE='explicit uncalibrated MAC/SRAM/tree declaration; no fitted device timing'


def block_memory_hardware(*,tiles=64,capacity=1024**2,mac_rows=16,mac_cols=16,
                          simt_lanes=8,sfu_lanes=1,clock_ghz=1.0,dataflow='os',
                          external_bytes=None,link_bandwidth=16.0,port_bandwidth=16.0,hop_latency=1.0):
    """Return (ADF, complete slot/compute/memory/network metadata).

    The two bandwidths are explicit bytes/ns per physical link and port. The
    clock is an explicit cycles/ns declaration. None is a measured device rate.
    external_bytes=None is the original onchip candidate, with no DRAM point.
    """
    if (type(tiles) is not int or not 1<=tiles<=64 or type(capacity) is not int or capacity<1
            or type(mac_rows) is not int or mac_rows not in (8,16,32,64,128,256)
            or type(mac_cols) is not int or mac_cols not in (8,16,32,64,128,256)
            or any(type(v) is not int or not 1<=v<=1024 for v in (simt_lanes,sfu_lanes))
            or dataflow not in ('os','ws','is')
            or external_bytes is not None and (type(external_bytes) is not int or external_bytes<1)):
        raise ValueError('block.hardware: invalid explicit tile/SRAM/MAC/scalar/external declaration')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0
           for v in (clock_ghz,link_bandwidth,port_bandwidth)) or type(hop_latency) not in (int,float) or not math.isfinite(hop_latency) or hop_latency<0:
        raise ValueError('block.hardware: clock/bandwidth must be positive and hop latency nonnegative')
    def p(value,unit):return Param(value,unit,SOURCE)
    def tree():return SpacePoint('tree','multicast_tree',dict(link_bandwidth=p(link_bandwidth,'B/ns'),
        port_bandwidth=p(port_bandwidth,'B/ns'),hop_latency=p(hop_latency,'ns')))
    compute=SpacePoint('sm','sm',dict(precision=p('fp32','1')),implementation=ImplementationBinding('sm.mac_events','1',
        dict(rows=p(mac_rows,'count'),cols=p(mac_cols,'count'),dataflow=p(dataflow,'1'),
             simt_lanes=p(simt_lanes,'count'),sfu_lanes=p(sfu_lanes,'count'),clock_ghz=p(clock_ghz,'GHz'))))
    memory=SpacePoint('sram','sram',dict(capacity=p(capacity,'B'),bandwidth=p(port_bandwidth,'B/ns'),latency=p(1.0,'ns')))
    children=[SpaceMatrix('tile',[compute,memory],[tree()],repeat=tiles)]
    if external_bytes is not None:
        children.append(SpacePoint('external','dram',dict(capacity=p(external_bytes,'B'),
            bandwidth=p(port_bandwidth,'B/ns'),latency=p(4.0,'ns'),technology=p('hbm','1'))))
    adf=SpaceMatrix('board',children,[tree()])
    from arch_explorer.adf.dynamic import instances
    actual=dict(instances(adf))
    computes=[path for path,point in actual.items() if point.component=='sm']
    memories=[path.rsplit('/',1)[0]+'/sram' for path in computes]
    assert len(computes)==tiles and all(path in actual for path in memories)
    external=None if external_bytes is None else next(path for path,point in actual.items() if point.component=='dram')
    networks=[dict(path=path,declaration=point.to_dict()) for path,point in actual.items() if point.component=='multicast_tree']
    metadata=dict(schema='arch-explorer.block-memory-hardware.v1',
        candidate='onchip-only' if external_bytes is None else 'explicit-external-memory',
        slot_order=[dict(slot=i,compute=c,memory=m) for i,(c,m) in enumerate(zip(computes,memories))],
        compute=computes,memories=memories,compute_storage=dict(zip(computes,memories)),
        external_memory=external,external_bytes=external_bytes or 0,
        per_slot_capacity_bytes=capacity,total_sram_bytes=tiles*capacity,
        mac_rows=mac_rows,mac_cols=mac_cols,simt_lanes=simt_lanes,sfu_lanes=sfu_lanes,
        clock_ghz=clock_ghz,dataflow=dataflow,networks=networks,
        network_contract='ordinary multicast_tree scope, endpoint order, ports and per-edge accounting; no free broadcast',
        implementation=dict(id='sm.mac_events',version='1',
            register='arch_explorer.components.sm.implementations.mac_events.install'),
        source=SOURCE,hardware_accuracy=None)
    return adf,metadata
