"""Independent values, traffic, synchronization and capacity checks."""
import json

import numpy as np
import pytest

from arch_explorer import devices
from arch_explorer.adf import MappingSpec, Param, WorkloadSpec, evaluate, from_dict
from arch_explorer.components.base import Demand
from arch_explorer.mapping.context import MappingContext
from arch_explorer.mapping.manycore import _Schedule, matrix_jobs
from arch_explorer.simulator.events.backend import run_mapping
from arch_explorer.simulator.events.builder import build_simulator
from arch_explorer.simulator.events.engine import simulate
from arch_explorer.workload.llm import workload_from_spec


def small(name="ipu_gc200", count=2, **kwargs):
    adf = devices.get(name, **kwargs)
    adf.elements[0].repeat = count
    net = adf.networks[0].params
    net["tiles"] = Param(count, "count", "synthetic test")
    if "rows" in net:
        net["rows"], net["cols"] = Param(1, "count", "synthetic"), Param(count, "count", "synthetic")
    return adf


def matrix(dtype="fp16", **extra):
    return WorkloadSpec("operator", dict(op="matmul", m=2, k=3, n=4, dtype=dtype, **extra))


@pytest.mark.parametrize("device,template,dtype", [("ipu_gc200", "ipu_bsp", "fp16"),
                                                   ("tianjicx", "tianjic_spatial", "int8")])
def test_hand_compute_traffic_and_cost_boundary(device, template, dtype):
    adf = small(device, count=1)
    sim, result = run_mapping(adf, matrix(dtype), MappingSpec(template, {"tile_m": 4, "tile_n": 32}))
    d = sim.mapping.details["manycore"]
    # 6 A +12 B elements staged once, output8 elements, disjoint gather/compute/scatter.
    width, out_width = (2, 2) if dtype == "fp16" else (1, 4)
    inputs, outputs = 18*width, 8*out_width
    assert d["tiles"][0]["local_read"] == 2*inputs+outputs
    assert d["tiles"][0]["local_write"] == inputs+2*outputs
    unit = sim.find_points(role="compute")[0]
    expected_compute = 24/(unit.rate*unit.clock_ghz)
    expected_sram = (inputs+outputs)/sim.find_points(component="sram")[0].port.capacity
    # Each staged copy reads and writes the same port, separately from compute.
    expected = 2*(inputs+outputs)/sim.find_points(component="sram")[0].port.capacity
    expected += max(expected_compute, expected_sram)+3*unit.sync_cycles/unit.clock_ghz
    assert result.latency_ns == pytest.approx(expected)
    r = evaluate(adf, matrix(dtype), MappingSpec(template, {}))
    assert r.cost_usd is None and r.area_mm2 is None
    assert "partial analytical" in r.details["manycore"]["estimate"]
    assert from_dict(json.loads(json.dumps(adf.to_dict()))).to_dict() == adf.to_dict()


def test_network_independence_contention_and_zero_bytes():
    sim = build_simulator(small(count=4))
    ctx = sim.mapping = MappingContext(sim)
    net = sim.root.networks[0]
    # Two distinct sending and receiving endpoints overlap at the same rate.
    ctx.task("a", net.route(0, 1).scaled(53), kind="comm")
    ctx.task("b", net.route(2, 3).scaled(53), kind="comm")
    assert simulate(ctx.graph).latency_ns == pytest.approx(10)
    ctx = MappingContext(sim)
    ctx.task("a", net.route(0, 1).scaled(53), kind="comm")
    ctx.task("b", net.route(0, 3).scaled(53), kind="comm")
    assert simulate(ctx.graph).latency_ns == pytest.approx(20)
    assert net.route(1, 1).resources == {}
    assert all(v == 0 for v in net.route(0, 1).scaled(0).resources.values())
    schedule = _Schedule(sim, workload_from_spec(matrix()), {"tile_m": 1, "tile_n": 32}, "ipu_tile")
    before = len(sim.mapping.graph.tasks)
    assert schedule.move(0, 1, 0, [], "test", "zero") is None
    assert len(sim.mapping.graph.tasks) == before


def test_bsp_global_phase_dependencies_and_capacity():
    adf = small()
    sim, result = run_mapping(adf, matrix(), MappingSpec("ipu_bsp", {"tile_m": 1, "tile_n": 32}))
    records = {r.task: r for r in result.records}
    stages = sim.mapping.details["manycore"]["stages"]
    last = 0
    for stage in stages:
        for task in stage["tasks"]:
            assert records[task].start_ns >= last-1e-9
        last = max([records[t].end_ns for t in stage["tasks"]] or [last])
    mem = adf.elements[0].elements[1]
    mem.params["capacity"] = Param(64, "B", "synthetic")
    with pytest.raises(ValueError, match="staging capacity"):
        evaluate(adf, matrix(), MappingSpec("ipu_bsp", {}))
    mem.params["capacity"] = Param(1, "B", "synthetic")
    with pytest.raises(ValueError, match="resident capacity"):
        evaluate(adf, matrix(), MappingSpec("ipu_bsp", {}))


@pytest.mark.parametrize("op,shared", [("matmul", False), ("matmul", True), ("mvm", False)])
def test_mapping_value_dependencies_against_numpy(op, shared):
    p = dict(op=op, m=5, k=7, batch=2, dtype="int8")
    if op == "matmul":
        p.update(n=9, shared_b=shared)
    ir = workload_from_spec(WorkloadSpec("operator", p))
    operation = ir.operators[0]
    rng = np.random.default_rng(17)
    values = {name: rng.integers(-3, 4, size=t.shape, dtype=np.int32) for name, t in ir.tensors.items()}
    output = np.zeros(ir.tensors[operation.outputs[0]].shape, dtype=np.int32).ravel()
    coverage = np.zeros(output.size, dtype=int)
    for j in matrix_jobs(operation, 2, 4):
        gathered = [np.concatenate([values[name].ravel()[start:start+size] for start, size in ranges])
                    for name, ranges in j["inputs"]]
        _, m, k, n = j["shape"]
        block = (gathered[0].reshape(m, k) @ gathered[1].reshape(k, n) if op == "matmul"
                 else gathered[1].reshape(1, k) @ gathered[0].reshape(n, k).T).ravel()
        offset = 0
        for start, size in j["outputs"]:
            output[start:start+size] = block[offset:offset+size]
            coverage[start:start+size] += 1
            offset += size
    expected = values["A"] @ values["B"] if op == "matmul" else values["x"] @ values["A"].T
    np.testing.assert_array_equal(output, expected.ravel())
    assert np.all(coverage == 1)


def test_decode_attention_semantics_and_shapes():
    w = WorkloadSpec("operator", dict(op="attention", batch=2, heads=2, head_dim=3, seq_len=5))
    ir = workload_from_spec(w)
    assert [op.name for op in ir.operators] == ["QK", "scale", "softmax", "AV"]
    assert ir.tensors["scores"].shape == (4, 1, 5)  # not S x S
    rng = np.random.default_rng(11)
    q, k, v = rng.normal(size=(4, 1, 3)), rng.normal(size=(4, 3, 5)), rng.normal(size=(4, 5, 3))
    logits = (q @ k)*ir.meta["scale"]
    probs = np.exp(logits-logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    # Reconstruct both GEMMs from the actual mapping's source/destination intervals.
    # This checks mathematical dependence, not a deployable hardware instruction stream.
    values = {"Q": q, "K": k, "V": v}
    for op in ir.operators:
        if op.op == "matmul":
            result = np.zeros(ir.tensors[op.outputs[0]].shape).ravel()
            for job in matrix_jobs(op, 1, 2):
                operands = [np.concatenate([values[name].ravel()[s:s+n] for s, n in ranges])
                            for name, ranges in job["inputs"]]
                _, m, inner, n = job["shape"]
                block = (operands[0].reshape(m, inner) @ operands[1].reshape(inner, n)).ravel()
                pos = 0
                for start, length in job["outputs"]:
                    result[start:start+length] = block[pos:pos+length]
                    pos += length
            values[op.outputs[0]] = result.reshape(ir.tensors[op.outputs[0]].shape)
        elif op.op == "scale":
            values[op.outputs[0]] = values[op.inputs[0]]*ir.meta["scale"]
        elif op.op == "softmax":
            a = values[op.inputs[0]]
            exp = np.exp(a-a.max(axis=-1, keepdims=True))
            values[op.outputs[0]] = exp/exp.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(values["out"], probs @ v)
    r = evaluate(small(), w, MappingSpec("ipu_bsp", {}))
    assert any("exponential" in item for item in r.details["manycore"]["unmodeled"])
    with pytest.raises(ValueError, match="query_len=1"):
        workload_from_spec(WorkloadSpec("operator", dict(w.params, query_len=5)))


def test_remote_explicit_source_bytes_and_rejections():
    w = WorkloadSpec("operator", dict(op="attention", heads=1, head_dim=3, seq_len=5, kv_storage="remote"))
    with pytest.raises(NotImplementedError, match="explicitly sourced"):
        evaluate(small(), w, MappingSpec("ipu_bsp", {}))
    with pytest.raises(ValueError, match="sourced Param"):
        devices.ipu_gc200(remote_bandwidth=20, remote_capacity=1024)
    adf = small(remote_bandwidth=Param(2, "B/ns", "synthetic test attachment"),
                remote_capacity=Param(1024, "B", "synthetic test capacity"))
    r = evaluate(adf, w, MappingSpec("ipu_bsp", {}))
    assert sum(t["remote_read"] for t in r.details["manycore"]["tiles"]) == 2*5*3*2
    assert r.details["manycore"]["remote_service_ns"] == 30
    with pytest.raises(NotImplementedError, match="dtype"):
        evaluate(small("tianjicx"), matrix(), MappingSpec("tianjic_spatial", {}))
    with pytest.raises(ValueError, match="iterations=1"):
        evaluate(small(), matrix(iterations=2), MappingSpec("ipu_bsp", {}))
    with pytest.raises(ValueError, match="iterations=1"):
        evaluate(small(), matrix(), MappingSpec("ipu_bsp", {}), iterations=2)
    with pytest.raises(NotImplementedError, match="collectives"):
        evaluate(small(), WorkloadSpec("operator", {"op": "allreduce", "bytes": 32}), MappingSpec("ipu_bsp", {}))


def test_tianjic_memory_units_and_unmodeled_mesh():
    a, b = devices.tianjicx(), devices.tianjicx(capacity_unit="binary")
    assert a.elements[0].elements[1].params["capacity"].value == 128000
    assert b.elements[0].elements[1].params["capacity"].value == 131072
    sim = build_simulator(small("tianjicx"))
    assert sim.root.networks[0].route(0, 1) == Demand()
    assert sim.find_points(component="sram")[0].port.capacity == 32


@pytest.mark.parametrize("device,template,dtype", [("ipu_gc200", "ipu_bsp", "fp16"),
                                                   ("tianjicx", "tianjic_spatial", "int8")])
def test_blocked_hand_timing_accumulator_and_peak(device, template, dtype):
    w = WorkloadSpec("operator", dict(op="matmul", m=2, k=35, n=3, dtype=dtype))
    sim, trace = run_mapping(small(device, count=1), w, MappingSpec(template, dict(schedule="blocked", tile_k=32)))
    d = sim.mapping.details["manycore"]
    unit = sim.find_points(role="compute")[0]
    mem = sim.find_points(component="sram")[0]
    width = 2 if dtype == "fp16" else 1
    acc, output = 24, 12 if dtype == "fp16" else 24
    rate = unit.rate/2 if dtype == "fp16" else unit.rate
    expected = acc/mem.port.capacity + 4*unit.sync_cycles/unit.clock_ghz
    for kk in (32, 3):
        inputs = 5*kk*width
        expected += 2*inputs/mem.port.capacity
        expected += max(6*kk/(rate*unit.clock_ghz), (inputs+2*acc)/mem.port.capacity)
    if dtype == "fp16":
        expected += (acc+output)/mem.port.capacity
    expected += 2*output/mem.port.capacity
    assert trace.latency_ns == pytest.approx(expected)
    assert d["macs"] == 2*35*3 and d["partial_blocks"] == 2
    read = 2*175*width+2*acc+output+(acc if dtype == "fp16" else 0)
    write = 175*width+3*acc+output+(output if dtype == "fp16" else 0)
    assert d["tiles"][0]["local_read"] == read
    assert d["tiles"][0]["local_write"] == write
    assert trace.memory_peak[mem.path] == d["capacity_peak_bytes"][0]
    assert trace.memory_peak[mem.path] <= mem.capacity
    records = {r.task:r for r in trace.records}
    first, second = (records[j["task"]] for j in d["jobs"])
    assert second.start_ns >= first.end_ns+unit.sync_cycles/unit.clock_ghz-1e-9
    labels = [t.meta["label"] for t in sim.mapping.graph.tasks]
    assert labels.count("AccumulatorInit") == 1
    assert labels.count("FinalStore") == (2 if dtype == "fp16" else 0)  # copy + dependency join


@pytest.mark.parametrize("op,shared", [("matmul", False), ("matmul", True), ("mvm", False)])
@pytest.mark.parametrize("dtype", ["fp16", "int8"])
def test_partial_rectangle_values_with_tail_batch_and_final_format(op, shared, dtype):
    from arch_explorer.mapping.manycore import output_blocks, block_inputs, block_output
    p = dict(op=op, batch=3, m=7, k=35, dtype=dtype)
    if op == "matmul":
        p.update(n=37, shared_b=shared)
    ir = workload_from_spec(WorkloadSpec("operator", p))
    operation = ir.operators[0]
    rng = np.random.default_rng(91)
    values = {name:rng.integers(-3,4,size=t.shape).astype(np.float32 if dtype=="fp16" else np.int32)
              for name,t in ir.tensors.items()}
    output_dtype = np.float16 if dtype == "fp16" else np.int32
    result = np.zeros(ir.tensors[operation.outputs[0]].shape, dtype=output_dtype)
    coverage = np.zeros_like(result, dtype=np.int32)
    macs = 0
    for job in output_blocks(operation, 4, 32):
        acc = np.zeros((job[2],job[4]), dtype=np.float32 if dtype=="fp16" else np.int32)
        for start in range(0,35,32):
            size = min(32,35-start)
            operands=[]
            for name, (b,r,nr,c,nc) in block_inputs(operation,job,start,size):
                a=values[name].reshape(-1,*values[name].shape[-2:])
                operands.append(a[b,r:r+nr,c:c+nc])
            acc += operands[0]@operands[1] if op=="matmul" else operands[1]@operands[0].T
            macs += job[2]*job[4]*size
        b,r,nr,c,nc = block_output(operation,job)
        result.reshape(-1,*result.shape[-2:])[b,r:r+nr,c:c+nc]=acc.astype(output_dtype)
        coverage.reshape(-1,*coverage.shape[-2:])[b,r:r+nr,c:c+nc]+=1
    expected = values['A']@values['B'] if op=='matmul' else values['x']@values['A'].T
    np.testing.assert_array_equal(result,expected.astype(output_dtype))
    assert np.all(coverage==1)
    assert macs == result.size*35


def test_block_layout_prefix_intersection_against_element_owners():
    from arch_explorer.mapping.manycore import BlockLayout
    from arch_explorer.workload.operators import Tensor
    layout=BlockLayout(Tensor("x",(2,5,7),"fp16"),3,4)
    owner=np.zeros((2,5,7),dtype=np.int32)
    for key,size in layout.specs():
        # Split inside rows, including one-element segments; no rectangle shortcut.
        segments=[]
        for start in range(0,size,2):
            tile=(start//2)%3
            segments.append((start,2,tile))
            b,br,bc=key
            stride=min(4,7-bc*4)
            rr,cc=divmod(start//2,stride)
            owner[b,br*3+rr,bc*4+cc]=tile
        layout.blocks[key]=segments
    for b in range(2):
        for r in range(5):
            for c in range(7):
                rect=(b,r,5-r,c,7-c)
                expected={i:int(np.sum(owner[b,r:,c:]==i))*2 for i in range(3)}
                assert layout.pieces(rect)=={i:n for i,n in expected.items() if n}
    assert layout.pieces((0,1,0,1,3)) == {}


def test_blocked_attention_tail_dependencies_bytes_and_remote():
    w=WorkloadSpec("operator",dict(op="attention",batch=2,heads=2,head_dim=5,seq_len=35))
    mapping=MappingSpec("ipu_bsp",dict(schedule="blocked",tile_k=32,tile_n=32))
    sim,trace=run_mapping(small(count=4),w,mapping)
    d=sim.mapping.details['manycore']
    assert d['macs']==2*4*35*5
    records={r.task:r for r in trace.records}
    last=0
    for stage in d['stages']:
        assert all(records[t].start_ns>=last-1e-8 for t in stage['tasks'])
        last=max([records[t].end_ns for t in stage['tasks']] or [last])
    for i,mem in enumerate(sim.find_points(component="sram")):
        assert trace.memory_peak[mem.path]==d['capacity_peak_bytes'][i]
    # Exact per-resource work in the solver agrees with independently accumulated byte counters.
    for i,mem in enumerate(sim.find_points(component="sram")):
        assert trace.resource_usage[mem.port.name] == pytest.approx(d['tiles'][i]['local_read']+d['tiles'][i]['local_write'])
    remote=WorkloadSpec("operator",dict(w.params,kv_storage="remote"))
    adf=small(count=4,remote_bandwidth=Param(2,"B/ns","synthetic"),remote_capacity=Param(10000,"B","synthetic"))
    e=evaluate(adf,remote,mapping)
    assert sum(t['remote_read'] for t in e.details['manycore']['tiles'])==2*4*35*5*2
    with pytest.raises(NotImplementedError,match='explicitly sourced'):
        evaluate(small(),remote,mapping)


def test_legacy_default_replay_and_legal_search_space():
    a=small()
    w=matrix()
    old=evaluate(a,w,MappingSpec("ipu_bsp",{}),trace=True)
    explicit=evaluate(a,w,MappingSpec("ipu_bsp",dict(tile_m=16,tile_n=128)),trace=True)
    assert old.to_dict()==explicit.to_dict()
    with pytest.raises(ValueError,match='requires'):
        evaluate(a,w,MappingSpec("ipu_bsp",dict(tile_k=32)))
    from arch_explorer.mapping.manycore import IpuBsp
    space=IpuBsp().space(a,w)
    assert space['schedule']==['blocked']  # every generated full mapping selects the new strategy


def test_blocked_attention_both_gemms_against_dense_reference():
    from arch_explorer.mapping.manycore import output_blocks, block_inputs, block_output
    w=workload_from_spec(WorkloadSpec('operator',dict(op='attention',batch=2,heads=2,head_dim=35,seq_len=37)))
    rng=np.random.default_rng(27)
    values={k:(rng.normal(size=w.tensors[k].shape)*0.2).astype(np.float16) for k in ('Q','K','V')}
    q,k,v=(values[n].astype(np.float32) for n in ('Q','K','V'))
    score=(q@k).astype(np.float16)
    scaled=(score.astype(np.float32)*w.meta['scale']).astype(np.float16).astype(np.float32)
    prob=np.exp(scaled-scaled.max(axis=-1,keepdims=True))
    prob=(prob/prob.sum(axis=-1,keepdims=True)).astype(np.float16)
    expected=(prob.astype(np.float32)@v).astype(np.float16)
    for op in w.operators:
        if op.op=='matmul':
            output=np.zeros(w.tensors[op.outputs[0]].shape,dtype=np.float16)
            for job in output_blocks(op,4,32):
                acc=np.zeros((job[2],job[4]),dtype=np.float32)
                for start in range(0,op.attrs['k'],32):
                    operands=[]
                    for name,(b,r,nr,c,nc) in block_inputs(op,job,start,min(32,op.attrs['k']-start)):
                        operands.append(values[name][b,r:r+nr,c:c+nc].astype(np.float32))
                    acc+=operands[0]@operands[1]
                b,r,nr,c,nc=block_output(op,job)
                output[b,r:r+nr,c:c+nc]=acc.astype(np.float16)
        elif op.op=='scale':
            output=(values[op.inputs[0]].astype(np.float32)*w.meta['scale']).astype(np.float16)
        else:
            a=values[op.inputs[0]].astype(np.float32)
            output=np.exp(a-a.max(axis=-1,keepdims=True))
            output=(output/output.sum(axis=-1,keepdims=True)).astype(np.float16)
        values[op.outputs[0]]=output
    np.testing.assert_allclose(values['out'],expected,rtol=2e-3,atol=1e-4)
    # Forward narrowing reuses the accumulator bytes without overwriting unread FP32 values.
    source=np.arange(21,dtype=np.float32)/8
    buffer=bytearray(source.tobytes())
    for i in range(len(source)):
        x=np.frombuffer(buffer,dtype=np.float32,count=1,offset=4*i)[0]
        buffer[2*i:2*i+2]=np.float16(x).tobytes()
    np.testing.assert_array_equal(np.frombuffer(buffer,dtype=np.float16,count=21),source.astype(np.float16))


@pytest.mark.parametrize('op,shared', [('matmul',False),('matmul',True),('mvm',False),('attention',False)])
def test_split_k_values_coverage_and_non_power_of_two_tree(op,shared):
    from arch_explorer.mapping.manycore import output_blocks,block_inputs,block_output
    from arch_explorer.mapping.manycore_split import k_partitions,tree_pairs
    params=dict(op=op,batch=2,m=5,k=131,n=37,dtype='fp16')
    if op=='mvm': params.pop('n')
    if shared: params['shared_b']=True
    if op=='attention': params=dict(op=op,batch=2,heads=2,head_dim=67,seq_len=131,dtype='fp16')
    w=workload_from_spec(WorkloadSpec('operator',params)); rng=np.random.default_rng(28)
    values={name:(rng.normal(size=t.shape)*0.2).astype(np.float16) for name,t in w.tensors.items()}
    for operator in w.operators:
        if operator.op not in ('matmul','mvm'):
            a=values[operator.inputs[0]].astype(np.float32)
            if operator.op=='scale': out=a*w.meta['scale']
            else:
                out=np.exp(a-a.max(axis=-1,keepdims=True)); out/=out.sum(axis=-1,keepdims=True)
            values[operator.outputs[0]]=out.astype(np.float16); continue
        a,b=(values[n].astype(np.float32) for n in operator.inputs)
        expected=a@b if operator.op=='matmul' else b@a.T
        actual=np.zeros_like(expected,dtype=np.float16); coverage=np.zeros(actual.shape,dtype=int)
        ranges=k_partitions(operator.attrs['k'],32,3)
        assert len(ranges)==3
        assert [k for lo,hi in ranges for k in range(lo,hi)]==list(range(operator.attrs['k']))
        macs=adds=0
        for job in output_blocks(operator,4,32):
            parts=[]
            for lo,hi in ranges:
                acc=np.zeros((job[2],job[4]),dtype=np.float32)
                for start in range(lo,hi,32):
                    operands=[]
                    for name,(batch,row,nr,col,nc) in block_inputs(operator,job,start,min(32,hi-start)):
                        x=values[name]; x=x.reshape((-1,*x.shape[-2:]))
                        operands.append(x[batch,row:row+nr,col:col+nc].astype(np.float32))
                    acc+=operands[0]@operands[1] if operator.op=='matmul' else operands[1]@operands[0].T
                    macs+=job[2]*job[4]*min(32,hi-start)
                parts.append(acc)
            for pairs in tree_pairs(len(parts)):
                for src,dst in pairs: parts[dst]=parts[dst]+parts[src]; adds+=parts[dst].size
            batch,row,nr,col,nc=block_output(operator,job)
            actual.reshape((-1,*actual.shape[-2:]))[batch,row:row+nr,col:col+nc]=parts[0].astype(np.float16)
            coverage.reshape((-1,*coverage.shape[-2:]))[batch,row:row+nr,col:col+nc]+=1
        np.testing.assert_allclose(actual,expected.astype(np.float16),rtol=2e-3,atol=2e-4)
        assert np.all(coverage==1) and adds==2*actual.size and macs==actual.size*operator.attrs['k']
        values[operator.outputs[0]]=actual


def test_split_k_exact_graph_traffic_addition_and_capacity():
    from reference import global_event_reference
    w=WorkloadSpec('operator',dict(op='matmul',m=2,k=67,n=3,dtype='fp16'))
    mapping=MappingSpec('ipu_split_k',dict(tile_k=32,partitions=3))
    sim,trace=run_mapping(small(count=3),w,mapping)
    d=sim.mapping.details['manycore']; a=24; inputs=5*67*2; output=12
    assert d['reduction_additions']==12 and d['reduction_transfer_bytes']==2*a
    assert d['partial_blocks']==3 and d['macs']==402
    # Gather+compute input bytes, init, partial read/write, tree transfers,
    # tree add reads/writes, final narrowing, scatter; independent of timing.
    expected=3*inputs+3*a+6*a+4*a+6*a+a+output+2*output
    assert sum(t['local_read']+t['local_write'] for t in d['tiles'])==expected
    assert sum(r['isolated_compute_ns'] for r in d['reductions'])==pytest.approx(12/(2*1.325))
    reference=global_event_reference(sim.mapping.graph)
    assert trace.latency_ns==pytest.approx(reference.latency_ns)
    records={x.task:x for x in trace.records}; refs={x.task:x for x in reference.records}
    for task,record in records.items():
        assert record.start_ns==pytest.approx(refs[task].start_ns)
        assert record.end_ns==pytest.approx(refs[task].end_ns)
    reductions=d['reductions']
    assert reductions[1]['level']==1 and reductions[1]['source']==2
    assert records[reductions[1]['transfer']].start_ns>=records[reductions[0]['task']].end_ns+150/1.325-1e-8
    for i,mem in enumerate(sim.find_points(component='sram')):
        assert trace.memory_peak[mem.path]==d['capacity_peak_bytes'][i]
        assert trace.resource_usage[mem.port.name]==d['tiles'][i]['local_read']+d['tiles'][i]['local_write']
    for i in range(3):
        for direction,key in [('tx','sent'),('rx','received')]:
            assert trace.resource_usage.get(sim.root.networks[0].port(i,direction).name,0)==d['tiles'][i][key]
    near=small(count=3)
    for point in near.elements[0].elements:
        if point.component=='sram': point.params['capacity']=Param(595,'B','synthetic total one byte below required')
    with pytest.raises(ValueError,match='workspaces|capacity'):
        evaluate(near,w,mapping)


def test_split_one_partition_and_legality():
    w=WorkloadSpec('operator',dict(op='matmul',m=2,k=67,n=3,dtype='fp16'))
    a=small(count=3)
    old=evaluate(a,w,MappingSpec('ipu_bsp',dict(schedule='blocked',tile_k=32)))
    new=evaluate(a,w,MappingSpec('ipu_split_k',dict(tile_k=32,partitions=1)))
    assert old.latency_ns==new.latency_ns
    assert old.details['manycore']['tiles']==new.details['manycore']['tiles']
    assert new.details['manycore']['reduction_additions']==0
    capped=evaluate(a,w,MappingSpec('ipu_split_k',dict(tile_k=128,partitions=3)))
    plan=next(iter(capped.details['manycore']['partition_plan'].values()))
    assert plan['requested']==3 and plan['effective']==1
    with pytest.raises(ValueError): evaluate(a,w,MappingSpec('ipu_split_k',dict(partitions=4)))
    with pytest.raises(ValueError): evaluate(small('tianjicx'),matrix('int8'),MappingSpec('ipu_split_k',{}))
    with pytest.raises(NotImplementedError,match='remote'):
        evaluate(a,WorkloadSpec('operator',dict(op='attention',head_dim=4,heads=1,seq_len=67,kv_storage='remote')),
                 MappingSpec('ipu_split_k',dict(partitions=2)))
