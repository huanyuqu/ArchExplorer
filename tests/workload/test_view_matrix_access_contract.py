"""Hand-addressed holey matrix operands, uneven regions and repeated reads."""
import json

import pytest

from adapter_boundary_cases import events_scene,roofline_scene
from precision_transformer_expectations import store
from test_operator_mapping_attention_contract import no_backend
from view_storage_contract_cases import independent_layouts,view_api
from view_storage_expectations import row_major_steps
from view_storage_runtime_cases import address_rows,audit_storage,observe_memory


def index_of(shape,flat):
    result=[]
    for stride in row_major_steps(shape):
        value,flat=divmod(flat,stride);result.append(value)
    return result


@pytest.mark.parametrize('mode',['events','roofline'])
@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_uneven_matrix_holes_count_each_FMA_operand_read(view_api,monkeypatch,tmp_path,mode,dtype):
    from arch_explorer.adf import WorkloadSpec,MappingSpec
    w,s=view_api
    adf,original,_,seed=events_scene() if mode=='events' else roofline_scene(m=3,k=3,n=5,tiles=dict(m=2,k=2,n=3))
    old=original.operators[0];a,b=[original.tensors[n] for n in old.inputs]
    output=original.tensors[old.outputs[0]]
    graph=w.Workload('hand-holey-matrix',dtype=dtype)
    graph.tensor('ROOT',b.shape,dtype=dtype)
    root_values=[[float(r*5+c) for c in range(5)] for r in range(3)]
    graph.meta['inputs']={'ROOT':[root_values] if len(b.shape)==3 else root_values}
    shape=list(b.shape);shape[-1]=3
    w.readonly_view(graph,'A.slice','ROOT',transform='slice',shape=tuple(shape),
        starts=[0]*len(shape),stops=shape,steps=[1]*len(shape))
    w.readonly_view(graph,'A','A.slice',transform='reshape',shape=a.shape)
    w.readonly_view(graph,'B','ROOT',transform='reshape',shape=b.shape)
    graph.tensor('Y',output.shape,dtype=dtype)
    graph.add('product',old.op,['A','B'],['Y'],**old.attrs)
    graph.meta['outputs']=['Y'];spec=WorkloadSpec('graph',graph.to_dict())
    mapping=MappingSpec('compile_graph_v3',dict(seed.params,operator_rules={}))
    with no_backend(monkeypatch):
        raw=w.execute_reference(spec)
        compilation=w.compile_workload(adf,spec,mapping)
        program=compilation['program'];layouts=independent_layouts(program)
        mapped_graph=w.workload_from_spec(WorkloadSpec('graph',program))
        mapped=w.execute_reference(WorkloadSpec('graph',program))
        steps={row['name']:row for row in compilation['plan']['steps']}
        read_bytes=write_bytes=0;covered=set();width=4 if dtype=='fp32' else 2
        assert layouts['A'].addresses()==[0,1,2,5,6,7,10,11,12]
        assert layouts['A'].read_bytes==9*width and layouts['A'].enclosing_span_bytes==13*width
        for op in program['operators']:
            if op['op']!='compiled.matmul':continue
            p=op['attrs'];expected=[]
            for batch in range(p['batch']):
                for row in range(p['row'],p['row']+p['rows']):
                    for col in range(p['col'],p['col']+p['cols']):
                        assert (batch,row,col) not in covered;covered.add((batch,row,col))
                        for k in range(3):
                            for name,flat,address in [('A',(batch*3+row)*3+k,row*5+k),
                                                      ('B',k*5+col,k*5+col)]:
                                expected.append(dict(direction='read',tensor=name,storage='ROOT',
                                    logical_index=index_of(layouts[name].shape,flat),element_offset=address,
                                    byte_offset=address*width,bytes=width))
            expected+=address_rows(layouts[op['outputs'][0]],op['outputs'][0],'write')
            assert w.logical_accesses(mapped_graph,op['name'])==expected
            reads=sum(r['bytes'] for r in expected if r['direction']=='read')
            writes=sum(r['bytes'] for r in expected if r['direction']=='write')
            assert steps[op['name']]['access']['read_bytes']==reads
            assert steps[op['name']]['access']['write_bytes']==writes
            read_bytes+=reads;write_bytes+=writes
        assert covered=={(0,r,c) for r in range(3) for c in range(5)}
        assert read_bytes==90*width and write_bytes==15*width
        request=s.prepare_compiled(compilation,output_dir=tmp_path/'matrix',options={'trace':True})
    expected=[[store(value,dtype) for value in row] for row in
        [[25.,28.,31.,34.,37.],[100.,118.,136.,154.,172.],[175.,208.,241.,274.,307.]]]
    if len(output.shape)==3:expected=[expected]
    assert raw['outputs']['Y'].tolist()==mapped['outputs']['Y'].tolist()==expected
    with observe_memory(monkeypatch) as events:result=s.evaluate_request(request)
    physical=audit_storage(compilation,result,events)
    with no_backend(monkeypatch):assert s.evaluate_request(result)==result
    (tmp_path/'HAND-MATRIX.json').write_text(json.dumps(dict(root_addresses=list(range(15)),
        actual_physical=physical,matmul_read_bytes=read_bytes,matmul_write_bytes=write_bytes,
        actual_result=result,expected_CPU_values=expected),indent=2)+'\n')
