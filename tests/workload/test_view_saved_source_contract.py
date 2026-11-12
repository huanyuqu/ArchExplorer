"""Bound layouts, effective selections and actual source bytes remain strict."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
import sys

import pytest

from test_operator_attention_search_contract import command
from test_operator_mapping_attention_contract import files,no_backend
from test_operator_mapping_extension_contract import reseal
from view_storage_contract_cases import REJECT,hand_graph,scene,view_api


def test_saved_view_result_rejects_resealed_layout_data_rule_and_plan_changes(view_api,monkeypatch,tmp_path):
    from arch_explorer.adf import WorkloadSpec,MappingSpec
    w,s=view_api;adf,_,_=scene('events-offset-fork')
    spec=WorkloadSpec('graph',hand_graph(w).to_dict())
    mapping=MappingSpec('compile_graph_v3',dict(compute='board/core/sm',memory='board/sram',rules={},operator_rules={}))
    with no_backend(monkeypatch):
        compilation=w.compile_workload(adf,spec,mapping)
        request=s.prepare_compiled(compilation,output_dir=tmp_path/'original',options={'trace':True})
    result=s.evaluate_request(request);before=files(tmp_path/'original')
    with no_backend(monkeypatch):
        assert s.evaluate_request(result)==result
        for mutation in ('offset','stride','root','allocation','address','effective-rule','data'):
            changed=deepcopy(compilation)
            tensor=next(row for row in changed['program']['tensors'] if row['name']=='S')
            if mutation=='offset':tensor['offset']+=1
            elif mutation=='stride':tensor['strides'][-1]+=1
            elif mutation=='root':tensor['alias_of']='Y'
            elif mutation=='allocation':changed['plan']['allocations'][0]['bytes']+=1
            elif mutation=='address':
                changed['plan']['steps'][-1]['address_map']['reads'][0]['offset']+=1
            elif mutation=='effective-rule':
                changed['source']['mapping']['params']['operator_rules']['S.view']['version']='altered'
            else:changed['source']['workload']['params']['meta']['inputs']['X'][0][0]+=1
            with pytest.raises(REJECT):w.validate_compilation(reseal(changed))
            bad=deepcopy(result);bad['request']['compilation']=reseal(changed)
            bad['request']=reseal(bad['request'])
            with pytest.raises(REJECT):s.evaluate_request(reseal(bad))
    assert files(tmp_path/'original')==before
    (tmp_path/'SAVED-RESULT.json').write_text(json.dumps(result,indent=2)+'\n')


def test_layout_resolver_source_bytes_invalidate_saved_compilation_in_isolated_copy(view_api,tmp_path):
    import arch_explorer
    assert 'site-packages' in arch_explorer.__file__
    source=Path(arch_explorer.__file__).parent
    copied=tmp_path/'package/arch_explorer';shutil.copytree(source,copied)
    adf,_,_=scene('events-offset-fork');(tmp_path/'ADF.json').write_text(json.dumps(adf.to_dict()))
    script=r'''
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import numpy,pytest,arch_explorer
from arch_explorer.adf import from_dict,WorkloadSpec,MappingSpec
from arch_explorer.workload import api as w
from arch_explorer.simulator import api as s
from view_storage_contract_cases import hand_graph,REJECT
from test_operator_mapping_attention_contract import no_backend
root=Path(sys.argv[3]);assert str(Path(arch_explorer.__file__).resolve()).startswith(sys.argv[1])
adf=from_dict(json.loads((root/'ADF.json').read_text()))
with pytest.MonkeyPatch.context() as patch,no_backend(patch):
    c=w.compile_workload(adf,WorkloadSpec('graph',hand_graph(w).to_dict()),
        MappingSpec('compile_graph_v3',dict(compute='board/core/sm',memory='board/sram',rules={},operator_rules={})))
    request=s.prepare_compiled(c,output_dir=root/'prepared')
    target=Path(arch_explorer.__file__).parent/'workload/storage.py';old=target.read_bytes()
    (root/'SOURCE-before.bin').write_bytes(old)
    try:
        target.write_bytes(old+b'\n# explicit QA source invalidation\n')
        (root/'SOURCE-altered.bin').write_bytes(target.read_bytes())
        with pytest.raises(REJECT):w.validate_compilation(c)
        with pytest.raises(REJECT):s.evaluate_request(request)
    finally:target.write_bytes(old)
    assert w.validate_compilation(c)==c
print('layout-source-strict')
'''
    reply=command([sys.executable,'-c',script,str(copied.parent),str(Path(__file__).parent),str(tmp_path)],tmp_path)
    assert reply.stdout.strip()=='layout-source-strict'
