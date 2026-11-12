"""ViT parameter, arithmetic, root/layout and source identities remain strict."""
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from test_operator_attention_search_contract import command
from test_operator_mapping_attention_contract import files,no_backend
from test_operator_mapping_extension_contract import reseal
from vit_contract_cases import REJECT,scene,vit_api


def test_resealed_ViT_root_rule_layout_and_semantic_changes_are_rejected(vit_api,monkeypatch):
    w,s=vit_api;adf,spec,mapping=scene('regular-2-fp32')
    with no_backend(monkeypatch):
        normal=w.compile_workload(adf,spec,mapping)
        assert w.validate_compilation(normal)==normal
        for mutation in ('weight','image','erf-to-tanh','epsilon','offset','allocation','source','rule'):
            bad=deepcopy(normal)
            if mutation=='weight':bad['source']['workload']['params']['inputs']['block1.Wq'][0][0]+=.125
            elif mutation=='image':bad['source']['workload']['params']['inputs']['images'][0][0][0][0]+=.125
            elif mutation=='erf-to-tanh':next(op for op in bad['program']['operators'] if op['op']=='gelu_exact')['op']='gelu_tanh'
            elif mutation=='epsilon':next(op for op in bad['program']['operators'] if op['op']=='affine_norm')['attrs']['epsilon']=1e-5
            elif mutation=='offset':next(t for t in bad['program']['tensors'] if t['alias_of'] is not None)['offset']+=1
            elif mutation=='allocation':bad['plan']['allocations'][0]['bytes']+=1
            elif mutation=='source':
                path=next(path for path in bad['sources']['files'] if 'vit' in path.lower())
                bad['sources']['files'][path]='0'*64
            else:
                binding=next(value for value in bad['source']['mapping']['params']['operator_rules'].values() if value['id']=='matmul.tiled')
                binding['options']['tiles']['k']+=1
            with pytest.raises(REJECT):w.validate_compilation(reseal(bad))


def test_actual_nested_ViT_builder_source_change_rejects_in_isolated_installed_copy(vit_api,tmp_path):
    import arch_explorer
    assert 'site-packages' in arch_explorer.__file__
    copied=tmp_path/'package/arch_explorer'
    shutil.copytree(Path(arch_explorer.__file__).parent,copied)
    script=r'''
import inspect,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import pytest,arch_explorer
from arch_explorer.workload import api as w
from vit_contract_cases import REJECT,scene
from test_operator_mapping_attention_contract import no_backend
root=Path(sys.argv[3]);assert str(Path(arch_explorer.__file__).resolve()).startswith(sys.argv[1])
adf,spec,mapping=scene('regular-2-fp32')
with pytest.MonkeyPatch.context() as patch,no_backend(patch):
    c=w.compile_workload(adf,spec,mapping)
    assert w.validate_compilation(c)==c
    source=Path(inspect.getsourcefile(w.vit_classifier))
    key=str(source.relative_to(Path(arch_explorer.__file__).parent))
    assert key in c['sources']['files']
    before=source.read_bytes();(root/'SOURCE-before.bin').write_bytes(before)
    try:
        source.write_bytes(before+b'\n# independent QA source invalidation\n')
        (root/'SOURCE-altered.bin').write_bytes(source.read_bytes())
        with pytest.raises(REJECT):w.validate_compilation(c)
    finally:source.write_bytes(before)
    assert w.validate_compilation(c)==c
print('vit-builder-source-strict')
'''
    answer=command([sys.executable,'-c',script,str(copied.parent),str(Path(__file__).parent),str(tmp_path)],tmp_path)
    assert answer.stdout.strip()=='vit-builder-source-strict'


def test_fresh_saved_ViT_result_readonly_uses_recorded_real_execution(tmp_path):
    # The acceptance driver supplies a result produced by a completed numeric
    # case. Never rerun an equivalent whole-model schedule to seed this reader.
    path=Path(os.environ['VAL59_SAVED_RESULT'])
    digest=os.environ['VAL59_SAVED_RESULT_SHA256']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    result=json.loads(path.read_text())
    assert result['request']['compilation']['source']['workload']['kind']=='vit_classifier'
    before=files(path.parent)
    from test_vit_user_consumers import fresh_readonly
    fresh_readonly(path,tmp_path,'compiled')
    assert files(path.parent)==before
