"""CLIP source, mask, token, rule and dual-output identities stay strict."""
from copy import deepcopy
import hashlib,json,os,shutil,sys
from pathlib import Path
import pytest
from clip_contract_cases import REJECT,clip_api,scene
from test_operator_mapping_attention_contract import no_backend,files
from test_operator_mapping_extension_contract import reseal
from test_operator_attention_search_contract import command
from test_vit_user_consumers import fresh_readonly

def test_resealed_token_mask_rules_layout_and_outputs_are_refused(clip_api,monkeypatch):
    w,s=clip_api;adf,spec,mapping=scene('regular-2-fp32')
    with no_backend(monkeypatch):
        normal=w.compile_workload(adf,spec,mapping);assert w.validate_compilation(normal)==normal
        for mutation in ('token','weight','QuickGELU','mask','layout','allocation','source','rule','outputs'):
            bad=deepcopy(normal)
            if mutation=='token':bad['source']['workload']['params']['inputs']['token_ids'][0][0]=2
            elif mutation=='weight':bad['source']['workload']['params']['inputs']['text.block1.Wq'][0][0]+=.125
            elif mutation=='QuickGELU':next(o for o in bad['program']['operators'] if o['op']=='quick_gelu')['op']='gelu_tanh'
            elif mutation=='mask':next(o for o in bad['program']['operators'] if o['op']=='masked_softmax')['op']='softmax'
            elif mutation=='layout':next(t for t in bad['program']['tensors'] if t['alias_of'] is not None)['offset']+=1
            elif mutation=='allocation':bad['plan']['allocations'][0]['bytes']+=1
            elif mutation=='source':
                name=next(n for n in bad['sources']['files'] if 'clip' in n.lower());bad['sources']['files'][name]='0'*64
            elif mutation=='rule':
                binding=next(v for v in bad['source']['mapping']['params']['operator_rules'].values() if v['id']=='matmul.tiled')
                binding['options']['tiles']['k']+=1
            else:bad['program']['meta']['outputs']=['logits_per_image']
            with pytest.raises(REJECT):w.validate_compilation(reseal(bad))

def test_nested_builder_actual_source_mutation_rejects_in_an_isolated_installed_copy(clip_api,tmp_path):
    import arch_explorer
    assert 'site-packages' in arch_explorer.__file__
    copied=tmp_path/'package/arch_explorer';shutil.copytree(Path(arch_explorer.__file__).parent,copied)
    script=r'''
import inspect,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import pytest,arch_explorer
from arch_explorer.workload import api as w
from clip_contract_cases import REJECT,scene
from test_operator_mapping_attention_contract import no_backend
assert str(Path(arch_explorer.__file__).resolve()).startswith(sys.argv[1])
adf,spec,mapping=scene('regular-2-fp32')
with pytest.MonkeyPatch.context() as patch,no_backend(patch):
 c=w.compile_workload(adf,spec,mapping);assert w.validate_compilation(c)==c
 source=Path(inspect.getsourcefile(w.clip_similarity));key=str(source.relative_to(Path(arch_explorer.__file__).parent))
 assert key in c['sources']['files'];before=source.read_bytes()
 try:
  source.write_bytes(before+b'\n# VAL60 independent source invalidation\n')
  with pytest.raises(REJECT):w.validate_compilation(c)
 finally:source.write_bytes(before)
 assert w.validate_compilation(c)==c
print('CLIP-builder-source-strict')
'''
    result=command([sys.executable,'-c',script,str(copied.parent),str(Path(__file__).parent)],tmp_path)
    assert result.stdout.strip()=='CLIP-builder-source-strict'

def test_completed_actual_result_fresh_readonly_and_token_payload_tamper_rejection(clip_api,monkeypatch,tmp_path):
    w,s=clip_api;path=Path(os.environ['VAL60_SAVED_RESULT']);raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==os.environ['VAL60_SAVED_RESULT_SHA256']
    result=json.loads(raw);assert result['request']['compilation']['source']['workload']['kind']=='clip_similarity'
    before=files(path.parent);fresh_readonly(path,tmp_path,'compiled');assert files(path.parent)==before
    for mutation in ('token','mask','output','latency'):
        bad=deepcopy(result)
        if mutation=='latency':bad['evaluation']['latency_ns']+=1
        else:
            c=bad['request']['compilation']
            if mutation=='token':c['source']['workload']['params']['inputs']['token_ids'][0][0]+=1
            elif mutation=='mask':next(o for o in c['program']['operators'] if o['op']=='masked_softmax')['op']='softmax'
            else:c['program']['meta']['outputs'].reverse()
        with no_backend(monkeypatch),pytest.raises(REJECT):s.evaluate_request(bad)
