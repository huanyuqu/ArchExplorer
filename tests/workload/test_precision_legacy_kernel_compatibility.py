"""Real installed unknown FP32 model/adapter retains plain Kernel restore."""
from copy import deepcopy
import json

import pytest

from test_operator_mapping_attention_contract import no_backend,files
from test_operator_mapping_extension_contract import reseal
from test_precision_transformer_contract import precision_api


def test_installed_unknown_fp32_plain_kernel_restore_and_saved_request(
        precision_api,monkeypatch,tmp_path):
    import arch_explorer
    import ae_component_implementations
    from ae_component_implementations.compilation import compilation_scene
    from ae_component_implementations.adapter import restore
    from arch_explorer.components.api import Kernel
    w,s=precision_api
    assert 'site-packages' in arch_explorer.__file__
    assert 'site-packages' in ae_component_implementations.__file__
    adf,spec,mapping=compilation_scene()
    with no_backend(monkeypatch):
        compilation=w.compile_workload(adf,spec,mapping)
        tasks=compilation['groups'][0]['payload']['tasks']
        assert tasks
        for row in tasks.values():
            work=row['work']
            assert set(work)=={'op','shape','precision','simt','sfu','row_simt','row_sfu'}
            assert work['precision']=='fp32'
            assert type(restore(work)) is Kernel
        request=s.prepare_compiled(compilation,output_dir=tmp_path/'external')
        # JSON roundtrip must restore the same external Kernel protocol.
        restored=json.loads(json.dumps(request))
        assert s.assess_request(restored)['supported']
    result=s.evaluate_request(restored)
    before=files(tmp_path/'external')
    with no_backend(monkeypatch):
        assert s.evaluate_request(result)==result
        broken=deepcopy(compilation)
        work=next(iter(broken['groups'][0]['payload']['tasks'].values()))['work']
        work['shape'][1]+=1
        with pytest.raises((ValueError,TypeError,NotImplementedError)):
            w.validate_compilation(reseal(broken))
    assert before==files(tmp_path/'external')


@pytest.mark.parametrize('dtype',['fp16','bf16'])
def test_unknown_model_without_conversion_input_support_refuses_low_precision(
        precision_api,monkeypatch,tmp_path,dtype):
    from ae_component_implementations.compilation import compilation_scene
    from arch_explorer.adf import WorkloadSpec
    w,s=precision_api;adf,spec,mapping=compilation_scene()
    graph=w.workload_from_spec(spec).to_dict()
    graph['dtype']=dtype
    for tensor in graph['tensors']:tensor['dtype']=dtype
    for operator in graph['operators']:
        for field in ('dtype','result_dtype','output_dtype'):
            if field in operator['attrs']:operator['attrs'][field]=dtype
    with no_backend(monkeypatch),pytest.raises((ValueError,TypeError,NotImplementedError),
            match='(?i)precision|conversion|fp32|support|adapter'):
        compilation=w.compile_workload(adf,WorkloadSpec('graph',graph),mapping)
        s.prepare_compiled(compilation,output_dir=tmp_path/'unsupported')
