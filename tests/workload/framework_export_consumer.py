"""Installed API/CLI ordering control, prepared from CORE34 ENTRYPOINTS.

No producer source is mixed into a package here. A caller supplies the real
framework module and frozen roots in an environment with optional torch/onnx.
"""
import copy
import inspect
import json
from pathlib import Path
import subprocess
import sys


def exercise_export(module,*,inputs,input_names,output_names,output_dir,dynamic_axes=None,external_data=False):
    import onnx
    import torch
    from arch_explorer.archs.framework_export import export_torch
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=False)
    # Sorted JSON reverses the MLP R/X insertion order. Explicit names must
    # control both call paths, independently of the parser's object ordering.
    sorted_inputs=json.loads(json.dumps(inputs,sort_keys=True))
    facts=dict(input_names=list(input_names),json_key_order=list(sorted_inputs),exports={},status='running')
    (root/'CONSUMER.json').write_text(json.dumps(facts,indent=2)+'\n')
    try:
        signature=inspect.signature(export_torch)
        assert signature.parameters['input_names'].default is None
        parameters={n:v.detach().clone() for n,v in module.state_dict().items()}
        training={n:m.training for n,m in module.named_modules()}
        kwargs=dict(inputs=sorted_inputs,input_names=list(input_names),output_names=list(output_names),
                    dynamic_axes=dynamic_axes,external_data=external_data)
        facts['api_manifest']=export_torch(module,output_dir=root/'api',**kwargs)
        assert {n:m.training for n,m in module.named_modules()}==training
        assert all(torch.equal(parameters[n],v) for n,v in module.state_dict().items())
        model=copy.deepcopy(module).cpu().eval()
        dtypes={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16,
                'int32':torch.int32,'int64':torch.int64,'bool':torch.bool}
        args=tuple(torch.tensor(inputs[n]['values'],dtype=dtypes[inputs[n]['dtype']]) for n in input_names)
        script=torch.jit.trace(model,args,check_trace=True)
        script.save(str(root/'module.pt'))
        (root/'TORCH-INPUTS.json').write_text(json.dumps(sorted_inputs,sort_keys=True)+'\n')
        cli=[str(Path(sys.executable).with_name('archexplorer')),'export-torch',str(root/'module.pt'),
             '--inputs',str(root/'TORCH-INPUTS.json'),'--input-names',*input_names,
             '--output-names',*output_names,'--output-dir',str(root/'cli')]
        if dynamic_axes is not None:
            (root/'AXES.json').write_text(json.dumps(dynamic_axes)+'\n')
            cli+=['--dynamic-axes',str(root/'AXES.json')]
        if external_data:cli+=['--external-data']
        facts['argv']=cli
        process=subprocess.run(cli,cwd=root,capture_output=True,text=True,timeout=120)
        (root/'stdout.log').write_text(process.stdout);(root/'stderr.log').write_text(process.stderr)
        facts['returncode']=process.returncode
        assert process.returncode==0,process.stderr
        for name in ['api','cli']:
            folder=root/name
            manifest=json.loads((folder/'EXPORT.json').read_text())
            assert manifest['roots']==list(input_names)
            proto=onnx.load(folder/'model.onnx',load_external_data=True)
            onnx.checker.check_model(proto,full_check=True)
            assert proto.ir_version==8 and [(o.domain,o.version) for o in proto.opset_import]==[('',17)]
            assert [v.name for v in proto.graph.input]==list(input_names)
            assert [v.name for v in proto.graph.output]==list(output_names)
            saved=json.loads((folder/'INPUTS.json').read_text())
            for i,n in enumerate(input_names):
                expected=args[i].float().tolist() if args[i].is_floating_point() else args[i].tolist()
                assert saved[n]==expected
            assert json.loads((folder/'INPUT-SHAPES.json').read_text())=={n:inputs[n]['shape'] for n in input_names}
            facts['exports'][name]=dict(manifest=manifest,nodes=len(proto.graph.node))
        facts['status']='succeeded'
        return facts
    except BaseException as error:
        facts.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        (root/'CONSUMER.json').write_text(json.dumps(facts,indent=2,allow_nan=False)+'\n')
