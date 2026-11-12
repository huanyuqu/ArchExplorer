"""Generate frozen standard ONNX inputs only, without expected values."""
import argparse,hashlib,json
from pathlib import Path
from onnx_import_cases import mlp_case,attention_case,save_case

def generate(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    rows={}
    for key,case,external in [('mlp',mlp_case(),False),('attention',attention_case(),False),('mlp_external',mlp_case(),True)]:
        path=save_case(case,directory/key,external=external)
        rows[key]=dict(manifest=key+'/INPUTS.json',model=key+'/model.onnx',external=external,
            model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result=dict(schema='arch-explorer.val62.independent-model-set.v1',ONNX_release='1.16.2',opset=17,IR=8,
        cases=rows,expected_answers_included=False,product_importer_used=False)
    (directory/'INPUTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(directory=str(directory.resolve()),models=rows,inputs_sha256=hashlib.sha256((directory/'INPUTS.json').read_bytes()).hexdigest())))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');generate(p.parse_args().directory)
