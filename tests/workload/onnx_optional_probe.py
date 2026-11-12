"""Fresh core import works with optional onnx resolution deliberately absent."""
import importlib.abc,importlib.util,json,sys
from pathlib import Path

class MissingONNX(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='onnx' or fullname.startswith('onnx.'):
            raise ModuleNotFoundError("No module named 'onnx'",name='onnx')

def main():
    sys.dont_write_bytecode=True
    assert not any(n=='onnx' or n.startswith('onnx.') for n in sys.modules)
    sys.meta_path.insert(0,MissingONNX())
    import arch_explorer
    from arch_explorer.adf import WorkloadSpec
    from arch_explorer.workload import api as w
    from arch_explorer.workload.onnx_import import import_onnx
    from arch_explorer.archs.onnx_compilation import onnx_workload
    # Existing generic graph path remains usable; no ONNX resolution needed.
    graph=w.Workload('without-onnx',dtype='fp32');graph.tensor('X',(1,2));graph.tensor('Y',(1,2))
    graph.meta['inputs']={'X':[[1.,-2.]]};graph.meta['outputs']=['Y']
    graph.add('scale','scale',['X'],['Y'],shape=[1,2],factor=2.)
    import numpy as np
    np.testing.assert_array_equal(w.execute_reference(WorkloadSpec('graph',graph.to_dict()))['outputs']['Y'],[[2.,-4.]])
    failures=[]
    for function in (import_onnx,onnx_workload):
        try:function(sys.argv[1],inputs={'X':[[0.,0.,0.],[0.,0.,0.]]})
        except ImportError as error:
            assert str(error).startswith('ONNX import requires the optional onnx extra:')
            failures.append(dict(type=type(error).__name__,message=str(error)))
        else:raise AssertionError('missing dependency was ignored')
    Path(sys.argv[2]).write_text(json.dumps(dict(core_import=True,generic_graph_reference=True,
        missing_optional_refusals=failures,onnx_loaded=any(n=='onnx' for n in sys.modules)),indent=2)+'\n')

if __name__=='__main__':main()
