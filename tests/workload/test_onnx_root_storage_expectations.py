import unittest
from onnx import TensorProto,helper
from onnx_root_storage_expectations import roots_from_model


class RootStorageControl(unittest.TestCase):
    def test_input_initializer_and_constant_have_distinct_storage_types(self):
        weight=helper.make_tensor('weight',TensorProto.BFLOAT16,[3],[0,0,0],raw=False)
        scalar=helper.make_tensor('scalar',TensorProto.FLOAT,[],[.5])
        graph=helper.make_graph([helper.make_node('Constant',[],['constant'],value=scalar)],'root-control',[],[],[weight])
        result=roots_from_model(helper.make_model(graph),{'X':[2]},'fp16')
        self.assertEqual({name:record['bytes'] for name,record in result.items()},dict(X=4,weight=6,constant=4))
        self.assertEqual(sum(v['bytes'] for v in result.values()),14)
        self.assertEqual(result['constant']['logical_shape'],[])


if __name__=='__main__':unittest.main()
