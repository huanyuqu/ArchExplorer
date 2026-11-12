"""Hand arithmetic controls, without building a product Graph or any arrays."""
from copy import deepcopy
import unittest
from transformer_metadata_expectations import decoder_ledger, tiny_request


class LedgerControls(unittest.TestCase):
    def test_two_layer_prefill_hand_storage_and_fmas(self):
        d=decoder_ledger(tiny_request(),1)
        # Per layer: 120 weight elements, 12 cache elements, 198 temporary
        # elements, 432 FMA. x0 has 12 elements, counted once across layers.
        expected=dict(operators=24,tensors=35,root_bytes=552,weight_bytes=480,kv_bytes=48,
            temporary_bytes=792,total_tensor_bytes=1344,matrix_fmas_per_iteration=864,
            total_matrix_fmas=2592,vector_elements_per_iteration=168,
            allreduce_bytes_per_iteration=0,max_rank=5,max_dimension=12,max_tensor_elements=48)
        for k,v in expected.items():self.assertEqual(d[k],v,k)
        self.assertEqual(d['materialized_root_bytes'],0)
        self.assertEqual(d['root_python_bytes'],0)

    def test_tp2_kv_is_replicated_and_reductions_are_per_rank(self):
        d=decoder_ledger(tiny_request(),2)
        expected=dict(operators=28,tensors=39,root_bytes=344,weight_bytes=272,kv_bytes=48,
            temporary_bytes=660,total_tensor_bytes=1004,matrix_fmas_per_iteration=480,
            total_matrix_fmas=1440,allreduce_bytes_per_iteration=96)
        for k,v in expected.items():self.assertEqual(d[k],v,k)
        self.assertNotEqual(d['weight_bytes']*2,decoder_ledger(tiny_request(),1)['weight_bytes'])

    def test_decode_keeps_cache_and_weights_but_uses_one_query(self):
        d=decoder_ledger(tiny_request(phase='decode'),1)
        for k,v in dict(root_bytes=536,weight_bytes=480,kv_bytes=48,temporary_bytes=264,
            total_tensor_bytes=800,matrix_fmas_per_iteration=288,total_matrix_fmas=864).items():
            self.assertEqual(d[k],v,k)

    def test_dtype_and_vocab_do_not_change_matrix_work(self):
        low=decoder_ledger(tiny_request(),1)
        single=decoder_ledger(tiny_request(dtype='fp32'),1)
        self.assertEqual(single['total_tensor_bytes'],2*low['total_tensor_bytes'])
        self.assertEqual(single['total_matrix_fmas'],low['total_matrix_fmas'])
        self.assertEqual(decoder_ledger(tiny_request(dtype='bf16'),1),low)
        request=deepcopy(tiny_request());request['model']['vocab']=1000000
        self.assertEqual(decoder_ledger(request,1),low)  # Decoder blocks exclude embedding/LM head.


if __name__=='__main__':unittest.main()
