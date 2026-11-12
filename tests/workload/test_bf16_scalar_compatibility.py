"""BF16 scalar storage must match array storage across supported NumPy versions."""
import unittest
import numpy as np

from arch_explorer.workload.precision import bf16,round_storage,storage_bytes


# Literal IEEE FP32 words around BF16 ties, subnormals, zero and overflow.
WORDS = (
    (0x00000000,0x00000000), (0x80000000,0x80000000),
    (0x00000001,0x00000000), (0x80000001,0x80000000),
    (0x00008000,0x00000000), (0x00018000,0x00020000),
    (0x3f000000,0x3f000000), (0x3f808000,0x3f800000),
    (0x3f808001,0x3f810000), (0x3f818000,0x3f820000),
    (0xbf808000,0xbf800000), (0xbf818000,0xbf820000),
    (0x3f7fffff,0x3f800000), (0x7f7f7fff,0x7f7f0000),
    (0x7f7f8000,0x7f800000), (0xff7f8000,0xff800000),
    (0x7f800000,0x7f800000), (0xff800000,0xff800000),
    (0x7fc12345,0x7fc12345), (0xffc12345,0xffc12345),
)


class BF16ScalarCompatibility(unittest.TestCase):
    def test_literal_words_for_scalar_and_array(self):
        inputs=np.asarray([a for a,_ in WORDS],dtype=np.uint32).view(np.float32)
        expected=np.asarray([b for _,b in WORDS],dtype=np.uint32)
        with np.errstate(over='raise',invalid='raise'):
            vector=bf16(inputs)
            np.testing.assert_array_equal(vector.view(np.uint32),expected)
            for word,(before,after) in zip(inputs,WORDS):
                for value in (word,np.asarray(word,dtype=np.float32)):
                    scalar=bf16(value)
                    self.assertEqual(scalar.shape,())
                    self.assertEqual(scalar.dtype,np.dtype(np.float32))
                    self.assertEqual(int(scalar.view(np.uint32)),after,hex(before))

    def test_shapes_and_noncontiguous_input(self):
        words=np.asarray([a for a,_ in WORDS],dtype=np.uint32).reshape(4,5)
        expected=np.asarray([b for _,b in WORDS],dtype=np.uint32).reshape(4,5)
        values=words.view(np.float32).T
        self.assertFalse(values.flags.c_contiguous)
        result=bf16(values)
        self.assertEqual(result.shape,(5,4))
        np.testing.assert_array_equal(result.view(np.uint32),expected.T)
        empty=bf16(np.empty((0,3),dtype=np.float32))
        self.assertEqual(empty.shape,(0,3))
        self.assertEqual(empty.dtype,np.dtype(np.float32))

    def test_python_scalar_and_little_endian_storage(self):
        for value in (.5,np.float32(.5),np.asarray(.5,dtype=np.float32)):
            result=round_storage(value,'bf16')
            self.assertEqual(result.shape,())
            self.assertEqual(result.dtype,np.dtype(np.float32))
            self.assertEqual(float(result),.5)
            self.assertEqual(storage_bytes(result,'bf16'),b'\x00\x3f')
        self.assertEqual(storage_bytes(round_storage([.5,-.5],'bf16'),'bf16'),b'\x00\x3f\x00\xbf')

    def test_nonfinite_and_rounded_overflow_remain_refused(self):
        overflow=np.asarray(0x7f7f8000,dtype=np.uint32).view(np.float32)
        for value in (float('inf'),float('-inf'),float('nan'),overflow):
            for shape in ('scalar','array'):
                with self.subTest(value=value,shape=shape):
                    with self.assertRaisesRegex(ValueError,'storage conversion requires finite bf16 values'):
                        round_storage(value if shape=='scalar' else [value],'bf16')


if __name__=='__main__':unittest.main()
