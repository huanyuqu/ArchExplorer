"""CPU SYNTHETIC fixtures for the independent raw-record reader/oracles.

No data here is a device capture. Original opcode bytes are read offline from
the compiled cubin solely to test the validator, never supplied to the GPU.
"""
from copy import deepcopy
import json
from pathlib import Path
import struct
import tempfile
import unittest

from verify_capture import verify

BUILD = None


class SyntheticReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='SIM28-SYNTHETIC-ONLY-')
        self.d=Path(self.temp.name)
        decoded=json.loads((BUILD/'OFFLINE-CHECK.json').read_text())['functions']['sim28_u16']['instructions']
        instructions=[]
        prefix=[]
        for r in decoded:
            sass=r['sass']; pred=sass.startswith('@')
            body=sass.split(None,1)[1] if pred else sass
            opcode=body.split()[0]
            row={'id':r['pc']//16,'function_id':0,'offset':r['pc'],'binary_hex':r['binary_hex'],
                'sass':sass,'opcode':opcode,'opcode_short':opcode.split('.')[0],'has_pred':pred,
                'pred_num':0 if pred else -1,'pred_uniform':False,'pred_neg':False,
                'size':2 if opcode.startswith(('LDG','STG')) else 0,
                'mref_count':1 if opcode.startswith(('LDG','STG')) else 0,'operands':[]}
            instructions.append(row)
            if not prefix or prefix[-1]['opcode_short']!='EXIT' or prefix[-1]['has_pred']:
                prefix.append(row)
        self.write('static.json',{'schema':'arch-explorer.sim28-nvbit-static.v1','nvbit':'1.8.1',
            'kernel':'sim28_u16','functions':[{'id':0,'name':'sim28_u16'}],'instructions':instructions,
            'fixture_origin':'SYNTHETIC CPU specimen, not NVBit capture'})
        self.write('launch.json',{'schema':'arch-explorer.sim28-launch.v1','case':'u16','function':'sim28_u16',
            'grid':[4,1,1],'block':[128,1,1],'n':501,'allocated_elements':512,
            'buffers':[{'name':'in16','address':0x100000,'element_bytes':2,'output':False},
                       {'name':'out16','address':0x200000,'element_bytes':2,'output':True}]})
        self.records=[]
        for cta in range(4):
            for warp in range(4):
                active=0xffffffff
                valid=sum(1<<lane for lane in range(32) if 128*cta+32*warp+lane<501)
                for sequence,ins in enumerate(prefix):
                    executed=(active&~valid) if ins['has_pred'] else active
                    addresses=[0]*32
                    if ins['opcode_short'] in ('LDG','STG'):
                        base=0x100000 if ins['opcode_short']=='LDG' else 0x200000
                        for lane in range(32):
                            if executed&(1<<lane): addresses[lane]=base+2*(128*cta+32*warp+lane)
                    self.records.append([ins['id'],cta,0,0,warp,active,executed,sequence,*addresses])
                    if ins['opcode_short']=='EXIT': active&=~executed
        self.save_records()
        expected=bytearray(b'\xa5'*1024)
        for i in range(501): struct.pack_into('<H',expected,2*i,i*13+24)
        for name in ('out16.expected.bin','out16.actual.bin'): (self.d/name).write_bytes(expected)

    def tearDown(self): self.temp.cleanup()
    def write(self,name,value): (self.d/name).write_text(json.dumps(value))
    def save_records(self):
        n=len(self.records)
        header=struct.pack('<8s14I',b'SIM28NV1',1,64,288,n,n,4096,0,1,4,1,1,128,1,1)
        (self.d/'records.bin').write_bytes(header+b''.join(struct.pack('<8I32Q',*r) for r in self.records))
        self.write('capture.json',{'status':'captured','overflow':0,'application_launches':1,'count':n,'attempted':n})

    def test_positive_is_synthetic(self):
        result=verify(self.d,BUILD/'kernels.cubin')
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['global_memory_active_lanes'],{'LDG_2':501,'STG_2':501})

    def test_observed_wrong_address_rejected(self):
        row=next(r for r in self.records if any(r[8:])); row[8]+=8
        self.save_records()
        with self.assertRaisesRegex(ValueError,'MREF address'): verify(self.d,BUILD/'kernels.cubin')

    def test_dropped_pred_off_record_even_if_renumbered(self):
        index=next(i for i,r in enumerate(self.records) if r[6]==0)
        dropped=self.records.pop(index)
        for row in self.records[index:]:
            if row[1]==dropped[1] and row[4]==dropped[4]: row[7]-=1
        self.save_records()
        with self.assertRaisesRegex(ValueError,'missing/extra dynamic original'): verify(self.d,BUILD/'kernels.cubin')

    def test_wrong_static_encoding_rejected(self):
        static=json.loads((self.d/'static.json').read_text())
        static['instructions'][0]['binary_hex']='00'*16;self.write('static.json',static)
        with self.assertRaisesRegex(ValueError,'raw instruction'): verify(self.d,BUILD/'kernels.cubin')

    def test_wrong_decoded_opcode_with_correct_bytes_rejected(self):
        static=json.loads((self.d/'static.json').read_text())
        static['instructions'][0]['opcode']='NOP';static['instructions'][0]['opcode_short']='NOP'
        self.write('static.json',static)
        with self.assertRaisesRegex(ValueError,'opcode metadata differs'): verify(self.d,BUILD/'kernels.cubin')

    def test_tail_value_rejected(self):
        raw=bytearray((self.d/'out16.actual.bin').read_bytes());raw[-1]=0
        (self.d/'out16.actual.bin').write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'actual output or tail'): verify(self.d,BUILD/'kernels.cubin')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True)
    args,rest=p.parse_known_args();BUILD=args.build.resolve()
    unittest.main(argv=['test_verify',*rest])
