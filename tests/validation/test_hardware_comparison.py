"""Synthetic arithmetic and strict-identity controls; no hardware claim."""
import copy
import unittest
from hardware_comparison import compare, compare_pair, SCHEMA


def pair(sample_id='control', measured=100, predicted=110):
    conditions = dict(device=dict(vendor='synthetic',sku='control',architecture='control'),
        software=dict(sdk='control',driver='not-applicable: CPU control',kernel='control'),
        clock=dict(mode='fixed',core_hz=1000,memory_hz=1000),
        operation=dict(op='MatMul',inputs={'A':[2,3],'B':[3,4]},outputs={'Y':[2,4]},attributes={}),
        precision=dict(inputs=['fp32','fp32'],outputs=['fp32'],accumulate='fp32',rounding='rne'),
        mapping=dict(tile=[2,3,4],layout='row-major'),
        timing=dict(start='dispatch',end='synchronized-completion',synchronization='control',warmup=0,statistic='median'))
    def record(kind, value):
        return dict(kind=kind,source=dict(locator='synthetic://unmeasured',sha256='0'*64,
            license='CC0 synthetic control',redistribution='public',evidence_class='synthetic-control',audit='reviewed'),
            conditions=copy.deepcopy(conditions),unit='ns',value=value,repetitions=3,raw_values=[value]*3)
    return dict(sample_id=sample_id,split='holdout',measured=record('measured',measured),predicted=record('predicted',predicted))


class ComparisonTests(unittest.TestCase):
    def test_signed_errors_zeros_units_and_splits(self):
        a,b,c=pair('positive'),pair('negative',100,80),pair('zero',0,3)
        b['split']='development'
        a['predicted'].update(unit='us',value=.11,raw_values=[.11]*3)
        r=compare(dict(schema=SCHEMA,samples=[a,b,c]))
        self.assertEqual(r['rejected'],[])
        self.assertAlmostEqual(r['samples'][0]['signed_error_ns'],10)
        self.assertEqual(r['samples'][1]['signed_relative_error'],-.2)
        self.assertIsNone(r['samples'][2]['absolute_relative_error'])
        self.assertEqual(len(r['groups']),2)
        self.assertFalse(r['provenance_authenticated'])

    def test_identity_mismatches_are_rejected(self):
        changes=[('device','sku','other'),('software','driver','other'),('clock','core_hz',2000),
                 ('operation','inputs',{'A':[3,2],'B':[2,4]}),('precision','accumulate','fp16'),
                 ('mapping','tile',[1,3,4]),('timing','end','host-return')]
        for group,field,value in changes:
            with self.subTest(group=group,field=field):
                p=pair();p['predicted']['conditions'][group][field]=value
                with self.assertRaisesRegex(ValueError,'conditions differ'):compare_pair(p)

    def test_missing_faked_and_nonfinite_records(self):
        for kind in ['measured','predicted']:
            for field in ['device','software','clock','operation','precision','mapping','timing']:
                p=pair();p[kind]['conditions'][field]=None
                with self.assertRaises(ValueError):compare_pair(p)
        p=pair();p['measured']['source']['evidence_class']='paper-digitization'
        with self.assertRaisesRegex(ValueError,'evidence classes'):compare_pair(p)
        p=pair();p['measured']['value']=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):compare_pair(p)
        p=pair();p['measured']['value']=1
        with self.assertRaisesRegex(ValueError,'statistic differs'):compare_pair(p)
        p=pair();del p['predicted']['unit']
        with self.assertRaisesRegex(ValueError,'expected fields'):compare_pair(p)

    def test_duplicate_and_zero_only_denominator(self):
        z=pair('zero',0,0)
        r=compare(dict(schema=SCHEMA,samples=[z,copy.deepcopy(z)]))
        self.assertEqual(len(r['rejected']),2)
        self.assertEqual(r['samples'],[])
        r=compare(dict(schema=SCHEMA,samples=[z]))
        self.assertIsNone(r['groups'][0]['mape_pct'])
        self.assertEqual(r['groups'][0]['relative_error_count'],0)


if __name__=='__main__':unittest.main()
