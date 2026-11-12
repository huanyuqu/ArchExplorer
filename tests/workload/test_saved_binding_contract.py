"""Digest controls only: these synthetic envelopes do not claim product replay."""
from copy import deepcopy
import unittest

from arch_explorer.workload import compilation as compiler
from saved_binding_contract import BindingLayer, check_layers, reseal_layers


LAYERS = (
    BindingLayer((), 'control.outer', frozenset(('schema', 'result', 'binding'))),
    BindingLayer(('result',), 'control.result', frozenset(('schema', 'request', 'binding'))),
    BindingLayer(('result', 'request'), 'control.request', frozenset(('schema', 'compilation', 'binding'))),
    BindingLayer(('result', 'request', 'compilation'), 'control.compilation',
                 frozenset(('schema', 'payload', 'binding'))),
)


def fixture():
    compilation = compiler.seal(dict(schema='control.compilation', payload=1))
    request = compiler.seal(dict(schema='control.request', compilation=compilation))
    result = compiler.seal(dict(schema='control.result', request=request))
    return compiler.seal(dict(schema='control.outer', result=result))


class BindingControls(unittest.TestCase):
    def test_retaining_old_binding_before_seal_fails_digest_check(self):
        original = fixture()['result']['request']['compilation']
        changed = deepcopy(original)
        changed['payload'] = 2
        invalid = compiler.seal(changed)
        with self.assertRaisesRegex(ValueError, '^compilation content binding differs$'):
            compiler.check(invalid, set(LAYERS[-1].fields), LAYERS[-1].schema)

    def test_all_nested_digests_are_valid_before_any_semantic_claim(self):
        original = fixture()
        changed = deepcopy(original)
        changed['result']['request']['compilation']['payload'] = 2
        before = deepcopy(changed)
        sealed, receipts = reseal_layers(changed, LAYERS)
        self.assertEqual(changed, before)
        self.assertEqual(original['result']['request']['compilation']['payload'], 1)
        self.assertEqual(sealed['result']['request']['compilation']['payload'], 2)
        self.assertEqual(receipts, check_layers(sealed, LAYERS))
        self.assertEqual(len(receipts), 4)
        for layer, receipt in zip(LAYERS, receipts):
            old, new = original, sealed
            for key in layer.path:
                old, new = old[key], new[key]
            self.assertNotEqual(old['binding'], new['binding'])
            self.assertTrue(receipt['binding_valid'])
            self.assertEqual(receipt['scope'], 'digest-only; product semantic check pending')

    def test_omitting_intermediate_bound_envelope_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '^binding layers omit a bound ancestor$'):
            reseal_layers(fixture(), (LAYERS[0], LAYERS[-1]))


if __name__ == '__main__':
    unittest.main()
