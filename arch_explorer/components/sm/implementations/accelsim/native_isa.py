# Concrete Accel-Sim implementation; owned by the component provider.
"""Pinned native binary-version dispatch, separate from adapter eligibility.

Source: Accel-Sim 1c9cc831c5d948f89fa9dc8f21f6c87952f5f2ea,
gpu-simulator/trace-driven/trace_driven.cc:198-216 and ISA_Def/*_opcode.h.
A dispatch entry certifies neither every operand nor physical timing accuracy.
"""
import re

from arch_explorer.components.accelsim.profile import ACCELSIM_SHA

NATIVE_ISAS = {35: 'kepler', 60: 'pascal', 61: 'pascal', 70: 'volta',
               75: 'turing', 80: 'ampere', 86: 'ampere', 90: 'hopper'}
ADAPTER_ISAS = frozenset({'sm90'})
SOURCE = {'commit': ACCELSIM_SHA,
          'dispatch': 'gpu-simulator/trace-driven/trace_driven.cc:198-216',
          'constants': 'gpu-simulator/ISA_Def/{kepler,pascal,volta,turing,ampere,hopper}_opcode.h'}


class ISAError(ValueError):
    def __init__(self, blocker):
        self.kind = blocker['kind']
        super().__init__(self.kind + ': ' + blocker['reason'])


def isa_facts(isa, *, engine=None):
    match = re.fullmatch(r'sm([0-9]{2,3})([af]?)', isa) if type(isa) is str else None
    number = int(match[1]) if match else None
    result = {'isa': isa, 'binary_version': number, 'native_dispatch': number in NATIVE_ISAS,
            'native_family': NATIVE_ISAS.get(number), 'adapter_supported': isa in ADAPTER_ISAS if type(isa) is str else False,
            'source': SOURCE}
    if engine is not None:
        from arch_explorer.components.accelsim.runner import load_engine
        from arch_explorer.components.accelsim.variant import SCHEMA,validate
        manifest,_,_,sha=load_engine(engine)
        result['engine_manifest_sha256']=sha
        if manifest['schema']==SCHEMA and number==120 and isa in ('sm120','sm120a'):
            variant=validate(manifest)
            result.update(native_dispatch=True,native_family='blackwell-ordinary-mechanism',adapter_supported=True,
                source={'upstream':SOURCE,'variant':variant},physical_profile='experimental-unvalidated')
    return result


def isa_blocker(isa, *, engine=None):
    facts = isa_facts(isa,engine=engine)
    if not facts['native_dispatch']:
        return {'kind': 'native_isa', 'reason': 'ISA '+str(isa)+' has no OpcodeMap dispatch in the fixed native source; relabeling forbidden'}
    if not facts['adapter_supported']:
        return {'kind': 'adapter_isa', 'reason': 'ISA '+str(isa)+' has native '+facts['native_family']+' dispatch, but this adapter validates ordinary SM90 only'}
    return None


def require_adapter_isa(isa):
    blocker = isa_blocker(isa)
    if blocker:
        raise ISAError(blocker)
