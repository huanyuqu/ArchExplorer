"""An attributed upstream SM90 seed; this does not describe a Blackwell GPU."""
from arch_explorer.adf import Param, SpaceMatrix, SpacePoint
from arch_explorer.components.accelsim.profile import CLOCKS, FIELDS, GPGPUSIM_SHA, ACCELSIM_SHA


def sm90_adf():
    """Return the complete supported H100 configuration as five native components."""
    def point(owner):
        params = {}
        for group, key, value, unit, location in FIELDS:
            if group == owner:
                sha = ACCELSIM_SHA if location.startswith('gpu-simulator/') else GPGPUSIM_SHA
                params[key] = Param(value, unit, f'Accel-Sim v2.0.1 {sha}/{location}')
        if owner in CLOCKS:
            params['clock_mhz'] = Param(CLOCKS[owner], 'MHz',
                f'GPGPU-Sim {GPGPUSIM_SHA}/configs/tested-cfgs/SM90_H100/gpgpusim.config:35')
        return SpacePoint(owner, f'accelsim_{owner}', params, repeat=2 if owner == 'l2' else 1)
    return SpaceMatrix('gpu', [
        SpaceMatrix('cluster', [point('sm'), point('l1_shared')], repeat=132),
        SpaceMatrix('partition', [point('l2'), point('dram')], repeat=40)], [point('icnt')])


def sm120_mechanism_adf():
    """Complete synthetic mechanism fixture. No parameter describes target hardware.

    Reuses the attributed native parameter vocabulary, with a two-SM/40-channel
    synthetic topology and explicit typed-bank model. Never use for accuracy.
    """
    from arch_explorer.adf import walk
    adf = sm90_adf()
    adf.name = 'synthetic_sm120_mechanism'
    adf.child('cluster').repeat = 2
    adf.child('partition').repeat = 40
    sm = adf.child('cluster').child('sm')
    sm.component = 'accelsim_sm120_mechanism'
    sm.params['sim28_typed_regs'] = Param(1, '1', 'synthetic SIM28 logical register-bank model; uncalibrated')
    adf.child('cluster').child('l1_shared').params['gpgpu_coalesce_arch'] = Param(120, '1', 'synthetic sector coalescing mechanism; not a device resource')
    adf.networks[0].params['booksim_k'] = Param(82, 'count', 'synthetic topology: two SMs plus eighty memory subpartitions; valid fixed IPOLY-MODULO factors')
    for _, node in walk(adf):
        if isinstance(node, SpacePoint):
            for key, value in list(node.params.items()):
                node.params[key] = Param(value.value, value.unit,
                    'synthetic mechanism fixture; inherited native vocabulary; NOT measured or target physical parameters; '+value.source)
    return adf


def device_scenario(assumptions, evidence):
    """Bind historical facts to an explicitly supplied complete synthetic ADF.

    This reads an existing file only, never a GPU. Unknown parameters have no
    implicit target defaults; select topology/clocks and mark their assumptions.
    The config compiler rejects contradictory facts, units or native topology.
    """
    from arch_explorer.components.accelsim.device import apply_device_facts
    return apply_device_facts(assumptions, evidence)
