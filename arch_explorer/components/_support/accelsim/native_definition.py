"""Shared native parameter validation, not a hardware category."""
from arch_explorer.adf.registry import ComponentType, ParamSpec
from arch_explorer.components.accelsim.profile import CHOICES, CLOCKS, FIELDS
from arch_explorer.components.accelsim.constraints import RANGES

class NativeComponent(ComponentType):
    implementation_policy = 'alias'
    execution_modes = ()
    execution_domain = 'accelsim'

    def validate(self, point):
        super().validate(point)
        for key, spec in self.params.items():
            if key not in point.params:
                continue
            domain = spec.domain
            valid = (True if domain is None else
                     domain['min'] <= point.value(key) <= domain['max']
                     if isinstance(domain, dict) else point.value(key) in domain)
            if not valid:
                raise ValueError(f"{point.name}.{key}: unsupported value {point.value(key)!r}; supported {spec.domain}")
            if not isinstance(point.params[key].source, str) or not point.params[key].source.strip():
                raise ValueError(f"{point.name}.{key}: source is required")

    def build(self, point, context):
        raise NotImplementedError(f"{self.name} requires a native accelsim/accelsim_components provider; no events Demand model")


def _params(owner):
    result = {key: ParamSpec(unit, 'int' if type(value) is int else 'str',
                            domain=({'min': RANGES[key][0], 'max': RANGES[key][1]}
                                    if key in RANGES else CHOICES.get(key, [value])), doc=source)
              for group, key, value, unit, source in FIELDS if group == owner}
    if owner in CLOCKS:
        result['clock_mhz'] = ParamSpec('MHz', 'int', domain={'min': 1, 'max': 1000000 if owner != 'sm' else 4294},
                                       doc='gpgpu_clock_domains: core:icnt:l2:dram in MHz')
    if owner == 'l1_shared':
        result['gpgpu_adaptive_cache_config'] = ParamSpec('1', 'int', domain=[0, 1], doc='native shared/L1 allocation mode')
        result['gpgpu_shmem_sizeDefault'] = ParamSpec('B', 'int', required=False,
            domain={'min': 1, 'max': 2**30}, doc='native initial shared allocation; derived from shared capacity when omitted')
    if owner == 'dram':
        result['gpgpu_simple_dram_model'] = ParamSpec('1', 'int', domain=[0, 1],
            doc='native detailed DRAM controller (0) or explicit fixed-service assumption (1)')
    if owner == 'sm':
        result['device_evidence'] = ParamSpec('1', 'str', required=False, domain=None,
            doc='SHA-bound original read-only device evidence; metadata only, no duplicate parameter values')
    return result

