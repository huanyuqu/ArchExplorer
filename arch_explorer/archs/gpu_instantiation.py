"""User comparison cases; native topology and mechanisms remain component-owned."""
from copy import deepcopy
import json
import math
from pathlib import Path

CASE_SCHEMA = "arch-explorer.gpu-instantiation-case.v1"


def case_specs(model, *, engine, hardware=None, options=None):
    """Build two standard scenes with identical hardware, program and mappings.

    ``hardware`` supplies public gpu_adf keyword arguments. One new component
    engine manifest selects separate component/original-cycle process modes.
    This function does not load native libraries or run either simulator.
    """
    from arch_explorer.components.assemblies.gpu import gpu_adf
    from arch_explorer.archs.accelsim import program_inputs
    if hardware is not None and not isinstance(hardware, dict):
        raise TypeError("hardware must be a gpu_adf keyword object")
    if options is not None and not isinstance(options, dict):
        raise TypeError("options must be an object")
    if not isinstance(engine, (str, Path)) or not str(engine).strip():
        raise ValueError("Supply the new component engine manifest path")
    engine = str(Path(engine).resolve())
    limits = {"engine": engine, "timeout_s": 600, "max_cycles": 10000000, "memory_limit_mb": 8192}
    limits.update(deepcopy(options or {}))
    if limits["engine"] != engine:
        raise ValueError("Conflicting engine manifest paths")
    timeout = limits["timeout_s"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 600:
        raise ValueError("This example requires timeout_s in (0, 600]")
    memory = limits["memory_limit_mb"]
    if type(memory) is not int or not 0 < memory <= 8192:
        raise ValueError("This example requires integer memory_limit_mb in [1, 8192]")
    if type(limits["max_cycles"]) is not int or limits["max_cycles"] < 1:
        raise ValueError("max_cycles must be a positive integer")
    base = program_inputs(model)
    adf = gpu_adf(**deepcopy(hardware or {}))
    scene = {"adf": adf.to_dict(), "workload": base["workload"], "mappings": base["mappings"],
             "options": limits, "scope": base["scope"]}
    case = {"schema": CASE_SCHEMA,
            "scope": "complete small synthetic " + model + "; component GPU versus independent original-cycle oracle; "
                     "reference-generated FP32 trace, no hardware accuracy claim",
            "candidate": {**deepcopy(scene), "provider": "accelsim_components"},
            "oracle": {**deepcopy(scene), "provider": "accelsim"}}
    return json.loads(json.dumps(case, allow_nan=False))
