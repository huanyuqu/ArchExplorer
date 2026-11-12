"""An independent, small synthetic execution domain, not an Accel-Sim adapter.

This example models only serial MAC work and a declared per-tile startup. It
does not model memory traffic, numerical outputs, power or hardware latency.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from arch_explorer.adf import Evaluation, MappingSpec, SpaceMatrix, SpacePoint, WorkloadSpec, from_dict
from arch_explorer.adf.registry import MappingTemplate
from .component import TutorialMac

REQUEST_SCHEMA = "tutorial.serial-mac-request.v1"
RESULT_SCHEMA = "tutorial.serial-mac-result.v1"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path.write_text(_canonical(value) + "\n", encoding="utf-8")


def _seal(value):
    return {**value, "binding": _hash(value)}


def _unseal(value, schema):
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"Expected {schema}")
    if value.get("binding") != _hash({key: item for key, item in value.items() if key != "binding"}):
        raise ValueError("Synthetic provider payload binding mismatch")


def _specs(adf, workload, mapping, options):
    if (not isinstance(adf, SpaceMatrix) or adf.repeat != 1 or adf.networks
            or len(adf.elements) != 1 or not isinstance(adf.elements[0], SpacePoint)
            or adf.elements[0].component != "tutorial_mac" or adf.elements[0].repeat != 1):
        raise ValueError("tutorial.serial_mac owns exactly one tutorial_mac; mixed execution domains are unsupported")
    TutorialMac().validate(adf.elements[0])
    params = workload.params
    if (workload.kind != "operator" or set(params) != {"op", "m", "k", "n", "dtype"}
            or params["op"] != "matmul" or params["dtype"] != "fp32"
            or any(type(params[key]) is not int or not 1 <= params[key] <= 64 for key in ("m", "k", "n"))):
        raise ValueError("tutorial.serial_mac requires explicit FP32 operator/matmul and integer m/k/n in [1, 64]")
    if (mapping.template != "tutorial_serial" or set(mapping.params) != {"tile_m"}
            or type(mapping.params["tile_m"]) is not int or mapping.params["tile_m"] not in (1, 2)):
        raise ValueError("tutorial_serial requires tile_m=1 or 2 (integer)")
    if (not isinstance(options, dict) or set(options) - {"startup_ns"}
            or type(options.get("startup_ns", 5)) not in (int, float)
            or not math.isfinite(options.get("startup_ns", 5)) or options.get("startup_ns", 5) < 0):
        raise ValueError("Only a finite nonnegative startup_ns provider option is supported")
    return {"adf": adf.to_dict(), "workload": workload.to_dict(),
            "mapping": mapping.to_dict(), "options": deepcopy(options)}


def _checked_request(request):
    _unseal(request, REQUEST_SCHEMA)
    if request.get("implementation_sha256") != _file_hash(__file__):
        raise ValueError("Synthetic provider implementation changed; prepare with this version")
    path = Path(request["record_path"])
    if not path.is_absolute() or _file_hash(path) != request["record_sha256"]:
        raise ValueError("Synthetic provider specification file changed")
    record = json.loads(path.read_text(encoding="utf-8"))
    if _canonical(record) != _canonical({key: request[key] for key in ("adf", "workload", "mapping", "options")}):
        raise ValueError("Synthetic provider request/specification mismatch")
    _specs(from_dict(record["adf"]), WorkloadSpec(**record["workload"]),
           MappingSpec(**record["mapping"]), record["options"])
    return record


def _evaluation(record):
    adf = from_dict(record["adf"])
    point = adf.elements[0]
    rate = point.params["macs_per_ns"].value
    workload = record["workload"]["params"]
    tile = record["mapping"]["params"]["tile_m"]
    startup = record["options"].get("startup_ns", 5)
    rows, now = [], 0.0
    for offset in range(0, workload["m"], tile):
        count = min(tile, workload["m"] - offset)
        duration = startup + count * workload["k"] * workload["n"] / rate
        rows.append({"operator": "matmul", "task_type": "synthetic-mac-tile", "resource": point.name,
                     "start_ns": now, "end_ns": now + duration, "row_start": offset, "row_count": count})
        now += duration
    provenance = [{"path": f"{adf.name}.{point.name}", "param": key, "value": param.value,
                   "unit": param.unit, "source": param.source} for key, param in point.params.items()]
    return Evaluation(latency_ns=now, breakdown=rows, backend="tutorial.serial_mac",
                      provenance=provenance, warnings=["Synthetic MAC schedule; not measured hardware latency"],
                      details={"scope": "compute only; no memory, numerical-output or energy model",
                               "startup_ns_per_tile": startup, "tiles": len(rows)}).to_dict()


def _checked_result(result):
    _unseal(result, RESULT_SCHEMA)
    record = _checked_request(result["request"])
    if _file_hash(result["execution_path"]) != result["execution_sha256"]:
        raise ValueError("Synthetic execution artifact changed")
    execution = json.loads(Path(result["execution_path"]).read_text(encoding="utf-8"))
    expected = {"request_identity": _hash(result["request"]), "evaluation": _evaluation(record)}
    if _canonical(execution) != _canonical(expected) or _canonical(result["evaluation"]) != _canonical(expected["evaluation"]):
        raise ValueError("Synthetic execution/result mismatch")
    return result


class TutorialSerial(MappingTemplate):
    name = "tutorial_serial"

    def applicable(self, adf, workload):
        try:
            _specs(adf, workload, MappingSpec(self.name, {"tile_m": 1}), {})
            return True
        except (TypeError, ValueError, KeyError):
            return False

    def space(self, adf, workload):
        return {"tile_m": [1, 2]}

    def apply(self, simulator, workload, params):
        raise ValueError("tutorial_serial requires provider='tutorial.serial_mac'; it has no event-graph adapter")


class SerialMacProvider:
    id = "tutorial.serial_mac"
    version = "1"
    component_types = ("tutorial_mac",)
    execution_domain = "single"
    capabilities = {"workloads": ["operator:matmul"], "precisions": ["fp32"],
                    "mappings": ["tutorial_serial"], "scope": "synthetic serial MAC tiles; compute only"}
    request_schemas = (REQUEST_SCHEMA,)
    result_schemas = (RESULT_SCHEMA,)

    def prepare_request(self, adf, workload, mapping, *, output_dir, options=None):
        record = _specs(adf, workload, mapping, {} if options is None else options)
        directory = Path(output_dir).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "specs.json"
        _write(path, record)
        return _seal({"schema": REQUEST_SCHEMA, **record,
                      "record_path": str(path), "record_sha256": _file_hash(path),
                      "implementation_sha256": _file_hash(__file__)})

    def assess_request(self, payload):
        try:
            if isinstance(payload, dict) and payload.get("schema") == RESULT_SCHEMA:
                _checked_result(payload)
            else:
                _checked_request(payload)
        except (KeyError, TypeError, ValueError, OSError) as error:
            return {"supported": False, "gaps": [str(error)]}
        return {"supported": True, "gaps": [], "scope": self.capabilities["scope"]}

    def evaluate_request(self, payload):
        if isinstance(payload, dict) and payload.get("schema") == RESULT_SCHEMA:
            # Verify existing artifacts without writing or running the engine again.
            return deepcopy(_checked_result(payload))
        record = _checked_request(payload)
        evaluation = _evaluation(record)
        path = Path(payload["record_path"]).parent / "execution.json"
        if path.exists():
            raise ValueError("Synthetic request already executed; replay its result or prepare a new directory")
        _write(path, {"request_identity": _hash(payload), "evaluation": evaluation})
        return _seal({"schema": RESULT_SCHEMA, "request": deepcopy(payload), "evaluation": evaluation,
                      "execution_path": str(path), "execution_sha256": _file_hash(path)})


PROVIDER = SerialMacProvider()
