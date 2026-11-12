"""Entry handling for a declared request whose executor is not yet registered."""
from arch_explorer.adf import MappingSpec
from arch_explorer.adf.multi_device import TEMPLATE, UNSUPPORTED, request_space, validate_request
from .errors import EnvError


def validate(adf, params, components):
    try:
        return validate_request(adf, params, registry=components)
    except (ValueError, TypeError) as error:
        raise EnvError("invalid_mapping", str(error), template=TEMPLATE) from error


def configured(config, count):
    if TEMPLATE not in config.get("mapping_templates", []):
        return None
    params = config.get("mapping_selectors", {}).get(TEMPLATE, {})
    return [MappingSpec(TEMPLATE, params) for _ in range(count)]


def description(adf, params, components):
    params = validate(adf, params, components)
    return {"request_space": request_space(adf, params, registry=components).to_dict(),
            "contract": {"kind": "conditional", "branches": [], "direct_defaults": {},
                         "sampling_defaults": {}, "unavailable": [{"code": "unsupported_mapping",
                                                                      "reason": UNSUPPORTED}]},
            "sampling_spaces": [], "executable": False,
            "requested_mapping": MappingSpec(TEMPLATE, params).to_dict()}
