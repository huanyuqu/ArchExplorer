"""Explicit roots and thin consumers of the public complete ViT builder.

The workload API owns graph construction, shape validation, numerical
boundaries, mapping expansion and layout. This module supplies data/config only.
"""
from copy import deepcopy

from .transformer import _values


def _case(input_case):
    if input_case not in ('regular', 'boundary', 'constant'):
        raise ValueError('input_case must be regular, boundary or constant')


def vit_inputs(*, layers=2, batch=2, image_size=4, patch=2, width=4, heads=2,
               mlp=7, classes=3, input_case='regular'):
    """Return complete flat roots from the public shape helper, without rounding.

    Each block receives independent weights/norm/biases. Images include signed
    data or rounding boundaries; no patch or intermediate activation is computed.
    """
    from arch_explorer.workload.api import vit_classifier_input_shapes
    _case(input_case)
    shapes = vit_classifier_input_shapes(batch=batch, image_size=image_size, patch=patch,
        width=width, layers=layers, heads=heads, mlp_hidden=mlp, classes=classes)
    roots = {}
    for index, (name, shape) in enumerate(sorted(shapes.items())):
        if name == 'images':
            roots[name] = _values(shape, pattern=input_case)
            continue
        local = name.split('.', 1)[-1]
        offset = index * 19 + 7
        if local.endswith('_gamma'):
            roots[name] = _values(shape, offset=offset, scale=0.125, shift=1.0 + index / 1024.0)
        elif local in ('bq', 'bk', 'bv', 'bo', 'b1', 'b2') or local.endswith(('_beta', '_bias')):
            roots[name] = _values(shape, offset=offset, scale=0.0625, shift=index / 4096.0)
        else:
            roots[name] = _values(shape, offset=offset, scale=0.25, shift=index / 1024.0)
    return roots


def vit_workload(*, layers=2, dtype='fp32', batch=2, image_size=4, patch=2, width=4,
                 heads=2, mlp=7, classes=3, input_case='regular', inputs=None):
    """Return the public ViT WorkloadSpec, preserving complete supplied roots."""
    from arch_explorer.workload.api import vit_classifier
    _case(input_case)
    if type(layers) is not int or not 2 <= layers <= 4:
        raise ValueError('layers must be an integer from 2 through 4')
    roots = vit_inputs(layers=layers, batch=batch, image_size=image_size, patch=patch,
        width=width, heads=heads, mlp=mlp, classes=classes, input_case=input_case) if inputs is None else deepcopy(inputs)
    return vit_classifier(inputs=roots, batch=batch, image_size=image_size, patch=patch,
        width=width, layers=layers, heads=heads, mlp_hidden=mlp, classes=classes, dtype=dtype)


def _selection(candidate, layout):
    if type(candidate) is not int or candidate not in (0, 1):
        raise ValueError('candidate must be 0 or 1')
    if layout not in ('materialized', 'views'):
        raise ValueError('layout must be materialized or views')


def _problem(*, layers, dtype, batch, image_size, patch, width, heads, mlp, classes,
             input_case, inputs, hardware):
    from .compilation import paired_board
    from .compiled_configs import per_operator_mappings
    adf, points, memory = paired_board(hardware)
    spec = vit_workload(layers=layers, dtype=dtype, batch=batch, image_size=image_size,
        patch=patch, width=width, heads=heads, mlp=mlp, classes=classes,
        input_case=input_case, inputs=inputs)
    return adf, spec, per_operator_mappings(spec, points, memory)


def vit_scene(*, layers=2, dtype='fp32', batch=2, image_size=4, patch=2, width=4,
              heads=2, mlp=7, classes=3, input_case='regular', inputs=None,
              hardware=None, candidate=0, layout='materialized'):
    """Return ordinary ADF, ViT spec and a complete public layout mapping.

    Stable public composition membership selects explicit per-instance compute
    points and matmul tiles. It is never parsed from a layer-name convention.
    """
    from arch_explorer.workload.api import layout_candidates
    _selection(candidate, layout)
    adf, spec, choices = _problem(layers=layers, dtype=dtype, batch=batch,
        image_size=image_size, patch=patch, width=width, heads=heads, mlp=mlp,
        classes=classes, input_case=input_case, inputs=inputs, hardware=hardware)
    return adf, spec, layout_candidates(spec, choices[candidate])[layout]


def vit_search_config(*, layers=2, dtype='fp32', batch=2, image_size=4, patch=2,
                      width=4, heads=2, mlp=7, classes=3, input_case='regular',
                      inputs=None, hardware=None, candidate=0, layout='materialized',
                      compare_layouts=True):
    """Same roots, two instance choices and optional layout comparison; seed 0."""
    from arch_explorer.workload.api import layout_candidates
    from .compiled_configs import bounded_search_config
    _selection(candidate, layout)
    if type(compare_layouts) is not bool:
        raise ValueError('compare_layouts must be a boolean')
    adf, spec, choices = _problem(layers=layers, dtype=dtype, batch=batch,
        image_size=image_size, patch=patch, width=width, heads=heads, mlp=mlp,
        classes=classes, input_case=input_case, inputs=inputs, hardware=hardware)
    layouts = (layout, 'views' if layout == 'materialized' else 'materialized') if compare_layouts else (layout,)
    requests = []
    for index in (candidate, 1 - candidate):
        products = layout_candidates(spec, choices[index])
        requests.extend(products[selected] for selected in layouts)
    return bounded_search_config(adf, spec, requests,
        source='bounded complete ViT example; explicit synthetic hardware proposal')


def vit_agent_config(*, layers=2, dtype='fp32', batch=2, image_size=4, patch=2,
                     width=4, heads=2, mlp=7, classes=3, input_case='regular',
                     inputs=None, hardware=None, candidate=0, layout='materialized'):
    """Offline discover/evaluate/submit on one exact layout/instance choice."""
    from .compiled_configs import scripted_agent_config
    return scripted_agent_config(vit_search_config(layers=layers, dtype=dtype,
        batch=batch, image_size=image_size, patch=patch, width=width, heads=heads,
        mlp=mlp, classes=classes, input_case=input_case, inputs=inputs,
        hardware=hardware, candidate=candidate, layout=layout, compare_layouts=False))
