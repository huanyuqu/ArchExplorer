"""Explicit CLIP roots and thin public API consumers; no model semantics here."""
from copy import deepcopy

from .transformer import _values


def _case(input_case):
    if input_case not in ('regular', 'boundary', 'constant'):
        raise ValueError('input_case must be regular, boundary or constant')


def _token_ids(shape, vocab):
    """Deterministic integer roots with repeats, endpoints and a maximum-ID tie.

    This generates data only. It never computes EOT indices, embedding rows,
    causal masks or any model answer, and it never casts caller-supplied IDs.
    """
    if (len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape)
            or type(vocab) is not int or vocab <= 0):
        raise ValueError('Synthetic tokens need a positive two-dimensional shape and vocabulary')
    batch, context = shape
    rows = []
    for index in range(batch):
        row = [(index * 3 + position) % max(1, vocab - 1) for position in range(context)]
        marked = (index + 1) % context
        row[marked] = vocab - 1
        if index == 0 and marked < context - 1:
            row[-1] = vocab - 1
        rows.append(row)
    return rows


def clip_inputs(*, image_batch=2, text_batch=3, image_size=4, patch=2, width=4,
                layers=2, heads=2, mlp=7, context=4, vocab=13, text_width=4,
                text_heads=2, text_layers=2, projection=3, input_case='regular'):
    """Return every flat root using public shapes, without computing the model.

    Floating values are unrounded. Token IDs are Python integers; logit_scale
    is explicit FP32 data under the builder's mixed-storage contract.
    """
    from arch_explorer.workload.api import clip_input_shapes
    _case(input_case)
    shapes = clip_input_shapes(image_batch=image_batch, text_batch=text_batch,
        image_size=image_size, patch=patch, width=width, layers=layers, heads=heads,
        mlp_hidden=mlp, context=context, vocab=vocab, text_width=text_width,
        text_heads=text_heads, text_layers=text_layers, projection=projection)
    roots = {}
    for index, (name, shape) in enumerate(sorted(shapes.items())):
        if name == 'token_ids':
            roots[name] = _token_ids(shape, vocab)
        elif name == 'logit_scale':
            roots[name] = _values(shape, pattern='constant', scale=0.5)
        elif name == 'images':
            roots[name] = _values(shape, pattern=input_case)
        else:
            local = name.rsplit('.', 1)[-1]
            offset = index * 19 + 7
            if local.endswith('_gamma'):
                roots[name] = _values(shape, offset=offset, scale=0.125,
                                      shift=1.0 + index / 1024.0)
            elif local in ('bq', 'bk', 'bv', 'bo', 'b1', 'b2') or local.endswith('_beta'):
                roots[name] = _values(shape, offset=offset, scale=0.0625,
                                      shift=index / 4096.0)
            else:
                roots[name] = _values(shape, offset=offset, scale=0.25,
                                      shift=index / 1024.0)
    return roots


def clip_workload(*, image_batch=2, text_batch=3, image_size=4, patch=2, width=4,
                  layers=2, heads=2, mlp=7, context=4, vocab=13, text_width=4,
                  text_heads=2, text_layers=2, projection=3, dtype='fp32',
                  input_case='regular', inputs=None):
    """Call public clip_similarity, retaining supplied roots without default fill."""
    from arch_explorer.workload.api import clip_similarity
    _case(input_case)
    if any(type(count) is not int or not 2 <= count <= 4 for count in (layers, text_layers)):
        raise ValueError('layers and text_layers must be integers from 2 through 4')
    roots = clip_inputs(image_batch=image_batch, text_batch=text_batch,
        image_size=image_size, patch=patch, width=width, layers=layers, heads=heads,
        mlp=mlp, context=context, vocab=vocab, text_width=text_width,
        text_heads=text_heads, text_layers=text_layers, projection=projection,
        input_case=input_case) if inputs is None else deepcopy(inputs)
    return clip_similarity(inputs=roots, image_batch=image_batch, text_batch=text_batch,
        image_size=image_size, patch=patch, width=width, layers=layers, heads=heads,
        mlp_hidden=mlp, context=context, vocab=vocab, text_width=text_width,
        text_heads=text_heads, text_layers=text_layers, projection=projection, dtype=dtype)


def _selection(candidate, layout):
    if type(candidate) is not int or candidate not in (0, 1):
        raise ValueError('candidate must be 0 or 1')
    if layout not in ('materialized', 'views'):
        raise ValueError('layout must be materialized or views')


def _problem(*, hardware, **options):
    from .compilation import paired_board
    from .compiled_configs import per_operator_mappings
    adf, points, memory = paired_board(hardware)
    spec = clip_workload(**options)
    choices = per_operator_mappings(spec, points, memory,
                                    matmul_tiles=((8, 8, 8), (6, 7, 10)))
    return adf, spec, choices


def clip_scene(*, image_batch=2, text_batch=3, image_size=4, patch=2, width=4,
               layers=2, heads=2, mlp=7, context=4, vocab=13, text_width=4,
               text_heads=2, text_layers=2, projection=3, dtype='fp32',
               input_case='regular', inputs=None, hardware=None,
               candidate=0, layout='materialized'):
    """Return ordinary (ADF, WorkloadSpec, MappingSpec), using public layouts."""
    from arch_explorer.workload.api import layout_candidates
    _selection(candidate, layout)
    adf, spec, choices = _problem(hardware=hardware, image_batch=image_batch,
        text_batch=text_batch, image_size=image_size, patch=patch, width=width,
        layers=layers, heads=heads, mlp=mlp, context=context, vocab=vocab,
        text_width=text_width, text_heads=text_heads, text_layers=text_layers,
        projection=projection, dtype=dtype, input_case=input_case, inputs=inputs)
    return adf, spec, layout_candidates(spec, choices[candidate])[layout]


def clip_search_config(*, image_batch=2, text_batch=3, image_size=4, patch=2, width=4,
                       layers=2, heads=2, mlp=7, context=4, vocab=13, text_width=4,
                       text_heads=2, text_layers=2, projection=3, dtype='fp32',
                       input_case='regular', inputs=None, hardware=None,
                       candidate=0, layout='materialized', compare_layouts=True):
    """Ordinary seeded search: two placements, both layouts and three proposals."""
    from arch_explorer.workload.api import layout_candidates
    from .compiled_configs import bounded_search_config
    _selection(candidate, layout)
    if type(compare_layouts) is not bool:
        raise ValueError('compare_layouts must be a boolean')
    adf, spec, choices = _problem(hardware=hardware, image_batch=image_batch,
        text_batch=text_batch, image_size=image_size, patch=patch, width=width,
        layers=layers, heads=heads, mlp=mlp, context=context, vocab=vocab,
        text_width=text_width, text_heads=text_heads, text_layers=text_layers,
        projection=projection, dtype=dtype, input_case=input_case, inputs=inputs)
    layouts = ((layout, 'views' if layout == 'materialized' else 'materialized')
               if compare_layouts else (layout,))
    requests = []
    for index in (candidate, 1 - candidate):
        products = layout_candidates(spec, choices[index])
        requests.extend(products[selected] for selected in layouts)
    return bounded_search_config(adf, spec, requests,
        source='bounded complete CLIP example; explicit synthetic hardware proposal')


def clip_agent_config(*, image_batch=2, text_batch=3, image_size=4, patch=2, width=4,
                      layers=2, heads=2, mlp=7, context=4, vocab=13, text_width=4,
                      text_heads=2, text_layers=2, projection=3, dtype='fp32',
                      input_case='regular', inputs=None, hardware=None,
                      candidate=0, layout='materialized'):
    """Offline discovery/evaluate/submit with one exact candidate and one unit."""
    from .compiled_configs import scripted_agent_config
    return scripted_agent_config(clip_search_config(image_batch=image_batch,
        text_batch=text_batch, image_size=image_size, patch=patch, width=width,
        layers=layers, heads=heads, mlp=mlp, context=context, vocab=vocab,
        text_width=text_width, text_heads=text_heads, text_layers=text_layers,
        projection=projection, dtype=dtype, input_case=input_case, inputs=inputs,
        hardware=hardware, candidate=candidate, layout=layout, compare_layouts=False))
