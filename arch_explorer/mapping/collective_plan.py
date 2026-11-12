"""Explicit rank-ordered collectives; no NCCL/firmware performance claim.

Tree is a rooted star (reduce to ranks[0], then broadcast). All-pairs gather
concatenates rank-ordered equal payloads. Direct reduce-scatter sums balanced
flat intervals at their owners. Unequal tails are exact, with zero padding.
All local reduction additions and memory ports execute on the same graph.
"""
from arch_explorer.components.base import Kernel, dtype_bytes
from arch_explorer.mapping.directed import Copy


def partitions(elements, ranks):
    """Half-open balanced flat ranges in caller rank order, no padded elements."""
    return [(elements*i//len(ranks), elements*(i+1)//len(ranks)) for i in range(len(ranks))]


def collective(ctx, transport, paths, states, values, spec):
    ranks, op, algo = spec['ranks'], spec['operation'], spec['algorithm']
    n, dtype = spec['elements'], spec['dtype']
    item = dtype_bytes(dtype)
    if any(values[r].size != n*item for r in ranks):
        raise ValueError('collective elements/dtype must equal each rank input payload')
    supported = {'allreduce': 'tree', 'allgather': 'all_pairs', 'reduce_scatter': 'direct'}
    if op not in supported or algo != supported[op]:
        raise NotImplementedError(f'collective {op}/{algo}; supported: {supported}')
    if op != 'allgather' and dtype != 'fp32':
        raise ValueError('sum collectives require explicit FP32 partials')
    name = spec['after']+'.'+op
    ledger = dict(**spec, padding_elements=0, reduction_additions=0, output_ranges={},
                  contributions=[], communication_start=len(transport.messages))

    def reduce(rank, left, right, count):
        memory = states[rank].dram
        held = [*left.held, *right.held]
        deps = [left.ready, right.ready, *held]
        size = count*item
        buf = ctx.storage(name+f'.rank{rank}.sum', memory, size, deps, name)
        kernel = Kernel('vector', (1, count), dtype, simt=1)
        unit = states[rank].units_for(kernel)[0]
        task = ctx.map_node(name+f'.rank{rank}.add', unit, [(kernel, 1)], [*deps, buf],
                            [(memory, 2*size), (memory, size, 'write')], name)
        done = ctx.sync(name+'.sum.ready', [task, buf, *held], name)
        ledger['reduction_additions'] += count
        return Copy(done, [buf], memory, size)

    if len(ranks) == 1:
        ledger['output_ranges'] = {str(ranks[0]): [0, n]}
        return dict(values), ledger
    if op == 'allreduce':
        root = ranks[0]
        acc = values[root]
        for rank in ranks[1:]:
            incoming = transport.send(paths[rank], paths[root], values[rank], name=name)
            acc = reduce(root, acc, incoming, n)
            ledger['contributions'].append(dict(src=rank, dst=root, start=0, stop=n, action='sum'))
        result = {root: acc}
        for rank in ranks[1:]:
            result[rank] = transport.send(paths[root], paths[rank], acc, name=name)
        ledger['output_ranges'] = {str(r): [0, n] for r in ranks}
    elif op == 'allgather':
        result = {}
        for dst in ranks:
            pieces = []
            for src in ranks:
                pieces.append(transport.send(paths[src], paths[dst], values[src], name=name))
                ledger['contributions'].append(dict(src=src, dst=dst, start=ranks.index(src)*n,
                                                    stop=(ranks.index(src)+1)*n, action='copy'))
            deps = [p.ready for p in pieces] + [h for p in pieces for h in p.held]
            mem = states[dst].dram
            buf = ctx.storage(name+f'.rank{dst}.assembled', mem, n*item*len(ranks), deps, name)
            copy = ctx.task(name+'.assemble', ctx.sim.port_demand(mem, 2*n*item*len(ranks)),
                            [*deps, buf], 'comm', name)
            done = ctx.sync(name+'.ready', [copy, buf, *deps], name)
            result[dst] = Copy(done, [buf], mem, n*item*len(ranks))
        ledger['output_ranges'] = {str(r): [0, n*len(ranks)] for r in ranks}
    else:
        result = {}
        for dst, (start, stop) in zip(ranks, partitions(n, ranks)):
            if stop == start:
                raise ValueError('reduce_scatter needs at least one element per rank')
            # Intervals select source bytes, not independent full-tensor sends.
            # The initial local interval aliases the full source allocation.
            v = values[dst]
            acc = Copy(v.ready, v.held, v.memory, (stop-start)*item)
            for src in ranks:
                if src == dst:
                    continue
                incoming = transport.send(paths[src], paths[dst], values[src],
                                          size=(stop-start)*item, name=name,
                                          region=dict(flat_start=start, flat_stop=stop, dtype=dtype))
                acc = reduce(dst, acc, incoming, stop-start)
                ledger['contributions'].append(dict(src=src, dst=dst, start=start, stop=stop, action='sum'))
            result[dst] = acc
            ledger['output_ranges'][str(dst)] = [start, stop]
    ledger['communication_stop'] = len(transport.messages)
    return result, ledger
