"""Explicit store-and-forward transport on the ordinary TaskGraph.

Each edge names an existing network resource. No edge, reverse direction or
bandwidth is inferred. Ports and links keep their component arbitration.
Buffers reside in endpoint memory; capacity is checked on the resulting live
allocation trace. Overflow rejects the mapping (there is no hidden infinite
queue). Propagation holds buffers, never the serialization resource.
"""
from collections import deque
from dataclasses import dataclass

from arch_explorer.components.base import Demand


@dataclass
class Copy:
    ready: int
    held: list
    memory: str
    size: int


class DirectedTransport:
    def __init__(self, ctx, edges, memories):
        self.ctx, self.memories = ctx, memories
        self.edges = edges
        self.messages = []
        self._routes = {}
        self.networks = {n.path: n for c in ctx.sim.containers() for n in c.networks}
        for edge in edges:
            model = self.networks[edge['network']]
            # An edge binds a single shared-capacity link, not an inferred mesh
            # port number. Internal local meshes remain available to local GSM.
            if not hasattr(model, 'bus'):
                raise NotImplementedError('explicit edges require a single bus resource')
            if model.bus.unit != 'B/ns':
                raise ValueError('edge bandwidth must use B/ns')
            if model.latency < 0:
                raise ValueError('edge propagation latency must be nonnegative')

    def hop(self, edge, size, deps, name):
        """Network-layer boundary: serialization then propagation, no ports.

        Used by send() and by isolated network reference measurements. A full
        workload always uses send() so its endpoint traffic is not omitted.
        """
        model = self.networks[edge['network']]
        tx = self.ctx.task(name+'.serialize', Demand({model.bus.name: size}), deps, 'comm', name)
        arrival = self.ctx.sync(name+'.propagate', [tx], name, latency_ns=model.latency)
        return tx, arrival

    def route(self, src, dst):
        key = (src, dst)
        if key in self._routes:
            return self._routes[key]
        queue, seen = deque([(src, [])]), {src}
        while queue:
            node, route = queue.popleft()
            if node == dst:
                self._routes[key] = route
                return route
            for edge in self.edges:  # input order is the deterministic tie rule
                if edge['src'] == node and edge['dst'] not in seen:
                    seen.add(edge['dst'])
                    queue.append((edge['dst'], route + [edge]))
        raise ValueError(f'No directed route from {src} to {dst}')

    def send(self, src, dst, value, *, size=None, name='transfer', deps=(), region=None):
        ctx = self.ctx
        size = value.size if size is None else size
        if type(size) is not int or size < 0 or size > value.size:
            raise ValueError('transfer byte count must fit the source allocation')
        route = self.route(src, dst)
        if not route:
            return value
        current = value
        hops = []
        serial = len(self.messages)
        for i, edge in enumerate(route):
            model = self.networks[edge['network']]
            memory = self.memories[edge['dst']]
            if size > ctx.sim.memory(memory).capacity:
                raise ValueError(f'Receive buffer exceeds capacity at {memory}')
            prefix = f'{name}.message{serial}.hop{i}'
            read = ctx.task(prefix+'.read', ctx.sim.port_demand(current.memory, size),
                            [current.ready, *current.held, *deps], 'comm', name)
            buffer = ctx.storage(prefix+'.receive_buffer', memory, size, [read], name)
            tx, arrival = self.hop(edge, size, [read, buffer], prefix)
            write = ctx.task(prefix+'.write', ctx.sim.port_demand(memory, size),
                             [arrival, buffer], 'comm', name)
            # Hold the source through serialization, and the receive allocation
            # through writeback and every later consumer using Copy.held.
            end = ctx.sync(prefix+'.ready', [write, buffer, *current.held], name)
            hops.append(dict(**edge, serialization_task=tx, arrival_task=arrival,
                             read_task=read, write_task=write, buffer_task=buffer,
                             bytes=size, propagation_ns=model.latency))
            current = Copy(end, [buffer], memory, size)
        self.messages.append(dict(name=name, src=src, dst=dst, logical_bytes=size,
                                  wire_bytes=size*len(route), source_region=region, hops=hops))
        return current
