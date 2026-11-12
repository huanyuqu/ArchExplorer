"""Declarative connections between physical component instances.

These records carry no device state or timing model. Ports and their protocol
are checked by the execution assembly before it starts its clocks.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Connection:
    source: str
    source_port: str
    destination: str
    destination_port: str
    protocol: str = 'transaction.v1'

    def __post_init__(self):
        for value in (self.source, self.source_port, self.destination,
                      self.destination_port, self.protocol):
            if not isinstance(value, str) or not value or any(c.isspace() for c in value):
                raise ValueError('connection endpoints/protocol must be nonempty tokens')

    def to_dict(self):
        return dict(source=self.source, source_port=self.source_port,
                    destination=self.destination, destination_port=self.destination_port,
                    protocol=self.protocol)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {
                'source', 'source_port', 'destination', 'destination_port', 'protocol'}:
            raise ValueError('exact dynamic connection fields required')
        return cls(**value)


def instances(root, maximum=100000):
    """Yield physical point paths, expanding repeats without changing the ADF."""
    from .model import SpaceMatrix

    if type(maximum) is not int or maximum <= 0:
        raise ValueError('physical expansion bound must be a positive integer')
    visited = 0

    def expand(node, prefix):
        nonlocal visited
        for index in range(node.repeat):
            visited += 1
            if visited > maximum:
                raise ValueError('dynamic physical instance expansion exceeds its CPU bound')
            suffix = '['+str(index)+']' if node.repeat != 1 else ''
            path = '/'.join(filter(None, (prefix, node.name+suffix)))
            if isinstance(node, SpaceMatrix):
                for child in (*node.elements, *node.networks):
                    yield from expand(child, path)
            else:
                yield path, node
    yield from expand(root, '')


def validate_connections(root):
    from .model import SpaceMatrix, walk
    matrices = [node for _, node in walk(root) if isinstance(node, SpaceMatrix)]
    if any(not isinstance(node.connections, list) for node in matrices):
        raise TypeError('matrix connections must be a list')
    if not any(node.connections for node in matrices):
        return
    paths = dict(instances(root))
    seen = set()
    for node in matrices:
        for connection in node.connections:
            if not isinstance(connection, Connection):
                raise TypeError('matrix connections must contain Connection records')
            if connection.source not in paths or connection.destination not in paths:
                raise ValueError('connection refers to an absent physical component instance')
            key = tuple(connection.to_dict().values())
            if key in seen:
                raise ValueError('duplicate dynamic connection')
            seen.add(key)
