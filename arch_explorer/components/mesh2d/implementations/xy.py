from __future__ import annotations
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import GATEWAY, Demand, NetworkModel

from arch_explorer.components._support.network import FAIR, PortedModel


class MeshModel(NetworkModel):
    """XY routing, optionally preceded by KING_MESH diagonal steps, on directed links."""

    def __init__(self, component, point, context):
        super().__init__(component, point, context)
        self.rows, self.cols = self.positive("rows"), self.positive("cols")
        self.link_bandwidth = self.positive("link_bandwidth")
        self.hop_latency = self.param("hop_latency")
        self.gateway_tile = self.param("gateway_tile", 0)
        self.diagonal = self.param("diagonal", False)
        tiles = self.rows * self.cols
        if self.endpoints > tiles:
            raise ValueError(f"{self.path}: {self.endpoints} endpoints do not fit a {self.rows}x{self.cols} mesh")
        if not 0 <= self.gateway_tile < tiles:
            raise ValueError(f"{self.path}: gateway_tile {self.gateway_tile} outside the mesh")
        self.links = {}

    def tile(self, endpoint):
        index = self.gateway_tile if endpoint == GATEWAY else endpoint
        return divmod(index, self.cols)

    def link(self, a, b):
        key = (a, b)
        if key not in self.links:
            self.links[key] = self.resource(f"link{a}->{b}", self.link_bandwidth, "B/ns")
        return self.links[key]

    def route(self, src, dst):
        self._check(src, dst)
        (r, c), (r2, c2) = self.tile(src), self.tile(dst)
        hops = []
        while self.diagonal and r != r2 and c != c2:
            nr, nc = r + (1 if r2 > r else -1), c + (1 if c2 > c else -1)
            hops.append(self.link((r, c), (nr, nc)))
            r, c = nr, nc
        while c != c2:
            step = 1 if c2 > c else -1
            hops.append(self.link((r, c), (r, c + step)))
            c += step
        while r != r2:
            step = 1 if r2 > r else -1
            hops.append(self.link((r, c), (r + step, c)))
            r += step
        return Demand({link.name: 1.0 for link in hops}, len(hops) * self.hop_latency)

