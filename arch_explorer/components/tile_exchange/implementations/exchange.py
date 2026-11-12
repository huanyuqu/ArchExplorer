import math
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import ComputeModel, Demand, NetworkModel



class TileExchangeModel(NetworkModel):
    def __init__(self, component, point, context):
        super().__init__(component, point, context)
        self.bandwidth = self.positive("endpoint_bandwidth")
        self.tiles = self.param("tiles")
        self.topology = self.param("topology")
        self.rows, self.cols = self.param("rows", 1), self.param("cols", self.tiles)
        if type(self.tiles) is not int or self.tiles < 1 or self.endpoints < self.tiles:
            raise ValueError("exchange tiles must match available endpoints")
        if self.topology not in ("all_to_all", "mesh_unknown"):
            raise ValueError("unsupported tile exchange topology")
        if self.topology == "mesh_unknown" and self.bandwidth is not None:
            raise ValueError("mesh_unknown has no established endpoint/link rate")
        if any(type(v) is not int or v < 1 for v in (self.rows, self.cols)) or self.rows * self.cols != self.tiles:
            raise ValueError("exchange rows*cols must equal tiles")
        self.ports = {}

    def port(self, endpoint, direction):
        key = (endpoint, direction)
        if key not in self.ports:
            self.ports[key] = self.resource(f"{direction}{endpoint}", self.bandwidth, "B/ns")
        return self.ports[key]

    def route(self, src, dst):
        self._check(src, dst)
        if src == dst or self.bandwidth is None:
            return Demand()
        demand = {}
        for endpoint, direction in ((src, "tx"), (dst, "rx")):
            if 0 <= endpoint < self.tiles:
                demand[self.port(endpoint, direction).name] = 1.0
        return Demand(demand)

