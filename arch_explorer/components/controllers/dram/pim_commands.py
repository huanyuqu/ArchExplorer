"""Optional NeuPIMs simulator semantics, not a hardware calibration.

Source: NeuPIMs f299af3 / extern/NewtonSim: configuration.cc (delays),
channel_state.cc::EstimatePIMOperationLatency / UpdateChannelTiming,
refresh.cc (rank stagger), neupims_controller.cc::IssueCommand.
No measured/reference latency is used here. All times are ns.
"""
from dataclasses import dataclass
import math


def rank_activations(banks, act_banks, ranks, trrd, tfaw, tick=1., start=0., available=None):
    """Earliest G_ACT on a single command bus with independent rank windows."""
    if banks % ranks or (banks // ranks) % act_banks:
        raise ValueError("rank banks must divide into complete activation groups")
    count = banks // ranks // act_banks
    times = [[] for _ in range(ranks)]
    bus = start
    while sum(map(len, times)) < count * ranks:
        options = []
        for rank, issued in enumerate(times):
            if len(issued) == count:
                continue
            t = max(bus, issued[-1]+trrd if issued else start)
            window = max(1, 4 // act_banks)
            if len(issued) >= window:
                t = max(t, issued[-window]+tfaw)
            if available is not None:
                t = available(t, rank)
            options.append((t, rank))
        t, rank = min(options)
        times[rank].append(t)
        bus = t + tick
    return sorted((t, r) for r, values in enumerate(times) for t in values)


@dataclass(frozen=True)
class RankRefresh:
    trfc: float
    trefi: float
    ranks: int
    tick: float = 1.

    def available(self, time, rank):
        first = (rank + 1) * self.trefi / self.ranks
        if time < first:
            return time
        last = first + math.floor((time-first)/self.trefi)*self.trefi
        return max(time, last+self.trfc)

    def admit(self, time, estimate):
        """CanMeetRefreshDeadline requires positive slack, not >= zero.

        Header may enter during an ongoing rank refresh; G_ACT waits per rank.
        Epoch is simulation zero, never reset at an Operation boundary.
        """
        period = self.trefi / self.ranks
        if estimate + self.tick >= period:
            raise ValueError("PIM header estimate cannot fit between staggered refresh deadlines")
        deadline = (math.floor(time/period)+1)*period
        if deadline - self.tick - time <= estimate:
            return deadline + self.tick
        return time


def softmax_cycles(size_bytes, lanes):
    """NeuPIMSystolicWS uses Instruction.size (allocated bytes), not elements."""
    def tree(size):
        if size <= lanes:
            return 1
        n = math.ceil(size/lanes)
        return n + tree(n)
    return 2*tree(size_bytes) + 3*math.ceil(size_bytes/lanes)


def combined_times(issue, comps, tccd, tcl, burst, tick=1.):
    """Issue, callback, next COMPS_READRES, and SRAM consumption boundaries.

    The last tick represents the SimpleInterconnect response leg; it is an
    explicit mapping choice, not included in the DRAM callback.
    """
    return dict(issue=issue, returned=issue+tcl+burst,
                bank_ready=issue+tccd*(max(comps, 6)+1),
                consumer_ready=issue+tcl+burst+tick)


def write_response(accepted, physical_end, tick, early=False):
    """NewtonSim AddTransaction WRITE callback is accept+1 tick when enabled."""
    return accepted+tick if early else physical_end
