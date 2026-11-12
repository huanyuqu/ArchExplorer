"""Causal mixed execution. No speculative scheduling or component rollback."""
from dataclasses import replace
import heapq
import math

from .engine import _Engine, allocate, _advance
from .protocol import valid_time


class ConservativeScheduler(_Engine):
    def run(self):
        superfluous = set(self.graph.execution_owners)
        if any(superfluous.intersection(t.demand) for t in self.graph.tasks):
            raise ValueError('Physical resource charged by both Demand and execution model')
        models = {name: factory() for name, factory in self.graph.execution_models.items()}
        if len({id(m) for m in models.values()}) != len(models):
            raise ValueError('Different bindings must not alias one model; share a single binding')
        active, blocked, accepted = [], [], {}
        requests = {}
        completed = set()
        completions, serial = [], 0
        self.events = []
        now = 0
        rate_key, cached_rates = None, {}

        def finish_at(inst, end):
            nonlocal serial
            heapq.heappush(completions, (end, serial, inst))
            serial += 1

        def activate(inst):
            inst.ready = inst.arrival = now
            self._allocate_memory(inst, now)
            if inst.task.execution is not None:
                req = inst.task.execution
                if req.component not in models:
                    raise ValueError(f'Unknown execution component {req.component}')
                blocked.append(inst)
            elif inst.task.demand:
                active.append(inst)
            else:
                inst.start = inst.release = now
                finish_at(inst, inst.task.calendar_end(now))

        def submit(inst):
            req = inst.task.execution
            ident = f'{req.request_id}:{inst.it}'
            if ident in requests:
                raise ValueError(f'Duplicate request id {ident}')
            request = replace(req, request_id=ident, available_time=max(now, req.available_time),
                              iteration=inst.it, operator=inst.task.operator or req.operator)
            if now < request.available_time:
                return
            ok = models[req.component].submit(request, now)
            if type(ok) is not bool:
                raise ValueError('submit must return bool')
            if ok:
                blocked.remove(inst)
                inst.start = now
                accepted[ident] = (inst, req.component)
                requests[ident] = req.component

        for inst in self._sources():
            activate(inst)
        # Retry only on first readiness, explicit queue/resource events, or availability.
        retry = True
        idle_rounds = 0
        while self.done < len(self.instances):
            while completions and completions[0][0] <= now:
                end, _, inst = heapq.heappop(completions)
                for succ in self._finish(inst, end):
                    activate(succ)
                    retry = True
            if self.done == len(self.instances):
                break
            if retry:
                for inst in list(blocked):
                    submit(inst)
                retry = False
            event_times = {}
            for name, model in models.items():
                t = model.next_event_time()
                if t is not None:
                    valid_time(t)
                    if t < now:
                        raise ValueError('Component time moved backwards')
                    event_times[name] = t
            future = [t for t in event_times.values()]
            if completions:
                future.append(completions[0][0])
            future.extend(i.task.execution.available_time for i in blocked
                          if i.task.execution.available_time > now)
            boundary = min(future, default=math.inf)
            # Immutable demands/priorities and unchanged contender order imply
            # identical shares. Calendar/state events alone do not change them.
            key = tuple(active)
            if key != rate_key:
                cached_rates = allocate(active, self.resources, self.normalized) if active else {}
                rate_key = key
            rates = cached_rates
            old = now
            if active:
                # Component boundaries are exact; Demand-only arithmetic retains
                # the established tolerance for simultaneous fluid completions.
                finish = min((now + i.remaining/r for i,r in rates.items() if r > 0), default=math.inf)
                if event_times and boundary < finish:
                    dt = boundary - now
                    for inst, rate in rates.items():
                        if rate > 0:
                            if inst.start is None: inst.start = now
                            inst.remaining -= rate * dt
                    now, done = boundary, set()
                else:
                    now, done = _advance(rates, now, boundary)
                for inst in sorted(done, key=lambda v: v.seq):
                    active.remove(inst)
                    inst.release = now
                    inst.holding = frozenset()
                    finish_at(inst, now + inst.task.latency_ns)
                for inst in rates:
                    inst.portions += 1
            else:
                now = boundary
            if math.isinf(now):
                raise RuntimeError('Execution deadlock: no completion or release can wake pending tasks')
            idle_rounds = idle_rounds + 1 if now == old else 0
            if idle_rounds > 100000:
                raise RuntimeError('Non-progressing same-time component events')
            emitted = []
            for name, t in event_times.items():
                if t == now:
                    for event in models[name].advance(now):
                        if event.time != now:
                            raise ValueError('advance must emit events at its requested time')
                        emitted.append((event, name))
            for event, name in sorted(emitted, key=lambda x: (x[0].time, x[0].phase, x[0].serial, x[1])):
                if event.request_id and requests.get(event.request_id) != name:
                    raise ValueError('Event references unknown request or wrong component')
                self.events.append((name, event))
                if event.kind in {'resource_ready', 'queue_available'}:
                    retry = True
                if event.kind == 'consumer_ready':
                    if event.request_id not in accepted:
                        raise ValueError('Unknown or duplicate consumer_ready request')
                    inst, owner = accepted.pop(event.request_id)
                    if owner != name:
                        raise ValueError('Response from wrong component')
                    inst.release = now
                    completed.add(event.request_id)
                    finish_at(inst, now)
            if any(i.task.execution.available_time == now for i in blocked):
                retry = True
            self.stats['event_steps'] += 1
        trace = self._trace()
        trace.stats['execution_events'] = [dict(component=n, time=e.time, phase=e.phase,
                    serial=e.serial, request_id=e.request_id, kind=e.kind, payload=e.payload)
                    for n, e in self.events]
        trace.stats['pending_execution'] = len(accepted)
        trace.stats['ready_times'] = {f'{i.task.id}:{i.it}': i.arrival for i in self.instances.values()}
        trace.stats['next_component_events'] = {n: m.next_event_time() for n,m in models.items()}
        return trace
