"""One objective for candidate selection and source-validated winner replay."""


def score(evaluations, cost, weights, limits, penalties):
    metrics = {key: cost[key] for key in ("cost_usd", "area_mm2")}
    unavailable = [key for key, limit in limits.items() if limit is not None and metrics[key] is None]
    latency = sum(weight * evaluation["latency_ns"] for weight, evaluation in zip(weights, evaluations))
    charges = {key: penalties[key] * max(0, metrics[key] - limit)
               for key, limit in limits.items() if limit is not None and metrics[key] is not None}
    feasible = not unavailable and all(limit is None or metrics[key] <= limit for key, limit in limits.items())
    return {"objective": None if unavailable else latency + sum(charges.values()),
            "weighted_latency_ns": latency, "metrics": metrics, "penalty": charges,
            "feasible": feasible, "unavailable_constraints": unavailable,
            "budget_utilization": {key: metrics[key] / limit if limit is not None and metrics[key] is not None else None
                                   for key, limit in limits.items()}}
