"""Traffic inputs for later public-factory consumption, not native API schemas.

Geometry is selected from the documented public configuration before execution.
No latency prediction, hardware calibration, GPU setup or device simulation.
The final driver must demonstrate real hits/misses/writeback/backpressure.
"""


def memory_requests(instance, *, base, line_bytes, resident_lines):
    if (not instance or type(base) is not int or base < 0
            or type(line_bytes) is not int or line_bytes < 32
            or base % line_bytes or type(resident_lines) is not int or resident_lines < 1):
        raise ValueError("aligned address, at least 32B lines and positive resident lines required")
    rows = []

    def add(operation, address, after):
        rid = len(rows) + 1
        rows.append(dict(instance=instance, id=rid, operation=operation,
                         address=address, bytes=32, byte_mask=[1] * 32, after=after))
        return rid

    cold = add("read", base, [])
    dirty = add("write", base, [cold])
    hot = add("read", base, [dirty])
    # More distinct lines than capacity, issued as a burst after the hot access.
    pressure = [add("read", base + index * line_bytes, [hot])
                for index in range(1, 2 * resident_lines + 2)]
    add("read", base, pressure)
    return rows
