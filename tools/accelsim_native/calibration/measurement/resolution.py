# SPDX-License-Identifier: Apache-2.0
"""Conditional elapsed-time bounds; no device access and no physical fitting.

CUDA specifies about 500 ns resolution, not a guaranteed accuracy bound. An
epsilon supplied here is a separate explicit bounded-error assumption.
"""
import math
import struct


def event_ms(bits):
    if isinstance(bits,bool) or not isinstance(bits,int) or not 0<=bits<2**32:
        raise ValueError('raw float32 bit pattern required')
    result=struct.unpack('<f',struct.pack('<I',bits))[0]
    if not math.isfinite(result) or result<0:
        raise ValueError('nonnegative finite elapsed value required; retain rejected raw bits')
    return result


def slope_interval(low_ns,high_ns,step_gap,epsilon_ns):
    """Bound difference per round under independent |elapsed error| <= epsilon.

    Averages of three samples have the same worst-case bound. No positive-
    latency clamp or core-frequency conversion hides a nonidentifiable result.
    """
    if isinstance(step_gap,bool) or not isinstance(step_gap,int) or step_gap<=0:
        raise ValueError('positive integer step gap required')
    if any(not math.isfinite(x) for x in (low_ns,high_ns,epsilon_ns)):
        raise ValueError('finite observed times and assumed bound required')
    if min(low_ns,high_ns,epsilon_ns)<0:raise ValueError('nonnegative values required')
    difference=high_ns-low_ns
    half_width=2*epsilon_ns/step_gap
    center=difference/step_gap
    return {'quantity':'conditional elapsed_ns per round, including confounders',
            'difference_ns':difference,'step_gap':step_gap,'assumed_epsilon_ns':epsilon_ns,
            'lower_ns_per_round':center-half_width,'upper_ns_per_round':center+half_width,
            'half_width_ns_per_round':half_width,
            'relative_quantization_le_10pct':difference>0 and 2*epsilon_ns<=difference*.1,
            'physical_parameter_identified':False}
