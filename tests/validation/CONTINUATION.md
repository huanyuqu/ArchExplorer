# Independent Reference for Real Export and Physical Sharding

These test tools generate references from actual ONNX ModelProto and original roots, independent of product import, lowering, layout, and executors. The default behavior of `framework_numpy_interpreter.evaluate` is preserved; new models explicitly call `framework_continuation_numpy.evaluate(model, feeds)`. Model names do not participate in dispatch.

## Logical Interface

`onnx_index_expectations` provides `first_argmax(data, axis_value=0, keepdims=1, select_last_index=0)`, `gather(data, indices, axis_value=0)`, `gather_elements`, `shape_of`, `concat`, `slice_tensor`, `expand`, `squeeze`, `unsqueeze`, and `integer_binary`. Indices and shapes remain signed int32/int64; float/bool are not automatically converted to indices. Integer arithmetic uses Python exact integers, and division truncates toward zero; signed 64-bit overflow is outside this reference's admission domain, and cannot be used to claim that ONNX requires wrapping or rejection.

`onnx_conv_expectations.conv_geometry(input_shape, weight_shape, *, strides=(1,1), pads=(0,0,0,0), dilations=(1,1), group=1, auto_pad='NOTSET')` accepts positive-dimension NCHW 2D Conv and returns the complete output shape and actual padding. `conv2d(data, weight, bias=None, *, dtype='fp32', **options)` performs FP32 FMA, FP32 bias addition, and target output storage rounding in channel/kernel row-column order. The logical FMA count for zero-padding multiplication is only for mathematical description; actual memory reads and component billing must be verified from the product execution originals.

`framework_continuation_numpy.evaluate` consumes real node types from IR8/opset17, extending integer/shape, Conv, indexing, broadcasting, reduction, Exp, and multiple outputs. It preserves every original value name, intermediate result, and output type; floating point uses float64 containers to hold rounded declared storage values, and integer values always remain in integer containers. Explicit integer-to-float Cast rounds directly from exact integers, and reductions use sequential FP32 accumulation; BF16 uses a rational-to-binary-grid ties-to-even reference. Parallel framework reductions and fused GELU may differ in rounding from per-ONNX-node evaluation, and differences should be saved.

```sh
python tests/workload/framework_continuation_reference.py \
  --package FROZEN_INPUT_DIRECTORY --case FROZEN_CASE_ID --output NEW_REFERENCE_DIRECTORY
```

Optional `--framework-module FROZEN_GENERATOR.py` only calls its real CPU model construction and forward, does not re-export, and does not call the product. Generates `INDEPENDENT.npz`, independent `OFFICIAL.npz`, optional `FRAMEWORK.npz`, and `RESULT.json`. FP32 compares each original value with preset atol/rtol=5e-5, and integers/bools are compared exactly. Low precision does not use the official evaluator as an oracle, and the status is `reference-generated`; real framework output errors are saved item by item without adjusting thresholds. This stage does not include Graph/component/installation pass conclusions.

## Physical Observation Protocol

`physical_shard_expectations.audit(document)` accepts `arch-explorer.qa-shard-observations.v1`, and all coordinates are non-negative integers:

| Field | Type and Meaning |
| --- | --- |
| tensors | Name to `{elements:int>0,dtype:fp32/fp16/bf16/int64/int32/bool,root:bool}` |
| capacities | Tile name to capacity bytes, positive integer |
| links | List of directed links `[source_tile,target_tile]` |
| buffers | List of `{id,tensor,tile,kind,start,stop,offset,bytes,born,ready,dead}`; kind is root/output/scratch/copy |
| transfers | List of `{id,source,target,start,stop,begin,end,bytes,route}`; source/target are buffer IDs, route is a tile sequence |
| reads | List of `{buffer,tile,at,start,stop}` |

Logical element slices are `[start,stop)`, allocation liveness is `[born,dead)`, and `ready` is the content-ready time. At the same time, deallocation happens before allocation, and read times must be strictly earlier than dead. The original slices of each logical root/output/temporary tensor exactly cover the complete tensor; copies have separate independent allocations and cannot serve as original slices. This protocol is limited to dense contiguous element slices; arbitrary strided views must first provide an auditable independent address expansion, and then be classified into this domain by the adapter.

Check per-tile actual physical addresses and simultaneous liveness peaks, active alias conflicts, last readers of each slice, complete coverage of target copies, source/target liveness and readiness, directed paths, and per-hop transfer bytes. Reads of copies must have corresponding target slices; remote reads cannot reference source buffers for free. Return peaks, peak times, last readers, copy allocation bytes, and per-link bytes.

`test_physical_shard_expectations.sample()` is a small manual control: two cached slices, a two-hop copy, and output reuse of freed copy space; 12B exact-fit, 11B insufficient capacity. It tests the audit tool; actual Events adaptation, node values, observation completeness, and execution identity still require the unique product candidate and originals. The protocol does not authenticate caller-reported times/routes.

## Standard Sources and Reference Divergence

Pin the actual version using local ONNX 1.16.2's `onnx.defs.get_schema(op,17)`. Official documentation supports these semantics: [ArgMax13](https://onnx.ai/onnx/operators/onnx__ArgMax.html#argmax-13), [Gather13](https://onnx.ai/onnx/operators/onnx__Gather.html#gather-13), [Conv11](https://onnx.ai/onnx/operators/onnx__Conv.html#conv-11), [Slice13](https://onnx.ai/onnx/operators/onnx__Slice.html#slice-13). opset17 Conv does not accept BF16; FP32 Conv in actual models and explicit BF16 storage conversions are measured by their original types.

ONNX 1.16.2 `onnx.reference.ops.op_conv._conv_implementation`'s auto_pad path uses `X.shape[i]` instead of the spatial dimension `X.shape[i+2]`. Control preserves its actual difference and lists it as xfail; it cannot be counted as official numerical pass. Official controls for explicit pads and hand-computed SAME_UPPER/LOWER geometry are preserved separately. Existing divergences in BF16 official reference retain their original classification.

All probes use `tests/validation/stage_capture.py` to write to disk and exit, with resource and process group cleanup. Small cases: 120s/8GiB; scale cases: 600s/16GiB and the caller explicitly passes a lock already agreed upon on the same host. First publication of full-scale cases belongs to the simulator author; QA prioritizes reviewing their originals.
