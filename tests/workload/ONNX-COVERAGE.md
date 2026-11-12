# Bounded ONNX Import Independent Acceptance

The standard file domain is ONNX 1.16.2, IR 8, standard empty domain, opset 17, static positive-size FP32 inference DAG. Models come from deterministic numerical MLP and explicit Q/K/V attention, including rectangular multiplication, batch/scalar broadcasting, duplicate or empty node display names, two return roots, shared intermediate values, and shared weights. int64 is used only for compile-time Reshape literals.

`onnx_import_cases.py` generates real ModelProtos and external weights; `onnx_generate_inputs.py` generates answer-free manifests. `onnx_import_expectations.py` independently writes per-node NumPy formulas for the two self-owned models; `onnx_numpy_interpreter.py` additionally implements ten standard operators and interprets the formal inputs. Neither calls ArchExplorer reference, lowering, layout resolver, or compiler to compute expectations. Original-model official ReferenceEvaluator, independent NumPy, product original Graph, product mapped Graph, and actual Events service are reported separately.

| Contract | Independent input/check | Candidate consumer |
| --- | --- | --- |
| MatMul/Gemm parameters, broadcast, Softmax arbitrary axis | closed form, hand-computed rectangular multiplication, independent interpreter vs official per-node comparison | test_onnx_import_edges.py, test_onnx_numeric_storage.py |
| Transpose/Reshape/Identity/Constant, stable original identity | actual files, literal int64, non-default permutation, multiple outputs | test_onnx_import_contract.py, test_onnx_import_refusals.py |
| External weights complete bytes, offset/length, explicit roots | non-zero offset, move/rename, missing files | test_onnx_import_refusals.py, snapshot migration |
| FP32, finite roots, symbolic specialization, initializer override policy | standard legal controls and product extra domain restrictions listed separately | specific ValueError/ImportError text; arbitrary Exception must not be passed off as rejection |
| Actual addresses, service, shared roots, return value holding, peaks | existing independent root walk, scalar FMA, actual allocate/free pass-through observation | two models × two mappings × two layouts; view peaks are not assumed to decrease |
| Exact capacity and one byte less | consume peaks from observed real intervals, reject repeated estimation | test_onnx_numeric_storage.py independent capacity entry |
| Public user closed loop, failure accounting, optional dependencies | complete ordinary WorkloadSpec, API/CLI/Env/Agent/search entries | test_onnx_user_consumers.py, onnx_search_consumer.py |
| fresh read-only/explicit rerun, source rejection | pure compile validation listed separately; prepare/engine/solver/native/network/write set actual guards | onnx_readonly_probe.py, reseal tamper rejection, independent new rerun directory |

Independent oracles and input preparation can run in environments without the product installed. Candidate consumers run only on the main wheel/sdist built uniformly by the responsible owner and same-source extensions; the source stage only performs AST checks, and collection does not count as pass. Required environment variables, dependency order, and actual execution records are written into the private acceptance deliverable.

This bounded domain does not close framework export, full ONNX operators/dynamic shapes, hardware-accurate latency, or IPU cost. Old failures and old matrices retain their original identities.
