# 2. Turning your own workloads into interpretable inputs

First list the operators, tensor shapes, precision,
data dependencies, and repetition counts for a fixed input scenario from model code or existing profiler records. ArchExplorer does not automatically download models or convert arbitrary ONNX/PyTorch graphs into supported simulations.
Only shapes are needed below; input values or weight values are not needed, and model predictions or accuracy are not computed.

## Vision foundation model: token projection

Assume a 224×224 image has already been converted into tokens by 16×16 patches, and does not contain a CLS token,
then the token count is `(224/16)² = 196`. To keep the tutorial lightweight, choose a synthetic embedding width of 192.
A combined Q/K/V projection can be abstracted as:

```text
X[196,192] @ W[192,576] -> Y[196,576]
m = token count = 196
k = input channels = 192
n = triple channels = 576
```

This is not complete attention: only the projection is computed, and QKᵀ, softmax, AV, output projection, residual, or MLP are not yet included.
It is even less an official configuration description of some vision foundation model. The patch encoding overhead is also outside the boundary of this example.
This shape corresponds to `examples/tutorials/vision-projection.json`:

```json
{"kind":"operator","name":"synthetic-vision-projection","params":{"op":"matmul","batch":1,"m":196,"k":192,"n":576,"dtype":"fp16"},"weight":1.0}
```

When changing resolution, first recalculate the token count; if the model has CLS, overlapping patches, dynamic cropping, or multi-scale inputs,
shapes should be computed according to the actual graph, and 196 cannot be applied as a fixed value. batch and token are different dimensions.
When batched images share one set of weights, `shared_b: true` should also be specified; this example has batch=1 so there is no such distinction.
The minimum data volume of input/weight/output can be estimated by `(m*k+k*n+m*n)*2` bytes,
but actual traffic depends on reuse, mapping, padding, capacity, and the communication model, and this formula cannot be used directly as memory access timing.

## Embodied AI: fused feature projection for a single decision

Assume the upstream encoder has already generated a 256-dimensional fused feature, which can come from vision, robot proprioceptive state, etc.; these upstream encoders are not simulated here. A synthetic policy head projects it to a 64-dimensional intermediate representation:

```text
X[1,256] @ W[256,64] -> H[1,64]
m = one observation at the current control step = 1
k = fused feature dimension = 256
n = intermediate representation dimension = 64
```

The accompanying `embodied-projection.json` simulates this GEMM. 64 is not the number of joints of the specified robot, and there is no bias, activation, action output layer, or action execution yet. It is also not closed-loop reinforcement learning or robot environment simulation. This example uses a synthetic non-GPU DMC; partitioning, local storage, and communication jointly affect the results, and larger hardware is not necessarily faster.

If the real computation is a matrix-vector multiply, create a separate `operator`/`mvm` scenario using `params={"op":"mvm","batch":1,"m":64,"k":256,"dtype":"fp16"}`. Here the MVM's `m` is the output dimension, which has a different meaning from the GEMM's `m` above; the model scheduling is also different. Record the actual computation form used, and do not retroactively treat the faster of the two results as the measured implementation. Changing the GEMM's m to 8 to represent processing 8 rows together does not automatically express 8 interdependent control steps.

The complete deadline of a control system must also include sampling, image processing, transmission, encoder, the rest of the policy, runtime, and actuators. The ns results of this fragment cannot be used to prove that the system achieves a certain control frequency.

## Multiple fragments and real graphs

Multiple entries in `workloads` are scenarios or fragments evaluated separately; the objective is `Σ weight * latency_ns`. They do not share intermediate tensors, do not establish interdependencies, and do not automatically obtain cross-operator fusion, cache reuse, or pipeline overlap. Weights are not automatically normalized. If layer repetition counts are used as weights, the result is only a corresponding weighted estimate, not a complete network simulation.

Complete custom graphs explicitly specify tensor/producer relationships through the Workload of `arch_explorer.workload.api`, save them as standard JSON using `graph.to_dict()` / `Workload.from_dict(data)`, and run them with `WorkloadSpec('graph', graph.to_dict())`, without modifying the main project or adding a model kind. New operators provide validate/lower through registered OperatorDefinition; finite floating-point attributes are checked by operator semantics, and arbitrary Python IR objects cannot be written directly into JSON. See the [external workload tutorial](../workload-extensions.md): read hardware from an ADF file, build a branching graph in a new operator package installed outside the real repository, and then execute using the public mapping and existing events. The interfaces of existing Transformer or single-query attention have their own assumptions and cannot replace arbitrary visual attention, diffusion, convolution, sparse operators, or dynamic control flow just because of similar names. New operators also require resource requirement and mapping support.

Record an experiment note: where the input shapes/precisions come from, where timing starts and ends, where weights and activations reside, which operators are modeled, and which remain unknown. Then proceed to [build the target architecture](03-architecture.md).

For the entry points, supported versions, and commands of named complete ViT/CLIP, see the [complete model tutorial](05-full-models.md). The two synthetic projection fragments above retain their original boundaries.
