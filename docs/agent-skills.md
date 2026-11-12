# ArchExplorer skills for external agents

The `skills/` directory in this distribution provides three independent, copyable skill directories:

| Skill | User task |
| --- | --- |
| [arch-explorer-run](../skills/arch-explorer-run/SKILL.md) | Prepare, execute, compare, and interpret simulation results |
| [arch-explorer-workload](../skills/arch-explorer-workload/SKILL.md) | Integrate new operators, new graphs, mappings, and custom architectures |
| [arch-explorer-component](../skills/arch-explorer-component/SKILL.md) | Integrate component execution models or existing simulators |

Copy the required directory to the skills directory supported by the external agent (for example, Codex's `$CODEX_HOME/skills`, or `~/.codex/skills` when unset). If the environment does not auto-discover skills, you can also explicitly instruct the agent to read the corresponding SKILL.md. No Portal account, internal workspace, or team run tools are required.

Documentation paths referenced by skills are relative to the matching version's ArchExplorer project directory; after copying the skill, the agent still needs to locate the public source distribution or corresponding documentation. Skills do not install dependencies or download weights; actual operations are performed by the agent according to the user task. You can give requests like these:

- "Read arch-explorer-run, use my architecture and scenario, compare two mappings, and explain the results."
- "Read arch-explorer-workload, add an operator for a graph with branches, and execute it on the ADF I provide."
- "Read arch-explorer-workload, provide two target implementations for the same operator in a standalone package, one with multiple steps and temporary tensors, then connect to the existing GPU target."
- "Read arch-explorer-component, integrate my SRAM model into the existing event framework."
- "Read arch-explorer-run, run a complete small ViT/CLIP following the matching version's component instantiation tutorial, and perform software differential comparison against an independent Accel-Sim oracle."

For GPU component tasks, first read the [GPU instantiation process](gpu-instantiation.md) and check the components,
transactions, and clock capabilities actually supported by the selected version. The skill's guidance does not constitute evidence that this version has implemented or verified these capabilities; the raw results from both sides and
the public comparison report carry the conclusions of this differential comparison. When the interface is not available, report it explicitly and do not interpret whole-engine repeated execution as componentization.

For new execution implementation processes, see the [external operator tutorial](operator-implementations.md). First distinguish semantics, target implementation, mapping, and
hardware mechanisms; ordinary graphs reuse common policies, and do not let the agent duplicate scheduling state machines or add provider branches for every operator.

Public source exports and sdist carry skills; the wheel provides the execution library and does not guarantee carrying these project-level materials. Skills do not interpret synthetic examples as real device accuracy. Whether a new workload can execute on a specific GPU engine is still determined by its capability contract.
