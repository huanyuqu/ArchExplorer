"""Public removal means no executable old sweep and a runnable cost example."""
import importlib.util
from pathlib import Path
import re
import subprocess
import sys

from arch_explorer.adf import SpacePoint, walk

ROOT = Path(__file__).resolve().parents[1]


def test_paper_gpu_sweep_module_and_cli_are_retired():
    assert importlib.util.find_spec('arch_explorer.devices.scans') is None
    result = subprocess.run([sys.executable, '-m', 'arch_explorer.devices.scans', '--help'],
        cwd='/tmp', text=True, capture_output=True, timeout=15)
    assert result.returncode != 0
    assert 'No module named arch_explorer.devices.scans' in result.stderr


def test_first_cost_example_runs_on_explicit_non_gpu_adf():
    text = (ROOT / 'docs/cost.md').read_text()
    block = re.search(r'```python\n(.*?)\n```', text, re.S)
    assert block
    namespace = {}
    exec(compile(block.group(1), 'docs/cost.md:first-example', 'exec'), namespace)
    adf, cost = namespace['adf'], namespace['cost']
    assert all(node.component not in ('gpu_sm', 'gpu_kernel_service') and not node.component.startswith('accelsim_')
               for _, node in walk(adf) if isinstance(node, SpacePoint))
    assert cost.area_mm2 > 0 and cost.raw_cost_usd > 0 and cost.full_cost_usd > 0
    # Running successfully never turns unknown physical DRAM die area into 0.
    assert any('DRAM' in str(issue) or 'dram' in str(issue) for issue in cost.unsupported)
