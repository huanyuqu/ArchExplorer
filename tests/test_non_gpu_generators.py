"""Public non-GPU precision capabilities and exported generator entrypoints."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from arch_explorer.adf import walk
from arch_explorer.archs.adf_seeds import non_gpu_adf
from tools.export_source import export


@pytest.mark.parametrize('precision', ['fp16', 'bf16', 'fp32'])
def test_explicit_matrix_precision_retains_fp32_vector_and_capacity(precision):
    adf = non_gpu_adf(num_cores=2, matrix_precision=precision, dram_capacity_bytes=2**30)
    assert adf.child('core').repeat == 2
    assert adf.child('core').child('array').value('precision') == precision
    assert adf.child('core').child('vector').value('precision') == 'fp32'
    assert adf.child('core').child('vector').value('matrix_kernels') is True
    assert adf.child('dram').value('capacity') == 2**30
    for _, node in walk(adf):
        if hasattr(node, 'component'):
            assert node.component not in ('gpu_sm', 'gpu_kernel_service')
            assert not node.component.startswith('accelsim_')
            assert all('synthetic' in param.source for param in node.params.values())
    fresh = non_gpu_adf(matrix_precision=precision)
    assert fresh.child('core').repeat == 1 and fresh.child('dram').value('capacity') == 64 * 2**30


@pytest.mark.parametrize('kwargs', [
    {'num_cores': True}, {'num_cores': 0}, {'num_cores': 1.5},
    {'dram_capacity_bytes': True}, {'dram_capacity_bytes': 0}, {'dram_capacity_bytes': 1.5},
    {'matrix_precision': 'float16'}, {'matrix_precision': 'int8'}, {'matrix_precision': None},
])
def test_invalid_seed_dimensions_or_precision_are_not_silently_coerced(kwargs):
    with pytest.raises(ValueError):
        non_gpu_adf(**kwargs)


@pytest.fixture(scope='module')
def source_export(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    destination = tmp_path_factory.mktemp('generator-export') / 'source'
    manifest = export(destination, root=root)
    assert 'LICENSE' in manifest and 'NOTICE' in manifest
    assert 'arch_explorer/devices/scans.py' not in manifest
    return destination


@pytest.mark.parametrize('script', ['generate_pi05.py', 'generate_multi_device.py', 'generate_multi_device_execution.py'])
def test_exported_generator_runs_with_installed_package(source_export, script, tmp_path):
    executable = source_export / 'tools' / script
    assert executable.is_file()
    output = tmp_path / 'generated'
    process = subprocess.run([sys.executable, str(executable), str(output)], cwd=tmp_path,
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    configs = [json.loads(path.read_text()) for path in sorted(output.glob('*.json'))]
    assert len(configs) == {'generate_pi05.py': 5, 'generate_multi_device.py': 2,
                            'generate_multi_device_execution.py': 4}[script]
    for config in configs:
        assert config['backend'] == 'events' and config['workloads']
        # These are explicit sourced structures, not removed GPU preset aliases.
        text = json.dumps(config['seed_adf'])
        assert 'systolic_array' in text and 'vector_unit' in text
        assert 'gpu_sm' not in text and 'accelsim_' not in text
        assert 'synthetic non-GPU' in text
