"""Unknown process placement cannot authorize a lease; CPU temp locks only."""
import xml.etree.ElementTree as ET

import pytest

from arch_explorer.simulator.accelsim import state_idle as idle
from test_state_idle import release, runner_for, setup, xml
from test_state_resource import GPU_UUID


CASES = (
    'container-wrapper', 'management-process', 'management-pid',
    'memory-process', 'utilization-pid', 'nested-process',
    'wrapped-pid-in-record', 'empty-stray-process', 'busy-with-stray-pid',
)


def misplaced(name):
    root = ET.fromstring(xml(pids=[321] if name == 'busy-with-stray-pid' else []))
    gpu = root.find('gpu')
    if name == 'container-wrapper':
        parent = ET.SubElement(gpu.find('processes'), 'wrapper')
    elif name in ('management-process', 'management-pid', 'empty-stray-process', 'busy-with-stray-pid'):
        parent = ET.SubElement(gpu, 'management_info')
    elif name == 'memory-process':
        parent = gpu.find('fb_memory_usage')
    elif name == 'utilization-pid':
        parent = ET.SubElement(gpu.find('utilization'), 'memory_util')
    else:
        process = ET.SubElement(gpu.find('processes'), 'process_info')
        ET.SubElement(process, 'pid').text = '321'
        parent = ET.SubElement(process, 'details')
    if name in ('management-pid', 'utilization-pid', 'wrapped-pid-in-record', 'busy-with-stray-pid'):
        ET.SubElement(parent, 'pid').text = '654'
    else:
        process = ET.SubElement(parent, 'process_info')
        if name != 'empty-stray-process':
            ET.SubElement(process, 'pid').text = '654'
    return ET.tostring(root)


@pytest.mark.parametrize('name', CASES)
def test_unowned_process_data_prevents_admission(tmp_path, name):
    raw = misplaced(name)
    assert idle.occupancy(raw, b'', 0, GPU_UUID) == ('unknown', None)
    _, _, _, request, transport = setup(tmp_path)
    try:
        with pytest.raises(ValueError, match='occupied or unknown'):
            idle.acquire(transport, request, runner=runner_for(raw))
        assert not (transport.root / 'IDLE-ADMISSION.json').exists()
        assert not (transport.root / 'lease.json').exists()
    finally:
        release(transport)


@pytest.mark.parametrize('kind', ('empty', 'none', 'compute', 'graphics', 'utilization'))
def test_valid_management_and_process_controls(kind):
    root = ET.fromstring(xml(pids=[321] if kind in ('compute', 'graphics') else [],
                             utilization='1 %' if kind == 'utilization' else '0 %'))
    gpu = root.find('gpu')
    ET.SubElement(root, 'driver_version').text = 'CPU fixture'
    ET.SubElement(gpu, 'product_name').text = 'management fixture'
    ET.SubElement(gpu.find('utilization'), 'memory_util').text = '0 %'
    if kind == 'none':
        gpu.find('processes').text = 'None'
    if kind == 'compute':
        gpu.find('processes/process_info/type').text = 'C'
    status, facts = idle.occupancy(ET.tostring(root), b'', 0, GPU_UUID)
    assert status == ('idle' if kind in ('empty', 'none') else 'busy')
    assert facts['process_ids'] == ([321] if kind in ('compute', 'graphics') else [])
