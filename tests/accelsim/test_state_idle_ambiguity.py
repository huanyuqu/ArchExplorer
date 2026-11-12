"""Ambiguous management bytes confer no authority; CPU temp locks only."""
import xml.etree.ElementTree as ET
import pytest
from arch_explorer.simulator.accelsim import state_idle as idle
from test_state_idle import xml,setup,release,runner_for
from test_state_resource import GPU_UUID


def malformed(name):
    root=ET.fromstring(xml());gpu=root.find('gpu')
    if name in ('processes','same-processes'):
        node=ET.SubElement(gpu,'processes')
        if name=='processes':ET.SubElement(ET.SubElement(node,'process_info'),'pid').text='777'
    elif name in ('uuid','same-uuid'):
        ET.SubElement(gpu,'uuid').text=GPU_UUID if name=='same-uuid' else 'GPU-00000000-0000-0000-0000-000000000000'
    elif name in ('utilization','same-utilization'):
        ET.SubElement(ET.SubElement(gpu,'utilization'),'gpu_util').text='0 %' if name=='same-utilization' else '99 %'
    elif name in ('gpu_util','same-gpu_util'):
        ET.SubElement(gpu.find('utilization'),'gpu_util').text='0 %' if name=='same-gpu_util' else '99 %'
    elif name=='nested-uuid':ET.SubElement(gpu.find('uuid'),'uuid').text=GPU_UUID
    elif name=='nested-util':ET.SubElement(gpu.find('utilization/gpu_util'),'value').text='0 %'
    elif name=='misplaced-util':ET.SubElement(gpu,'gpu_util').text='0 %'
    elif name=='attributed-uuid':gpu.find('uuid').set('other','value')
    elif name in ('zero-pid','same-pid','nested-pid','attributed-pid'):
        proc=ET.SubElement(gpu.find('processes'),'process_info');pid=ET.SubElement(proc,'pid');pid.text='0' if name=='zero-pid' else '777'
        if name=='same-pid':ET.SubElement(proc,'pid').text='777'
        if name=='nested-pid':ET.SubElement(pid,'value').text='777'
        if name=='attributed-pid':pid.set('shadow','777')
    elif name=='misplaced-process-info':ET.SubElement(ET.SubElement(gpu,'process_info'),'pid').text='777'
    elif name=='misplaced-pid':ET.SubElement(gpu,'pid').text='777'
    else:raise AssertionError(name)
    return ET.tostring(root)


CASES=['processes','same-processes','uuid','same-uuid','utilization','same-utilization','gpu_util','same-gpu_util',
       'nested-uuid','nested-util','misplaced-util','attributed-uuid','zero-pid','same-pid','nested-pid','attributed-pid',
       'misplaced-process-info','misplaced-pid']


@pytest.mark.parametrize('name',CASES)
def test_ambiguous_or_misplaced_critical_data_is_unknown(name):
    assert idle.occupancy(malformed(name),b'',0,GPU_UUID)==('unknown',None)


@pytest.mark.parametrize('name',['processes','same-processes','same-uuid','same-gpu_util','misplaced-process-info','misplaced-pid'])
def test_direct_admission_never_issues_on_ambiguous_bytes(tmp_path,name):
    _,_,_,request,t=setup(tmp_path)
    try:
        with pytest.raises(ValueError,match='occupied or unknown'):idle.acquire(t,request,runner=runner_for(malformed(name)))
        assert not (t.root/'IDLE-ADMISSION.json').exists() and not (t.root/'lease.json').exists()
    finally:release(t)


def test_normal_additional_fields_do_not_reject_idle():
    root=ET.fromstring(xml());gpu=root.find('gpu')
    ET.SubElement(root,'driver_version').text='EXPLICIT_CPU_FIXTURE'
    ET.SubElement(gpu,'product_name').text='parser fixture'
    ET.SubElement(gpu.find('utilization'),'memory_util').text='0 %'
    assert idle.occupancy(ET.tostring(root),b'',0,GPU_UUID)[0]=='idle'
