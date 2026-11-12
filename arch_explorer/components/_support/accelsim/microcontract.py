# Concrete Accel-Sim implementation; owned by the component provider.
"""Finite measurement identity and budgets, independent of experiment names.

The original SIM30 wire records remain readable. New registries carry their
work item and derive all denominators from the frozen case list. A transport
format (including SIM30NV1) is never an authorization identity.
"""
import re
from pathlib import Path
from arch_explorer.components.accelsim.trace import exact,same_json

REGISTRY_V2='arch-explorer.measurement-registry.v2'
CAPS={'cases':12,'host_processes':13,'host_timeout_s':120,'total_window_s':1800,
      'capture_launches':12,'timing_launches':36,'samples_per_case':3,'warmup_launches':0,
      'helper_launches':0,'app_bytes':2**20,'capture_bytes':16*2**20,
      'host_raw_bytes':64*2**20,'warp_records_per_case':4096}

def modern(bundle):return bundle.document.get('schema') in (REGISTRY_V2,'arch-explorer.measurement-registry.v3')
def work_item(bundle):return bundle.document['work_item'] if modern(bundle) else 'WP-SIM-30'
def budget(bundle):return bundle.document.get('budget',CAPS)
def schema(bundle,role):
    if bundle.document.get('schema')=='arch-explorer.measurement-registry.v3':return 'arch-explorer.measurement-'+role+'.v3'
    return ('arch-explorer.measurement-'+role+'.v2') if modern(bundle) else ('arch-explorer.sim30-'+role+'.v1')
def identity(bundle):return {'work_item':work_item(bundle)} if modern(bundle) else {}
def identity_fields(bundle,fields):return set(fields)|set(identity(bundle))
def check_identity(bundle,row,role):
    if row.get('schema')!=schema(bundle,role) or any(row.get(k)!=v for k,v in identity(bundle).items()):
        raise ValueError('measurement version/work item differs: '+role)

def validate_budget(values,count,*,legacy=False):
    exact(values,set(CAPS),'finite measurement budget')
    if legacy:
        if not same_json(values,CAPS):raise ValueError('legacy finite budget changed')
        return values
    if type(count) is not int or not 1<=count<=CAPS['cases']:raise ValueError('finite candidate count')
    for key,value in values.items():
        if type(value) is not int or value<0 or value>CAPS[key]:raise ValueError('measurement cap exceeded: '+key)
    derived={'cases':count,'host_processes':count+1,'capture_launches':count,'timing_launches':3*count,
             'samples_per_case':3,'warmup_launches':0,'helper_launches':0}
    if any(values[k]!=v for k,v in derived.items()) or any(values[k]<=0 for k in set(CAPS)-set(derived)):
        raise ValueError('budget denominator is not derived from frozen candidates')
    return values

def validate_header(data):
    if not isinstance(data['work_item'],str) or not re.fullmatch(r'WP-SIM-[0-9]+[A-Z]?',data['work_item']):raise ValueError('explicit work item required')
    if not isinstance(data['function_prefix'],str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,31}',data['function_prefix']):raise ValueError('explicit compiled function prefix required')
    validate_budget(data['budget'],len(data['cases']))

def validate_handoff(bundle,design,data):
    if design.get('schema')=='arch-explorer.measurement-state-device-design.v1':
        from arch_explorer.components.accelsim.state_resource import validate_handoff as state_handoff
        from arch_explorer.components.accelsim.microrecords import original
        from arch_explorer.components.accelsim.microbundle import json_file
        synthetic=json_file(original(design['source']))[0].get('kind')=='synthetic'
        return state_handoff(bundle,design,data,synthetic=synthetic)
    fields={'schema','work_item','run_id','resource_released','external_compute_empty','coordinator',
            'resource_owner','coordination_evidence','issued_realtime_ns','lock_file'}
    if modern(bundle):fields|={'absolute_deadline_realtime_ns','close_reserve_s'}
    exact(data,fields,'fresh resource handoff')
    if (data['schema']!=schema(bundle,'resource-handoff') or data['work_item']!=work_item(bundle)
        or data['run_id']!=design['run_id'] or data['resource_released'] is not True
        or data['external_compute_empty'] is not True or data['coordinator']!='resident_2e1684fdfc1b'
        or not isinstance(data['resource_owner'],str) or not data['resource_owner']
        or type(data['issued_realtime_ns']) is not int or data['issued_realtime_ns']<=design['created_realtime_ns']
        or not isinstance(data['coordination_evidence'],list) or not data['coordination_evidence']
        or not Path(data['lock_file']).is_absolute()):raise ValueError('new actual work-item-bound handoff required')
    if modern(bundle):
        end=data['absolute_deadline_realtime_ns'];reserve=data['close_reserve_s']
        if (type(end) is not int or not data['issued_realtime_ns']<end<=data['issued_realtime_ns']+budget(bundle)['total_window_s']*10**9
            or type(reserve) is not int or not 10<=reserve<=120):raise ValueError('absolute handoff deadline/cleanup reserve differs')
    return data
