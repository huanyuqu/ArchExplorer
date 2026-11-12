"""Frozen identity/numeric oracles. All embedded evidence is explicitly synthetic."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from arch_explorer.adf import Param, from_dict
from arch_explorer.simulator.accelsim.config import compile_adf
from arch_explorer.simulator.accelsim.protocol import case_identity, preflight_protocol, main
from arch_explorer.simulator.accelsim.trace_stream import file_digest
from protocol_fixture import review, write, frozen, replace_role, edit_file


@pytest.fixture(autouse=True)
def no_native():
    with patch('arch_explorer.simulator.accelsim.backend.run_native',side_effect=AssertionError('native forbidden')) as backend, \
         patch('arch_explorer.simulator.accelsim.runner.run_native',side_effect=AssertionError('native forbidden')) as runner, \
         patch('subprocess.Popen',side_effect=AssertionError('subprocess forbidden')) as popen:
        yield
        assert backend.call_count == runner.call_count == popen.call_count == 0


def result(spec):
    output=preflight_protocol(spec)
    assert output['native_started'] is False and output['hardware_executed'] is False
    request=json.loads(Path(spec['request']).read_text())
    assert not Path(request['options']['output_dir']).exists()
    return output


def reject(spec, kind=None):
    output=result(spec)
    assert output['status']=='blocked',output
    if kind:
        assert output['blockers'][0]['kind']==kind,output
    return output


def test_complete_software_review_and_hardware_block(review):
    output=result(review)
    assert output['status']=='passed' and output['native']['latency_ns']==1000
    review.update(purpose='hardware-accuracy',target={'gpu':'existing identity','isa':'sm120','evidence_sha256':'b'*64})
    output=result(review)
    assert output['status']=='blocked'
    assert {'native_isa','physical_parameters','measurement_association','new_window'} <= {b['kind'] for b in output['blockers']}


def test_f1_unrelated_trace_even_with_recomputed_case(review,tmp_path):
    wrong=tmp_path/'unrelated.txt';wrong.write_text('not the executed trace')
    replace_role(review,'trace',wrong)
    review['case_identity']=file_digest(wrong)
    reject(review,'frozen_binding')


def test_f2_shallow_prediction(review,tmp_path):
    replace_role(review,'prediction',write(tmp_path/'shallow.json',{'latency_ns':0.0}))
    reject(review,'prediction_schema')


@pytest.mark.parametrize('change', [
    lambda d: d.pop('device_facts'),
    lambda d: d.update(device_facts={}),
    lambda d: d.update(device_facts={'live_lease': True, 'hardware_accuracy': 'validated'}),
    lambda d: d.update(extra_field='ignored?'),
    lambda d: d.pop('component_map_schema'),
    lambda d: d.update(component_map_schema='arch-explorer.accelsim-component-map.v1'),
    lambda d: d.pop('schema'),
    lambda d: d.update(schema='arch-explorer.accelsim-result-details.v1'),
])
def test_current_result_requires_complete_applicable_facts_and_version(review, change):
    edit_file(review, 'prediction', lambda p: change(p['evaluation']['details']))
    reject(review)


def test_current_source_cannot_masquerade_as_legacy_result(review):
    def change(prediction):
        details = prediction['evaluation']['details']
        for key in ('schema', 'device_facts', 'component_map_schema'):
            details.pop(key)
    edit_file(review, 'prediction', change)
    reject(review, 'prediction_schema')


@pytest.mark.parametrize('field,value', [('source', 'borrowed provenance'),
                                       ('constraint', None), ('clock_domain', 'fabricated'),
                                       ('status', 'device_fact'), ('instances', 99)])
def test_current_component_map_metadata_is_not_ignored(review, field, value):
    edit_file(review, 'prediction', lambda p: p['evaluation']['details']['component_map'][0].update({field: value}))
    reject(review, 'prediction_identity')


def test_stale_complete_prediction_with_new_valid_configuration(review):
    request=json.loads(Path(review['request']).read_text())
    adf=from_dict(request['adf'])
    for node,value in ((adf.child('cluster').child('sm'),990),(adf.child('icnt'),1980),(adf.child('partition').child('l2'),1980)):
        old=node.params['clock_mhz'];node.params['clock_mhz']=Param(value,old.unit,old.source)
    config=compile_adf(adf)
    request['adf']=adf.to_dict();write(Path(review['request']),request)
    replace_role(review,'request',Path(review['request']))
    write(Path(frozen(review,'adf')['path']),request['adf'])
    replace_role(review,'adf',Path(frozen(review,'adf')['path']))
    write(Path(frozen(review,'config')['path']),config.artifacts)
    replace_role(review,'config',Path(frozen(review,'config')['path']))
    reject(review,'prediction_identity')


@pytest.mark.parametrize('latency',[0,-1,True,'1000',1001.0])
def test_full_prediction_top_latency_closes_to_raw_output(review,latency):
    edit_file(review,'prediction',lambda p:p['evaluation'].update(latency_ns=latency))
    reject(review,'prediction_numeric')


@pytest.mark.parametrize('change',[
    lambda native:native.update(gpu_sim_cycle=3960,gpu_tot_sim_cycle=3960,latency_ns=2000.0),
    lambda native:native.update(scope='whole-application'),
    lambda native:native.update(core_mhz=990),
])
def test_copied_consistent_json_values_are_not_native_evidence(review,change):
    run=json.loads(Path(frozen(review,'native-run')['path']).read_text())
    change(run['stats'])
    write(Path(frozen(review,'native-run')['path']),run)
    replace_role(review,'native-run',Path(frozen(review,'native-run')['path']))
    def modify(prediction):
        details=prediction['evaluation']['details']
        details['run']=deepcopy(run);details['native']=deepcopy(run['stats'])
        prediction['evaluation']['latency_ns']=run['stats']['latency_ns']
    edit_file(review,'prediction',modify)
    reject(review,'prediction_numeric')


@pytest.mark.parametrize('state,code',[('executed',0),('complete',1),('complete',False)])
def test_incomplete_native_record_is_not_complete_prediction(review,state,code):
    run=edit_file(review,'native-run',lambda r:r.update(status=state,exit_code=code))
    edit_file(review,'prediction',lambda p:p['evaluation']['details'].update(run=run))
    reject(review,'native_record')


def test_source_archive_must_match_recorded_execution_sources(review):
    def modify(run):run['adapter_sources_sha256']['simulator/accelsim/runner.py']='0'*64
    run=edit_file(review,'native-run',modify)
    edit_file(review,'prediction',lambda p:p['evaluation']['details'].update(run=run))
    reject(review,'source_identity')


def test_source_origin_requires_matching_archive_and_object(review):
    # Rehashing malformed origin metadata cannot turn it into execution proof.
    origin=Path(frozen(review,'source-origin')['path'])
    write(origin,[])
    replace_role(review,'source-origin',origin)
    edit_file(review,'source-commit',lambda p:p['origin'].update(sha256=file_digest(origin)))
    reject(review,'source_identity')


@pytest.mark.parametrize('mutation',[
    lambda rows:rows.pop(),lambda rows:rows.append(deepcopy(rows[0])),
    lambda rows:rows[0].update(sha256=True),lambda rows:rows[0].update(extra='unexpected'),
])
def test_roles_require_complete_unique_typed_bindings(review,mutation):
    mutation(review['frozen'])
    reject(review)


def test_second_root_cause_unrelated_commands_file(review,tmp_path):
    wrong=tmp_path/'unrelated-commands.g';wrong.write_text('kernel-free.traceg\n')
    replace_role(review,'commands',wrong)
    reject(review,'frozen_binding')


@pytest.mark.parametrize('option,value',[('max_cycles',2000000),('memory_limit_mb',2048),('timeout_s',90)])
def test_execution_guards_not_ignored_as_output_metadata(review,option,value):
    edit_file(review,'request',lambda r:r['options'].update({option:value}))
    reject(review,'prediction_identity')


def test_output_location_and_display_name_normalize(review,tmp_path):
    edit_file(review,'request',lambda r:(r['options'].update(output_dir=str(tmp_path/'different-no-run')),
                                       r['workload'].update(name='different display only',weight=2.0)))
    assert result(review)['status']=='passed'


def test_history_identity_not_manifest_provenance_or_frozen_role(review):
    request=json.loads(Path(review['request']).read_text())
    original=case_identity(request)
    edit_file(review,'history',lambda h:h['case_identities'].append(original['case_identity']))
    manifest_path=Path(frozen(review,'trace-manifest')['path'])
    manifest=json.loads(manifest_path.read_text())
    manifest['source']['description']='new provenance description only'
    write(manifest_path,manifest);replace_role(review,'trace-manifest',manifest_path)
    request['workload']['params']['accelsim']['sha256']=file_digest(manifest_path)
    write(Path(review['request']),request);replace_role(review,'request',Path(review['request']))
    assert case_identity(request)==original
    reject(review,'historical_overlap')


def test_legacy_registered_bytes_survive_manifest_metadata_change(review):
    trace_sha=frozen(review,'trace')['sha256']
    edit_file(review,'history',lambda h:(h.clear(),h.update(schema='arch-explorer.sim27-history.v1',
        case_identities=[trace_sha+':'+('f'*64)],closed_binary_sha256={},policy='old immutable registry')))
    reject(review,'historical_overlap')


def test_strict_json_nonfinite_and_duplicate_keys(review):
    file=Path(frozen(review,'prediction')['path'])
    for raw in ('{"latency_ns":NaN}','{"schema":"a","schema":"b"}','{"latency_ns":1e999}'):
        file.write_text(raw);replace_role(review,'prediction',file)
        reject(review)


def test_public_module_cli_pass_and_block_no_subprocess(review,tmp_path,capsys):
    file=write(tmp_path/'protocol.json',review)
    assert main([str(file)])==0
    assert json.loads(capsys.readouterr().out)['status']=='passed'
    review['review_kind']='claimed-premeasurement-freeze'
    write(file,review)
    assert main([str(file)])==2
    assert json.loads(capsys.readouterr().out)['blockers'][0]['kind']=='protocol_schema'

