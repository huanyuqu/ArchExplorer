# Concrete Accel-Sim implementation; owned by the component provider.
"""Explicit CPU boundary rehearsal of Collector's unchanged phase orchestration.

The boundary file is frozen in the installed source archive. It only creates
synthetic originals; actual Journal, raw parsers, freeze/protocol checks, solver,
comparison and terminal closure run without mocks or parser exceptions.
"""
import argparse
import importlib.util
from pathlib import Path
import time

from arch_explorer.components.accelsim.microcampaign import Collector, check_source_installation, native_assumption_scenario
from arch_explorer.components.accelsim.microexperiment import Journal, ref, write_new, prediction_freeze
from arch_explorer.components.accelsim.microrecords import original
from arch_explorer.components.accelsim.microbundle import json_file
from arch_explorer.components.accelsim.microclosure import _replay_run
from arch_explorer.components.accelsim.microcontract import budget,schema,identity,modern


class Rehearsal(Collector):
    def __init__(self, fixture, *, abort=None):
        if abort not in (None,'first_failure','timeout'):raise ValueError('unknown CPU boundary abort')
        self.fixture=fixture;self.abort=abort
        self.design_file=fixture.design_file;self.design=fixture.design;self.bundle=fixture.bundle
        self.budget=budget(self.bundle)
        self.root=fixture.root;self.processes=[];self.lock_fd=None;self.started=None;self.deadline=None
        self.opened=False;self.lease_ref=fixture.lease_ref;self.capture_bytes=0
        self.journal=Journal(self.root/'journal',self.bundle,self.design)

    def acquire(self):
        # No flock, lease issuance, subprocess, CUDA or device management.
        self.started=time.monotonic();self.deadline=self.started+self.budget['total_window_s']

    def host(self,phase,freeze,cid=None):
        if phase!='structure' and len(self.processes)==1 and self.abort:
            if self.abort=='timeout':raise TimeoutError('synthetic first timing boundary timed out; no process launched')
            raise RuntimeError('synthetic first timing boundary failed; no process launched')
        output,receipt=self.fixture.process(phase,freeze,cid)
        self.processes.append(receipt);self.record_reap(receipt)
        return output,receipt

    def predict(self,capture_output,bindings):
        # This is the native process boundary: reuse its immutable completed
        # software protocols, never execute or forge a new native prediction.
        from arch_explorer.components.accelsim.config import compile_adf
        adf=native_assumption_scenario(capture_output/'preflight.json')
        config=compile_adf(adf)
        if adf.child('cluster').child('l1_shared').params['gpgpu_adaptive_cache_config'].value!=0:
            raise ValueError('rehearsal must exercise fixed shared allocation')
        write_new(self.root/'SCENARIO-CHECK.json',{'kind':'synthetic','scope':'CPU domain/config validation only',
                  'adf':adf.to_dict(),'config':config.artifacts,'hardware_accuracy':'not validated'})
        predictions=self.fixture.native_predictions;parameters=self.fixture.parameters
        frozen=prediction_freeze(self.bundle,self.design,predictions,parameters,self.root/'freezes',stage='initial')
        self.journal.initial_predictions(frozen)
        return frozen,predictions,parameters

    def release(self):
        if self.lock_fd is not None:raise AssertionError('CPU rehearsal acquired a real lock')

    def replay_outcome(self):
        return _replay_run(self.design_file,self.root,synthetic=True)


def rehearse(source_file,bundle_file,template_root,output,*,mode='success',native_fixture=None):
    if mode not in ('success','first_failure','timeout','quality_refused'):raise ValueError('unknown rehearsal mode')
    source_ref=ref(source_file);origin=check_source_installation(source_ref)
    generator=Path(origin['source_tree'])/'tools/accelsim_native/calibration/rehearsal_fixture.py'
    # The archive and source tree are tied to the installation by SOURCE. The
    # generator itself is additionally bound in each immutable fixture marker.
    import tarfile
    archive=original({'path':origin['archive'],'sha256':origin['archive_sha256']})
    with tarfile.open(archive) as tf:
        member=tf.extractfile('tools/accelsim_native/calibration/rehearsal_fixture.py')
        if member is None or member.read()!=generator.read_bytes():raise ValueError('CPU generator differs from frozen source archive')
    spec=importlib.util.spec_from_file_location('_sim30a_boundary_fixture',generator)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fixture=module.Fixture(output,bundle_file,template_root,source=source_ref,quality_refused=mode=='quality_refused',native_fixture=native_fixture)
    campaign=Rehearsal(fixture,abort=mode if mode in ('first_failure','timeout') else None)
    error=campaign.execute()
    result=campaign.replay_outcome()
    if (mode=='success')!=(error is None):raise AssertionError('synthetic rehearsal did not reach its intended terminal')
    # Exercise a non-decimal rational endpoint through the real comparison path.
    # This explicit boundary perturbation is not a fitted/hardware interval.
    if mode=='success':
        from arch_explorer.components.accelsim.microcampaign import compare
        import copy
        intervals=copy.deepcopy(result['fit_intervals']);cid=fixture.design['order']['heldout'][0]
        intervals['heldout_service_intervals_ns'][cid][0]='31/3'
        rational=compare(fixture.bundle,fixture.native_predictions,campaign.journal.heldout,intervals)
        if rational['rows'][0]['conditional_service_interval_ns'][0]!=31/3:raise AssertionError('rational endpoint was not consumed')
        write_new(fixture.root/'RATIONAL-ENDPOINT-CHECK.json',{'kind':'synthetic','endpoint':'31/3',
                  'replaced_boundary':'conditional interval endpoint fixture only','comparison':rational})
    write_new(fixture.root/'REHEARSAL.json',{'schema':schema(fixture.bundle,'rehearsal') if modern(fixture.bundle) else 'arch-explorer.sim30a-rehearsal.v1',**identity(fixture.bundle),'kind':'synthetic',
              'mode':mode,'source':source_ref,'generator':ref(generator),'result':ref(fixture.root/'REPLAY.json'),
              'error':error,'device_queries':0,'cuInit_calls':0,'host_launches':0,'kernel_launches':0,
              'event_API_calls':0,'native_runs':0,'common_locks_acquired':0,'actual_leases':0,
              'hardware_accuracy':'not validated','physical_parameters':None})
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--bundle',required=True)
    p.add_argument('--template',required=True);p.add_argument('--output',required=True)
    p.add_argument('--native-fixture',help='explicit same-registry synthetic native protocol plan for new registries')
    p.add_argument('--mode',choices=('success','first_failure','timeout','quality_refused'),default='success')
    a=p.parse_args(argv);result=rehearse(a.source,a.bundle,a.template,a.output,mode=a.mode,native_fixture=a.native_fixture)
    print({'kind':result['kind'],'collection_outcome':result['collection_outcome'],'attempt_counts':result['attempt_counts']})


if __name__=='__main__':main()
