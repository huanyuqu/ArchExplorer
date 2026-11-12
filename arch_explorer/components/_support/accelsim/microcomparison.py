# Concrete Accel-Sim implementation; owned by the component provider.
"""Every registered program's native prediction versus retained timing rows.

Unknown and failed rows remain explicit. Native time uses its declared clock;
the conditional measurement bounds below are assumptions, never calibration.
"""
from fractions import Fraction
from arch_explorer.components.accelsim.microbundle import json_file
from arch_explorer.components.accelsim.microrecords import original,load_attempt
from arch_explorer.components.accelsim.microexperiment import sample_quality
from arch_explorer.components.accelsim.microcontract import schema,identity


def errors(native_ns,values,epsilon_ns):
    if not values:raise ValueError('actual finite samples required')
    values=[Fraction(x) for x in values];pred=Fraction(native_ns);epsilon=Fraction(epsilon_ns)
    if min(values)<=0 or pred<=0 or epsilon<0:raise ValueError('positive native/observed time and nonnegative conditional error required')
    mean=sum(values)/len(values);signed=pred-mean;lower=mean-epsilon;upper=mean+epsilon
    relation='unknown within assumed measurement error'
    if signed>epsilon:relation='native overprediction under assumed bound'
    elif signed < -epsilon:relation='native underprediction under assumed bound'
    relative=None
    if lower>0:
        ends=[(pred-upper)/upper,(pred-lower)/lower]
        relative={'signed':[float(x) for x in ends],
                  'absolute':[0.0 if lower<=pred<=upper else float(min(abs(x) for x in ends)),float(max(abs(x) for x in ends))]}
    return {'observed_mean_ns':float(mean),'observed_mean_ns_exact':str(mean),
            'signed_error_ns':float(signed),'absolute_error_ns':float(abs(signed)),
            'signed_relative_error':float(signed/mean),'absolute_percentage_error':float(abs(signed)/mean),
            'assumed_event_interval_abs_error_ns':str(epsilon),'conditional_signed_error_ns':[float(signed-epsilon),float(signed+epsilon)],
            'conditional_relative_error':relative,'conditional_direction':relation,
            'measurement_bound_is_documented_guarantee':False,'samples_reduce_assumed_bound':False,
            'physical_parameters_identified':False}


def compare_registered(bundle,design,outcomes,journal,intervals,*,kind):
    """Called after closure from the same reader for actual and CPU boundaries."""
    denominator={(r['case'],r['mode'],r['sample']):r for r in outcomes['denominator']}
    initial=json_file(original(journal.initial))[0] if journal.initial else None
    final=json_file(original(journal.final))[0] if journal.final else None
    predictions=initial['native_predictions'] if initial else {}
    observed={**journal.fit,**journal.heldout};rows=[]
    for cid,case in bundle.cases.items():
        native=None;prediction=predictions.get(cid)
        if prediction:
            result=json_file(original(prediction['prediction']))[0]
            native=result['evaluation']['details']['native']
        samples=[]
        for i in range(design['budget']['samples_per_case']):
            slot=denominator[cid,'timing',i]
            raw=json_file(original(slot['original']))[0] if slot['original'] else None
            samples.append({'sample':i,'status':slot['status'],'started':slot['started'],'original':slot['original'],
                            'event_elapsed_ms_float32_bits':None if raw is None else raw.get('event_elapsed_ms_float32_bits'),
                            'event_elapsed_ms':None if raw is None else raw.get('event_elapsed_ms'),
                            'elapsed_ns':None if raw is None else raw.get('elapsed_ns'),
                            'unit':None if raw is None else raw.get('unit'),
                            'hardware_core_mhz':None if raw is None else raw.get('hardware_core_mhz'),
                            'clock_state':None if raw is None else raw.get('clock_state')})
        attempts=observed.get(cid);quality=sample_quality(bundle,attempts) if attempts else None
        error=None;service=None
        if native is not None and attempts:
            times=[load_attempt(bundle,r)['elapsed_ns'] for r in attempts]
            error=errors(native['latency_ns'],times,design['quality']['assumed_event_interval_abs_error_ns'])
            if case['partition']=='heldout' and intervals and intervals['status']=='passed':
                lo,hi=[Fraction(x) for x in intervals['heldout_service_intervals_ns'][cid]]
                mean=Fraction(error['observed_mean_ns_exact'])
                service={'conditional_interval_ns_exact':[str(lo),str(hi)],'observed_mean_inside':lo<=mean<=hi,
                         'scope':'fit-only conditional composite interval; not native physical parameters'}
        eligible=quality is not None and quality['status']=='passed' and error is not None
        reason=None if eligible else ('no preregistered initial native freeze' if initial is None else
                                     'timing not observed/accepted' if attempts is None else 'preregistered sample quality refused')
        rows.append({'case':cid,'family':case['family'],'partition':case['partition'],'function':case['function'],
                     'registry_sha256':bundle.sha256,'text_sha256':case['text_sha256'],'input':case['input'],'auxiliary':case['auxiliary'],
                     'full_flow_instructions':case['full_flow_instructions'],'resources':case['resources'],
                     'structure_status':denominator[cid,'structure',0]['status'],'prediction':prediction,
                     'initial_freeze':journal.initial,'final_freeze':journal.final if case['partition']=='heldout' else None,
                     'native':None if native is None else {'core_cycles':native['gpu_tot_sim_cycle'],
                         'clock_domain':native['clock_domain'],'assumed_core_mhz':native['core_mhz'],
                         'assumed_latency_ns':native['latency_ns'],'clock_is_hardware_observation':False},
                     'samples':samples,'quality':quality,'error':error,'conditional_service_heldout':service,
                     'eligible_for_quality_summary':eligible,'exclusion_reason':reason})
    groups=[]
    for partition in ('fit','heldout'):
        for family in [None]+list(dict.fromkeys(r['family'] for r in rows)):
            members=[r for r in rows if r['partition']==partition and (family is None or r['family']==family)]
            if not members:continue
            valid=[r for r in members if r['eligible_for_quality_summary']]
            groups.append({'partition':partition,'family':family,'registered_count':len(members),'quality_eligible_count':len(valid),
                           'excluded':[{'case':r['case'],'reason':r['exclusion_reason']} for r in members if not r['eligible_for_quality_summary']],
                           'native_mape':sum(r['error']['absolute_percentage_error'] for r in valid)/len(valid) if valid else None,
                           'native_worst':max(r['error']['absolute_percentage_error'] for r in valid) if valid else None,
                           'native_mean_signed_relative_error':sum(r['error']['signed_relative_error'] for r in valid)/len(valid) if valid else None})
    return {'schema':schema(bundle,'native-timing-comparison'),**identity(bundle),'kind':kind,
            'run_id':design['run_id'],'collection_outcome':outcomes['status'],'registered_rows':rows,'groups':groups,
            'structure_known_timing_heldout':True,'physical_parameters':None,'hardware_accuracy':'not validated',
            'scope':'complete finite programs only; assumed native clock, unknown physical clock/cache/scheduling; no target-model precision claim',
            'final_native_assumptions_changed':bool(final and final['parameters']!=initial['parameters'])}
