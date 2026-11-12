# Concrete Accel-Sim implementation; owned by the component provider.
"""Pure CPU freeze ordering, complete denominators and conditional interval fit.

Observed microbenchmark service intervals are not physical native parameters.
No device calls, runner, parameter search or heldout-dependent fitting occurs.
"""
from fractions import Fraction
import itertools,json,time,uuid
from pathlib import Path
from arch_explorer.components.accelsim.trace import exact,digest,same_json
from arch_explorer.components.accelsim.microbundle import BUDGET,json_file
from arch_explorer.components.accelsim.microrecords import original,load_attempt
from arch_explorer.components.accelsim.microcontract import budget,identity,identity_fields,schema,check_identity,modern,work_item

DESIGN_SCHEMA='arch-explorer.sim30-design.v1'
FREEZE_SCHEMA='arch-explorer.sim30-prediction-freeze.v1'
QUALITY={'max_sample_range_over_mean':'0.03','assumed_event_interval_abs_error_ns':'1000',
         'event_resolution_description_ns':500,'error_bound_is_documented_guarantee':False,
         'samples_shrink_event_error_bound':False,'native_mape_goal':'0.10','native_worst_goal':'0.20'}

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def seal(x):return digest(canonical(x))
def ref(p):return {'path':str(Path(p).resolve()),'sha256':digest(Path(p).read_bytes())}
def write_new(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:f.write(json.dumps(x,indent=2,allow_nan=False)+'\n')
    return ref(p)

def make_design(bundle,source,engine,hosts,tracer,history,method,*,catalog=None):
    """Caller freezes complete real offline identities before any device operation."""
    for row in [source,engine,tracer,history,method]+list(hosts.values()):original(row)
    if set(hosts)!={'capture','timing'}:raise ValueError('separate frozen capture and timing executables required')
    result={'schema':schema(bundle,'design'),**identity(bundle),'run_id':uuid.uuid4().hex,'registry':ref(bundle.path),
            'source':source,'engine':engine,'hosts':hosts,'tracer':tracer,'history':history,
            'budget':budget(bundle),'quality':QUALITY,'method':method,'order':{'structure':list(bundle.cases),
             'fit':[c['id'] for c in bundle.cases.values() if c['partition']=='fit'],
             'heldout':[c['id'] for c in bundle.cases.values() if c['partition']=='heldout']},
            'created_realtime_ns':time.time_ns(),'validation_scope':'structure-known; timing-heldout',
            'physical_parameters':None,'hardware_accuracy':'not validated'}
    if modern(bundle):
        if catalog is None:raise ValueError('explicit registry/actual-ELF catalog binding required before new design')
        result.update(schema='arch-explorer.measurement-design.v3',catalog=catalog)
        from arch_explorer.components.accelsim.microcatalog import require_design_catalog
        require_design_catalog(result,source=source,run_policy=True)
    result['design_sha256']=seal(result)
    return result

def validate_design(bundle,design):
    device=design.get('schema')=='arch-explorer.measurement-state-device-design.v1'
    registered=device or design.get('schema')=='arch-explorer.measurement-design.v3'
    fields={'schema','run_id','registry','source','engine','hosts','tracer','history','budget','quality','method','order','created_realtime_ns','validation_scope','physical_parameters','hardware_accuracy','design_sha256'}
    exact(design,identity_fields(bundle,fields|({'catalog'} if registered else set())|({'state_protocol','plan_sha256'} if device else set())),'frozen design')
    if registered:
        if not modern(bundle) or design['work_item']!=work_item(bundle):raise ValueError('registered design identity differs')
        # Historical replay validates frozen references, not a current policy.
        from arch_explorer.components.accelsim.microcatalog import design_binding
        design_binding(bundle,design)
    else:check_identity(bundle,design,'design')
    if device:
        from arch_explorer.components.accelsim.state_device import validate_extension
        validate_extension(bundle,design)
    data=dict(design);claimed=data.pop('design_sha256')
    if seal(data)!=claimed or design['registry']['sha256']!=bundle.sha256 or not same_json(design['budget'],budget(bundle)) or not same_json(design['quality'],QUALITY):raise ValueError('pre-device design changed/version/budget mismatch')
    if original(design['registry'])!=bundle.path or design['physical_parameters'] is not None or design['hardware_accuracy']!='not validated':raise ValueError('registry or unsupported physical claim')
    for row in [design['source'],design['engine'],design['tracer'],design['history'],design['method']]+list(design['hosts'].values()):original(row)
    for part in ('structure','fit','heldout'):
        expected=[c['id'] for c in bundle.cases.values() if part=='structure' or c['partition']==part]
        if design['order'][part]!=expected:raise ValueError('frozen execution order/partition changed')
    return design

def prediction_document(bundle,design,predictions,parameters,*,stage,fit_result=None,parent=None,created_realtime_ns=None):
    validate_design(bundle,design)
    if stage not in ('initial','final') or set(predictions)!=set(bundle.cases):raise ValueError('all initial/final predictions required before opening timings')
    if stage=='initial' and (fit_result is not None or parent is not None):raise ValueError('initial freeze cannot consume fit or heldout observations')
    if stage=='final' and (fit_result is None or parent is None):raise ValueError('final freeze requires fit-only conclusion and original prediction parent')
    exact(parameters,{'physical_parameters','hardware_accuracy','scenario','adf','native_config','unknown'},'explicit unchanged native assumptions')
    assumption_adf,_=json_file(original(parameters['adf']));assumption_config,_=json_file(original(parameters['native_config']))
    for cid,row in predictions.items():
        exact(row,{'request','prediction','review','protocol'},'complete native prediction closure')
        for r in row.values():original(r)
        from arch_explorer.components.accelsim.protocol import preflight_protocol,consume_request
        spec,_=json_file(original(row['review']));proof,_=json_file(original(row['protocol']))
        request,_=json_file(original(row['request']));current=consume_request(request)
        if spec.get('request')!=row['request']['path'] or current['actual']['name']!=bundle.cases[cid]['function'] or not same_json(request['adf'],assumption_adf) or not same_json(current['config'].artifacts,assumption_config):raise ValueError('missing/duplicated/cross-case prediction or native parameter binding')
        fixed={r['role']:r for r in spec['frozen']}
        if fixed['prediction']['path']!=row['prediction']['path'] or fixed['prediction']['sha256']!=row['prediction']['sha256']:raise ValueError('frozen complete prediction differs')
        if not same_json(preflight_protocol(spec),proof):raise ValueError('native protocol semantic replay differs')
        if proof.get('status')!='passed' or proof.get('hardware_executed') is not False:raise ValueError('completed public native software protocol required')
    if parameters.get('physical_parameters') is not None or parameters.get('hardware_accuracy')!='not validated':raise ValueError('unsupported physical identification claim')
    if stage=='final':
        prior,_=json_file(original(parent))
        if prior['stage']!='initial' or prior['design_sha256']!=design['design_sha256'] or prior['registry_sha256']!=bundle.sha256:raise ValueError('final freeze parent differs')
        if fit_result.get('status')!='passed' or fit_result.get('physical_parameters') is not None:raise ValueError('no certified physical mapping in this finite protocol')
        if not same_json(parameters,prior['parameters']) or not same_json(predictions,prior['native_predictions']):raise ValueError('unidentified service intervals cannot retune native parameters or choose predictions')
    result={'schema':schema(bundle,'prediction-freeze'),**identity(bundle),'stage':stage,'run_id':design['run_id'],'design_sha256':design['design_sha256'],'registry_sha256':bundle.sha256,
            'created_realtime_ns':time.time_ns() if created_realtime_ns is None else created_realtime_ns,'native_predictions':predictions,'parameters':parameters,
            'fit_result':fit_result,'parent':parent,'structure_scope':'known complete captured structures; no timing read to produce initial predictions'}
    return result

def prediction_freeze(bundle,design,predictions,parameters,directory,*,stage,fit_result=None,parent=None):
    result=prediction_document(bundle,design,predictions,parameters,stage=stage,fit_result=fit_result,parent=parent)
    return write_new(Path(directory)/(stage+'-predictions.json'),result)

def validate_prediction_freeze(bundle,design,reference,stage,*,fit_result=None,parent=None):
    frozen,_=json_file(original(reference))
    exact(frozen,identity_fields(bundle,{'schema','stage','run_id','design_sha256','registry_sha256','created_realtime_ns','native_predictions','parameters','fit_result','parent','structure_scope'}),'complete prediction freeze')
    check_identity(bundle,frozen,'prediction-freeze')
    stamp=frozen['created_realtime_ns']
    if type(stamp) is not int or stamp<=design['created_realtime_ns']:raise ValueError('prediction not frozen after its design')
    expected=prediction_document(bundle,design,frozen['native_predictions'],frozen['parameters'],stage=stage,fit_result=fit_result,parent=parent,created_realtime_ns=stamp)
    if not same_json(frozen,expected):raise ValueError('prediction freeze run/stage/source/fit binding differs')
    return frozen

def validate_timing_phase(bundle,design,freeze,attempt):
    validate_design(bundle,design);snapshot,_=json_file(original(freeze))
    attempt=load_attempt(bundle,attempt)
    process=attempt['process'];sample=attempt['sample'];case=bundle.cases[sample['case']]
    expected='initial' if case['partition']=='fit' else 'final'
    check_identity(bundle,snapshot,'prediction-freeze')
    if snapshot.get('stage')!=expected or snapshot.get('registry_sha256')!=bundle.sha256 or snapshot.get('design_sha256')!=design['design_sha256']:
        raise ValueError('wrong freeze phase; heldout observations cannot enter fit')
    if process['freeze_sha256']!=freeze['sha256'] or process['design_sha256']!=design['design_sha256'] or process['phase']!=case['partition'] or snapshot['created_realtime_ns']>=process['started_realtime_ns']:
        raise ValueError('prediction freeze does not precede actual independent timing process')
    if snapshot['stage']=='initial' and design['created_realtime_ns']>=process['started_realtime_ns']:raise ValueError('design not frozen before device operation')
    return attempt

def sample_quality(bundle,attempts):
    attempts=[load_attempt(bundle,a) for a in attempts]
    if len(attempts)!=3 or {a['sample']['sample_index'] for a in attempts}!={0,1,2}:raise ValueError('all three prespecified samples required; no replacement')
    if any(a['kind']!='hardware' or a['status']!='verified' or a['elapsed_ns'] is None for a in attempts):raise ValueError('all timing observations must be genuine verified attempts')
    if len({(a['sample']['case'],a['sample']['pid'],a['sample']['context'],a['process']['freeze_sha256']) for a in attempts})!=1:raise ValueError('three samples must belong to the same independent host/case/freeze')
    values=[Fraction(a['elapsed_ns']) for a in attempts];mean=sum(values)/3
    ratio=(max(values)-min(values))/mean
    return {'status':'passed' if ratio<=Fraction(QUALITY['max_sample_range_over_mean']) else 'refused',
            'all_sample_ns':[str(v) for v in values],'mean_ns':str(mean),'range_over_mean':str(ratio),
            'assumed_event_interval_abs_error_ns':QUALITY['assumed_event_interval_abs_error_ns'],
            'bound_is_conditional_not_documented_guarantee':True,'no_quantization_error_reduction_from_three_samples':True}

def _vertices(points,error):
    # a>=0,b>=0 and each individual elapsed sample lies within an assumed
    # interval around a+b*n. No clock conversion or native parameter inference.
    lines=[(Fraction(-1),Fraction(0),Fraction(0)),(Fraction(0),Fraction(-1),Fraction(0))]
    for n,y in points:
        lines += [(Fraction(1),n,y+error),(Fraction(-1),-n,-y+error)]
    vertices=set()
    for (a,b,c),(d,e,f) in itertools.combinations(lines,2):
        det=a*e-b*d
        if det:
            x,y=(c*e-b*f)/det,(a*f-c*d)/det
            if all(i*x+j*y<=k for i,j,k in lines):vertices.add((x,y))
    return sorted(vertices)

def fit_intervals(bundle,design,fit_cases):
    """Accept only the complete FIT partition; no heldout values in this API."""
    validate_design(bundle,design)
    expected=set(design['order']['fit'])
    if set(fit_cases)!=expected:raise ValueError('fit data missing/extra or heldout-contaminated')
    for cid,rows in fit_cases.items():
        for row in rows:
            if row.get('case')!=cid:raise ValueError('observation assigned to another fit case')
            actual=load_attempt(bundle,row)
            if actual['process']['phase']!='fit' or actual['process']['design_sha256']!=design['design_sha256']:raise ValueError('heldout/foreign observations cannot enter fit')
    quality={cid:sample_quality(bundle,rows) for cid,rows in fit_cases.items()}
    result={'schema':schema(bundle,'fit-intervals'),**identity(bundle),'status':'passed','quality':quality,
            'families':{},'heldout_service_intervals_ns':{},'physical_parameters':None,'hardware_accuracy':'not validated',
            'parameter_mapping':'no certified clock/path/issue proof; native parameters remain explicit unchanged assumptions',
            'confounders':['event resolution is approximate, assumed endpoint errors not guaranteed','fixed launch/prologue overhead and three samples','unroll changes instruction-cache footprint','register allocation stable within each family, not pipeline proof','physical clocks unknown','cache state/queue path uncontrolled; chase is one composite observation']}
    if any(q['status']!='passed' for q in quality.values()):result.update(status='refused',reason='preregistered sample dispersion failed; no additional observation');return result
    error=Fraction(QUALITY['assumed_event_interval_abs_error_ns'])
    for family in dict.fromkeys(c['family'] for c in bundle.cases.values()):
        points=[(Fraction(bundle.cases[cid]['steps']),Fraction(load_attempt(bundle,a)['elapsed_ns'])) for cid,rows in fit_cases.items() if bundle.cases[cid]['family']==family for a in rows]
        vertices=_vertices(points,error)
        if not vertices:result.update(status='refused',reason='conditional affine interval model inconsistent on fit');result['families'][family]={'status':'inconsistent'};continue
        rates=[b for a,b in vertices];offsets=[a for a,b in vertices]
        result['families'][family]={'status':'resolution_limited' if min(rates)==0 else 'conditionally_bounded',
            'composite_ns_per_round':[str(min(rates)),str(max(rates))],'fixed_overhead_ns':[str(min(offsets)),str(max(offsets))],
            'feasible_vertices':[[str(a),str(b)] for a,b in vertices],'physical_native_parameter':None}
        for cid in design['order']['heldout']:
            c=bundle.cases[cid]
            if c['family']==family:
                ys=[a+b*c['steps'] for a,b in vertices]
                result['heldout_service_intervals_ns'][cid]=[str(min(ys)),str(max(ys))]
    result['fit_only_sha256']=seal({'quality':quality,'families':result['families'],'predictions':result['heldout_service_intervals_ns']})
    return result

class Journal:
    """Append-only evidence transitions; terminal failure closes all unused slots."""
    def __init__(self,directory,bundle,design,*,_replay_events=None,_denominator=None):
        self.root=Path(directory);self.bundle=bundle;self.design=validate_design(bundle,design)
        self.files=[];self.state='registered';self.processes=[];self.fit={};self.heldout={};self.initial=None;self.final=None
        self._replay_events=_replay_events;self._denominator=_denominator
        self.append('registered',{'design_sha256':design['design_sha256']})
    def append(self,state,payload):
        if self.state in ('closed','failed'):raise ValueError('terminal experiment cannot be reopened')
        event={'schema':schema(self.bundle,'journal-event'),**identity(self.bundle),'run_id':self.design['run_id'],'index':len(self.files),'previous':self.files[-1] if self.files else None,
               'state':state,'realtime_ns':time.time_ns(),'payload':payload}
        if self._replay_events is None:
            row=write_new(self.root/(f'{len(self.files):04d}-{state}.json'),event)
        else:
            if len(self.files)>=len(self._replay_events):raise ValueError('journal has no matching transition')
            row,actual=self._replay_events[len(self.files)]
            expected=dict(event,realtime_ns=actual['realtime_ns'])
            if not same_json(actual,expected):raise ValueError('journal transition/payload differs from verified observations')
        self.files.append(row);self.state=state
        return row
    def structures(self,proofs):
        if self.state!='registered' or set(proofs)!=set(self.bundle.cases):raise ValueError('all structures before initial predictions required')
        from arch_explorer.components.accelsim.microcapture import derive,references
        for cid,row in proofs.items():
            binding,_=json_file(original(row));files=references(binding)
            process,_=json_file(files['supervisor'])
            if binding['case']!=cid or process['design_sha256']!=self.design['design_sha256']:raise ValueError('structure belongs to another case/run/design')
            derive(binding)
        self.append('structures',proofs)
    def initial_predictions(self,freeze):
        if self.state!='structures':raise ValueError('initial predictions require complete structures')
        validate_prediction_freeze(self.bundle,self.design,freeze,'initial')
        self.initial=freeze;self.append('initial_predictions',freeze)
    def timing(self,cid,attempts,freeze):
        part=self.bundle.cases[cid]['partition'];target=self.fit if part=='fit' else self.heldout
        if cid in target or self.state not in (('initial_predictions','fit') if part=='fit' else ('final_predictions','heldout')):raise ValueError('duplicate/out-of-phase timing, no retry')
        if not same_json(freeze,self.initial if part=='fit' else self.final):raise ValueError('timing freeze differs')
        if cid!=self.design['order'][part][len(target)]:raise ValueError('unregistered timing order')
        for a in attempts:validate_timing_phase(self.bundle,self.design,freeze,a)
        quality=sample_quality(self.bundle,attempts);target[cid]=attempts
        self.append(part,{'case':cid,'quality':quality,'process':load_attempt(self.bundle,attempts[0])['process']})
        if quality['status']!='passed':self.fail('sample_quality',cid)
        return quality
    def final_predictions(self,freeze):
        if self.state!='fit' or set(self.fit)!=set(self.design['order']['fit']):raise ValueError('complete fit partition required before final freeze')
        validate_prediction_freeze(self.bundle,self.design,freeze,'final',fit_result=fit_intervals(self.bundle,self.design,self.fit),parent=self.initial)
        self.final=freeze;self.append('final_predictions',freeze)
    def fail(self,reason,at):
        remaining=[{'case':cid,'sample':i,'status':'not_started'} for cid in self.bundle.cases for i in range(3) if cid not in self.fit and cid not in self.heldout]
        if self._denominator is not None:
            remaining=[{k:r[k] for k in ('case','sample','status')} for r in self._denominator if r['mode']=='timing' and r['case'] not in self.fit and r['case'] not in self.heldout]
        return self.append('failed',{'reason':reason,'at':at,'remaining':remaining,'retry_allowed':False})
    def close(self):
        if self.state!='heldout' or set(self.heldout)!=set(self.design['order']['heldout']):raise ValueError('all timing-heldout observations required to close success')
        return self.append('closed',{'fit':list(self.fit),'heldout':list(self.heldout),'hardware_accuracy':'not validated'})
