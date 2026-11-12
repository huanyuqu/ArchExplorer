"""CPU-only, explicitly synthetic finite process/device boundary fixtures.

This module never starts a process, loads a library, acquires a lock, or accesses
a device. The strict v1 raw-record parser has no synthetic mode: its internal
hardware-shaped fields are emulated here, inside a root marked SYNTHETIC.json.
Only an explicitly synthetic rehearsal may consume this root. It must never be
passed to a hardware evidence or resource authorization entrypoint.

The fixture deliberately reuses the production raw parser without monkeypatches
or provenance flags. It replaces only the process/device boundary. Preserved
capture records/static tables and completed native protocols remain read-only;
new timing values, event handles, process IDs, maps, and resource observations are
all CPU-generated simulation data. A file named fake-lease.json is a format
fixture, never a GPU lease. Its lock path is not created or opened.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import struct
import time
import uuid

from arch_explorer.components.accelsim.microbundle import load_registry, oracle
from arch_explorer.components.accelsim.microcontract import budget, modern, work_item, schema, identity
from arch_explorer.components.accelsim.microexperiment import (
    make_design, prediction_freeze, ref, sample_quality, write_new,
)
from arch_explorer.components.accelsim.microrecords import (
    BASE, ENV_KEYS, EVENTS, attempt_reference, case_files, load_attempt, original,
    validate_case_attempt, validate_loader_environment,
)
from arch_explorer.components.accelsim.trace import digest


def _parts(text):
    """Split the deliberately supported straight-line SASS operand grammar."""
    depth = 0
    start = 0
    result = []
    for index, char in enumerate(text):
        if char == '[':
            depth += 1
        elif char == ']':
            depth -= 1
        elif char == ',' and depth == 0:
            result.append(text[start:index].strip())
            start = index + 1
        if depth < 0:
            raise ValueError('unbalanced synthetic SASS operand')
    if depth:
        raise ValueError('unbalanced synthetic SASS operand')
    if text[start:].strip():
        result.append(text[start:].strip())
    return result


def _operand(text, width=4):
    common = dict(type=0, type_name='', nbytes=width, str=text,
                  is_neg=False, is_not=False, is_abs=False)
    register = re.fullmatch(r'(UR|R)(Z|[0-9]+)(\.reuse)?', text)
    if register:
        uniform = register[1] == 'UR'
        number = 255 if register[2] == 'Z' else int(register[2])
        return dict(common, type=4 if uniform else 2,
                    type_name='UREG' if uniform else 'REG',
                    reg={'num':number, 'prop':register[3] or ''})
    constant = re.fullmatch(r'c\[(0x[0-9a-f]+)\]\[(0x[0-9a-f]+)\]', text, re.I)
    if constant:
        return dict(common, type=6, type_name='CBANK', cbank={
            'id':int(constant[1], 16), 'has_imm_offset':True,
            'imm_offset':int(constant[2], 16), 'has_reg_offset':False, 'reg_offset':0})
    memory = re.fullmatch(r'desc\[UR([0-9]+)\]\[R([0-9]+)(\.64)?(?:\+(0x[0-9a-f]+))?\]', text, re.I)
    if memory:
        return dict(common, type=7, type_name='MREF', mref={
            'has_ra':True, 'ra_num':int(memory[2]), 'ra_mod':6 if memory[3] else 0,
            'has_ur':False, 'ur_num':0, 'ur_mod':0,
            'has_imm':memory[4] is not None, 'imm':int(memory[4],16) if memory[4] else 0,
            'has_desc':True, 'desc_ureg_num':int(memory[1])})
    if re.fullmatch(r'SR_(TID|CTAID)\.[XYZ]', text):
        return dict(common, type=8, type_name='GENERIC', generic=text)
    if re.fullmatch(r'(?:0x[0-9a-f]+|[0-9]+)', text, re.I):
        return dict(common, type=0, type_name='IMM_UINT64', nbytes=8,
                    imm_uint64=int(text,16 if text.lower().startswith('0x') else 10))
    raise ValueError('unsupported complete synthetic operand: '+text)


def _description(instruction, exemplars, branch_offsets):
    """Use observed descriptor layouts while retaining the new complete SASS."""
    sass = instruction['sass'].strip().removesuffix(';').strip()
    if not sass or sass.startswith('@'):
        raise ValueError('predicated/empty complete flow requires an explicit fixture mechanism')
    fields = sass.split(None,1)
    opcode = fields[0]
    if opcode not in exemplars:
        raise ValueError('no observed descriptor layout for complete opcode: '+opcode)
    result = copy.deepcopy(exemplars[opcode])
    result['sass'] = sass+';'
    result['operands'] = []
    for text in _parts(fields[1] if len(fields)>1 else ''):
        if opcode=='BRA' and re.fullmatch(r'`\([A-Za-z0-9_.]+\)',text):
            # nvdisasm prints a symbol for compiler padding whereas NVBit uses
            # its resolved target. Reuse only a byte-identical observed branch
            # encoding and its proven PC-relative offset. Keep the new SASS and
            # machine bytes intact; a reachable branch still fails emulation.
            if instruction['reachability']=='in complete candidate flow' or instruction['binary_hex'] not in branch_offsets:
                raise ValueError('unresolved complete branch metadata')
            text=hex(instruction['pc']+branch_offsets[instruction['binary_hex']])
        width = result['size'] if opcode.split('.')[0] in ('LDC','LDCU','LDG','STG') else 4
        result['operands'].append(_operand(text,width))
    # Descriptor flags describe the observed opcode; registers, immediates and
    # CBANK/MREF fields above belong to this new complete instruction only.
    return result


def _emulate(case, main, aux, expected, allocation, instructions):
    """Evaluate finite integer instructions for CPU fixture MREF/output bytes.

    This has no timing/physical model. Unhandled reachable instructions fail;
    unreachable post-EXIT padding remains in the static table, never discarded.
    The independent production address/output oracle validates the result.
    """
    from arch_explorer.components.accelsim.microbundle import words
    registers = {'R':{}, 'UR':{}}
    buffers = {b['name']:dict(b, data=bytearray(
        main if b['name']=='input' else aux if b['name']=='auxiliary' else b'\xa5'*len(expected)))
        for b in allocation['buffers']}
    constants = bytearray(4096)
    struct.pack_into('<Q', constants, 0x358, 0x700000000)
    arguments = [buffers['input']['address']]
    if 'auxiliary' in buffers:
        arguments += [buffers['auxiliary']['address'], buffers['output']['address']]
    else:
        mul, add = words(aux)
        arguments += [buffers['output']['address'], mul | (add<<32)]
    for index,value in enumerate(arguments):
        struct.pack_into('<Q', constants, 0x380+index*8, value)

    def reg(kind,number,lane):
        if number==255:
            return 0
        return registers[kind].get(number,[0]*32)[lane]

    def value(operand,lane,bits=32):
        kind = operand['type_name']
        if kind in ('REG','UREG'):
            space = 'UR' if kind=='UREG' else 'R'
            number = operand['reg']['num']
            if number==255:
                return 0
            if bits not in (32,64,128) or number+bits//32>255:
                raise ValueError('unsupported synthetic register vector extent')
            return sum(reg(space,number+part,lane)<<(32*part) for part in range(bits//32))
        if kind=='IMM_UINT64':
            return operand['imm_uint64']
        if kind=='CBANK':
            cb = operand['cbank']
            if cb['id'] or not cb['has_imm_offset'] or cb['has_reg_offset']:
                raise ValueError('unhandled synthetic CBANK')
            offset = cb['imm_offset']
            if not 0<=offset<=len(constants)-bits//8:
                raise ValueError('synthetic CBANK offset bound')
            return int.from_bytes(constants[offset:offset+bits//8],'little')
        if kind=='GENERIC':
            return lane if operand['generic']=='SR_TID.X' else 0
        raise ValueError('unhandled synthetic numeric operand: '+kind)

    def assign(operand, values, bits=32):
        kind = 'UR' if operand['type_name']=='UREG' else 'R'
        number = operand['reg']['num']
        if number==255:
            return
        if bits not in (32,64,128) or number+bits//32>255:
            raise ValueError('unsupported synthetic destination vector extent')
        for part in range(bits//32):
            registers[kind][number+part] = [(v>>(32*part))&0xffffffff for v in values]

    def location(address,size):
        matches = [(b,address-b['address']) for b in buffers.values()
                   if b['address']<=address and address+size<=b['address']+b['bytes']]
        if len(matches)!=1:
            raise ValueError('synthetic MREF outside frozen input/output allocation')
        return matches[0]

    records=[]
    for iid,ins in instructions:
        opcode=ins['opcode'];base=ins['opcode_short'];ops=ins['operands'];addresses=[0]*32
        if base in ('LDC','LDCU'):
            bits=ins['size']*8
            assign(ops[0],[value(ops[1],lane,bits) for lane in range(32)],bits)
        elif base=='S2R':
            assign(ops[0],[value(ops[1],lane) for lane in range(32)])
        elif base=='IMAD':
            if opcode not in ('IMAD','IMAD.WIDE.U32'):
                raise ValueError('unhandled reachable synthetic arithmetic: '+opcode)
            bits=64 if opcode=='IMAD.WIDE.U32' else 32
            assign(ops[0],[(value(ops[1],lane)*value(ops[2],lane)+value(ops[3],lane,bits))&((1<<bits)-1)
                           for lane in range(32)],bits)
        elif base=='SHF':
            if opcode!='SHF.L.U32' or any(value(ops[3],lane) for lane in range(32)):
                raise ValueError('unhandled reachable synthetic funnel shift')
            assign(ops[0],[(value(ops[1],lane) << (value(ops[2],lane)&31))&0xffffffff for lane in range(32)])
        elif base in ('LDG','STG'):
            memory = next(o['mref'] for o in ops if o['type_name']=='MREF')
            if not memory['has_ra'] or memory['has_ur']:
                raise ValueError('unhandled synthetic MREF base')
            addresses=[reg('R',memory['ra_num'],lane)
                       +(reg('R',memory['ra_num']+1,lane)<<32 if memory['ra_mod']==6 else 0)
                       +(memory['imm'] if memory['has_imm'] else 0) for lane in range(32)]
            if base=='LDG':
                loaded=[]
                for a in addresses:
                    b,offset=location(a,ins['size'])
                    loaded.append(int.from_bytes(b['data'][offset:offset+ins['size']],'little'))
                assign(ops[0],loaded,ins['size']*8)
            else:
                for lane,a in enumerate(addresses):
                    b,offset=location(a,ins['size'])
                    if b['name']!='output':
                        raise ValueError('synthetic store targets an input')
                    b['data'][offset:offset+ins['size']]=value(ops[1],lane,ins['size']*8).to_bytes(ins['size'],'little')
        elif base!='EXIT':
            raise ValueError('unhandled reachable synthetic opcode; retain and request mechanism: '+opcode)
        records.append(struct.pack('<8I32Q',iid,0,0,0,0,0xffffffff,0xffffffff,len(records),*addresses))
        if base=='EXIT':
            break
    if len(records)!=case['full_flow_instructions'] or not instructions or instructions[len(records)-1][1]['opcode_short']!='EXIT':
        raise ValueError('synthetic complete flow/EXIT differs from registered program')
    if bytes(buffers['output']['data'])!=expected:
        raise ValueError('emulated new SASS disagrees with independent frozen output oracle')
    return records


def synthesize_capture(bundle, case, allocation, template_root):
    """Return explicitly synthetic capture bytes for this complete ELF/STATIC.

    The old delivery supplies descriptor layouts only. No old record, static
    table, instruction address or case-name substitution is consumed here.
    """
    template=Path(template_root)/'gpu-window/host-00-structure'
    exemplars={};branch_offsets={}
    for p in sorted(template.glob('*/static.json')):
        meta=json.loads(p.read_text())
        for description in meta['descriptions']:
            exemplars.setdefault(description['opcode'],description)
        table=(p.parent/'static.bin').read_bytes()
        for offset in range(0,len(table),40):
            iid,fid,pc,index,di,reserved,machine=struct.unpack_from('<6I16s',table,offset)
            d=meta['descriptions'][di]
            if d['opcode']=='BRA' and len(d['operands'])==1 and d['operands'][0]['type_name']=='IMM_UINT64':
                delta=d['operands'][0]['imm_uint64']-pc
                if machine.hex() in branch_offsets and branch_offsets[machine.hex()]!=delta:
                    raise ValueError('observed branch-layout examples disagree')
                branch_offsets[machine.hex()]=delta
    compiled=bundle.static[case['function']]
    descriptions=[];indexes={};table=[];decoded=[]
    for j,ins in enumerate(compiled['instructions']):
        description=_description(ins,exemplars,branch_offsets)
        key=json.dumps(description,sort_keys=True,separators=(',',':'))
        if key not in indexes:
            if len(descriptions)>=64 or len(key.encode())>2048:
                raise ValueError('synthetic full static dictionary exceeds frozen channel')
            indexes[key]=len(descriptions);descriptions.append(description)
        machine=bytes.fromhex(ins['binary_hex'])
        if len(machine)!=16 or machine!=bundle.texts[case['function']][j*16:j*16+16] or ins['pc']!=j*16:
            raise ValueError('new complete static instruction does not bind original ELF')
        table.append(struct.pack('<6I16s',j,0,ins['pc'],j,indexes[key],0,machine))
        if ins['reachability']=='in complete candidate flow':
            decoded.append((j,description))
    main,aux,expected=bundle.inputs(case)
    if oracle(case,main,aux)!=expected:
        raise ValueError('new frozen input/output oracle changed')
    records=_emulate(case,main,aux,expected,allocation,decoded)
    meta={'schema':'arch-explorer.sim30-nvbit-static.v1','nvbit':'1.8.1','kernel':case['function'],
          'grid':case['grid'],'block':case['block'],'buffer_bytes':1179720,
          'functions':[{'id':0,'name':case['function']}],
          'encoding':'lossless-description-dictionary-v1','table_record_bytes':40,
          'static_count':len(table),'descriptions':descriptions}
    count=len(records)
    header=struct.pack('<8s14I',b'SIM30NV1',1,64,288,count,count,4096,0,1,*case['grid'],*case['block'])
    summary={'schema':'arch-explorer.sim30-nvbit-capture.v1','status':'captured','attempted':count,
             'count':count,'overflow':0,'application_launches':1,
             'total_application_launches':list(bundle.cases).index(case['id'])+1,
             'hardware_timing_consumed':False}
    return {'records.bin':header+b''.join(records),'static.bin':b''.join(table),
            'static.json':(json.dumps(meta,indent=2)+'\n').encode(),
            'capture.json':(json.dumps(summary,indent=2)+'\n').encode()}


class Fixture:
    """Generate exactly one frozen CPU rehearsal in a previously absent root.

    ``template_recovery_root`` supplies immutable SIM30 descriptor layouts and
    device-record field examples. Legacy registries retain their old record and
    native boundary behavior; v2 registries synthesize complete new captures
    from their own ELF/STATIC and require ``native_fixture`` to identify their
    own host/tracer/engine/history and completed native protocol set. ``source``
    may be an installed source reference. A fresh synthetic source envelope is
    always written; without ``source`` it makes no installation claim.

    ``process`` follows the existing Collector.host contract and returns
    (output_directory, process_receipt_reference). It never runs the host bytes.
    ``quality_refused`` changes only the first fit slot's three simulated values;
    it does not add or retry a sample. The caller owns Journal, fit, compare,
    closure, and the production-versus-synthetic entrypoint guard.
    """

    def __init__(self, root, bundle_path, template_recovery_root, *, source=None,
                 quality_refused=False, native_fixture=None, state_cpu=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.bundle = load_registry(Path(bundle_path).resolve())
        self.budget = budget(self.bundle)
        self.work_item = work_item(self.bundle)
        self.template = Path(template_recovery_root).resolve()
        self.capture_template = self.template / 'gpu-window' / 'host-00-structure'
        self.quality_refused = quality_refused
        self.processes = []
        self._supervisor_pid = 1_000_000_000
        self._bytes(self.root / 'fixture-generator.py', Path(__file__).read_bytes())
        fixture_module = ref(self.root / 'fixture-generator.py')
        self._template_design = self._json(self.template / 'frozen' / 'DESIGN-v2.json')
        self._native_fixture_ref = None
        self._native_predictions = None
        self.catalog_binding = None
        if native_fixture is not None:
            self._native_fixture_ref = copy.deepcopy(native_fixture) if isinstance(native_fixture,dict) else ref(native_fixture)
            native_plan = self._json(original(self._native_fixture_ref))
            if (native_plan.get('schema') != 'arch-explorer.synthetic-native-fixture.v1'
                or native_plan.get('kind') != 'synthetic'
                or native_plan.get('registry') != ref(self.bundle.path)
                or set(native_plan.get('boundaries',{})) != {'engine','hosts','tracer','history'}):
                raise ValueError('new synthetic native fixture must bind the full current registry and boundaries')
            self._boundaries = copy.deepcopy(native_plan['boundaries'])
            self.catalog_binding = native_plan.get('catalog_binding')
            if set(self._boundaries['hosts']) != {'capture','timing'}:
                raise ValueError('separate new capture/timing boundary identities required')
            for row in [self._boundaries[k] for k in ('engine','tracer','history')]+list(self._boundaries['hosts'].values()):
                original(row)
        elif modern(self.bundle):
            raise ValueError('new registry requires its own synthetic native fixture plan; legacy predictions cannot be renamed')
        else:
            self._boundaries = {k:copy.deepcopy(self._template_design[k]) for k in ('engine','hosts','tracer','history')}
        template_source = self._json(original(self._template_design['source']))
        supplied_source = self._json(original(source)) if source is not None else {}
        source_envelope = dict(supplied_source)
        source_envelope.update(
            schema='arch-explorer.measurement-synthetic-source.v1', kind='synthetic',
            no_resource_authority=True, fixture_uuid=uuid.uuid4().hex,
            runtime_tools=copy.deepcopy(supplied_source.get('runtime_tools',template_source['runtime_tools'])),
            installed_source=copy.deepcopy(source),
            fixture_module=fixture_module,
            generated_realtime_ns=time.time_ns(),
            description='CPU fixture source epoch; no device observation or execution',
            work_item=self.work_item, registry=ref(self.bundle.path),
            native_fixture=self._native_fixture_ref,
        )
        if source is None:
            source_envelope.update(clean=False, installation_claim=False)
        source_ref = write_new(self.root / 'SOURCE.json', source_envelope)
        self.method_file = self.root / 'METHOD.json'
        method = {
            'schema': 'arch-explorer.measurement-synthetic-method.v1',
            'kind': 'synthetic', 'no_resource_authority': True,
            'replaced_boundary': [
                'process creation/wait/reap', 'device/live function observations',
                'allocation addresses', 'event handles and event elapsed values',
                'loader maps and resource coordination observations',
            ],
            'retained_read_only': [
                'registered compiled cubin and frozen inputs',
                'observed descriptor layouts and device-record field examples',
                'completed CPU native request/prediction/review/protocol records for this registry only',
            ],
            'structure_boundary':('new full ELF/STATIC instructions and CPU SASS-evaluated MREF/output bytes'
                                  if modern(self.bundle) else 'unchanged historical captured record/static bytes'),
            'timing_model': 'ns = 10000 + slope[family] * steps; slopes dep=8, quad=32, chase=80',
            'samples': 'three identical float32-ms rounded samples; quality-refused first fit uses 0.8/1/1.2 multipliers',
            'hardware_accuracy': 'not validated', 'physical_parameters': None,
            'not_an_authorization': True,
        }
        if state_cpu is not None:
            method.update(timing_model='shared C++ CPU Driver with explicitly injected intervals; not GPU timing',samples='one diagnostic, two preparation, three formal from immutable state plan',CPU_process_accounting='actual CPU executable/wait records in device-state frames and resource close')
        write_new(self.method_file,method)
        maker=make_design
        if state_cpu is not None:
            from arch_explorer.components.accelsim.state_device import make_design as device_design
            maker=lambda *args,**kwargs:device_design(*args,**kwargs,cpu=state_cpu)
        self.design = maker(
            self.bundle, source_ref, self._boundaries['engine'],
            copy.deepcopy(self._boundaries['hosts']), self._boundaries['tracer'],
            self._boundaries['history'], ref(self.method_file),catalog=self.catalog_binding,
        )
        self.design_file = self.root / 'DESIGN.json'
        write_new(self.design_file, self.design)
        self.design_ref = ref(self.design_file)
        self.run_id = self.design['run_id']
        # Only a file-format fixture. This path is never opened or flocked.
        self.lock_path = self.root / 'NOT-A-RESOURCE-LOCK'
        marker = {
            'schema': 'arch-explorer.measurement-synthetic-fixture.v1',
            'kind': 'synthetic', 'no_resource_authority': True,
            'run_id': self.run_id, 'design': self.design_ref,
            'source': source_ref, 'fixture_module': fixture_module,
            'work_item':self.work_item, 'registry':ref(self.bundle.path),
            'native_fixture':self._native_fixture_ref,
            'created_realtime_ns': time.time_ns(),
            'hardware_shaped_records_are_simulated': True,
            'resource_records_are_format_fixtures': True,
            'boundary_only': True, 'real_world_resource_authority': False,
            'descriptor_template_root': str(self.capture_template),
            'structure_mode':'generated-from-current-ELF' if modern(self.bundle) else 'historical-captured-bytes',
            'device_queries': 0, 'cuInit_calls': 0, 'external_processes': 0,
            'GPU_launches': 0, 'event_API_calls': 0, 'native_runs': 0,
            'locks_opened': 0, 'leases_acquired': 0,
            'warning': 'Only a controlled CPU rehearsal may consume this root; never hardware evidence.',
        }
        if state_cpu is not None:
            marker.update(boundary_scope='synthetic structure generator; actual CPU executor processes separately recorded',external_processes=None,state_CPU=state_cpu)
        self.marker_ref = write_new(self.root / 'SYNTHETIC.json', marker)
        evidence = write_new(self.root / 'FAKE-COORDINATION.json', {
            'schema': 'arch-explorer.measurement-synthetic-boundary.v1',
            'kind': 'synthetic', 'no_resource_authority': True,
            'run_id': self.run_id, 'actual_coordination': False,
            'actual_lock_held': False, 'actual_device_observed': False,
        })
        issued = time.time_ns()
        self.handoff_file = self.root / 'fake-handoff.json'
        handoff = {
            'schema': schema(self.bundle,'resource-handoff'),
            'work_item': self.work_item, 'run_id': self.run_id,
            'resource_released': True, 'external_compute_empty': True,
            'coordinator': 'resident_2e1684fdfc1b',
            'resource_owner': 'SYNTHETIC-CPU-BOUNDARY-NO-OWNER',
            'coordination_evidence': [evidence], 'issued_realtime_ns': issued,
            'lock_file': str(self.lock_path),
        }
        if modern(self.bundle):
            handoff.update(absolute_deadline_realtime_ns=issued+self.budget['total_window_s']*10**9,close_reserve_s=30)
        if state_cpu is not None:
            from arch_explorer.components.accelsim.measurement_state import sha_plan
            from arch_explorer.components.accelsim.state_resource import SYNTHETIC_HANDOFF
            handoff.update(schema=SYNTHETIC_HANDOFF,coordinator='synthetic-coordinator',resource_owner='synthetic-no-device-owner',no_resource_authority=True,registry_sha256=self.bundle.sha256,plan_sha256=sha_plan(self.bundle),budget=self.budget,design_sha256=self.design['design_sha256'],close_reserve_s=60)
        self.handoff_ref = write_new(self.handoff_file,handoff)
        lease = {
            'schema': schema(self.bundle,'lease'),
            'work_item': self.work_item, 'run_id': self.run_id,
            'design_sha256': self.design['design_sha256'],
            'registry_sha256': self.bundle.sha256, 'new_window': True,
            'handoff': self.handoff_ref, 'issued_realtime_ns': issued,
            'deadline_realtime_ns': issued + (self.budget['total_window_s']-handoff.get('close_reserve_s',0)) * 10**9,
            'device_ordinal': 0, 'lock_file': str(self.lock_path), 'budget': self.budget,
        }
        if state_cpu is not None:lease['plan_sha256']=sha_plan(self.bundle)
        self.lease_ref=write_new(self.root/'fake-lease.json',lease)
        self._parameters = None

    @property
    def native_predictions(self):
        """The native boundary is an immutable, same-program CPU protocol set."""
        if self._native_predictions is None:
            from arch_explorer.components.accelsim.protocol import consume_request
            if self._native_fixture_ref is None:
                results=self._json(self.template/'cpu-native-recovery-2/RESULTS.json')
            else:
                results=self._json(original(self._native_fixture_ref))
            rows=results.get('rows')
            if not isinstance(rows,list) or len(rows)!=len(self.bundle.cases) or {r.get('case') for r in rows}!=set(self.bundle.cases):
                raise ValueError('native fixture does not cover every current complete function exactly once')
            predictions={}
            for row in rows:
                current={role:copy.deepcopy(row[role]) for role in ('request','prediction','review','protocol')}
                for reference in current.values():
                    original(reference)
                consumed=consume_request(self._json(original(current['request'])))
                case=self.bundle.cases[row['case']];actual=consumed['actual']
                if (actual['name']!=case['function'] or actual['grid']!=case['grid'] or actual['block']!=case['block']
                    or actual['warp_instructions']!=case['full_flow_instructions']
                    or actual['thread_instructions']!=32*case['full_flow_instructions']
                    or actual['opcodes']!=case['opcode_histogram']
                    or actual['nregs']!=case['resources']['registers_per_thread']):
                    raise ValueError('legacy/foreign native prediction cannot be renamed to a new case')
                if modern(self.bundle) and consumed['manifest']['source']['kind']!='synthetic':
                    raise ValueError('new CPU native fixture must be explicit synthetic, never a relabelled old capture')
                predictions[row['case']]=current
            self._native_predictions=predictions
        return copy.deepcopy(self._native_predictions)

    @staticmethod
    def _json(path):
        return json.loads(Path(path).read_text())

    @staticmethod
    def _bytes(path, value):
        with Path(path).open('xb') as stream:
            stream.write(value)

    @property
    def parameters(self):
        """Unchanged native assumptions; no native execution or solver copy."""
        if self._parameters is None:
            from arch_explorer.components.accelsim.protocol import consume_request
            row = next(iter(self.native_predictions.values()))
            request = self._json(original(row['request']))
            consumed = consume_request(request)  # Parsing/config compilation only.
            adf = write_new(self.root / 'retained-native-adf.json', request['adf'])
            config = write_new(self.root / 'retained-native-config.json', consumed['config'].artifacts)
            self._parameters = {
                'physical_parameters': None, 'hardware_accuracy': 'not validated',
                'scenario': 'CPU synthetic rehearsal with unchanged retained native assumptions',
                'adf': adf, 'native_config': config,
                'unknown': ['physical clocks', 'latency and issue', 'cache/queue decomposition',
                            'interconnect', 'instruction cache'],
            }
        return copy.deepcopy(self._parameters)

    def _preflight(self, pid, rt, mono):
        pre = copy.deepcopy(self._json(self.capture_template / 'preflight.json'))
        pre.update(
            gpu='SYNTHETIC CPU DEVICE -- no hardware observation',
            uuid_hex=self.run_id, pid=pid, parent_pid=self._supervisor_pid,
            host_realtime_ns=rt, host_monotonic_ns=mono,
            module='SYNTHETIC-MODULE-' + str(pid),
            context='SYNTHETIC-CONTEXT-' + str(pid),
            driver_version_file='SYNTHETIC ABI observation; no driver queried',
            driver_version='SYNTHETIC-595-ABI',
            registry_sha256=self.bundle.sha256,cubin_sha256=self.bundle.document['cubin']['sha256'],
            grid=self.bundle.document['grid'],block=self.bundle.document['block'],
        )
        prototype=pre['functions'][0]
        pre['functions']=[]
        for index,case in enumerate(self.bundle.cases.values()):
            function=copy.deepcopy(prototype)
            function.update(case=case['id'],function=case['function'],
                            handle='SYNTHETIC-FUNCTION-'+str(pid)+'-'+str(index),
                            registers=case['resources']['registers_per_thread'],
                            static_shared=case['resources']['static_shared_bytes'],
                            local_bytes=case['resources']['local_section_bytes'],
                            active_blocks_per_sm=1,legal=True)
            pre['functions'].append(function)
        return pre

    def _allocation(self, case, directory, pre, freeze_sha, mode, index):
        main,aux,expected=self.bundle.inputs(case)
        from arch_explorer.components.accelsim.microbundle import words
        names=['input','auxiliary','output'] if case['family']=='chase' else ['input','output']
        sizes={'input':len(main),'auxiliary':len(aux),'output':len(expected)}
        mul,add=(0,0) if case['family']=='chase' else words(aux)
        allocation={'schema':'arch-explorer.sim30-allocation.v1',
                    'registry_sha256':self.bundle.sha256,'case':case['id'],'family':case['family'],
                    'partition':case['partition'],'steps':case['steps'],'function':case['function'],
                    'cubin_sha256':self.bundle.document['cubin']['sha256'],'text_sha256':case['text_sha256'],
                    'input_sha256':case['input']['sha256'],'auxiliary_sha256':case['auxiliary']['sha256'],
                    'expected_sha256':case['expected']['sha256'],'grid':case['grid'],'block':case['block'],
                    'dynamic_shared':0,'stream':0,'correct':None,'output_status':'awaiting_registered_launch',
                    'mul':mul,'add':add,
                    'buffers':[{'name':name,'address':0x100000000+index*0x1000000+j*0x200000,
                                'bytes':sizes[name],'element_bytes':4,'output':name=='output'} for j,name in enumerate(names)]}
        allocation.update(
            context=pre['context'], pid=pre['pid'], freeze_sha256=freeze_sha,
            lease_sha256=self.lease_ref['sha256'],
            input_file=str(directory / 'input.bin'),
            auxiliary_file=str(directory / 'auxiliary.bin'),
            expected_file=str(directory / 'expected.bin'),
            output_initial_file=str(directory / 'output.initial.bin'),
            actual_output_files=[str(directory / (f'sample-{i}.actual.bin'))
                                 for i in range(1 if mode == 'structure' else 3)],
        )
        if mode=='structure' and not modern(self.bundle):
            # Legacy replay keeps actual captured addresses and complete bytes.
            allocation['buffers']=copy.deepcopy(self._json(self.capture_template/case['id']/'allocation.json')['buffers'])
        return allocation

    def _sample(self, case, allocation, sample_index, mode, event_base):
        sample = {key: copy.deepcopy(allocation[key]) for key in BASE}
        start = None if mode == 'structure' else hex(event_base + sample_index * 2)
        end = None if mode == 'structure' else hex(event_base + sample_index * 2 + 1)
        ns = 10000 + {'dep': 8, 'quad': 32, 'chase': 80}[case['family']] * case['steps']
        if self.quality_refused and case['id'] == self.design['order']['fit'][0]:
            ns *= (0.8, 1.0, 1.2)[sample_index]
        ms = struct.unpack('<f', struct.pack('<f', ns / 1e6))[0]
        bits = struct.unpack('<I', struct.pack('<f', ms))[0]
        sample.update(
            schema='arch-explorer.sim30-host-sample.v1', sample_index=sample_index,
            kind='hardware', mode=mode, start_event=start, end_event=end,
            status='returned', launch_return=0, synchronize_return=0,
            event_elapsed_ms=None if mode == 'structure' else ms,
            event_elapsed_ms_float32_bits=None if mode == 'structure' else bits,
            elapsed_ns=None if mode == 'structure' else float(ms) * 1e6,
            unit='not_timed' if mode == 'structure' else 'ns', correct=True,
            output_sha256=case['expected']['sha256'], all_function_handles_live=True,
            hidden_warmup_launches=0, helper_launches=0, hardware_core_mhz=None,
            clock_state='unknown', cache_state='uncontrolled; sample order retained',
        )
        return sample

    def process(self, phase, freeze, cid=None):
        """Materialize one simulated slot; structure None binds the current design.

        Only the bound pure CPU policy harness may run; no host or native engine
        is invoked. A structure slot always generates a fresh raw
        directory with retained record/static bytes and new boundary metadata.
        """
        marker = self._json(original(self.marker_ref))
        if (marker['kind'] != 'synthetic' or marker['no_resource_authority'] is not True
                or marker['run_id'] != self.run_id or marker['design'] != self.design_ref):
            raise ValueError('synthetic root marker no longer binds this CPU fixture')
        order = ['structure'] + self.design['order']['fit'] + self.design['order']['heldout']
        index = len(self.processes)
        if phase not in ('structure', 'fit', 'heldout') or index >= len(order):
            raise ValueError('unknown/extra synthetic process slot')
        slot = 'structure' if phase == 'structure' else cid
        if order[index] != slot or (phase == 'structure' and cid is not None):
            raise ValueError('synthetic process order/slot differs from frozen schedule')
        if phase != 'structure' and self.bundle.cases[cid]['partition'] != phase:
            raise ValueError('synthetic case partition differs')
        if phase == 'structure':
            if freeze is None:
                freeze = self.design['design_sha256']
            elif freeze != self.design['design_sha256']:
                raise ValueError('structure fixture must bind its own new design')
        freeze_ref = None if phase == 'structure' else freeze
        freeze_sha = freeze if phase == 'structure' else freeze['sha256']
        if freeze_ref is not None:
            original(freeze_ref)
        output = self.root / (f'host-{index:02d}-' + (cid or 'structure'))
        output.mkdir(exist_ok=False)
        ids = list(self.bundle.cases) if phase == 'structure' else [cid]
        quota = len(self.bundle.cases) if phase == 'structure' else self.budget['samples_per_case']
        pid = self._supervisor_pid + index + 1
        rt = time.time_ns()
        mono = time.monotonic_ns()
        pre = self._preflight(pid, rt + 2, mono + 2)
        write_new(output / 'preflight.json', pre)
        host = self.design['hosts']['capture' if phase == 'structure' else 'timing']
        env = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'CUDA_VISIBLE_DEVICES': '0',
               'CUDA_MODULE_LOADING': 'EAGER'}
        if phase == 'structure':
            source = self._json(original(self.design['source']))
            nvdisasm = original(source['runtime_tools']['nvdisasm'])
            env.update(LD_PRELOAD=self.design['tracer']['path'], NVDISASM=str(nvdisasm))
        selected_env = {key: env.get(key) for key in ENV_KEYS}
        observed_env = dict(selected_env)
        # Emulate precisely NVBit's documented loader-variable removal; the raw
        # validator still requires the exact frozen tracer in both map files.
        if phase == 'structure':
            observed_env['LD_PRELOAD'] = None
        lock = {'path': str(self.lock_path), 'device': 0, 'inode': 0, 'fd': 198,
                'exclusive': True}
        command = [host['path'], '--bundle', str(self.bundle.path.parent),
                   '--output', str(output), '--lease', self.lease_ref['path'],
                   '--lease-sha', self.lease_ref['sha256'], '--freeze', freeze_sha,
                   '--lock-file', str(self.lock_path), '--lock-fd', str(lock['fd'])]
        if phase != 'structure':
            command += ['--case', cid]
        if self.design.get('state_protocol'):
            lease=self._json(original(self.lease_ref));command+=['--run',self.run_id,'--start-ns',str(rt),'--deadline-ns',str(min(lease['deadline_realtime_ns'],rt+self.budget['host_timeout_s']*10**9))]
        write_new(output / 'command.json', {
            'schema': 'arch-explorer.sim30-host-command.v1', 'argv': command,
            'cwd': str(self.root),
        })
        write_new(output / 'environment.json', {
            'schema': 'arch-explorer.sim30-loader-environment.v1',
            'scope': 'SYNTHETIC in-process loader observation', 'values': observed_env,
        })
        maps = '10000000-10001000 r-xp 00000000 00:00 0 /SYNTHETIC/libcuda.so.1\n'
        if phase == 'structure':
            maps += '20000000-20001000 r-xp 00000000 00:00 0 ' + self.design['tracer']['path'] + '\n'
        for name in ('maps-before-cuInit.txt', 'maps-before-close.txt'):
            self._bytes(output / name, maps.encode())
        write_new(output / 'invocation.json', {
            'schema': schema(self.bundle,'host-invocation'), **identity(self.bundle),
            'registry_sha256': self.bundle.sha256, 'host_sha256': host['sha256'],
            'pid': pid, 'parent_pid': self._supervisor_pid, 'host_realtime_ns': rt + 1,
            'mode': 'structure' if phase == 'structure' else 'timing',
            'lease_file': self.lease_ref['path'], 'lease_sha256': self.lease_ref['sha256'],
            'freeze_sha256': freeze_sha, 'common_lock': str(self.lock_path),
            'lock_fd': lock['fd'], 'receipt_validation': 'SYNTHETIC CPU format only',
            'monotonic_ns': mono + 1,
        })
        if self.design.get('state_protocol'):
            from arch_explorer.components.accelsim.measurement_state import sha_plan,plan_bytes
            from arch_explorer.components.accelsim.trace import digest
            start=int(command[command.index('--start-ns')+1]);deadline=int(command[command.index('--deadline-ns')+1])
            write_new(output/'STATE-INVOCATION.json',{'schema':'arch-explorer.measurement-state-invocation.v1','run':self.run_id,'registry_sha256':self.bundle.sha256,'plan_sha256':sha_plan(self.bundle),'directory_sha256':digest(plan_bytes(self.bundle)),'case':cid,'freeze_sha256':freeze_sha,'start_ns':start,'deadline_ns':deadline,'planned_application_launches':len(self.bundle.cases) if phase=='structure' else 6,'resource_authority':'SYNTHETIC boundary; no actual resource'})
            write_new(output/'STATE-SESSION-CLEANUP.json',{'schema':'arch-explorer.measurement-state-session-cleanup.v2','run':self.run_id,'registry_sha256':self.bundle.sha256,'state_executor_entered':False,'module':'SYNTHETIC-module','context':pre['context'],'module_unload_attempted':True,'context_destroy_attempted':True,'module_unload_return':0,'context_destroy_return':0,'all_module_context_handles_destroyed':True,'cleanup_complete':True,'state_resources_released':False})
        if phase=='structure' and self.catalog_binding is not None:
            from arch_explorer.components.accelsim.microcatalog import policy_sequence
            commands=[command for cid in ids for command in ('begin '+self.bundle.cases[cid]['function'],'launch','finish')]
            write_new(self.root/'CPU-CATALOG-POLICY.json',policy_sequence(self.bundle,self.catalog_binding,commands))
        for case_id in ids:
            case = self.bundle.cases[case_id]
            directory = output / case_id
            directory.mkdir()
            main, aux, expected = self.bundle.inputs(case)
            actual = oracle(case, main, aux)
            if actual != expected:
                raise ValueError('CPU oracle differs from the registered complete output')
            for name, data in (('input.bin', main), ('auxiliary.bin', aux),
                               ('expected.bin', expected), ('output.initial.bin', b'\xa5' * len(expected))):
                self._bytes(directory / name, data)
            mode = 'structure' if phase == 'structure' else 'timing'
            allocation = self._allocation(case, directory, pre, freeze_sha, mode, index)
            write_new(directory / 'allocation.json', allocation)
            write_new(directory / 'launch.json', dict(allocation, schema='arch-explorer.sim30-launch.v1'))
            for i in range(1 if mode == 'structure' else 3):
                sample = self._sample(case, allocation, i, mode, 0xABC000 + index * 16)
                started = {key: copy.deepcopy(sample[key]) for key in BASE | EVENTS | {'sample_index'}}
                started.update(schema='arch-explorer.sim30-started.v1',
                               host_realtime_ns=time.time_ns(), timed=mode == 'timing',
                               status='about_to_attempt_launch')
                write_new(directory / (f'sample-{i}.started.json'), started)
                self._bytes(directory / (f'sample-{i}.actual.bin'), actual)
                write_new(directory / (f'sample-{i}.json'), sample)
                if mode == 'timing':
                    event = {key: copy.deepcopy(sample[key]) for key in BASE | EVENTS |
                             {'sample_index', 'event_elapsed_ms_float32_bits', 'event_elapsed_ms'}}
                    event.update(schema='arch-explorer.sim30-event-raw.v1', event_api_return=0)
                    write_new(directory / (f'sample-{i}.event.json'), event)
            if mode == 'structure':
                if modern(self.bundle):
                    for name,raw in synthesize_capture(self.bundle,case,allocation,self.template).items():
                        self._bytes(directory/name,raw)
                else:
                    for name in ('records.bin', 'static.json', 'static.bin', 'capture.json'):
                        shutil.copyfile(self.capture_template / case_id / name, directory / name)
        write_new(output / 'closed.json', {
            'schema': 'arch-explorer.sim30-host-close.v1', 'status': 'returned',
            'application_launches': quota, 'planned_application_launches': quota,
            'unstarted_application_launches': 0,
            'all_module_context_handles_destroyed': True, 'retry_allowed': False,
        })
        stdout = self.root / (f'process-{index:02d}.stdout')
        stderr = self.root / (f'process-{index:02d}.stderr')
        self._bytes(stdout, b'SYNTHETIC CPU fixture: no executable started\n')
        self._bytes(stderr, b'')
        receipt = {
            'schema': schema(self.bundle,'process-receipt'), **identity(self.bundle), 'kind': 'hardware',
            'phase': phase, 'registry_sha256': self.bundle.sha256,
            'design_sha256': self.design['design_sha256'], 'design': self.design_ref,
            'freeze_sha256': freeze_sha, 'prediction_freeze': freeze_ref,
            'lease': self.lease_ref, 'command': command, 'environment': selected_env,
            'launch_environment': env, 'pid': pid, 'supervisor_pid': self._supervisor_pid,
            'exit_code': 0, 'status': 'returned', 'started_realtime_ns': rt,
            'ended_realtime_ns': time.time_ns(), 'started_monotonic_ns': mono,
            'ended_monotonic_ns': time.monotonic_ns(), 'host': host,
            'output': str(output), 'stdout': ref(stdout), 'stderr': ref(stderr), 'lock': lock,
            'budget_account': {'process_index': index, 'case_ids': ids, 'launch_quota': quota},
        }
        receipt_ref = write_new(self.root / (f'process-{index:02d}.json'), receipt)
        self.processes.append(receipt_ref)
        return output, receipt_ref


def selfcheck(root, bundle_path, template_recovery_root, *, quality_refused=False, source=None,native_fixture=None):
    """Use unmodified strict raw parsers; retain every generated CPU original."""
    fixture = Fixture(root, bundle_path, template_recovery_root,
                      quality_refused=quality_refused,source=source,native_fixture=native_fixture)
    output, process = fixture.process('structure', None)
    from arch_explorer.components.accelsim.microcapture import SCHEMA, derive
    structure_hashes = {}
    for cid, case in fixture.bundle.cases.items():
        files = case_files(output, cid, 0, process['path'])
        validate_case_attempt(fixture.bundle, case, files, 'structure')
        files.update(records=output/cid/'records.bin', static=output/cid/'static.json',
                     static_table=output/cid/'static.bin', capture=output/cid/'capture.json',
                     bundle=fixture.bundle.path,
                     cubin=fixture.bundle.file(fixture.bundle.document['cubin']))
        binding = {'schema': SCHEMA, 'case': cid,
                   'files': {key: ref(path) for key, path in files.items()}}
        # Independent bytes/address coverage and the production renderer handle
        # new synthetic streams exactly as the old retained record transport.
        structure_hashes[cid] = digest(derive(binding))
    initial = prediction_freeze(fixture.bundle, fixture.design,
                                fixture.native_predictions, fixture.parameters,
                                fixture.root / 'freezes', stage='initial')
    cid = fixture.design['order']['fit'][0]
    timing, process = fixture.process('fit', initial, cid)
    attempts = [attempt_reference(fixture.bundle, cid, timing, i, process['path'])
                for i in range(3)]
    loaded = [load_attempt(fixture.bundle, row) for row in attempts]
    quality = sample_quality(fixture.bundle, attempts)
    if quality['status'] != ('refused' if quality_refused else 'passed'):
        raise AssertionError('fixed three-sample quality fixture differs')
    launched = {key: None for key in ENV_KEYS}
    launched['LD_PRELOAD'] = fixture.design['tracer']['path']
    observed = dict(launched, LD_PRELOAD=None)
    maps = ['1000-2000 r-xp 0 00:00 0 ' + fixture.design['tracer']['path']] * 2
    validate_loader_environment('structure', launched, observed, maps, fixture.design['tracer'])
    rejection = {}
    for label, mode, mapped in (('capture_without_mapped_tracer', 'structure', ['', '']),
                                 ('timing_cannot_clear_preload', 'timing', maps)):
        try:
            validate_loader_environment(mode, launched, observed, mapped, fixture.design['tracer'])
        except ValueError as error:
            rejection[label] = str(error)
        else:
            raise AssertionError('loader counterexample accepted: ' + label)
    if fixture.lock_path.exists():
        raise AssertionError('fixture unexpectedly created a resource lock')
    result = {
        'schema': 'arch-explorer.measurement-synthetic-fixture-selfcheck.v1',
        'kind': 'synthetic', 'no_resource_authority': True, 'status': 'passed',
        'run_id': fixture.run_id, 'design': fixture.design_ref,
        'work_item':fixture.work_item,'registry':ref(fixture.bundle.path),
        'strict_parser_unmodified': True, 'raw_parser_bypass': False,
        'structure_cases_validated': list(fixture.bundle.cases),
        'retained_structure_trace_sha256': structure_hashes,
        'timing_attempts': attempts,
        'elapsed_ns': [row['elapsed_ns'] for row in loaded], 'quality': quality,
        'loader_counterexamples_rejected': rejection,
        'external_processes': 0, 'GPU_calls': 0, 'event_API_calls': 0,
        'locks_opened': 0, 'native_runs': 0,
        'physical_parameters': None, 'hardware_accuracy': 'not validated',
    }
    write_new(fixture.root / 'SELF-CHECK.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--template-recovery-root', required=True)
    parser.add_argument('--quality-refused', action='store_true')
    parser.add_argument('--source')
    parser.add_argument('--native-fixture')
    args = parser.parse_args(argv)
    result = selfcheck(args.root, args.bundle, args.template_recovery_root,
                       quality_refused=args.quality_refused,
                       source=None if args.source is None else ref(args.source),native_fixture=args.native_fixture)
    print(json.dumps({'kind': result['kind'], 'status': result['status'],
                      'quality': result['quality']['status'], 'root': str(Path(args.root).resolve())}))


if __name__ == '__main__':
    main()
