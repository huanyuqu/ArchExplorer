"""Independent CPU reader/oracles for SIM28 raw capture; never calls CUDA.

This checks captured bytes/masks/addresses/exit coverage against the original
cubin and the controlled application's memory allocation, not native timing.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import struct

HEADER = struct.Struct('<8s14I')
RECORD = struct.Struct('<8I32Q')
CAPACITY = 4096


def require(value, reason):
    if not value:
        raise ValueError(reason)


def text_sections(file):
    raw = Path(file).read_bytes()
    require(raw[:6] == b'\x7fELF\x02\x01', 'little-endian ELF64 cubin required')
    header = struct.unpack_from('<16sHHIQQQIHHHHHH', raw)
    require(header[2] == 190, 'CUDA ELF machine required')
    offset, entry, count, names = header[6], header[11], header[12], header[13]
    require(entry == 64 and count > 0, 'bounded ELF section header table required')
    require(offset+entry*count <= len(raw), 'ELF section table outside file')
    sections = [struct.unpack_from('<IIQQQQIIQQ', raw, offset+i*entry) for i in range(count)]
    require(names < count, 'ELF shstrndx out of range')
    strings = raw[sections[names][4]:sections[names][4]+sections[names][5]]
    result = {}
    for section in sections:
        name = strings[section[0]:].split(b'\0', 1)[0].decode()
        if name.startswith('.text.'):
            require(section[4]+section[5] <= len(raw), 'text section outside cubin')
            require(section[5] % 16 == 0, 'SM120 instruction alignment required')
            result[name[6:]] = raw[section[4]:section[4]+section[5]]
    return result


def read_records(file):
    raw = Path(file).read_bytes()
    require(64 <= len(raw) <= 64+CAPACITY*288, 'record file size is outside the fixed budget')
    h = HEADER.unpack_from(raw)
    magic, version, header, size, count, attempted, capacity, overflow, launches, *geometry = h
    require((magic,version,header,size,capacity) == (b'SIM28NV1',1,64,288,CAPACITY), 'record ABI differs')
    require(0 < count == attempted <= capacity and overflow == 0, 'capture is empty, truncated, or overflowed')
    require(launches == 1 and geometry == [4,1,1,128,1,1], 'capture launch scope differs')
    require(len(raw) == header+size*count, 'record file has missing/trailing bytes')
    rows = []
    for row in RECORD.iter_unpack(raw[64:]):
        iid,x,y,z,warp,active,executed,sequence,*addresses = row
        require(x < 4 and y == z == 0 and warp < 4, 'record outside CTA/warp scope')
        require(active > 0 and not executed & ~active, 'impossible active/exec mask')
        require(all(a == 0 for lane,a in enumerate(addresses) if not executed & (1 << lane)), 'inactive lane address must be zero')
        rows.append({'instr_id':iid,'cta':x,'warp':warp,'active_mask':active,'exec_mask':executed,
            'sequence':sequence,'addresses':addresses})
    return rows


def width(opcode):
    tokens = opcode.split('.')
    for token in tokens[1:]:
        if re.fullmatch(r'(?:U|S)?(?:8|16|32|64|128)', token):
            return int(token.lstrip('US'))//8
        require(not re.fullmatch(r'[US]?\d+',token),'unsupported numeric memory width')
    # The audited ordinary LDG.E/STG.E form is 32-bit by ISA definition.
    require(tokens[0] in ('LDG','STG'), 'width must come from an audited opcode')
    return 4


def expected_output(case, name):
    width_by_name = {'out16':2,'out32':4,'out64':8,'out128':16}
    w = width_by_name[name]
    data = bytearray(b'\xa5'*(512*w))
    for i in range(501):
        if name == 'out16': struct.pack_into('<H',data,2*i,(i*13+24)&0xffff)
        elif name == 'out32': struct.pack_into('<I',data,4*i,(i*17+9)^0x01020304)
        elif name == 'out64': struct.pack_into('<Q',data,8*i,((i << 33)^0x0011223344556677)+0x12345)
        else: struct.pack_into('<4I',data,16*i,i,i+65536,3*i+7,0xabcdef00^i)
    require(case in ('u16','u32_u64','u128'), 'unknown case')
    return bytes(data)


def verify(directory, cubin, static_oracle=None):
    directory = Path(directory)
    static = json.loads((directory/'static.json').read_text())
    launch = json.loads((directory/'launch.json').read_text())
    capture = json.loads((directory/'capture.json').read_text())
    require(static['schema'] == 'arch-explorer.sim28-nvbit-static.v1' and static['nvbit'] == '1.8.1', 'static contract differs')
    require(launch['schema'] == 'arch-explorer.sim28-launch.v1' and launch['grid'] == [4,1,1]
        and launch['block'] == [128,1,1] and launch['n'] == 501 and launch['allocated_elements'] == 512, 'launch contract differs')
    require(capture['status'] == 'captured' and capture['overflow'] == 0 and capture['application_launches'] == 1,
        'capture is not complete')
    require(static['kernel'] == launch['function'] == 'sim28_'+launch['case'], 'kernel identity differs')
    sections = text_sections(cubin)
    oracle=json.loads(Path(static_oracle or Path(cubin).parent/'OFFLINE-CHECK.json').read_text())
    require(oracle['status']=='passed','independent static disassembly oracle did not pass')
    functions = {r['id']:r['name'] for r in static['functions']}
    register_pattern=r'\b(?:URZ|RZ|UPT|PT|UR\d+|UP\d+|R\d+|P\d+)\b'
    instructions = {}
    for instruction in static['instructions']:
        iid = instruction['id']
        require(iid not in instructions, 'duplicate static instruction id')
        instructions[iid] = instruction
        body = sections[functions[instruction['function_id']]]
        pc = instruction['offset']
        require(pc % 16 == 0 and pc+16 <= len(body), 'static PC outside actual cubin')
        require(bytes.fromhex(instruction['binary_hex']) == body[pc:pc+16], 'NVBit raw instruction differs from original cubin')
        original=oracle['functions'][functions[instruction['function_id']]]
        require(original['text_sha256']==hashlib.sha256(body).hexdigest(),'static oracle belongs to different cubin text')
        decoded=next(r for r in original['instructions'] if r['pc']==pc)
        require(decoded['binary_hex']==instruction['binary_hex'],'static oracle instruction bytes differ')
        sass=decoded['sass']
        pred=re.match(r'^@(!?)(UP|P)(\d+)\s+',sass)
        bare=sass[pred.end():] if pred else sass
        opcode=bare.split()[0]
        require(instruction['opcode']==opcode and instruction['opcode_short']==opcode.split('.')[0],
            'NVBit opcode metadata differs from independent disassembly')
        require(instruction['has_pred']==bool(pred),'NVBit predicate presence differs')
        if pred:
            require(instruction['pred_num']==int(pred[3]) and instruction['pred_uniform']==(pred[2]=='UP')
                and instruction['pred_neg']==(pred[1]=='!'),'NVBit guard metadata differs')
        require(re.findall(register_pattern,instruction['sass'])==re.findall(register_pattern,sass),
            'NVBit SASS register list differs from independent disassembly')
        require(type(instruction['operands']) is list, 'all raw operands required')
        for operand in instruction['operands']:
            require({'type','type_name','nbytes','str','is_neg','is_not','is_abs'} <= set(operand), 'incomplete raw operand')
            kind=operand['type_name']
            tokens=re.findall(register_pattern,operand['str'])
            if kind in ('REG','UREG','PRED','UPRED'):
                require(tokens,'typed register operand lacks printed register')
                token=tokens[0]
                number=255 if token in ('RZ','URZ') else 7 if token in ('PT','UPT') else int(re.search(r'\d+',token)[0])
                field='reg' if kind in ('REG','UREG') else 'pred'
                require(operand[field]['num']==number,'NVBit typed register number differs from operand text/SM120 zero identity')
            if kind=='MREF' and operand['mref']['has_desc']:
                desc=re.search(r'desc\[UR(\d+)\]',operand['str'])
                require(desc and int(desc[1])==operand['mref']['desc_ureg_num'],'NVBit descriptor metadata differs')
    rows = read_records(directory/'records.bin')
    require(len(rows) == capture['count'] == capture['attempted'], 'capture count differs')
    groups = defaultdict(list)
    buffers = {r['name']:r for r in launch['buffers']}
    for row in rows:
        require(row['instr_id'] in instructions, 'dynamic instruction lacks static metadata')
        groups[row['cta'],row['warp']].append(row)
    require(set(groups) == {(c,w) for c in range(4) for w in range(4)}, 'missing/extra warp streams')
    observations = Counter()
    size_disagreements = []
    entry_id=next(fid for fid,name in functions.items() if name==static['kernel'])
    entry_instructions=sorted((r for r in instructions.values() if r['function_id']==entry_id),key=lambda r:r['offset'])
    final_exit=next(r['offset'] for r in entry_instructions if r['opcode_short']=='EXIT' and not r['has_pred'])
    expected_pcs=list(range(0,final_exit+16,16))
    require([r['offset'] for r in entry_instructions if r['offset']<=final_exit]==expected_pcs,
        'incomplete original straight-line prefix')
    require(all(r['opcode_short'] not in ('BRA','CALL','RET','BRX') for r in entry_instructions if r['offset']<=final_exit),
        'this controlled address oracle requires the compiled straight-line cases')
    memory_expected = {'u16':{('LDG',2),('STG',2)}, 'u32_u64':{('LDG',4),('STG',4),('LDG',8),('STG',8)},
        'u128':{('LDG',16),('STG',16)}}[launch['case']]
    for (cta,warp),stream in groups.items():
        stream.sort(key=lambda r:r['sequence'])
        require([r['sequence'] for r in stream] == list(range(len(stream))), 'missing/duplicate warp sequence')
        require([instructions[r['instr_id']]['offset'] for r in stream]==expected_pcs,
            'missing/extra dynamic original instruction, including predicated-off EXIT')
        alive = 0xffffffff
        seen = Counter()
        for row in stream:
            ins = instructions[row['instr_id']]
            require(not row['active_mask'] & ~alive, 'an exited lane reappeared')
            short = ins['opcode_short']
            if not ins['has_pred']:
                require(row['exec_mask']==row['active_mask'],'unpredicated instruction lost active lanes')
            if short=='EXIT' and ins['has_pred']:
                tail=sum(1<<lane for lane in range(32) if cta*128+warp*32+lane>=501)
                require(row['exec_mask']==row['active_mask']&tail,'predicate EXIT mask differs from controlled tail')
            if short in ('LDG','STG'):
                w = width(ins['opcode'])
                require((short,w) in memory_expected, 'unexpected memory operation/width')
                expected_mask = sum(1 << lane for lane in range(32) if cta*128+warp*32+lane < 501)
                require(row['exec_mask'] == expected_mask, 'global memory tail mask differs from controlled launch')
                name = ('in' if short=='LDG' else 'out')+str(w*8)
                buf = buffers[name]
                require(buf['element_bytes'] == w, 'record width differs from actual allocation')
                require(ins['mref_count'] == 1, 'exactly one original MREF required')
                for lane,address in enumerate(row['addresses']):
                    if row['exec_mask'] & (1 << lane):
                        expected = buf['address'] + (cta*128+warp*32+lane)*w
                        require(address == expected, 'observed MREF address disagrees with controlled buffer+index')
                if ins['size'] != w:
                    size_disagreements.append({'instr_id':ins['id'],'nvbit_size':ins['size'],'sass_width':w})
                seen[short,w] += 1
                observations[short,w] += row['exec_mask'].bit_count()
            if short == 'EXIT': alive &= ~row['exec_mask']
        require(alive == 0, 'incomplete EXIT coverage')
        require(set(seen) == memory_expected and all(n == 1 for n in seen.values()), 'not a complete single-pass controlled kernel')
    for name,buf in buffers.items():
        if not buf['output']: continue
        expected = expected_output(launch['case'],name)
        require((directory/(name+'.expected.bin')).read_bytes() == expected, 'retained CPU oracle differs from independent formula')
        require((directory/(name+'.actual.bin')).read_bytes() == expected, 'actual output or tail differs from independent CPU formula')
    return {'schema':'arch-explorer.sim28-independent-capture-check.v1','status':'passed','case':launch['case'],
        'records':len(rows),'warp_streams':len(groups),'cubin_sha256':hashlib.sha256(Path(cubin).read_bytes()).hexdigest(),
        'raw_static_instructions_matched':len(instructions),'global_memory_active_lanes':{f'{k[0]}_{k[1]}':v for k,v in observations.items()},
        'nvbit_size_disagreements':size_disagreements,'all_exits':True,'full_output_and_tail':True,
        'scope':'CPU re-read of captured structure and functional outputs; no hardware timing or native functional recomputation'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--cubin',type=Path,required=True)
    parser.add_argument('--static-oracle',type=Path)
    args=parser.parse_args()
    try: result=verify(args.directory,args.cubin,args.static_oracle)
    except (ValueError,KeyError,TypeError,OSError,struct.error) as e:
        print(json.dumps({'status':'blocked','reason':str(e)})); return 2
    print(json.dumps(result,indent=2)); return 0


if __name__=='__main__': raise SystemExit(main())
