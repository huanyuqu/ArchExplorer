"""Check every SM120 execution byte and the existing adapter's static grammar.

This does not manufacture a dynamic trace. Eligibility remains conditional on a
future complete capture and actual all-function resources. No native is started.
"""
from collections import Counter
import hashlib
import re
import struct
from arch_explorer.components.accelsim.capture_input import elf_texts
from arch_explorer.components.accelsim.typed_trace import opcode, register, width


def decode(cubin, disassembly):
    texts = elf_texts(cubin)
    rows = {name: [] for name in texts}
    # Read the actual CUDA ELF EIATTR_REGCOUNT/FRAME_SIZE records. On this
    # toolchain .text sh_info is a symbol-table index, not a register count.
    pos = struct.unpack_from('<Q', cubin, 40)[0]
    size, count, names = struct.unpack_from('<HHH', cubin, 58)
    sections = [struct.unpack_from('<IIQQQQIIQQ', cubin, pos + i * size) for i in range(count)]
    strings = sections[names]
    table = cubin[strings[4]:strings[4] + strings[5]]
    named = {}
    for section in sections:
        name_at = section[0]
        section_name = table[name_at:table.find(b'\0', name_at)].decode('ascii')
        named[section_name] = section
    symtab = named['.symtab']
    string_section = sections[symtab[6]]
    symbol_strings = cubin[string_section[4]:string_section[4] + string_section[5]]
    symbols = []
    if symtab[9] != 24 or symtab[5] % 24:
        raise ValueError('unsupported ELF symbol table layout')
    for at in range(symtab[4], symtab[4] + symtab[5], 24):
        offset = struct.unpack_from('<I', cubin, at)[0]
        symbols.append(symbol_strings[offset:symbol_strings.find(b'\0', offset)].decode('ascii'))
    info = named['.nv.info']
    resources = {n: {'static_shared_bytes': named.get('.nv.shared.' + n, (0,) * 10)[5],
                     'local_section_bytes': named.get('.nv.local.' + n, (0,) * 10)[5],
                     'compiler_reserved_shared_bytes': named.get('.nv.shared.reserved.0', (0,) * 10)[5]}
                 for n in texts}
    end = info[4] + info[5]
    at = info[4]
    while at < end:
        fmt, attribute, length = struct.unpack_from('<BBH', cubin, at)
        if fmt != 4 or at + 4 + length > end:
            raise ValueError('unsupported/truncated global CUDA resource metadata')
        if attribute in (0x2f, 0x11):  # REGCOUNT, FRAME_SIZE: symbol index, value
            if length != 8:
                raise ValueError('unexpected CUDA function resource metadata size')
            index, value = struct.unpack_from('<II', cubin, at + 4)
            if index >= len(symbols) or symbols[index] not in resources:
                raise ValueError('unbound CUDA function resource metadata')
            key = 'registers_per_thread' if attribute == 0x2f else 'stack_frame_bytes'
            if key in resources[symbols[index]]:
                raise ValueError('duplicate CUDA function resource metadata')
            resources[symbols[index]][key] = value
        at += 4 + length
    if any(not 1 <= r.get('registers_per_thread', 0) <= 255 or 'stack_frame_bytes' not in r
           for r in resources.values()):
        raise ValueError('all actual function resource declarations required')
    name = None
    pending = None
    for line in disassembly.splitlines():
        section = re.search(r'\.section\s+\.text\.([^,\s]+)', line)
        if section:
            name = section[1]
        if name not in rows:
            continue
        first = re.search(r'^\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*/\* 0x([0-9a-f]{16}) \*/', line)
        if first:
            if pending:
                raise ValueError('missing high instruction bytes')
            pending = (int(first[1], 16), first[2], int(first[3], 16))
            continue
        second = re.search(r'^\s*/\* 0x([0-9a-f]{16}) \*/', line)
        if second and pending:
            pc, sass, lo = pending
            raw = struct.pack('<QQ', lo, int(second[1], 16))
            if raw != texts[name][pc:pc + 16]:
                raise ValueError('disassembly/cubin byte mismatch')
            normalized = re.split(r'\s[&?]', sass, maxsplit=1)[0].rstrip('; ').strip()
            rows[name].append({'pc': pc, 'sass': normalized, 'disassembler_annotations': sass,
                               'binary_hex': raw.hex()})
            pending = None
    if pending or any([r['pc'] for r in rows[n]] != list(range(0, len(body), 16)) for n, body in texts.items()):
        raise ValueError('complete .text disassembly required; no dropped instructions')
    if set(resources) != set(texts):
        raise ValueError('all function resource declarations required')
    return texts, rows, resources


def instruction(sass, nregs):
    guard = None
    if sass.startswith('@'):
        guard, sass = sass.split(None, 1)
        register(guard[1:].removeprefix('!'), nregs)
    op, *tail = sass.split(None, 1)
    tokens = opcode(op)  # Reuse the actual installed adapter grammar.
    operands = tail[0] if tail else ''
    base = tokens[0]
    if base in ('BRA', 'UBRA'):
        raise ValueError('captured dialect refuses taken branches')
    if base == 'CS2R' or re.search(r'SR_(?:CLOCK|GLOBALTIMER|SMID)', operands):
        raise ValueError('clock/special-register operand not consumed by current capture reader')
    for special in re.findall(r'\bSR_[A-Z0-9_.]+', operands):
        if not re.fullmatch(r'SR_(TID|CTAID)\.[XYZ]', special):
            raise ValueError('unsupported special register ' + special)
    for reg in re.findall(r'\b(?:URZ|RZ|UPT|PT|UR\d+|UP\d+|R\d+|P\d+)\b', operands):
        register(reg, nregs)
    if re.search(r'\bc\[(?!0x0\])', operands) or re.search(r'\bc\[0x0\]\[(?!0x[0-9a-f]+\])', operands):
        raise ValueError('indexed or nonzero constant bank')
    if base in ('LDG', 'STG', 'LDC', 'LDCU', 'ULDC'):
        width(op)
    if base in ('LDG', 'STG'):
        if not re.search(r'desc\[UR\d+\]\[R\d+(?:\.64)?(?:\+0x[0-9a-f]+)?\]', operands):
            raise ValueError('unrecognized complete global address operand')
    return op, guard


def assess(cubin, disassembly):
    texts, rows, resources = decode(cubin, disassembly)
    result = {}
    for name, body in texts.items():
        blockers, after_exit, live = [], False, []
        for row in rows[name]:
            try:
                # The compiler emits a BRA-to-self and NOP fill after an
                # unconditional EXIT. Retain and classify these exact bytes;
                # only this proven dead suffix is outside the dynamic scope.
                sass = row['sass']
                if after_exit:
                    if not (sass.startswith('NOP') or re.fullmatch(r'BRA\s+`?\(\.L_[A-Za-z0-9_]+\)', sass)):
                        raise ValueError('unproven executable suffix after EXIT')
                    row['reachability'] = 'unreachable after unconditional EXIT; bytes retained'
                else:
                    op, guard = instruction(sass, resources[name]['registers_per_thread'])
                    live.append(op)
                    row['reachability'] = 'in complete candidate flow'
                    if op == 'EXIT' and guard is None:
                        after_exit = True
            except ValueError as error:
                blockers.append({'pc': row['pc'], 'sass': row['sass'], 'reason': str(error)})
        if not after_exit:
            blockers.append({'reason': 'no proven complete unconditional EXIT'})
        result[name] = {
            'text_sha256': hashlib.sha256(body).hexdigest(), 'text_bytes': len(body),
            'resources': resources[name], 'instructions': rows[name],
            'flow_opcode_histogram': dict(Counter(live)),
            'static_native_subset': 'blocked' if blockers else 'eligible', 'blockers': blockers,
            'dynamic_status': 'blocked: full real captured masks/operands/addresses absent',
            'hardware_accuracy': 'blocked: physical parameters and observations absent',
        }
    return result
