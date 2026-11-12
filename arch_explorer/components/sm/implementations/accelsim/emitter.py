"""Public scalar SM90 emission context; instructions execute on CPU before publication.

One logical output element maps to one active GPU lane. Tail lanes only run the
bounds prologue and EXIT. Native timing is performed by a separate caller.
"""
from dataclasses import dataclass
import math
import numpy as np
from arch_explorer.components.accelsim.lowering_limits import LIMITS, sha

@dataclass
class Buffer:
    name: str
    base: int
    bits: np.ndarray
    initialized: np.ndarray
    kind: str


class Memory:
    def __init__(self):
        self.next = 0x10000000
        self.buffers = []
        self.addresses = {}
        self.tables = {}
        self.dtypes = {}

    def allocate(self, name, shape, dtype, value=None, kind='activation'):
        if name in self.addresses:
            raise ValueError('duplicate program buffer '+name)
        if not shape or any(type(v) is not int or v < 1 for v in shape):
            raise ValueError('program buffer shape must contain positive integers')
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype('float32'), np.dtype('int32'), np.dtype('uint32')):
            raise ValueError('program memory supports 32-bit scalar storage only')
        n = math.prod(shape)
        if n > LIMITS['tensor_elements']:
            raise ValueError('program buffer element bound exceeded')
        self.dtypes[name] = dtype
        base = (self.next+255)//256*256
        self.next = base+4*n
        if self.next-0x10000000 > LIMITS['global_bytes']:
            raise ValueError('reference program global memory expansion exceeds limit')
        bits = np.zeros(n, dtype=np.uint32) if value is None else np.ascontiguousarray(value, dtype=dtype).reshape(-1).view(np.uint32).copy()
        if bits.size != n:
            raise ValueError('program initializer size mismatch')
        self.buffers.append(Buffer(name, base, bits, np.full(n, value is not None, dtype=bool), kind))
        address = (base+4*np.arange(n, dtype=np.uint32)).reshape(shape)
        self.addresses[name] = address
        return address

    def read(self, addresses):
        addresses = np.asarray(addresses, dtype=np.uint32)
        result = np.zeros(addresses.shape, dtype=np.uint32)
        seen = np.zeros(addresses.shape, dtype=bool)
        for b in self.buffers:
            select = (addresses >= b.base) & (addresses < b.base+4*b.bits.size)
            if np.any(select):
                offset = addresses[select].astype(np.uint64)-b.base
                if np.any(offset % 4): raise ValueError('unaligned reference program address')
                indices = (offset//4).astype(np.int64)
                if not np.all(b.initialized[indices]): raise ValueError('read before producer store: '+b.name)
                result[select] = b.bits[indices]; seen[select] = True
        if not np.all(seen): raise ValueError('reference program address outside allocated buffers')
        return result

    def write(self, name, values):
        b = next(b for b in self.buffers if b.name == name)
        bits = np.ascontiguousarray(values).reshape(-1).view(np.uint32)
        if bits.size != b.bits.size or np.any(b.initialized):
            raise ValueError('program output size or single-producer invariant differs')
        b.bits[:] = bits; b.initialized[:] = True

    def table(self, addresses):
        addresses = np.asarray(addresses, dtype=np.uint32).reshape(-1)
        key = sha(addresses.tobytes())
        if key not in self.tables:
            name = 'gather-'+key
            self.table_address = self.allocate(name, addresses.shape, np.uint32, addresses, 'shape-only-address-table')
            self.tables[key] = int(self.table_address[0])
        return self.tables[key]


class Kernel:
    """Register operations execute before their native trace is rendered."""
    def __init__(self, memory, name, size, threads):
        self.memory, self.name, self.size = memory, name, size
        self.threads = min(threads, max(32, math.ceil(size/32)*32))
        self.reg = {}
        self.code = []
        self.index = np.arange(size, dtype=np.int64)
        self.emit('S2R', 1, [], (self.index % self.threads).astype(np.uint32), special='threadIdx.x')
        self.emit('S2R', 2, [], (self.index // self.threads).astype(np.uint32), special='blockIdx.x')
        self.integer(3, self.threads)
        self.imad(0, 2, 3, 1)
        self.integer(4, 4)
        self.integer(18, size)
        self.emit('ISETP.GE.U32', None, [0, 18], None, predicate='global_index >= output_elements')
        self.prologue = len(self.code)

    def emit(self, op, dst, src, value, **extra):
        allowed = {'S2R','MOV32I','IMAD','LDG.E','STG.E','ISETP.GE.U32',
                   'FADD','FMUL','FFMA','FMNMX','IMNMX.U32','MUFU.EX2','MUFU.RCP',
                   'MUFU.RSQ','I2F.U32','F2I.U32.TRUNC','IADD3'}
        if op not in allowed:
            raise ValueError('unsupported scalar SM90 instruction '+str(op))
        if (dst is not None and (type(dst) is not int or not 0 <= dst < 64)
            or any(type(r) is not int or not 0 <= r < 64 for r in src)):
            raise ValueError('instruction register outside R0..R63')
        if (len(self.code)+2) * math.ceil(self.size/32) >= LIMITS['trace_records']:
            raise ValueError('program instruction expansion bound exceeded')
        if dst is not None:
            value = np.broadcast_to(value, (self.size,)).copy()
            self.reg[dst] = value
        self.code.append({'op': op, 'dst': dst, 'src': src, **extra})

    def integer(self, dst, value):
        value = int(value) & 0xffffffff
        self.emit('MOV32I', dst, [], np.uint32(value), immediate=value)

    def constant(self, dst, value):
        value = np.float32(value)
        self.emit('MOV32I', dst, [], value, immediate=int(value.view(np.uint32)))

    def imad(self, dst, a, b, c):
        value = (self.reg[a].astype(np.uint64)*self.reg[b].astype(np.uint64)+self.reg[c].astype(np.uint64)).astype(np.uint32)
        self.emit('IMAD', dst, [a, b, c], value)

    def pointer(self, dst, addresses):
        base = self.memory.table(addresses)
        self.integer(5, base)
        self.imad(6, 0, 4, 5)
        self.ldg(dst, self.reg[6], np.uint32, [6])

    def ldg(self, dst, addresses, dtype, src):
        bits = self.memory.read(addresses)
        self.emit('LDG.E', dst, src, bits.view(dtype), addresses=np.asarray(addresses, dtype=np.uint32), width=4)

    def load(self, dst, name, indices, dtype=np.float32):
        indices = np.asarray(indices)
        if indices.dtype.kind not in 'iu' or np.dtype(dtype).itemsize != 4:
            raise ValueError('load requires integer element indices and 32-bit storage')
        indices = np.broadcast_to(indices, (self.size,))
        source = self.memory.addresses[name].reshape(-1)
        if np.any(indices < 0) or np.any(indices >= source.size):
            raise ValueError('program load element index outside declared tensor')
        addresses = source[indices.astype(np.int64)]
        self.pointer(7, addresses)
        self.ldg(dst, self.reg[7], dtype, [7])

    def gather(self, dst, addresses, dtype=np.float32):
        self.pointer(7, addresses)
        self.ldg(dst, self.reg[7], dtype, [7])

    def arithmetic(self, op, dst, *src):
        x = [self.reg[r] for r in src]
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            if op == 'FADD': value = np.float32(x[0]+x[1])
            elif op == 'FMUL': value = np.float32(x[0]*x[1])
            elif op == 'FFMA': value = np.float32(x[0].astype(np.float64)*x[1].astype(np.float64)+x[2].astype(np.float64))
            elif op == 'FMNMX': value = np.maximum(x[0], x[1]).astype(np.float32)
            elif op == 'IMNMX.U32': value = np.maximum(x[0], x[1]).astype(np.uint32)
            elif op == 'MUFU.EX2': value = np.float32(np.exp2(x[0]))
            elif op == 'MUFU.RCP': value = np.float32(1/x[0])
            elif op == 'MUFU.RSQ': value = np.float32(1/np.sqrt(x[0]))
            elif op == 'I2F.U32': value = x[0].astype(np.float32)
            elif op == 'F2I.U32.TRUNC': value = np.trunc(x[0]).astype(np.uint32)
            elif op == 'IADD3': value = (x[0].astype(np.uint64)+x[1].astype(np.uint64)+x[2].astype(np.uint64)).astype(np.uint32)
            else: raise ValueError('unimplemented native arithmetic '+op)
        self.emit(op, dst, list(src), value)

    def exp(self, dst, source, temp=30):
        self.constant(temp, math.log2(math.e))
        self.arithmetic('FMUL', dst, source, temp)
        self.arithmetic('MUFU.EX2', dst, dst)

    def finish(self, output, register, dtype=None):
        if dtype is None:
            dtype = self.memory.dtypes[output]
        if np.dtype(dtype) != self.memory.dtypes[output]:
            raise ValueError('kernel store dtype differs from declared output')
        value = self.reg[register].astype(dtype)
        self.integer(5, int(self.memory.addresses[output].reshape(-1)[0]))
        self.imad(6, 0, 4, 5)
        expected = self.memory.addresses[output].reshape(-1)
        if not np.array_equal(self.reg[6], expected): raise ValueError('store address disagrees with output layout')
        self.emit('STG.E', None, [6, register], None, addresses=expected.copy(), width=4)
        self.memory.write(output, value)
        return value

    def render_bytes(self, uid):
        ctas = math.ceil(self.size/self.threads)
        lines = [f'-kernel name = {self.name}', f'-kernel id = {uid}',
            f'-grid dim = ({ctas},1,1)', f'-block dim = ({self.threads},1,1)',
            '-shmem = 0', '-nregs = 64', '-binary version = 90', '-cuda stream id = 0',
            '-enable lineinfo = 0', '-nvbit version = reference-generated-synthetic',
            '-accelsim tracer version = 3', '-shmem base_addr = 0x0', '-local mem base_addr = 0x0',
            '# Complete scalar FP32 reference program; synthetic addresses, no device capture']
        # Reserve a unique predicated EXIT immediately after the prologue.
        def line(pc, mask, ins, indices):
            dst = [] if ins['dst'] is None else ['R'+str(ins['dst'])]
            src = ['R'+str(x) for x in ins['src']]
            width = ins.get('width', 0)
            words = [f'{pc:04x}', f'{mask:08x}', str(len(dst)), *dst, ins['op'], str(len(src)), *src, str(width)]
            if width: words += ['0', *(f'{int(ins["addresses"][i]):x}' for i in indices)]
            words += [str(ins.get('immediate', 0))]
            return ' '.join(words)
        total = 0
        for cta in range(ctas):
            lines += ['#BEGIN_TB', f'thread block = {cta},0,0']
            for warp in range(self.threads//32):
                indices = [cta*self.threads+warp*32+i for i in range(32)]
                active = [i for i in indices if i < self.size]
                mask = (1 << len(active))-1
                records = [line(i*16, 0xffffffff, ins, active) for i, ins in enumerate(self.code[:self.prologue])]
                if mask != 0xffffffff:
                    records.append(f'{self.prologue*16:04x} {0xffffffff ^ mask:08x} 0 EXIT 0 0 0')
                if active:
                    records += [line((i+1)*16, mask, ins, active) for i, ins in enumerate(self.code[self.prologue:], self.prologue)]
                    records.append(f'{(len(self.code)+1)*16:04x} {mask:08x} 0 EXIT 0 0 0')
                lines += [f'warp = {warp}', f'insts = {len(records)}', *records]
                total += len(records)
            lines += ['#END_TB']
        return ('\n'.join(lines)+'\n').encode('ascii'), total

    def render(self, path, uid):
        raw, total = self.render_bytes(uid)
        path.write_bytes(raw)
        return total



@dataclass(frozen=True)
class Emission:
    """One operator's actual instruction kernels, aliases and temporary storage."""
    kernels: tuple
    aliases: tuple
    temporaries: tuple


class ProgramEmitter:
    """Evaluation-local target context exposed as ExecutionContext.options['emitter'].

    The emitter owns allocation/production; lower returns finish_operator().
    Kernel index has shape (logical_elements,), independent of CTA tail lanes.
    """
    def __init__(self, memory, graph, operator, threads=32):
        self.memory, self.graph, self.operator, self.threads = memory, graph, operator, threads
        self.kernels, self.aliases, self.temporaries = [], [], []
        self._allowed = set(operator.inputs) | set(operator.outputs)
        self._finished = False

    def temporary(self, name, shape, dtype='fp32'):
        if (not isinstance(name, str) or not name or any(c not in
                'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name)):
            raise ValueError('temporary needs a nonempty local ASCII identifier')
        key = self.operator.name+'::temporary::'+name
        storage = {'fp32':np.float32,'int32':np.int32}.get(dtype)
        if storage is None:
            raise ValueError('temporary dtype must be fp32 or int32')
        self.memory.allocate(key, tuple(shape), storage)
        self._allowed.add(key)
        self.temporaries.append({'name':key,'shape':list(shape),'dtype':dtype,
                                 'bytes':4*math.prod(shape)})
        return key

    def kernel(self, output):
        if len(self.kernels) >= LIMITS['kernels']:
            raise ValueError('operator emitted kernel count bound exceeded')
        if output not in self._allowed or output in self.operator.inputs:
            raise ValueError('kernel output must be a declared output or owned temporary')
        if output not in self.memory.addresses:
            tensor = self.graph.tensors[output]
            if tensor.alias_of is not None:
                raise ValueError('declared alias requires emitter.alias, not a materialized kernel')
            self.memory.allocate(output, tensor.shape,
                                 np.float32 if tensor.dtype == 'fp32' else np.int32)
        address = self.memory.addresses[output]
        if any(np.any(b.initialized) for b in self.memory.buffers if b.name == output):
            raise ValueError('kernel output already produced')
        name = 'ae_'+str(len(self.kernels)+1)+'_'+self.operator.op
        if any(ord(c) < 32 or ord(c) > 126 for c in name):
            raise ValueError('program operator name must produce an ASCII native kernel label')
        kernel = Kernel(self.memory, name, int(address.size), self.threads)
        kernel.output = output
        self.kernels.append(kernel)
        return kernel

    def literal(self, name, values, dtype='fp32'):
        """Own a materialized constant/table with explicit instruction loads."""
        if (not isinstance(name,str) or not name or any(c not in
                'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name)):
            raise ValueError('literal needs a local ASCII identifier')
        storage = {'fp32':np.float32,'int32':np.int32}.get(dtype)
        if storage is None:
            raise ValueError('literal dtype must be fp32 or int32')
        values = np.asarray(values,dtype=storage)
        key = self.operator.name+'::literal::'+name
        self.memory.allocate(key,values.shape,storage,values,'explicit-literal')
        self._allowed.add(key)
        return key

    def alias(self, output, source, axes=None):
        if output not in self.operator.outputs or source not in self.operator.inputs:
            raise ValueError('alias must connect declared operands/results')
        tensor = self.graph.tensors[output]
        if tensor.alias_of != source or output in self.memory.addresses:
            raise ValueError('alias disagrees with unique graph storage identity')
        address = self.memory.addresses[source]
        address = address.transpose(axes) if axes is not None else address.reshape(tensor.shape)
        if address.shape != tensor.shape:
            raise ValueError('alias shape mismatch')
        self.memory.addresses[output] = address
        self.memory.dtypes[output] = self.memory.dtypes[source]
        self.aliases.append({'output':output,'source':source,'axes':axes})
        return output

    def finish_operator(self):
        if self._finished:
            raise ValueError('operator emission already finished')
        for name in self.operator.outputs:
            if name not in self.memory.addresses:
                raise ValueError('implementation did not produce output '+name)
            self.memory.read(self.memory.addresses[name])
        allowed = [self.memory.addresses[n].reshape(-1) for n in self._allowed if n in self.memory.addresses]
        allowed += [np.arange(b.base,b.base+4*b.bits.size,4,dtype=np.uint32)
                    for b in self.memory.buffers if b.kind=='shape-only-address-table']
        allowed = np.concatenate(allowed) if allowed else np.array([],dtype=np.uint32)
        for kernel in self.kernels:
            if (not kernel.code or kernel.code[-1]['op'] != 'STG.E'
                    or kernel.output not in self._allowed):
                raise ValueError('each instruction kernel must finish its owned output')
            if sum(ins['op']=='STG.E' for ins in kernel.code) != 1:
                raise ValueError('instruction kernels must have one declared output store')
            for instruction in kernel.code:
                if instruction['op']=='LDG.E' and not np.isin(instruction['addresses'],allowed).all():
                    raise ValueError('instruction kernel read outside declared input/temporary/literal dependencies')
            if not np.array_equal(kernel.code[-1]['addresses'],
                                  self.memory.addresses[kernel.output].reshape(-1)):
                raise ValueError('instruction kernel store targets another output')
        self._finished = True
        return Emission(tuple(self.kernels), tuple(self.aliases), tuple(self.temporaries))

