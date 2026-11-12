"""Explicit matrix instruction IR for the optional NeuPIMs executor.

Behavioral sources: f299af3 MatMul::calculate_loops, sram_size_needed,
initialize_instructions; NeuPIMSystolicWS::issue_ex_inst. No reference timings.
Memory regions contain row-major element spans, not aggregate traffic factors.
"""
from arch_explorer.components.instruction_ir import Region, Instruction, Tile
from arch_explorer.components.execution_profiles import NEUPIMS_INSTRUCTIONS
from math import ceil

# f299af3 Common.cc initialize_config assigns BOTH spad_size and
# accum_spad_size from config["sram_size"], ignoring the same-name JSON keys.
# configs/systolic_ws_128x128_dev.json: sram_size=134218 KiB. This is a
# simulator configuration, not the paper's hardware SRAM capacity.
SOURCE_SPAD_BYTES = NEUPIMS_INSTRUCTIONS.spad_bytes


def footprint(shape, width, element_bytes):
    m,k,n = (ceil(d/width)*width for d in shape)
    return (m*k+k*n+m*n)*element_bytes


def outer_shape(m, k, n, width, spad_bytes, element_bytes=2):
    """Upstream halves the largest dimension (first on ties), using half SPAD."""
    dims=[m,k,n]
    if any(type(d) is not int or d <= 0 for d in (*dims,width,spad_bytes,element_bytes)):
        raise ValueError('positive integer dimensions and capacities required')
    if footprint((1,1,1),width,element_bytes)>spad_bytes//2:
        raise ValueError('one aligned instruction tile exceeds half SPAD')
    while footprint(dims,width,element_bytes)>spad_bytes//2:
        i=max(range(3),key=lambda j:dims[j])
        dims[i]=(dims[i]+1)//2
    return tuple(dims)


def matmul_tiles(m,k,n,width,spad_bytes,element_bytes=2,*,transposed=False,policy=NEUPIMS_INSTRUCTIONS):
    """Emit M/N/K outer tiles and N/K/M instruction loops; no padding traffic.

    Addresses are logical tensor coordinates. Physical transaction generation
    deduplicates aligned memory bursts within each MOVIN/MOVOUT instruction.
    Bias is not implicit: callers must supply a separate initialization op.
    """
    if width % policy.size_divisor:raise ValueError('width must be divisible by profile instruction size divisor')
    tm,tk,tn=outer_shape(m,k,n,width,spad_bytes,element_bytes)
    # Weighted MatMul in f299af3 sets _is_transposed in its constructor,
    # AFTER capacity selection reverses M/N, and computes B.T @ A.T.
    # Keep logical output coordinates correct even for rectangular tensors.
    original_n=n
    if transposed:m,n,tm,tn=n,m,tn,tm
    for mo in range(0,m,tm):
        for no in range(0,n,tn):
            for ko in range(0,k,tk):
                mi,ki,ni=min(tm,m-mo),min(tk,k-ko),min(tn,n-no)
                code=[]
                for nc in range(0,ni,width):
                    for kc in range(0,ki,width):
                        for mc in range(0,mi,width):
                            mr,kr,nr=min(width,mi-mc),min(width,ki-kc),min(width,ni-nc)
                            ai=('a',mc,kc);bi=('b',kc,nc);ci=('c',mc,nc)
                            if nc==0:
                                r=Region('a',mo+mc,ko+kc,mr,kr,k,element_bytes)
                                if transposed:r=Region('b',ko+kc,mo+mc,kr,mr,original_n,element_bytes)
                                code.append(Instruction('MOVIN',ai,region=r,size=r.bytes))
                            if mc==0:
                                r=Region('b',ko+kc,no+nc,kr,nr,n,element_bytes)
                                if transposed:r=Region('a',no+nc,ko+kc,nr,kr,k,element_bytes)
                                code.append(Instruction('MOVIN',bi,region=r,size=r.bytes))
                            code.append(Instruction('GEMM_PRELOAD' if mc==0 else 'GEMM',ci,
                                                    (ai,bi),size=width//policy.size_divisor,shape=(mr,kr,nr)))
                            if ko+ki==k and kc+kr==ki:
                                r=Region('c',mo+mc,no+nc,mr,nr,n,element_bytes)
                                if transposed:r=Region('c',no+nc,mo+mc,nr,mr,original_n,element_bytes)
                                code.append(Instruction('MOVOUT',ci,(ci,),r,r.bytes))
                yield Tile((mo,ko,no),ko!=0,footprint((tm,tk,tn),width,element_bytes),tuple(code))


def issue_compute(cycle, instruction, height, width, predecessor=None,policy=NEUPIMS_INSTRUCTIONS):
    """Return modeled start/finish; predecessor is (start, finish, size).

    A predecessor still in the pipeline overlaps PRELOAD. Upstream can assign
    a modeled start before the issue tick; keep that behavior visible rather
    than silently adding a throughput efficiency. Consumers retire no earlier
    than the issue tick in the executor.
    """
    if instruction.opcode not in ('GEMM','GEMM_PRELOAD'):
        raise ValueError('matrix instruction required')
    preload=instruction.opcode=='GEMM_PRELOAD'
    if predecessor is not None:
        start=predecessor[0]+(height if preload else max(predecessor[2],policy.minimum_issue))
    else:
        start=cycle+(2*height-1 if preload else 0)
    return start,start+height+width-2+max(instruction.size,policy.minimum_issue)
