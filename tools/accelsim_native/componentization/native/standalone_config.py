"""Extract configuration definitions without linking the GPU host TU."""
import hashlib
from pathlib import Path

from extract import block, replace_once


EXPECTED_GPU_CC = 'fd528a420be6bb93199e3c5ecbf48c9f57dc1e5e331c8f017477aee6d2be00e1'
EXPECTED_ISA_CC = 'f28f1d825fd570a03796af8378cde31a170a1979a9b73723e123f0432556d1b2'
FUNCTIONS = (
    'void power_config::reg_options',
    'void memory_config::reg_options',
    'void shader_core_config::reg_options',
    'void gpgpu_sim_config::reg_options',
    'void gpgpu_sim_config::init_clock_domains',
    'void dram_t::dram_log',
    'void increment_x_then_y_then_z',
)


def configuration_source(root: Path, output: Path):
    path = Path(root)/'src/gpgpu-sim/gpu-sim.cc'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_GPU_CC:
        raise ValueError('configuration extraction requires the untouched typed-v1 GPU TU')
    original = raw.decode()
    selected, identities = [], {}
    for marker in FUNCTIONS:
        begin, end, body = block(original,marker)
        signature = original[original.index(marker):begin]
        if marker == 'void gpgpu_sim_config::reg_options':
            body = '\n  if (!gpgpu_ctx || !gpgpu_ctx->stats)\n    throw std::invalid_argument("SM configuration requires explicit instruction statistics");\n'+body
            for old,new in (
                ('gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification','ae_ptx_instruction_classification'),
                ('gpgpu_ctx->func_sim->g_ptx_sim_mode','ae_sim_mode'),
                ('gpgpu_ctx->device_runtime->g_kernel_launch_latency','ae_kernel_launch_latency'),
                ('gpgpu_ctx->device_runtime->g_TB_launch_latency','ae_tb_launch_latency'),
                ('gpgpu_ctx->device_runtime->g_cdp_enabled','ae_cdp_enabled')):
                body = replace_once(body,old,new)
        selected.append(signature+'{'+body+'}\n')
        identities[marker] = hashlib.sha256(body.encode()).hexdigest()
    isa_path = Path(root)/'src/cuda-sim/cuda-sim.cc'
    isa_raw = isa_path.read_bytes()
    if hashlib.sha256(isa_raw).hexdigest() != EXPECTED_ISA_CC:
        raise ValueError('instruction configuration requires the fixed native registration')
    _,_,isa_body = block(isa_raw.decode(),'void cuda_sim::ptx_opcocde_latency_options')
    selected.append('void ae_accelsim::standalone::InstructionConfig::reg_options(option_parser_t opp) {'+isa_body+'}\n')
    identities['cuda_sim::ptx_opcocde_latency_options'] = hashlib.sha256(isa_body.encode()).hexdigest()
    text = original[:original.index('#include')]+'''// Generated configuration methods; no GPU host definition.
#include "gpu-sim.h"
#include "../../../libcuda/gpgpu_context.h"
#include "option_parser.h"
#include "shader.h"
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx-stats.h"
#include "cuda-sim/ptx_ir.h"
#include "trace.h"
#include "statwrapper.h"
#include "debug.h"
#include "dram.h"
#include "ae_component/instruction_config.hpp"
#include <cstdio>
#include <cstring>
#include <sstream>
bool g_interactive_debugger_enabled = false;
'''+'\n'.join(selected)
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(text)
    return {'source':str(path),'source_sha256':EXPECTED_GPU_CC,
            'instruction_options_source':str(isa_path),'instruction_options_source_sha256':EXPECTED_ISA_CC,
            'definitions':identities,'generated':str(output),
            'generated_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'scope':'configuration method bodies only; GPU constructors/cycle not linked'}


def trace_type_source(root: Path, output: Path):
    path=Path(root)/'src/cuda-sim/ptx_ir.cc'
    raw=path.read_bytes()
    expected='054b51aaa1cd24fdb022d368c85f797de04eaba51a35911ea390b40a501fbfa7'
    if hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError('trace function metadata requires the fixed native type implementation')
    text=raw.decode();marker='function_info::function_info(int entry_point, gpgpu_context *ctx)'
    begin,end,body=block(text,marker)
    generated=text[:text.index('#include')]+'''// Native function metadata constructor only; no PTX executor/parser.
#include "cuda-sim/ptx_ir.h"
#include "../../../libcuda/gpgpu_context.h"
'''+marker+' {'+body+'}\n'
    lookup='const ptx_instruction *gpgpu_context::pc_to_instruction(unsigned pc)'
    _,_,lookup_body=block(text,lookup)
    generated+=lookup+' {'+lookup_body+'}\n'
    symbol='const ptx_version &symbol_table::get_ptx_version() const'
    _,_,symbol_body=block(text,symbol)
    generated+=symbol+' {'+symbol_body+'}\n'
    isa=Path(root)/'src/cuda-sim/cuda-sim.cc'
    isa_raw=isa.read_bytes()
    if hashlib.sha256(isa_raw).hexdigest()!=EXPECTED_ISA_CC:
        raise ValueError('kernel resource metadata requires fixed native CUDA type source')
    native='const struct gpgpu_ptx_sim_info *ptx_sim_kernel_info('
    a,b,body_info=block(isa_raw.decode(),native)
    signature=isa_raw.decode()[isa_raw.decode().index(native):a]
    generated+=signature+' {'+body_info+'}\n'
    output.write_text(generated)
    return {'source':str(path),'source_sha256':expected,'definition':marker,
            'body_sha256':hashlib.sha256(body.encode()).hexdigest(),
            'generated':str(output),'generated_sha256':hashlib.sha256(generated.encode()).hexdigest()}
