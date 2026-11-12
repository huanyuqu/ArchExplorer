"""Explicit MAC array plus scalar/SFU service, using existing Python timing.

The repository's ``native_gemm`` is a pure-Python closed form. This model
never invokes SCALE-Sim, a native executable, SDK, compiler or hardware probe.
Array pipeline timing and external semantic memory traffic are separate.
"""
import hashlib
import inspect
import math
from pathlib import Path

from arch_explorer.adf.registry import Conventions, ParamSpec
from arch_explorer.components.base import ComputeModel, Kernel
from arch_explorer.components.implementations import ComponentImplementation
from arch_explorer.components.systolic_array.implementations.timing import SystolicArrayModel
from arch_explorer.components.systolic_array.implementations.systolic_model import native_gemm
from arch_explorer.components.vector_unit.implementations.lanes import _vector_instructions
from .events import PrecisionWork


def _dependencies():
    callbacks=(SystolicArrayModel.kernel_time,native_gemm,_vector_instructions,PrecisionWork.__post_init__)
    return {fn.__module__:hashlib.sha256(Path(inspect.getsourcefile(fn)).read_bytes()).hexdigest() for fn in callbacks}


_FROZEN_DEPENDENCIES=_dependencies()


def validate(point):
    if point.value('precision')!='fp32':
        raise ValueError('sm.mac_events: explicit FP32 arithmetic required')
    if _dependencies()!=_FROZEN_DEPENDENCIES:
        raise ValueError('sm.mac_events: selected Python timing dependency changed')


class MacEventsModel(ComputeModel):
    def __init__(self,component,point,context):
        super().__init__(component,point,context)
        self.rows=self.positive('rows');self.cols=self.positive('cols')
        self.dataflow=self.param('dataflow');self.buffering='single'
        self.simt_lanes=self.positive('simt_lanes');self.sfu_lanes=self.positive('sfu_lanes')

    def gemm(self,m,k,n):
        # One authority: reuse the unchanged repository fold/padding/startup
        # convention, including its reported-cycle-index convention.
        result=native_gemm(self.rows,self.cols,self.dataflow,m,k,n)
        self.provenance.record(self.type_path,'gemm_model','pure-python-systolic-closed-form','1',
            'existing native_gemm Python function; single buffering; no external executable or calibration')
        return result

    def kernel_time(self,work):
        conversions=0
        if isinstance(work,PrecisionWork):
            work.__post_init__();conversions=work.conversion_instructions;work=work.kernel
        if type(work) is not Kernel or work.op not in ('gemm','mvm','vector'):
            raise ValueError('sm.mac_events: expected a GEMM/MVM/vector Kernel or PrecisionWork')
        rank={'gemm':4,'mvm':3,'vector':2}[work.op]
        if (type(work.shape) not in (tuple,list) or len(work.shape)!=rank or any(type(n) is not int or n<1 for n in work.shape)
                or any(type(n) not in (int,float) or not math.isfinite(n) or n<0
                       for n in (work.simt,work.sfu,work.row_simt,work.row_sfu))):
            raise ValueError('sm.mac_events: positive integer shape and finite instruction counts required')
        self.precision(work)
        if work.op in ('gemm','mvm'):
            cycles=SystolicArrayModel.kernel_time(self,work)
        else:
            simt,sfu=_vector_instructions(work)
            cycles=max(simt/self.simt_lanes,sfu/self.sfu_lanes)
        # Conversion occupies the declared scalar service serially. Storage
        # dtype does not silently change the MAC rate or arithmetic precision.
        cycles+=conversions/self.simt_lanes
        if not math.isfinite(cycles) or cycles<0:
            raise ValueError('sm.mac_events: service cycles outside finite domain')
        return cycles


def factory(component,point,context):
    return MacEventsModel(component,point,context)


DEFINITION=ComponentImplementation(component='sm',id='sm.mac_events',version='1',
    source=dict(basis='existing pure-Python systolic fold timing plus explicit scalar/SFU service',
                calibration='none',dependencies=_FROZEN_DEPENDENCIES),
    factory=factory,validate=validate,execution_domain='events',execution_modes=('demand',),adapters=('events',),
    input_contract='events.kernel-or-precision.v1',hardware_parameters=('precision',),
    params=dict(rows=ParamSpec('count','int',domain=[8,16,32,64,128,256]),
                cols=ParamSpec('count','int',domain=[8,16,32,64,128,256]),
                dataflow=ParamSpec('1','str',domain=['os','ws','is']),
                simt_lanes=ParamSpec('count','int',domain=(1,1024)),
                sfu_lanes=ParamSpec('count','int',domain=(1,1024)),clock_ghz=ParamSpec('GHz')),
    conventions=Conventions(time_base='cycle',arbitration='fifo',pipelined=False,includes_startup=True),
    result_scope='Single-buffered MAC pipeline and explicit scalar/SFU demand with external memory streams; uncalibrated',
    capabilities=dict(mechanism='systolic-MAC-plus-SIMT-SFU',memory='external streams',
                      buffering='single; no hidden SRAM or cache',hardware_accuracy=None))


def install():
    from arch_explorer.components.implementations import register_component_implementation
    register_component_implementation(DEFINITION)
