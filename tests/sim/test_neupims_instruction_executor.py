from arch_explorer.components.neupims_instructions import matmul_tiles
from arch_explorer.simulator.events.neupims_core import InstructionCore


class Memory:
    def __init__(self,latency,capacity=2):
        self.latency=latency;self.capacity=capacity;self.queue=[];self.bytes=[0,0];self.rejected=0
    def submit(self,region,write,now,callback):
        if len(self.queue)==self.capacity:self.rejected+=1;return False
        self.bytes[write]+=region.bytes;self.queue.append((now+self.latency,callback));return True
    def step(self,now):
        for end,callback in list(self.queue):
            if end<=now:self.queue.remove((end,callback));callback(now)


def run(latency):
    core=InstructionCore(matmul_tiles(2,65,67,64,1024*1024),64,64)
    memory=Memory(latency)
    for now in range(10000):
        memory.step(now);core.step(now,memory)
        if not core.active:return core,memory,now
    assert False,'deadlock'


def test_instruction_dependencies_and_backpressure():
    fast,mem,finish=run(2)
    slow,smem,sfinish=run(200)
    assert mem.bytes==smem.bytes==[(2*65+65*67)*2,2*67*2]
    assert sfinish>finish and smem.rejected>0
    assert len([r for r in fast.trace if r['event']=='GEMM_PRELOAD'])==4
    assert len(fast.finished)==1
    for trace in (fast.trace,slow.trace):
        loaded={}
        for row in trace:
            if row['event']=='load-response':loaded[row['dest']]=row['time']
            if row['event']=='GEMM_PRELOAD':
                mc,nc=row['dest'][1:];kc=0 if row['shape'][1]==64 else 64
                assert loaded[('a',mc,kc)]<=row['time']
                assert loaded[('b',kc,nc)]<=row['time']
