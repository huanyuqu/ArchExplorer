"""Hand-checkable generic fork/merge facts for future public composition tests.

Semantic graph: u=2*x; left=3*u; right=-u; y=left+right. The two u
readers must both be real graph edges. No backend timing code is imported.
"""
from precision_transformer_expectations import store,tensor_bytes


def fork_values(x,*,dtype):
    x=[store(value,dtype) for value in x]
    u=[store(2*value,dtype) for value in x]
    left=[store(3*value,dtype) for value in u]
    right=[store(-value,dtype) for value in u]
    y=[store(a+b,dtype) for a,b in zip(left,right)]
    return dict(x=x,u=u,left=left,right=right,y=y)


def fork_storage(*,dtype):
    """Synthetic isolated schedule: fork readers end at 2 and 4, merge at 5.

    Allocation occurs at each producing task's activation; final y remains
    live until return at 6. At time 4 u is freed before y is allocated.
    Peak is three four-element tensors, not the five declared tensors.
    """
    unit=tensor_bytes((1,4),dtype)
    intervals=[dict(name=name,shape=(1,4),dtype=dtype,allocation=start,
                    readers=readers,returned=name=='y') for name,start,readers in
               [('x',0,['make']),('u',0,['left','right']),('left',1,['merge']),
                ('right',1,['merge']),('y',4,[])]]
    tasks=dict(make=dict(start=0,end=1),left=dict(start=1,end=2),
               right=dict(start=1,end=4),merge=dict(start=4,end=5))
    return dict(tensors=intervals,tasks=tasks,finish=6,
        expected_peak=3*unit,declared_sum=5*unit,
        expected_release=dict(x=1,u=4,left=5,right=5,y=6),
        operator_read_bytes=5*unit,operator_write_bytes=4*unit,
        conversion_instructions=0 if dtype=='fp32' else 36)
