"""Native program mapping discovery, owned by the target rather than events."""
from arch_explorer.adf.registry import MappingTemplate
from arch_explorer.adf.evaluation import MappingSpec

class ProgramMapping(MappingTemplate):
    name='accelsim_program'

    def applicable(self,adf,workload):
        # Request-sized selections are validated by request_space, not inferred
        # before an explicit request can disambiguate implementations.
        try:
            from arch_explorer.components.accelsim.program import _compile
            _compile(adf)
            return True
        except (ValueError,TypeError,KeyError,NotImplementedError):
            return False

    def request_space(self,adf,workload,params=None):
        from arch_explorer.workload.graph import finite_json
        from arch_explorer.workload.reference_api import execute_reference
        from arch_explorer.components.accelsim.open_program import inspect_graph
        from arch_explorer.components.accelsim.program import _compile
        _compile(adf)
        params={} if params is None else finite_json(params)
        if type(params) is not dict or set(params)-{'threads_per_block','implementations'}:
            raise ValueError('accelsim_program accepts threads_per_block and implementations only')
        reference=execute_reference(workload)
        threads=params.get('threads_per_block')
        choices=[threads] if threads is not None else [32,64,128]
        for count in choices:
            assessment=inspect_graph(reference['graph'],MappingSpec(self.name,
                {'threads_per_block':count,**({'implementations':params['implementations']} if 'implementations' in params else {})}))
            if assessment['status']!='supported':
                raise ValueError('native mapping selection: '+str(assessment['gaps']))
        result={'threads_per_block':choices}
        if 'implementations' in params:
            result['implementations']=[params['implementations']]
        return result

    def space(self,adf,workload):
        return self.request_space(adf,workload)

    def apply(self,simulator,workload,params):
        raise NotImplementedError('accelsim_program must execute via a native provider; no events fallback')

