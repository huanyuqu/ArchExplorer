"""Separate returned tensor values from unmaterialized static controls.

No folded value is inferred to have run on CPU. Metadata and source-attribute
identity are explicit, separate checks. This module imports no product code.
"""
import hashlib
import numpy as np


def classify(description,types,oracle,model):
    materialized=[];static=[];values=description['values'];producers={};consumers={}
    described={name:node for node in description['nodes'] for name in node['outputs']}
    for index,node in enumerate(model.graph.node):
        source=dict(id=f'n{index:05d}.{node.op_type}',op_type=node.op_type,inputs=list(node.input),outputs=list(node.output),
            attributes=[dict(name=a.name,type=int(a.type),sha256=hashlib.sha256(a.SerializeToString()).hexdigest()) for a in node.attribute])
        for name in node.output:producers[name]=source
        for name in node.input:consumers.setdefault(name,[]).append(source)
    for name,dtype in types.items():
        if name not in values or name not in oracle:raise ValueError('original value metadata/reference missing')
        meta=values[name];shape=list(np.asarray(oracle[name]).shape)
        if meta['dtype']!=dtype or meta['logical_shape']!=shape:raise ValueError('original value dtype/logical shape differs from frozen reference')
        source=producers[name];node=described.get(name)
        if node is None or node['op_type']!=source['op_type'] or node['inputs']!=source['inputs'] or node['outputs']!=source['outputs']:
            raise ValueError('original node input/output identity differs')
        tensor=meta['tensor']
        if tensor is not None:
            if not isinstance(tensor,str) or not tensor:raise ValueError('invalid materialized tensor identity')
            materialized.append(dict(value=name,tensor=tensor,dtype=dtype,logical_shape=shape))
            continue
        use=[]
        for consumer in consumers.get(name,[]):
            # Check each consuming node's input/output identities against the
            # original ModelProto. The attributes themselves are source facts;
            # a missing producer attribute is not manufactured from this hash.
            actual=next((n for n in description['nodes'] if n['outputs']==consumer['outputs']),None)
            if actual is None or actual['op_type']!=consumer['op_type'] or actual['inputs']!=consumer['inputs']:
                raise ValueError('static control consumer identity differs')
            use.append(dict(**consumer,output_shapes={n:values[n]['logical_shape'] for n in consumer['outputs']},
                source_attributes_bound=True,product_attributes_checked=False))
        static.append(dict(value=name,dtype=dtype,logical_shape=shape,role=meta.get('role'),
            producer=source,declared_constant_folded=node.get('constant_folded'),
            expected_value_sha256=hashlib.sha256(np.asarray(oracle[name]).tobytes()).hexdigest(),
            consumers=use,shape_metadata_checked=True,cpu_value_returned=False,folded_value_numerically_verified=False,
            scope='declared static control; source/shape identity checked, product consumer attribute values need separate inspection'))
    outputs=[]
    for output in description['outputs']:
        name=output['name']
        if name not in values or not isinstance(values[name]['tensor'],str) or not values[name]['tensor']:
            raise ValueError('public output must have a materialized tensor')
        if output['tensor']!=values[name]['tensor']:raise ValueError('public output tensor identity differs')
        outputs.append(name)
    if outputs!=[v.name for v in model.graph.output]:raise ValueError('public output order differs from original ModelProto')
    return dict(materialized=materialized,static=static,public_outputs=outputs,
        all_original_values_cpu_observable=not static,
        static_cpu_values_counted_as_passed=0,
        pending_static_product_attribute_values=bool(static))


def cast_observability(node,values):
    source=values[node['inputs'][0]];target=values[node['outputs'][0]]
    if source['tensor'] is None or target['tensor'] is None:
        return dict(observable=False,source=node['inputs'][0],target=node['outputs'][0],
            reason='static input or output has no returned tensor',cpu_cast_passed=False,
            declared_constant_folded=node.get('constant_folded'))
    return dict(observable=True,source=node['inputs'][0],target=node['outputs'][0])


def inspect_consumer_shapes(coverage,description,graph):
    """Bind consuming source nodes to ordinary Graph output shape attributes.

    This is deliberately a shape check. It does not label uninspected offsets,
    integer constants or folded numeric values as verified product attributes.
    """
    tensors={t['name']:t for t in graph['tensors']};rows=[]
    for control in coverage['static']:
        for consumer in control['consumers']:
            selected=[op for op in graph['operators'] if op['name'].startswith(consumer['id'])]
            targets=[]
            for name in consumer['outputs']:
                meta=description['values'][name]
                if meta['tensor'] is None:continue
                physical=meta['logical_shape'] or [1];tensor=meta['tensor']
                if tensor not in tensors or list(tensors[tensor]['shape'])!=physical:
                    raise ValueError('static consumer Graph output shape differs from original reference')
                operators=[op for op in selected if tensor in op['outputs']]
                checked=[]
                for op in operators:
                    if 'shape' in op['attrs']:
                        if list(op['attrs']['shape'])!=physical:raise ValueError('static consumer shape attribute differs')
                        checked.append(dict(operator=op['name'],kind=op['op'],field='shape',value=physical))
                targets.append(dict(value=name,tensor=tensor,physical_shape=physical,shape_attributes_checked=checked))
            rows.append(dict(static_value=control['value'],consumer=consumer['id'],targets=targets,
                source_integer_or_shape_value_returned=False,
                uninspected_attribute_values=True,
                scope='Graph tensor shape and available shape attributes; no CPU folded value or all-attribute claim'))
    return rows
