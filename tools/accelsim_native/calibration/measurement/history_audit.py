# SPDX-License-Identifier: Apache-2.0
"""Read explicit historical artifacts without unpacking archives or executing code."""
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path
from design import WORK_ITEM,write_json
from verify_programs import offline_backend


def audit(bundle,roots,output):
    denied,helper=offline_backend()
    from arch_explorer.components.accelsim.capture_input import elf_texts
    bundle=Path(bundle).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    registry=json.loads((bundle/'REGISTRY.json').read_text());static=json.loads((bundle/'STATIC.json').read_text())
    cache={};facts={};records=[];metadata=[];gaps=[];archives=[];binaries=[];functions=[]
    def read(p):return json.loads(p.read_text())
    def ref(p):
        p=p.resolve();s=p.stat();key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        if key not in cache:cache[key]=hashlib.sha256(p.read_bytes()).hexdigest()
        return {'path':str(p),'bytes':s.st_size,'sha256':cache[key]}
    def resolve(base,item):
        name=item.get('name',item.get('path'))
        if not name:return None
        p=Path(name);choices=[base/p] if not p.is_absolute() else [p,base/p.name,base/'inputs'/p.name]
        return next((p for p in choices if p.is_file()),None)
    def inputs(base,c):
        result=[]
        for role in ('input','auxiliary','expected'):
            item=c.get(role)
            if not isinstance(item,dict):gaps.append({'metadata':str(base),'case':c.get('id'),'role':role,'reason':'no structured input identity'});continue
            p=resolve(base,item)
            if p is None:gaps.append({'metadata':str(base),'case':c.get('id'),'role':role,'reference':item,'reason':'input original unavailable in scoped directory'});continue
            r=ref(p)
            if r['sha256']!=item['sha256']:raise ValueError('historical input identity contradiction: '+str(p))
            result.append(dict(role=role,**r))
        return result
    def prefix(body,instructions):
        if len(instructions)*16!=len(body) or any(r['pc']!=16*i or bytes.fromhex(r['binary_hex'])!=body[16*i:16*i+16] for i,r in enumerate(instructions)):
            raise ValueError('historical disassembly not bound to every text byte')
        for i,r in enumerate(instructions):
            if r['sass']=='EXIT':return hashlib.sha256(body[:16*(i+1)]).hexdigest(),i+1
        raise ValueError('no verified complete unconditional EXIT prefix')
    scopes=[]
    for rawroot in roots:
        root=Path(rawroot).resolve()
        if not root.is_dir():raise ValueError('explicit history root missing: '+str(root))
        scopes.append(str(root))
        # Wrapper registries refer to already enumerated complete program roots;
        # they do not supply an independent cubin or new input identity.
        for file in sorted(root.rglob('REGISTRY.json')):
            d=read(file)
            if 'cubin' not in d or not isinstance(d.get('cases'),list):continue
            body=resolve(file.parent,d['cubin']);f=resolve(file.parent,d.get('static',{'name':'STATIC.json'}))
            if body is None or f is None:gaps.append({'metadata':str(file),'reason':'registry cubin/static original unavailable'});continue
            metadata.extend([ref(file),ref(f)]);h=ref(body)['sha256'];fs=read(f)
            for c in d['cases']:
                if c['function'] in fs:facts[h,c['function']]=fs[c['function']]['instructions']
                else:gaps.append({'metadata':str(file),'function':c['function'],'reason':'no complete static entry'})
                records.append({'case':c['id'],'function':c['function'],'metadata':str(file),'grid':c['grid'],'block':c['block'],'inputs':inputs(file.parent,c)})
        for file in sorted(root.rglob('BUILD.json')):
            d=read(file)
            if not isinstance(d.get('candidates'),list):continue
            metadata.append(ref(file))
            for c in d['candidates']:
                if not isinstance(c,dict) or not isinstance(c.get('cubin'),dict) or 'facts' not in c:continue
                facts[c['cubin']['sha256'],c['function']]=c['facts']['instructions']
                records.append({'case':c['id'],'function':c['function'],'metadata':str(file),'grid':c['grid'],'block':c['block'],'inputs':inputs(file.parent,c)})
        for file in sorted(root.rglob('OFFLINE-CHECK.json')):
            d=read(file);body=file.parent/'kernels.cubin'
            if not isinstance(d.get('functions'),dict) or not body.is_file():continue
            metadata.append(ref(file));h=ref(body)['sha256']
            for name,fact in d['functions'].items():facts[h,name]=fact['instructions']
        for file in sorted(root.glob('captures-v4/*/static.json')):
            d=read(file);metadata.append(ref(file));items=[]
            for p in sorted(file.parent.glob('*.bin')):
                if p.name.endswith(('.input.bin','.expected.bin')):items.append(dict(role=p.name,**ref(p)))
            records.append({'case':file.parent.name,'function':d.get('kernel'),'metadata':str(file),'grid':d.get('grid'),'block':d.get('block'),'inputs':items})
        for file in sorted(root.rglob('*')):
            if not file.is_file():continue
            if file.name.endswith(('.tar','.tar.gz','.tgz','.zip','.tar.bz2','.tar.xz')):
                s=file.stat();archives.append({'path':str(file),'bytes':s.st_size,'reason':'archive not expanded or hashed as historical program contents'})
            if file.suffix!='.cubin':continue
            item=ref(file);binaries.append(item)
            for name,body in elf_texts(file.read_bytes()).items():
                ins=facts.get((item['sha256'],name));ph=None;count=None
                if ins is not None:ph,count=prefix(body,ins)
                else:gaps.append({'cubin':str(file),'function':name,'reason':'no complete static instruction metadata; complete raw text compared'})
                functions.append({'cubin':item,'function':name,'text_sha256':hashlib.sha256(body).hexdigest(),
                                  'text_bytes':len(body),'complete_prefix_sha256':ph,'prefix_instructions':count})
    by_input=defaultdict(list);by_pair=defaultdict(list);by_triplet=defaultdict(list)
    for c in records:
        row={'case':c['case'],'function':c['function'],'metadata':c['metadata']}
        for r in c['inputs']:by_input[r['sha256']].append(dict(role=r['role'],**row))
        values={r['role']:r['sha256'] for r in c['inputs']}
        if 'input' in values and 'auxiliary' in values:by_pair[values['input'],values['auxiliary']].append(row)
        if all(k in values for k in ('input','auxiliary','expected')):by_triplet[tuple(values[k] for k in ('input','auxiliary','expected'))].append(row)
    cubin=ref(bundle/registry['cubin']['name']);new_texts=elf_texts(Path(cubin['path']).read_bytes());rows=[]
    for c in registry['cases']:
        body=new_texts[c['function']];ph,count=prefix(body,static[c['function']]['instructions']);items={}
        for role in ('input','auxiliary','expected'):
            r=ref(bundle/c[role]['name']);assert r['sha256']==c[role]['sha256']
            items[role]={'file':r,'historical_matches':by_input[r['sha256']]}
        values=tuple(items[k]['file']['sha256'] for k in ('input','auxiliary','expected'))
        rows.append({'case':c['id'],'family':c['family'],'function':c['function'],'new_text_sha256':c['text_sha256'],
          'complete_prefix_sha256':ph,'prefix_instructions':count,'inputs':items,
          'cubin_matches':[r for r in binaries if r['sha256']==cubin['sha256']],
          'text_matches':[r for r in functions if r['text_sha256']==c['text_sha256']],
          'prefix_matches':[r for r in functions if r['complete_prefix_sha256']==ph],
          'main_auxiliary_tuple_matches':by_pair[values[:2]],'all_input_oracle_tuple_matches':by_triplet[values],
          'geometry_matches':[{'case':r['case'],'metadata':r['metadata']} for r in records if r['grid']==c['grid'] and r['block']==c['block']]})
    result={'schema':'arch-explorer.measurement-scoped-history.v1','work_item':WORK_ITEM,'registry':ref(bundle/'REGISTRY.json'),
      'scope':scopes,'historical_cubins':binaries,'historical_functions':functions,'historical_cases':records,
      'metadata':metadata,'rows':rows,'gaps':gaps,'unexpanded_archives':archives,
      'limits':'Only listed extant delivery files. Archives and external stale references are not an exhaustive project history.',
      'common_mechanisms':['ordinary IMAD/LDG/STG','dep/chase families shared with older programs','single CTA x32 and fixed toolchain',
                           'dep runtime MUL/ADD equal some older constants; individual auxiliary matches reported'],
      'pure_helper_selection':helper,'forbidden_calls':denied,'hardware_operations':0,'native_builds_runs':0,
      'physical_parameters':None,'hardware_accuracy':'not validated'}
    write_json(out/'HISTORY.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True)
    p.add_argument('--history-root',action='append',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=audit(a.bundle,a.history_root,a.output)
    print(json.dumps({'cases':len(r['rows']),'cubin_files':len(r['historical_cubins']),
      'function_texts':len(r['historical_functions']),'verified_prefixes':sum(x['complete_prefix_sha256'] is not None for x in r['historical_functions']),
      'gaps':len(r['gaps']),'unexpanded_archives':len(r['unexpanded_archives']),
      'overlap_cases':{k:sum(bool(x[k]) for x in r['rows']) for k in ('cubin_matches','text_matches','prefix_matches','main_auxiliary_tuple_matches','all_input_oracle_tuple_matches')}}))
