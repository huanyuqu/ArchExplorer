# SPDX-License-Identifier: Apache-2.0
"""Compare explicit historical delivery bytes; never inspect a device.

This is a scoped overlap audit, not an unseen-family certificate. Raw ELF text
is compared without its symbol name. A complete prefix requires independent
instruction bytes through unconditional EXIT, retaining predicated EXITs.
Missing metadata and archives that were not expanded are reported as gaps.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from design import WORK_ITEM, sha, write_json
from arch_explorer.components.accelsim.capture_input import elf_texts


def read(path):
    return json.loads(Path(path).read_text())


def reference(path):
    p=Path(path).resolve()
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}


def prefix_hash(body, instructions):
    """Every original text byte must be described, including dead padding."""
    if len(instructions)*16!=len(body) or any(r['pc']!=i*16 or bytes.fromhex(r['binary_hex'])!=body[16*i:16*i+16]
                                              for i,r in enumerate(instructions)):
        raise ValueError('historical disassembly is not bound to complete ELF text')
    for i,row in enumerate(instructions):
        if row['sass']=='EXIT':
            return hashlib.sha256(body[:16*(i+1)]).hexdigest(),i+1
    raise ValueError('historical function has no verified unconditional EXIT prefix')


def audit(bundle, roots, output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    bundle=Path(bundle).resolve();registry=read(bundle/'REGISTRY.json');static=read(bundle/'STATIC.json')
    cubin=bundle/registry['cubin']['name'];new_cubin=reference(cubin)
    new_text=elf_texts(cubin.read_bytes())
    roots=[Path(p).resolve() for p in roots]
    artifacts=[];functions=[];records=[];metadata=[];gaps=[];fact_index={}
    for root in roots:
        if not root.is_dir():raise ValueError('explicit history root missing: '+str(root))
        # Record the immutable metadata used to associate each exact ELF with
        # case inputs. Failed/early builds are retained rather than discarded.
        for file in sorted(root.rglob('REGISTRY.json')):
            data=read(file)
            if 'cubin' not in data or not isinstance(data.get('cases'),list):continue
            body=file.parent/data['cubin'].get('name',data['cubin'].get('path',''))
            facts_file=file.parent/data.get('static',{}).get('name','STATIC.json')
            if not body.is_file() or not facts_file.is_file():
                gaps.append({'metadata':str(file),'reason':'registry cubin/static absent'});continue
            metadata.extend([reference(file),reference(facts_file)])
            chash=sha(body);facts=read(facts_file)
            for c in data['cases']:
                fact_index[(chash,c['function'])]=facts[c['function']]['instructions']
                inputs=[]
                for role in ('input','auxiliary','expected'):
                    item=c[role];p=file.parent/item.get('name',item.get('path',''))
                    if not p.is_file() or sha(p)!=item['sha256']:raise ValueError('historical input missing/changed: '+str(p))
                    inputs.append(dict(role=role,**reference(p)))
                records.append({'case':c['id'],'function':c['function'],'metadata':str(file),
                                'grid':c['grid'],'block':c['block'],'inputs':inputs})
        for file in sorted(root.rglob('BUILD.json')):
            data=read(file);candidates=data.get('candidates')
            if not isinstance(candidates,list):continue
            metadata.append(reference(file))
            for c in candidates:
                if not isinstance(c,dict) or 'cubin' not in c or 'facts' not in c:continue
                chash=c['cubin']['sha256'];fact_index[(chash,c['function'])]=c['facts']['instructions']
                inputs=[]
                for role in ('input','auxiliary','expected'):
                    item=c[role];p=file.parent/item.get('name',item.get('path',''))
                    if not p.is_file() or sha(p)!=item['sha256']:raise ValueError('historical build input missing/changed: '+str(p))
                    inputs.append(dict(role=role,**reference(p)))
                records.append({'case':c['id'],'function':c['function'],'metadata':str(file),
                                'grid':c['grid'],'block':c['block'],'inputs':inputs})
        # SIM28's complete one-cubin static oracle predates the registry format.
        for file in sorted(root.rglob('OFFLINE-CHECK.json')):
            data=read(file);facts=data.get('functions');body=file.parent/'kernels.cubin'
            if not isinstance(facts,dict) or not body.is_file():continue
            metadata.append(reference(file));chash=sha(body)
            for name,fact in facts.items():fact_index[(chash,name)]=fact['instructions']
        for file in sorted(root.glob('captures-v4/*/static.json')):
            data=read(file);inputs=[]
            for p in sorted(file.parent.glob('*.bin')):
                if p.name.endswith(('.input.bin','.expected.bin')):inputs.append(dict(role=p.name,**reference(p)))
            metadata.append(reference(file))
            records.append({'case':file.parent.name,'function':data.get('kernel'),
                            'metadata':str(file),'grid':data.get('grid'),'block':data.get('block'),
                            'inputs':inputs})
        for file in sorted(root.rglob('*.cubin')):
            item=reference(file);artifacts.append(item)
            texts=elf_texts(file.read_bytes())
            for name,body in texts.items():
                instructions=fact_index.get((item['sha256'],name));flow=None;count=None
                if instructions is not None:flow,count=prefix_hash(body,instructions)
                else:gaps.append({'cubin':str(file),'function':name,'reason':'no complete instruction metadata in scoped delivery; full ELF text still compared'})
                functions.append({'cubin':item,'function':name,'text_sha256':hashlib.sha256(body).hexdigest(),
                                  'text_bytes':len(body),'complete_prefix_sha256':flow,'prefix_instructions':count})
    by_input=defaultdict(list)
    for c in records:
        for item in c['inputs']:by_input[item['sha256']].append({'case':c['case'],'metadata':c['metadata'],'role':item['role']})
    rows=[]
    for c in registry['cases']:
        body=new_text[c['function']];flow,count=prefix_hash(body,static[c['function']]['instructions'])
        inputs={}
        for role in ('input','auxiliary','expected'):
            p=bundle/c[role]['name']
            if sha(p)!=c[role]['sha256']:raise ValueError('new frozen input differs')
            inputs[role]={'file':reference(p),'historical_matches':by_input.get(c[role]['sha256'],[])}
        rows.append({'case':c['id'],'family':c['family'],'function':c['function'],
                     'new_text_sha256':hashlib.sha256(body).hexdigest(),'new_text_bytes':len(body),
                     'new_complete_prefix_sha256':flow,'prefix_instructions':count,
                     'cubin_matches':[x for x in artifacts if x['sha256']==new_cubin['sha256']],
                     'text_matches':[x for x in functions if x['text_sha256']==c['text_sha256']],
                     'complete_prefix_matches':[x for x in functions if x['complete_prefix_sha256']==flow],
                     'inputs':inputs,'geometry_matches':[{'case':x['case'],'metadata':x['metadata']}
                        for x in records if x['grid']==c['grid'] and x['block']==c['block']]})
    result={'schema':'arch-explorer.measurement-history.v2','work_item':WORK_ITEM,
            'scope':[str(r) for r in roots],'registry':reference(bundle/'REGISTRY.json'),
            'cubin':new_cubin,'historical_cubins':artifacts,'historical_functions':functions,
            'historical_cases':records,'metadata':metadata,'rows':rows,'gaps':gaps,
            'unexpanded_archives':'archive members and external historical paths were not enumerated; not an all-project unseen certificate',
            'common_mechanisms':['dep and chase families overlap historical mechanisms; quad not selected this round',
                                 'ordinary integer/load/store primitives',
                                 'same grid1/block32 as SIM29/SIM30/SIM31','same NVBit transport and fixed toolchain'],
            'no_hardware_execution':True,'physical_parameters':None,'hardware_accuracy':'not validated'}
    write_json(output/'HISTORY.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--history-root',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.bundle,a.history_root,a.output)
    print(json.dumps({'cases':len(r['rows']),'historical_cubins':len(r['historical_cubins']),
                      'historical_functions':len(r['historical_functions']),'gaps':len(r['gaps']),
                      'cubin_matches':sum(bool(x['cubin_matches']) for x in r['rows']),
                      'text_matches':sum(bool(x['text_matches']) for x in r['rows']),
                      'prefix_matches':sum(bool(x['complete_prefix_matches']) for x in r['rows'])}))
