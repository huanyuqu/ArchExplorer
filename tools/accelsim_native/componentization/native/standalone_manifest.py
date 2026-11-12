"""Publish source-bound SDK/image manifests after explicit native builds.

This command reads artifacts and stages private native Git indexes. It never
compiles, loads, or executes an image. The independent oracle stays unchanged.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from standalone_build import sha,write
from build import BASES,PATCHES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','oracle','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve(strict=True);oracle=args.oracle.resolve(strict=True)
    repo=Path(__file__).resolve().parents[4];sys.path.insert(0,str(repo))
    from arch_explorer.components.accelsim.component_execution import check_ref,file_ref,json_read
    from arch_explorer.components.accelsim.native_sdk import headers,load_sdk,SDK_SCHEMA
    from arch_explorer.components.accelsim.standalone_engine import SCHEMA,load
    base=json_read(root/'BASE.json');extraction=json_read(root/'EXTRACTION-STANDALONE.json')
    if (base.get('new_tree_only') is not True or
        any(base['sources'][name]['commit']!=commit for name,commit in BASES.items())):
        raise ValueError('manifest publication requires a private fixed standalone source tree')
    for ref in extraction['inputs'].values():check_ref(ref)
    links={'gpu':root/'build-gpu/SDK-LINK.json',
           'memory':root/'build-memory/LINK.json','icnt':root/'build-icnt/LINK.json'}
    archives={};dependencies={};receipts={}
    for target,path in links.items():
        document=json_read(path);receipts[target]=document
        if document['status']!='built' or document['target']!=target or document['GPU_host_translation_units']:
            raise ValueError('independent factory archive build incomplete')
        for ref in document['sources']+document['compiler_dependencies']:check_ref(ref)
        check_ref(document['binary']);archives[target]=document['binary'];dependencies[target]=[]
        for arg in document['link_argv']:
            if isinstance(arg,str) and arg.endswith('/libzstd.a'):
                dependencies[target].append(file_ref(str(Path(arg).resolve(strict=True))))
        # The archive ar command has no transitive linker inputs. Its image
        # receipt records the exact static dependency used by the same objects.
        if target=='gpu':
            for arg in json_read(root/'build-gpu/LINK.json')['link_argv']:
                if isinstance(arg,str) and arg.endswith('/libzstd.a'):
                    ref=file_ref(str(Path(arg).resolve(strict=True)))
                    if ref not in dependencies[target]:dependencies[target].append(ref)
    image=json_read(root/'build-gpu/LINK.json')
    check_ref(image['binary']);frozen=json_read(oracle)
    for ref in frozen['libraries']:check_ref(ref)
    inputs={'runtime.hpp':extraction['inputs']['runtime.hpp']}
    from arch_explorer.components._support.accelsim.native_resources import native_files
    for name,file in native_files().items():
        inputs['component_native/'+name]=file_ref(str(file.resolve()))
    for file in sorted(Path(__file__).parent.glob('*.py')):
        inputs['native/'+file.name]=file_ref(str(file.resolve()))
    for name in BASES:
        path=repo/'tools/accelsim_native/patches'/(name+'-sm120-typed-v1.patch')
        if sha(path)!=PATCHES[name]:raise ValueError('fixed typed patch differs')
        inputs[name+'-typed-v1.patch']=file_ref(str(path.resolve()))
    trees={}
    for name in BASES:
        source=root/name
        subprocess.run(['git','add','-A'],cwd=source,check=True)
        trees[name]=subprocess.check_output(['git','write-tree'],cwd=source,text=True).strip()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    sdk={'schema':SDK_SCHEMA,'headers':headers(),'archives':archives,
         'build':{name:file_ref(str(path.resolve())) for name,path in links.items()},'dependencies':dependencies}
    write(output/'sdk.json',sdk);load_sdk(str(output/'sdk.json'))
    engine={'schema':SCHEMA,'sources':BASES,'binary':image['binary'],'libraries':frozen['libraries'],
        'variant':{'id':'standalone-factories-v1','modes':['components','oracle'],
                   'typed_patches':PATCHES,'profile_status':'experimental-unvalidated'},
        'build':{'status':'built','jobs':1,'source_trees':trees,'inputs':inputs,
                 'link':file_ref(str((root/'build-gpu/LINK.json').resolve()))},
        'oracle':file_ref(str(oracle)),'sdk':file_ref(str(output/'sdk.json'))}
    write(output/'engine.json',engine);load(str(output/'engine.json'))
    print(json.dumps({'sdk':file_ref(str(output/'sdk.json')),'engine':file_ref(str(output/'engine.json'))}))


if __name__=='__main__':main()
