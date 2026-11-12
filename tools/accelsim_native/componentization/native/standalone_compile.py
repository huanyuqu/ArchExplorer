"""Single-job finite native source builds; no implicit whole-GPU linkage."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex

from standalone_build import sha, write


def compiler_dependencies(objects):
    dependencies=set()
    for obj in objects:
        text=obj.with_suffix('.d').read_text().replace('\\\n','')
        rule=text.splitlines()[0].split(':',1)[1]
        for name in shlex.split(rule):
            path=Path(name).resolve(strict=True)
            if not path.is_file():
                raise ValueError('compiler dependency is not a regular source input')
            dependencies.add(path)
    return dependencies


def compile_components(args, commands, repo):
    root = args.root
    extraction = root/'EXTRACTION-STANDALONE.json'
    if not extraction.is_file():
        raise ValueError('extract the component-only source before compilation')
    gpu = root/'gpgpusim'
    source = gpu/'src'
    includes = [source,source/'gpgpu-sim',source/'intersim2',source/'gpgpu-sim/ae_component',source/'accelwattch',gpu/'libcuda']
    book = source/'intersim2/ae_standalone'
    icnt = json.loads((book/'SOURCES.json').read_text())
    # Native generated subdirectories use short includes. Resolve their
    # isolated headers before similarly named legacy BookSim declarations.
    includes = [book/p for p in icnt['include_dirs']]+includes
    if args.target == 'icnt':
        sources = [book/p for p in icnt['sources']]+[source/'gpgpu-sim/ae_component/icnt.cc']
        libraries = []
        archive = True
    elif args.target in {'memory','gpu'}:
        native = source/'gpgpu-sim'
        sources = [native/p for p in (
            'dram.cc','dram_sched.cc','l2cache.cc','gpu-cache.cc','mem_fetch.cc',
            'mem_latency_stat.cc','addrdec.cc','hashing.cc','gpu-misc.cc',
            'stat-tool.cc','histogram.cc')]
        sources += [source/p for p in ('abstract_hardware_model.cc','option_parser.cc',
                    'statwrapper.cc','cuda-sim/ptx-stats.cc',
                    'intersim2/stats.cpp','intersim2/module.cpp')]
        sources += [native/'ae_component'/p for p in
                    ('configuration.cc','l2.cc','dram.cc','memory_statistics.cc','transactions.cc','memory_links.cc','memory_system.cc')]
        libraries = []
        archive = True
        if args.target == 'gpu':
            accel=root/'accelsim/gpu-simulator'
            includes += [accel,accel/'trace-driven',accel/'trace-parser',
                         root/'accelsim/util/tracer_nvbit/tracer_tool',args.tools/'usr/include']
            sources += [native/p for p in ('shader.cc','scoreboard.cc','stack.cc','traffic_breakdown.cc')]
            sources += [source/'trace.cc',source/'cuda-sim/memory.cc',
                        accel/'trace-driven/trace_driven.cc',accel/'trace-parser/trace_parser.cc']
            sources += [native/'ae_component'/p for p in
                        ('trace_types.cc','shader.cc','execution.cc','icnt.cc',
                         'gpu_transport.cc','gpu_assembly.cc','gpu_entry.cc')]
            sources += [book/p for p in icnt['sources']]
            libraries=[str(args.tools/'usr/lib/x86_64-linux-gnu/libzstd.a'),'-lz','-lm','-pthread'];archive=False
        write(root/('COMPONENT-SOURCES-'+args.target+'.json'),{
            'schema':'arch-explorer.standalone-sources.v1','target':args.target,
            'sources':[str(p.relative_to(root)) for p in sources],
            'include_dirs':[str(p) for p in includes]})
    else:
        raise ValueError('unknown finite component source target')
    forbidden = {'gpu-sim.cc','gpgpusim_entrypoint.cc','accel-sim.cc','main.cc','cuda_runtime_api.cc'}
    if any(p.name in forbidden for p in sources):
        raise ValueError('GPU host TU is forbidden in a standalone link')
    if not sources or len(set(sources)) != len(sources) or any(not p.is_file() for p in sources):
        raise ValueError('a unique finite source closure is required')
    output = root/('build-'+args.target)
    output.mkdir(exist_ok=True)
    objects = [output/(f'{i:03d}-'+p.name.replace('.','_')+'.o') for i,p in enumerate(sources)]
    binary = output/('libae_'+args.target+'.a' if archive else 'ae-'+args.target)
    flags = ['-std=c++17','-O1','-g','-ffunction-sections','-fdata-sections','-MMD','-MP']
    flags += ['-I'+str(p) for p in dict.fromkeys(includes)]
    if args.target != 'icnt':
        header = (args.cuda/'include/cuda_runtime_api.h').read_text()
        version = re.search(r'^#\s*define\s+CUDART_VERSION\s+(\d+)',header,re.M)
        if not version:
            raise ValueError('compiler header must declare its CUDART_VERSION')
        flags += ['-I'+str(args.cuda/'include'),'-DCUDART_VERSION='+version.group(1)]
    quote = shlex.quote
    make = ['.PHONY: all','all: '+str(binary),'']
    for src,obj in zip(sources,objects):
        make += [str(obj)+': '+str(src)+' '+str(output/'Makefile'),
                 '\t'+' '.join(map(quote,['g++',*flags,'-c',str(src),'-o',str(obj)])),'']
    make += [str(binary)+': '+' '.join(map(str,objects))]
    command = (['ar','rcs',str(binary),*map(str,objects)] if archive else
               ['g++','-Wl,--gc-sections',*map(str,objects),*libraries,'-o',str(binary)])
    make += ['\t'+' '.join(map(quote,command)),'',
             '-include '+' '.join(str(p.with_suffix('.d')) for p in objects)]
    makefile = output/'Makefile'
    rendered = '\n'.join(make)+'\n'
    if not makefile.exists() or makefile.read_text() != rendered:
        makefile.write_text(rendered)
    environment = dict(os.environ,CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none')
    commands.run(['make','-k','-j1','-f',makefile],root,env=environment,timeout=3600)
    # Every successful image/archive keeps an immutable identity object before
    # another development build may replace the working output path.
    digest = sha(binary)
    objects_root = root/'objects'/digest
    objects_root.mkdir(parents=True,exist_ok=True)
    frozen = objects_root/binary.name
    if not frozen.exists():
        commands.run(['rsync','-aH','--info=progress2','--',binary,frozen],root)
    identities={p.resolve():sha(p) for p in sources}
    dependencies=compiler_dependencies(objects)
    for path in sorted(dependencies):
        if path not in identities:identities[path]=sha(path)
    record = {'schema':'arch-explorer.standalone-link.v1','target':args.target,
        'source_extraction':{'path':str(extraction),'sha256':sha(extraction)},
        'compiler_flags':flags,'link_argv':command,'jobs':1,
        'sources':[{'path':str(p),'sha256':identities[p.resolve()]} for p in sources],
        'compiler_dependencies':[{'path':str(p),'sha256':identities[p]} for p in sorted(dependencies)],
        'binary':{'path':str(frozen),'sha256':digest},'commands':commands.entries,
        'GPU_host_translation_units':[],'physical_GPU_calls':0,'status':'built'}
    write(root/f'LINK-{args.target}-{commands.stamp}.json',record)
    write(output/'LINK.json',record)
    if args.target == 'gpu':
        # The external caller links the same factory/mechanism object files
        # as the actual GPU image, excluding only this image's CLI main.
        selected=[(p,o) for p,o in zip(sources,objects) if p.name!='gpu_entry.cc']
        sdk_archive=output/'libae_gpu.a'
        ar=['ar','rcs',sdk_archive,*[o for _,o in selected]]
        commands.run(ar,root,timeout=600)
        archive_sha=sha(sdk_archive);folder=root/'objects'/archive_sha;folder.mkdir(parents=True,exist_ok=True)
        frozen_archive=folder/sdk_archive.name
        if not frozen_archive.exists():
            commands.run(['rsync','-aH','--info=progress2','--',sdk_archive,frozen_archive],root)
        archive_record=dict(record,binary={'path':str(frozen_archive),'sha256':archive_sha},
                            sources=[{'path':str(p),'sha256':identities[p.resolve()]} for p,_ in selected],
                            link_argv=list(map(str,ar)),commands=commands.entries)
        write(root/f'LINK-gpu-archive-{commands.stamp}.json',archive_record)
        write(output/'SDK-LINK.json',archive_record)
