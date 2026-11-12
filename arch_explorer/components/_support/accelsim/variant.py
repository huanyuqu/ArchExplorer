# Concrete Accel-Sim implementation; owned by the component provider.
"""Approved, byte-identified native mechanism builds; never physical accuracy."""
ID='sm120-typed-v1'
SCHEMA='arch-explorer.accelsim-engine.v2'
CAPABILITIES=['sm120_ordinary','typed_registers_v1','traceg_v7','four_synthetic_bank_spaces','memory_width_v1','explicit_predicate_control_v1']
PROFILE='SM120_MECHANISM_SYNTHETIC'
# Populated only from a built and independently exercised exact patch/toolchain.
# Rebuilding with other bytes needs an explicit new reviewed entry.
APPROVED={'490b1eaadf7373343be0a0c5b3299cf28fbb183ea504bb98baa1f297fe59fb98': {'patches': {'accelsim': 'f90e0be472a1e657e02b3a3de5c2f7217d8be8dcb169bf2eb46e41eefa37b477', 'gpgpusim': 'b250da0dfd4d69154f7280e4311cf4641ba14e969738568d839f64038b389891'}, 'libraries': ['74aa8c5f3377db9fbbab7dad51c090ccdb07021429b6efc7153a702b91012bd1'], 'source_trees': {'accelsim': 'c10b3aa74131028945f3ad6934051471d219b5ed', 'gpgpusim': 'f0e51673d728894f20b0a0db7929caa3a39df800'}}}

def validate(manifest):
    from arch_explorer.components.accelsim.trace import exact
    exact(manifest['variant'], {'id','patches','capabilities','profile_status'}, 'engine variant')
    variant=manifest['variant']
    if (variant['id']!=ID or variant['capabilities']!=CAPABILITIES
            or variant['profile_status']!='experimental-unvalidated'):
        raise ValueError('unsupported or falsely physical native variant capability')
    approved=APPROVED.get(manifest['binary']['sha256'])
    if approved is None or approved['patches']!=variant['patches']:
        raise ValueError('native variant binary/patches are not an approved exact build')
    if manifest['build'].get('source_trees')!=approved['source_trees']:
        raise ValueError('native variant source trees differ from approved exact build')
    if sorted(row['sha256'] for row in manifest['libraries'])!=approved['libraries']:
        raise ValueError('native variant library bytes differ from approved exact build')
    return variant

def require_trace_engine(trace_manifest, engine_manifest):
    typed=trace_manifest['schema']=='arch-explorer.accelsim-trace.v3'
    variant=engine_manifest.get('schema')==SCHEMA
    if typed!=variant:
        raise ValueError('typed SM120 trace requires its actual approved native variant; legacy SM90 requires its unchanged engine')
    if typed:
        validate(engine_manifest)
        if trace_manifest['engine']['variant']!=ID:
            raise ValueError('typed trace/native variant identity differs')

