"""Indexed AiM timing provenance, without executing the upstream undefined read.

Source revision 0f28a07b, GDDR6.cpp:182-218,391-425,580-716.
Index names, not preset comments, govern vector assignment. No optional
upstream timing overrides are selected; refresh service is a caller input.
"""
SOURCE = 'https://github.com/arkhadem/aim_simulator/tree/0f28a07bdb83e42b9305ad3d45410ebd3aa2c091'
NAMES = ('rate nBL nCL nRCDRD nRCDRDMAC nRCDEWMUL nRCDRDAF nRCDRDCP '
         'nRCDWR nRCDWRCP nRP nRAS nRC nWR nRTP nCWL nCCDS nCCDL '
         'nRRDS nRRDL nWTRS nWTRL nFAW nRFC nREFI nRREFD nRFCpb '
         'nCLREG nCLGB nCWLREG nCWLGB nWPRE nMODCH tCK_ps').split()
STATIC = (2000,2,50,36,56,25,86,66,28,48,32,54,89,33,12,6,2,2,
          11,11,9,11,42,210,105,21,3333,0,1,1,1,1,32,500)
INITIALIZATION = {
    'tCK_ps': (1000, '1000000 / (rate2000 / 2); GDDR6.cpp:602-603'),
    'nRRDS': (5, 'x16/rate2000 secondary table [1][0]; GDDR6.cpp:630-649'),
    'nRRDL': (6, 'x16/rate2000 secondary table [1][0]; GDDR6.cpp:635-649'),
    'nFAW': (28, 'x16/rate2000 secondary table [1][0]; GDDR6.cpp:640-649'),
    'nRFC': (None, 'density131072Mb -> density_id=-1; tRFC_TABLE[0][-1] is invalid; GDDR6.cpp:663-676'),
    'nREFI': (7800, 'JEDEC_rounding(7800ns,1000ps); GDDR6.cpp:662,677; utils.cpp:37-44'),
    'nCLREG': (0, 'final AiM assignment after optional user overrides; GDDR6.cpp:702'),
    'nCLGB': (1, 'final AiM assignment after optional user overrides; GDDR6.cpp:704'),
    'nCWLREG': (1, 'final AiM assignment after optional user overrides; GDDR6.cpp:708'),
    'nCWLGB': (1, 'final AiM assignment after optional user overrides; GDDR6.cpp:710'),
    'nWPRE': (1, 'final AiM assignment after optional user overrides; GDDR6.cpp:713'),
}
EFFECTIVE = dict(zip(NAMES, STATIC))
EFFECTIVE.update({key: value for key, (value, _) in INITIALIZATION.items()})
EXCLUDED = {
    'nRCDEWMUL':'EWMUL unsupported', 'nRCDRDAF':'RDAF unsupported',
    'nRCDRDCP':'RDCP unsupported', 'nRCDWRCP':'WRCP unsupported',
    'nFAW':'source four-ACT window commented out; not enforced',
    'nRFCpb':'REFpb unsupported; indexed static value3333, comment says nREFI',
    'nRREFD':'REFpb unsupported', 'nCLGB':'no supported global-buffer read',
}


def parameter_sources(profile):
    """Machine-readable static/init/caller decisions in the public result."""
    rows = []
    for index, (name, static) in enumerate(zip(NAMES, STATIC)):
        value = EFFECTIVE[name]
        category = 'defined_effective_initialization' if name in INITIALIZATION else 'static_preset'
        reason = INITIALIZATION.get(name, (value, 'indexed preset; no selected user override'))[1]
        selected = profile.rfc if name == 'nRFC' else value
        row = dict(name=name, index=index, static_literal=static,
                   static_comment_name={24:'nRFCpb',26:'nREFI'}.get(index,name),
                   effective_value=value, defined=name!='nRFC', selected_value=selected,
                   unit='MT/s' if name=='rate' else 'ps' if name=='tCK_ps' else 'cycle',
                   category=category, source=SOURCE, derivation=reason,
                   override_order=['indexed preset','rate/organization initialization',
                                   'optional user n*/t* override (none selected)', 'final AiM assignments'],
                   applicability='single fixed AiM organization, dq16, rate2000',
                   exclusion=EXCLUDED.get(name,''))
        if name == 'nRFC':
            row.update(category='caller_configuration', configured_ns=profile.refresh_service_ns,
                       configured_source=profile.refresh_source,
                       derivation=reason+'; independent ceil(caller ns / controller tick); never read UB')
        rows.append(row)
    return rows
