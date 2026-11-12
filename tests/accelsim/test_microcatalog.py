"""Registry and actual-file boundary tests; never load an ELF or touch a device."""
from types import SimpleNamespace
import hashlib,struct
import pytest
from arch_explorer.simulator.accelsim.microcatalog import catalog_bytes,embedded_catalog,elf_sections,MAGIC
from arch_explorer.simulator.accelsim.microcontract import REGISTRY_V2,CAPS
from arch_explorer.simulator.accelsim.microexperiment import make_design,ref


def registered():
    return SimpleNamespace(document={'schema':REGISTRY_V2,'work_item':'WP-SIM-909','cubin':{'sha256':'c'*64},
                          'budget':dict(CAPS,cases=2,capture_launches=2,host_processes=3,timing_launches=6)},
        sha256='a'*64,cases={'first':{'id':'first','function':'new_one','partition':'fit'},
                           'second':{'id':'second','function':'new_two','partition':'heldout'}})


def elf(path,blob,*,flags=2,duplicate=False):
    # An explicit synthetic ELF64 section fixture. There is no executable code.
    names=b'\0.shstrtab\0.ae_catalog\0';body=bytearray(64);nameoff=len(body);body+=names;dataoff=len(body);body+=blob
    table=len(body);sections=[(0,0,0,0,0,0,0,0,0,0),(1,3,0,0,nameoff,len(names),0,0,1,0),
                             (11,1,flags,0,dataoff,len(blob),0,0,1,0)]
    if duplicate:sections.append(sections[-1])
    for row in sections:body+=struct.pack('<IIQQQQIIQQ',*row)
    body[:7]=b'\x7fELF\x02\x01\x01';struct.pack_into('<HHI',body,16,3,62,1)
    struct.pack_into('<Q',body,40,table);struct.pack_into('<HHHH',body,58,64,len(sections),1,0)
    path.write_bytes(body);return path


def test_generation_binds_order_and_registry_not_just_names():
    b=registered();raw=catalog_bytes(b)
    assert raw[:16]==MAGIC and struct.unpack_from('<5I',raw,16)==(1,len(raw),2,2,4096)
    assert hashlib.sha256(raw[:-32]).digest()==raw[-32:]
    b.sha256='b'*64;assert catalog_bytes(b)!=raw
    b=registered();b.cases=dict(reversed(list(b.cases.items())));assert catalog_bytes(b)!=raw


def test_exact_allocated_readonly_ELF_section_is_consumed(tmp_path):
    raw=catalog_bytes(registered());path=elf(tmp_path/'synthetic-elf',raw)
    assert embedded_catalog(path,raw)['catalog_sha256']==hashlib.sha256(raw).hexdigest()
    # A valid changed payload and freshly recomputed checksum cannot satisfy the
    # registry's original expected directory, even with a fresh ELF file hash.
    changed=raw[:-32].replace(b'new_one',b'bad_one')
    changed+=hashlib.sha256(changed).digest()
    mutant=elf(tmp_path/'rehashed-elf',changed)
    with pytest.raises(ValueError,match='complete ordered registry'):embedded_catalog(mutant,raw)


@pytest.mark.parametrize('flags,duplicate',[(3,False),(0,False),(2,True)])
def test_ambiguous_or_writable_actual_ELF_catalog_is_refused(tmp_path,flags,duplicate):
    raw=catalog_bytes(registered());path=elf(tmp_path/'synthetic-elf',raw,flags=flags,duplicate=duplicate)
    with pytest.raises(ValueError):embedded_catalog(path,raw)


def test_missing_catalog_cannot_form_new_design(tmp_path):
    path=tmp_path/'opaque-evidence';path.write_text('{}');evidence=ref(path);b=registered();b.path=path
    with pytest.raises(ValueError,match='actual-ELF catalog binding required'):
        make_design(b,evidence,evidence,{'capture':evidence,'timing':evidence},evidence,evidence,evidence)


def test_non_ELF_is_refused_without_loader(tmp_path):
    p=tmp_path/'text';p.write_text('not a dynamic library')
    with pytest.raises(ValueError):elf_sections(p)


@pytest.mark.parametrize('link',[99,2])
def test_dynamic_metadata_requires_a_bounded_string_table(tmp_path,link):
    p=elf(tmp_path/'synthetic-dynamic',bytes(16));raw=bytearray(p.read_bytes())
    table=struct.unpack_from('<Q',raw,40)[0];entry=table+2*64
    struct.pack_into('<I',raw,entry+4,6)  # SHT_DYNAMIC
    struct.pack_into('<I',raw,entry+40,link)
    struct.pack_into('<Q',raw,entry+56,16)
    p.write_bytes(raw)
    with pytest.raises(ValueError,match='dynamic string'):elf_sections(p)
