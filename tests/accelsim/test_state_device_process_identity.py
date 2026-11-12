"""Saved real-host identity is parsed without querying a live process."""
import pytest
from arch_explorer.simulator.accelsim.state_device_reader import _saved_process_identity


def stat(pid=42,parent=7,group=42,session=42,tick=123):
    fields=['R',str(parent),str(group),str(session)]+['0']*15+[str(tick)]
    return str(pid)+' (host with ) spaces) '+' '.join(fields)+'\n'


def test_saved_process_identity(tmp_path):
    path=tmp_path/'process-stat-before.txt';raw={'pid':42,'supervisor_pid':7}
    path.write_text(stat())
    assert _saved_process_identity(tmp_path,raw,{'42':123})==123
    with pytest.raises(ValueError,match='start tick'):
        _saved_process_identity(tmp_path,raw,{'42':124})


@pytest.mark.parametrize('change',[{'pid':43},{'parent':8},{'group':43},{'session':43},{'tick':0}])
def test_saved_process_scope_refused(tmp_path,change):
    (tmp_path/'process-stat-before.txt').write_text(stat(**change))
    with pytest.raises(ValueError,match='saved process identity'):
        _saved_process_identity(tmp_path,{'pid':42,'supervisor_pid':7})


@pytest.mark.parametrize('content',[None,'42 broken','42 (host) R','x'*65537])
def test_incomplete_process_identity_refused(tmp_path,content):
    if content is not None:(tmp_path/'process-stat-before.txt').write_text(content)
    with pytest.raises(ValueError,match='process identity'):
        _saved_process_identity(tmp_path,{'pid':42,'supervisor_pid':7})
