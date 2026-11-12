"""Actual-candidate consumer; independent arithmetic is the acceptance oracle.

Call exercise(compare_hardware) or execute this file in the installed package
environment. Preparation/AST checks are separate from product executions.
"""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
from hardware_comparison import compare,SCHEMA
from test_hardware_comparison import pair


def exercise(compare_hardware,*,observations=None):
    checks={} if observations is None else observations
    samples=[pair('positive',100,110),pair('negative',100,80),pair('zero',0,3),pair('zero-equal',0,0)]
    samples[1]['split']='development'
    samples[0]['predicted'].update(unit='us',value=.11,raw_values=[.11]*3)
    doc=dict(schema=SCHEMA,samples=samples)
    expected=compare(doc);actual=compare_hardware(copy.deepcopy(doc))
    checks['positive']=actual
    assert actual['schema']=='arch-explorer.hardware-comparison.v1'
    assert actual['all_samples_accepted'] and actual['rejected']==[]
    assert actual['fitting_performed'] is False
    indexed={r['sample_id']:r for r in actual['accepted']}
    assert set(indexed)=={r['sample_id'] for r in expected['samples']}
    numeric=['measured_ns','predicted_ns','signed_error_ns','absolute_error_ns','signed_relative_error','absolute_relative_error']
    for row in expected['samples']:
        got=indexed[row['sample_id']]
        assert got['split']==row['split'] and got['zero_reference']==row['zero_reference']
        assert got['evidence_class']=='synthetic-control'
        for field in numeric:
            if row[field] is None:assert got[field] is None
            else:assert math.isclose(got[field],row[field],rel_tol=1e-12,abs_tol=1e-12),(field,got,row)
    groups={r['split']:r for r in actual['groups']}
    assert len(groups)==len(actual['groups'])==len(expected['groups'])
    for row in expected['groups']:
        got=groups[row['split']]
        assert got['count']==row['count']
        assert got['relative_denominator_count']==row['relative_error_count']
        assert got['zero_reference_count']==row['zero_reference_count']
        if row['mape_pct'] is None:assert got['mape'] is None
        else:assert math.isclose(got['mape'],row['mape_pct']/100,rel_tol=1e-12)
        for f in ['worst_absolute_sample_id','worst_relative_sample_id']:assert got[f]==row[f]
    variants=[]
    for name,group,field,value in [('device','device','sku','other'),('timing','timing','end','host-return'),
                                  ('precision','precision','accumulate','fp16'),('mapping','mapping','tile',[1,2,3])]:
        p=pair(name);p['predicted']['conditions'][group][field]=value;variants.append(p)
    missing=pair('missing-unit');del missing['measured']['unit'];variants.append(missing)
    stat=pair('bad-statistic');stat['measured']['value']=1;variants.append(stat)
    nan=pair('nonfinite');nan['measured']['value']=float('nan');variants.append(nan)
    paper=pair('digitized');paper['measured']['source']['evidence_class']='paper-digitization';variants.append(paper)
    duplicate=pair('duplicate');variants.extend([duplicate,copy.deepcopy(duplicate)])
    bad=compare_hardware(dict(schema=SCHEMA,samples=variants))
    checks['negative']=bad
    assert bad['all_samples_accepted'] is False and bad['accepted']==[]
    assert len(bad['rejected'])==len(variants)
    assert all(isinstance(r['reason'],str) and r['reason'] for r in bad['rejected'])
    assert all('index'in r and 'sample_id'in r for r in bad['rejected'])
    return checks


def main():
    from arch_explorer.archs.hardware_comparison import compare_hardware
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
    result=dict(status='running',api={})
    try:
        exercise(compare_hardware,observations=result['api'])
        for label,samples,expected_code in [('positive',[pair()],0),('duplicate',[pair(),pair()],2)]:
            input_path=root/(label+'-INPUT.json');output_path=root/(label+'-OUTPUT.json')
            input_path.write_text(json.dumps(dict(schema=SCHEMA,samples=samples),allow_nan=False)+'\n')
            argv=[str(Path(sys.executable).with_name('archexplorer')),'compare-hardware',str(input_path),'--output',str(output_path)]
            process=subprocess.run(argv,cwd=root,capture_output=True,text=True,timeout=120)
            (root/(label+'-stdout.log')).write_text(process.stdout)
            (root/(label+'-stderr.log')).write_text(process.stderr)
            result[label+'-cli']=dict(argv=argv,returncode=process.returncode)
            assert process.returncode==expected_code,process.stderr
            report=json.loads(output_path.read_text());result[label+'-cli']['report']=report
            assert report['all_samples_accepted']==(expected_code==0)
            if label=='duplicate':assert len(report['rejected'])==2 and report['accepted']==[]
        result['status']='succeeded'
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        (root/'CONSUMER.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
