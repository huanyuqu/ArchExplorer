import json
import fcntl
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest import mock
import stage_capture
from stage_capture import run, stage
from runner_descendant_control import exercise


class StageTests(unittest.TestCase):
    def test_parent_exit_failure_and_timeout_clean_same_group_descendant(self):
        with tempfile.TemporaryDirectory() as d:
            for mode in ('exit','failure','timeout'):
                with self.subTest(mode=mode):
                    result=exercise(stage_capture,'qa',Path(d)/mode,mode=mode)
                    self.assertTrue(result['passed'],result)
                    self.assertFalse(result['lock_free_with_running_members'])
                    record=result['runner_result']
                    self.assertTrue(record['parent_reaped'] and record['process_group_closed'])
                    self.assertEqual(record['process_group_cleanup']['status'],'confirmed-no-running-members')
                    self.assertTrue(record['scale_lock_released'])

    def test_unconfirmed_cleanup_does_not_claim_closed_or_release_lock(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);lock=root/'temporary.lock'
            with mock.patch.object(stage_capture,'close_group',return_value=dict(status='unconfirmed',confirmed=False)):
                with self.assertRaisesRegex(RuntimeError,'process group cleanup unconfirmed'):
                    run([sys.executable,'-c','pass'],cwd=root,dest=root/'stage',scale=True,scale_lock=lock)
            record=json.loads((root/'stage/PROCESS.json').read_text())
            try:
                self.assertEqual(record['returncode'],0)
                self.assertTrue(record['parent_reaped'])
                self.assertFalse(record['process_group_closed'] or record['scale_lock_released'])
                self.assertEqual(record['status'],'cleanup-unconfirmed')
                with lock.open('a') as contender:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
            finally:
                retained=stage_capture._unconfirmed_groups.pop(record['pid'])
                retained['lock'].close()

    def test_missing_scale_lock_never_starts_child(self):
        with tempfile.TemporaryDirectory() as d:
            for index,selected in enumerate((None,'','relative.lock',b'/bytes.lock')):
                with self.subTest(scale_lock=selected):
                    dest=Path(d)/str(index)
                    with mock.patch.object(stage_capture.subprocess,'Popen') as spawn:
                        with self.assertRaisesRegex(ValueError,'^scale=True requires an explicit absolute scale_lock path$'):
                            run([sys.executable,'-c','raise AssertionError("must not start")'],
                                cwd=d,dest=dest,scale=True,scale_lock=selected)
                    spawn.assert_not_called()
                    saved=json.loads((dest/'PROCESS.json').read_text())
                    self.assertEqual(saved['status'],'rejected')
                    self.assertEqual(saved['reason'],'missing-scale-lock' if selected is None else
                                     'scale-lock-must-be-explicit-absolute-path')
                    self.assertEqual(saved['error_type'],'ValueError')
                    self.assertIsNone(saved['returncode'])
                    self.assertFalse(saved['scale_lock_acquired'])

    def test_cli_requires_and_forwards_explicit_scale_lock(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);marker=root/'spawned';lock=root/'private-control.lock'
            code='from pathlib import Path;Path('+repr(str(marker))+').write_text("ran")'
            common=[sys.executable,str(Path(stage_capture.__file__).resolve()),'--cwd',d,'--scale']
            missing=subprocess.run(common+['--dest',str(root/'missing'),'--',sys.executable,'-c',code],capture_output=True,text=True,timeout=10)
            self.assertEqual(missing.returncode,2)
            self.assertIn('scale=True requires an explicit absolute scale_lock path',missing.stderr)
            self.assertFalse(marker.exists())
            self.assertEqual(json.loads((root/'missing/PROCESS.json').read_text())['reason'],'missing-scale-lock')
            positive=subprocess.run(common+['--scale-lock',str(lock),'--dest',str(root/'explicit'),'--',sys.executable,'-c',code],capture_output=True,text=True,timeout=10)
            self.assertEqual(positive.returncode,0,positive.stderr)
            saved=json.loads((root/'explicit/PROCESS.json').read_text())
            self.assertEqual(saved['scale_lock'],str(lock))
            self.assertTrue(saved['scale_lock_acquired'] and saved['scale_lock_released'])
            self.assertEqual(marker.read_text(),'ran')

    def test_busy_scale_lock_never_starts_child(self):
        with tempfile.TemporaryDirectory() as d:
            lock_path=Path(d)/'scale.lock'
            with lock_path.open('a') as owner:
                fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with mock.patch.object(stage_capture.subprocess,'Popen') as spawn:
                    r=run([sys.executable,'-c','raise AssertionError("must not start")'],
                          cwd=d,dest=Path(d)/'busy',scale=True,scale_lock=lock_path)
                spawn.assert_not_called()
            saved=json.loads((Path(d)/'busy/PROCESS.json').read_text())
            self.assertEqual(r['status'],saved['status'],'busy')
            self.assertEqual(saved['status'],'busy')
            self.assertEqual(saved['reason'],'scale-lock-held')
            self.assertIsNone(saved['returncode'])
            self.assertFalse(saved['scale_lock_acquired'])
            self.assertFalse(saved['process_group_closed'])
            self.assertFalse((Path(d)/'busy/stdout.log').exists())

    def test_scale_lock_is_held_during_child_and_released_after_exit(self):
        with tempfile.TemporaryDirectory() as d:
            lock_path=Path(d)/'scale.lock'
            code='''import fcntl,sys,time
with open(sys.argv[1],'a') as contender:
    for i in range(3):
        try: fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: pass
        else: raise AssertionError('runner released shared lock before child exit')
        print('held',flush=True)
        time.sleep(.01)
'''
            r=run([sys.executable,'-c',code,str(lock_path)],cwd=d,dest=Path(d)/'held',scale=True,scale_lock=lock_path)
            self.assertEqual(r['status'],'succeeded')
            self.assertTrue(r['scale_lock_acquired'] and r['scale_lock_released'])
            self.assertEqual((Path(d)/'held/stdout.log').read_text().splitlines(),['held']*3)
            with lock_path.open('a') as subsequent:
                fcntl.flock(subsequent,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_timeout_reaps_scale_child_before_unlock(self):
        with tempfile.TemporaryDirectory() as d:
            lock_path=Path(d)/'scale.lock'
            r=run([sys.executable,'-c','import time;time.sleep(2)'],cwd=d,
                  dest=Path(d)/'timeout',scale=True,seconds=.1,scale_lock=lock_path)
            self.assertEqual(r['status'],'timeout')
            self.assertIsNotNone(r['returncode'])
            self.assertTrue(r['process_group_closed'])
            self.assertTrue(r['scale_lock_acquired'] and r['scale_lock_released'])
            with lock_path.open('a') as subsequent:
                fcntl.flock(subsequent,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_assertion_preserves_prior_observation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'stage.json'
            with self.assertRaises(AssertionError):
                with stage(p,'component',candidate='synthetic-control') as seen:
                    seen['allocations']=[dict(tensor='Y',bytes=16)]
                    raise AssertionError('after allocation capture')
            r=json.loads(p.read_text())
            self.assertEqual(r['status'],'failed')
            self.assertEqual(r['observations']['allocations'][0]['bytes'],16)

    def test_success_failure_and_timeout_are_distinct(self):
        with tempfile.TemporaryDirectory() as d:
            for label,body,seconds,want in [('ok','print(3)',2,'succeeded'),
                ('failed','raise ValueError("control")',2,'failed'),
                ('timeout','import time; time.sleep(2)',.1,'timeout')]:
                r=run([sys.executable,'-c',body],cwd=d,dest=Path(d)/label,seconds=seconds)
                self.assertEqual(r['status'],want)
                self.assertEqual(json.loads((Path(d)/label/'PROCESS.json').read_text())['status'],want)
                self.assertTrue(r['process_group_closed'])


if __name__=='__main__':unittest.main()
