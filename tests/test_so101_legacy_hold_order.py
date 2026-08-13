import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_so101_legacy30_audited_pilot import install_hold_before_finalize


class FakeStrategy:
    def __init__(self,events,fail):
        self.events=events
        self.fail=fail
    def _policy_loop(self):
        self.events.append('policy_loop')
        if self.fail:
            raise RuntimeError('guard rejected')
    def run(self):
        try:self._policy_loop()
        finally:self.events.append('video_save')


class ImmediateHoldOrderTest(unittest.TestCase):
    def test_hold_precedes_video_save_after_rejection(self):
        events=[];strategy=FakeStrategy(events,True)
        install_hold_before_finalize(strategy,lambda:events.append('hold'),lambda _:None,clock=lambda:1)
        with self.assertRaisesRegex(RuntimeError,'guard rejected'):
            strategy.run()
        self.assertEqual(events,['policy_loop','hold','video_save'])
    def test_hold_precedes_video_save_on_normal_end(self):
        events=[];strategy=FakeStrategy(events,False)
        install_hold_before_finalize(strategy,lambda:events.append('hold'),lambda _:None,clock=lambda:1)
        strategy.run()
        self.assertEqual(events,['policy_loop','hold','video_save'])


if __name__=='__main__':unittest.main()
