import unittest
from pathlib import Path
import cv2


class SceneClearTest(unittest.TestCase):
    def dark_fraction(self,path):
        image=cv2.imread(str(path))
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        return float((gray[250:330,190:310]<100).mean())
    def test_empty_baseline_passes_and_current_keys_are_rejected(self):
        clean=self.dark_fraction(Path('artifacts/so101-pose-alignment-postcheck-20260926/00-follower.png'))
        clutter=self.dark_fraction(Path('artifacts/so101-bounded-pilot-preflight-20260926/00-follower.png'))
        self.assertLess(clean,0.05)
        self.assertGreater(clutter,0.05)
        self.assertGreater(clutter-clean,0.4)


if __name__=='__main__':unittest.main()
