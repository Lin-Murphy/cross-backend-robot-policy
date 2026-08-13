import unittest
import cv2
import numpy as np
from scripts.calibrate_s1_camera_offline import fit_views


class CalibrationTest(unittest.TestCase):
    def test_known_camera_and_holdout(self):
        obj = np.zeros((54, 3), np.float32)
        obj[:, :2] = np.mgrid[:9, :6].T.reshape(-1, 2) * .018
        k = np.array([[610., 0, 320], [0, 605, 240], [0, 0, 1]])
        rng = np.random.default_rng(128)
        points = []
        for _ in range(20):
            r = rng.uniform(-.5, .5, 3)
            t = np.array([rng.uniform(-.12, -.04), rng.uniform(-.09, -.02), rng.uniform(.45, .7)])
            p, _ = cv2.projectPoints(obj, r, t, k, np.zeros(5))
            points.append(p + rng.normal(0, .02, p.shape))
        result = fit_views(points, (640, 480), (9, 6), 18)
        np.testing.assert_allclose(np.array(result['K'])[:2, :], k[:2, :], atol=2)
        self.assertLess(max(v['rms_px'] for v in result['views']), .1)
        self.assertEqual(sum(v['split'] == 'holdout' for v in result['views']), 4)
        self.assertFalse(result['physical_alignment_verified'])

    def test_insufficient_and_duplicate(self):
        with self.assertRaises(ValueError):
            fit_views([], (640, 480), (9, 6), 18)
        with self.assertRaisesRegex(ValueError, 'repeated'):
            fit_views([np.zeros((54, 1, 2))] * 15, (640, 480), (9, 6), 18)
        near=[np.zeros((54, 1, 2),dtype=np.float32) for _ in range(15)]
        near[1][:,:,0]=1.0
        with self.assertRaisesRegex(ValueError, 'near-duplicate'):
            fit_views(near, (640, 480), (9, 6), 18)


if __name__ == '__main__':
    unittest.main()
