"""The cube task uses the cube's box footprint rather than the old tape radius."""
import unittest
from pathlib import Path

from cross_backend.tape_sim_backend import TapeSimBackend
from cross_backend.tape_task_observer import observe_task


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "artifacts/sim-cube-v1-20260925/scene.xml"


class CubeTaskObserverTest(unittest.TestCase):
    def test_cube_fits_on_mat_at_five_centimetre_offset(self):
        backend = TapeSimBackend(SCENE)
        try:
            backend.reset(tape_xy=[-0.13, 0.065], mat_xy=[-0.18, 0.065])
            facts = observe_task(backend)
            self.assertEqual(facts.zone, "inside")
            backend.reset(tape_xy=[-0.11, 0.065], mat_xy=[-0.18, 0.065])
            self.assertEqual(observe_task(backend).zone, "boundary")
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
