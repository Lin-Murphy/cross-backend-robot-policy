import unittest
from cross_backend.so101_pose_positioning import (NAMES, REFERENCE, TARGET, STEP_COUNT,
    PositioningStop, restricted_bus_class, verify_plan, waypoint)


def calibration():
    return {n:{'range_min':min(a,b)-1,'range_max':max(a,b)+1} for n,a,b in zip(NAMES,REFERENCE,TARGET)}


class PosePositioningTest(unittest.TestCase):
    def test_full_path_is_one_tick_per_joint_within_frozen_endpoints(self):
        plan=verify_plan(calibration())
        self.assertEqual(waypoint(0),REFERENCE)
        self.assertEqual(waypoint(STEP_COUNT),TARGET)
        self.assertEqual(plan['steps'],514)
        self.assertAlmostEqual(plan['minimum_duration_s'],51.4)
        self.assertFalse(plan['actual_motor_speed_limit_verified'])

    def test_changed_calibration_rejected(self):
        data=calibration();data['shoulder_lift']['range_max']=1000
        with self.assertRaisesRegex(PositioningStop,'outside calibration'):
            verify_plan(data)

    def test_write_allowlist_rejects_other_registers_and_outside_targets(self):
        class FakeBase:
            def __init__(self):
                self.motors={n:type('M',(),{'id':i})() for i,n in enumerate(NAMES,1)}
                self.calibration={n:type('C',(),bounds)() for n,bounds in calibration().items()}
                self.writes=[]
            def write(self,register,name,value,normalize=False,num_retry=0):
                address,length=(42,2) if register=='Goal_Position' else (40,1)
                return self._write(address,length,self.motors[name].id,value,num_retry=num_retry)
            def _write(self,addr,length,motor_id,value,*,num_retry=0,raise_on_error=True,err_msg=''):
                self.writes.append((addr,length,motor_id,value))
        bus=restricted_bus_class(FakeBase)()
        with self.assertRaises(PositioningStop):bus.write('Goal_Position','shoulder_pan',REFERENCE[0])
        with self.assertRaises(PositioningStop):bus.allowed_write('Operating_Mode','shoulder_pan',0)
        with self.assertRaises(PositioningStop):bus.allowed_write('Goal_Position','shoulder_pan',0)
        bus.allowed_write('Goal_Position','shoulder_pan',REFERENCE[0])
        bus.allowed_write('Torque_Enable','shoulder_pan',1)
        self.assertEqual(bus.writes,[(42,2,1,REFERENCE[0]),(40,1,1,1)])


if __name__=='__main__':unittest.main()
