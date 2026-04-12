import unittest
import numpy as np

from opt_targets import ResultBundle, OptTarget, TargetType, Reduction, CompareMode, Mode


class TestOptTargets(unittest.TestCase):
    def setUp(self):
        fields = {'stress_vm': np.array([1.0, 2.0, 3.0])}
        rbe = {'residual': np.array([0.0, 0.0, 10.0])}
        node_disps = {0: np.array([0.0, 0.0, 1.0])}
        modes = [Mode(index=0, frequency=10.0, vector=np.array([1.0, 0.0])),
                 Mode(index=1, frequency=20.0, vector=np.array([0.0, 1.0]))]

        self.bundle = ResultBundle(fields=fields, rbe_reactions=rbe, node_disps=node_disps, mass=5.0, modes=modes)

        ref_modes = [Mode(index=0, frequency=10.1, vector=np.array([0.9, 0.1])),
                     Mode(index=1, frequency=19.9, vector=np.array([0.1, 0.9]))]
        self.ref_bundle = ResultBundle(fields={'stress_vm': np.array([2.0, 2.0, 2.0])},
                                       rbe_reactions={'residual': np.array([0.0, 0.0, 8.0])},
                                       node_disps=node_disps,
                                       mass=5.0,
                                       modes=ref_modes)

    def test_mass(self):
        ot = OptTarget(target_type=TargetType.MASS, compare_mode=CompareMode.ABSOLUTE, weight=1.0)
        err, details = ot.compute_error(self.bundle, ref_bundle=self.ref_bundle)
        self.assertAlmostEqual(err, 0.0)

    def test_field_stat(self):
        ot = OptTarget(target_type=TargetType.FIELD_STAT, field='stress_vm', reduction=Reduction.MAX,
                       compare_mode=CompareMode.ABSOLUTE, ref_value=2.0, weight=1.0)
        err, details = ot.compute_error(self.bundle, ref_bundle=None)
        # val max=3, ref=2 => err=1
        self.assertAlmostEqual(err, 1.0)

    def test_rbe(self):
        ot = OptTarget(target_type=TargetType.RBE_REACTION, rbe_id='residual', component='magnitude',
                       compare_mode=CompareMode.RELATIVE, weight=1.0)
        err, details = ot.compute_error(self.bundle, ref_bundle=self.ref_bundle)
        # val mag =10, ref mag=8 => relative err=(10-8)/8=0.25
        self.assertAlmostEqual(err, 0.25)

    def test_modes_mac(self):
        ot = OptTarget(target_type=TargetType.MODES, compare_mode=CompareMode.MAC, weight=1.0)
        err, details = ot.compute_error(self.bundle, ref_bundle=self.ref_bundle)
        self.assertTrue(0.0 <= err <= 1.0)


if __name__ == '__main__':
    unittest.main()
