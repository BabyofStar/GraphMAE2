import importlib.util
import math
from pathlib import Path
import random
import unittest

spec=importlib.util.spec_from_file_location('_route_math',Path(__file__).resolve().parents[1]/'models/density_route_math.py')
route=importlib.util.module_from_spec(spec)
spec.loader.exec_module(route)


class RouteMathTests(unittest.TestCase):
    def test_official_budgets_have_uniform_warmup_and_density_tail(self):
        for epochs in (500,2000):
            p=route.probability_schedule(epochs)
            length=epochs//5
            self.assertEqual(p[:length],[0.]*length)
            self.assertEqual(p[-length:],[1.]*length)

    def test_expected_density_exposure_is_half_not_full_s(self):
        for epochs in (500,2000):
            self.assertTrue(math.isclose(sum(route.probability_schedule(epochs)),epochs/2,abs_tol=1e-9))

    def test_short_and_single_epoch_boundaries(self):
        self.assertEqual(route.probability_schedule(1),[0.])
        actual=route.probability_schedule(5)
        for a,b in zip(actual,[0.,1/12,.5,11/12,1.]):
            self.assertAlmostEqual(a,b)

    def test_invalid_schedule_rejected(self):
        for args in ((0,),(-1,),(True,),(1.5,),(10,.8,.2),(10,float('nan'),.8),(10,.2,float('inf'))):
            with self.assertRaises(ValueError):
                route.probability_schedule(*args)

    def test_probability_calculation_preserves_python_rng_state(self):
        random.seed(113)
        before=random.getstate()
        route.probability_schedule(2000)
        self.assertEqual(before,random.getstate())


if __name__=='__main__':
    unittest.main(verbosity=2)
