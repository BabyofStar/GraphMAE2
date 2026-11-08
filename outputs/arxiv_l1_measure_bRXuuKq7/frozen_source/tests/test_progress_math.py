import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('_rank_math',Path(__file__).parents[1]/'models/progress_math.py')
mathmod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mathmod)


class ProgressMathTests(unittest.TestCase):
    def test_first_observation_is_neutral(self):
        self.assertEqual(mathmod.update_ema(0,0,2,False),(2,2))
        self.assertEqual(mathmod.progress_score(2,2),0)
    def test_decreasing_error_produces_positive_progress(self):
        f,s=2.,2.
        for _ in range(8): f,s=mathmod.update_ema(f,s,1,True)
        self.assertGreater(mathmod.progress_score(f,s),0)
    def test_increasing_error_has_no_positive_progress(self):
        f,s=mathmod.update_ema(1,1,2,True)
        self.assertEqual(mathmod.progress_score(f,s),0)
    def test_every_score_vector_has_identical_weight_multiset(self):
        vectors=([0,0,0,0],[1,5,2,4],[1000,1e-9,4,5])
        for alpha in (0.,.1,.5,1.):
            values=[sorted(mathmod.rank_weights(x,[3,0,2,1],alpha)) for x in vectors]
            self.assertEqual(values[0],values[1]); self.assertEqual(values[1],values[2])
            self.assertAlmostEqual(sum(values[0]),1.)
    def test_high_scores_get_high_weights(self):
        w=mathmod.rank_weights([3,1,2],[0,1,2],.5)
        self.assertGreater(w[0],w[2]); self.assertGreater(w[2],w[1])
    def test_invalid_input_fails(self):
        for scores,ties,alpha in (([],[],.5),([1,2],[0,0],.5),([float('nan')],[0],.5),([1],[0],2)):
            with self.assertRaises(ValueError): mathmod.rank_weights(scores,ties,alpha)
    def test_tie_priorities_control_equal_scores(self):
        weights=mathmod.rank_weights([0,0],[1,0],.5)
        self.assertAlmostEqual(weights[0],7/12); self.assertAlmostEqual(weights[1],5/12)
    def test_shuffle_stream_is_epoch_and_layer_specific(self):
        seeds={mathmod.shuffle_seed(s,g) for s in range(20) for g in range(4)}
        self.assertEqual(len(seeds),80)
        with self.assertRaises(ValueError): mathmod.shuffle_seed(-1,0)

if __name__=='__main__': unittest.main(verbosity=2)
