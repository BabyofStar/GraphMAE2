import importlib.util
import itertools
from fractions import Fraction
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('_joint_math',Path(__file__).parents[1]/'models/joint_math.py')
j=importlib.util.module_from_spec(spec); spec.loader.exec_module(j)

class JointMathTests(unittest.TestCase):
    def test_matching_covers_each_node_once(self):
        pairs,singleton,flags=j.pairing(list(range(7)),[(.9,0,1),(.8,1,2),(.7,2,3)],list(reversed(range(7))))
        nodes=[x for pair in pairs for x in pair]+([singleton] if singleton is not None else [])
        self.assertEqual(sorted(nodes),list(range(7))); self.assertEqual(flags[:2],[True,True])
    def test_nonpositive_edges_are_not_semantic_matches(self):
        _,_,flags=j.pairing([0,1,2,3],[(0.,0,1),(-.2,2,3)],[0,1,2,3]); self.assertEqual(flags,[False,False])
    def test_deterministic_score_tie_break(self):
        a=j.pairing([0,1,2,3],[(1.,2,3),(1.,0,1)],[3,2,1,0])
        b=j.pairing([0,1,2,3],[(1.,0,1),(1.,2,3)],[3,2,1,0]); self.assertEqual(a,b)
    def test_pair_selection_exact_budget_and_unique(self):
        for count in range(5):
            for q in range(2*count+1):
                for coupling in ('anti','co'):
                    selected=j.selected_pair_positions(count,list(range(count)),[i%2 for i in range(count)],q,coupling)
                    self.assertEqual(len(selected),q); self.assertEqual(len(set(selected)),q)
    def test_equal_marginals_exhaustively_including_odd_singleton(self):
        for n in range(2,8):
            p=n//2
            for q in range(n+1):
                for coupling in ('anti','co'):
                    totals=[Fraction(0) for _ in range(n)]
                    singleton=[(0,Fraction(n-q,n)),(1,Fraction(q,n))] if n%2 else [(0,Fraction(1))]
                    orders=list(itertools.permutations(range(p))); coins=list(itertools.product((0,1),repeat=p))
                    for b,prob in singleton:
                        if not prob: continue
                        for order in orders:
                            for coin in coins:
                                selected=j.selected_pair_positions(p,list(order),list(coin),q-b,coupling)
                                if b: selected.append(n-1)
                                for index in selected: totals[index]+=prob/(len(orders)*len(coins))
                    self.assertEqual(totals,[Fraction(q,n)]*n,(n,q,coupling,totals))
    def test_anti_and_co_have_opposite_pair_disagreement(self):
        for coupling,expected in (('anti',1.),('co',0.)):
            selected=set(j.selected_pair_positions(4,list(range(4)),[0]*4,4,coupling))
            difference=sum((2*i in selected)!=(2*i+1 in selected) for i in range(4))/4
            self.assertEqual(difference,expected)
    def test_invalid_candidates_and_sampling_fail(self):
        with self.assertRaises(ValueError): j.pairing([0,1],[(1.,0,2)],[0,1])
        with self.assertRaises(ValueError): j.selected_pair_positions(2,[0,0],[0,1],2,'anti')
        with self.assertRaises(ValueError): j.selected_pair_positions(2,[0,1],[0,1],5,'co')
    def test_fixed_seeds_are_distinct(self): self.assertNotEqual(j.SEMANTIC_SEED,j.RANDOM_SEED)

if __name__=='__main__': unittest.main(verbosity=2)
