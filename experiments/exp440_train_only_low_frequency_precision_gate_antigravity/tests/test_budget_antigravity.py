from __future__ import annotations
import json,random,unittest
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.budget_antigravity import select_global_candidates_antigravity
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.contracts_antigravity import ProvisionalCandidateAntigravity


def _candidate(i:int)->ProvisionalCandidateAntigravity:return ProvisionalCandidateAntigravity(f'D{i:03d}',i,16,(1,2),2,1.0+(i%3),0.1+(i%5)/100)


class Exp440BudgetAntigravityTests(unittest.TestCase):
    def test_hard_exclusion_precedes_deterministic_global_top_100(self)->None:
        values=[_candidate(i) for i in range(105)];hard=frozenset({values[0].coordinate,values[1].coordinate});random.Random(44).shuffle(values)
        first=select_global_candidates_antigravity(values,hard);random.Random(99).shuffle(values);second=select_global_candidates_antigravity(values,hard)
        self.assertEqual(len(first),100);self.assertEqual(tuple(x.global_rank for x in first),tuple(range(1,101)));self.assertTrue(all(x.coordinate not in hard for x in first));self.assertEqual(json.dumps([x.to_payload() for x in first],sort_keys=True),json.dumps([x.to_payload() for x in second],sort_keys=True))

    def test_sparse_candidates_are_not_padded(self)->None:
        result=select_global_candidates_antigravity([_candidate(i) for i in range(7)],frozenset());self.assertEqual(len(result),7)

    def test_duplicate_coordinate_and_budget_override_fail_closed(self)->None:
        item=_candidate(1)
        with self.assertRaises(ValueError):select_global_candidates_antigravity([item,item],frozenset())
        with self.assertRaises(ValueError):select_global_candidates_antigravity([item],frozenset(),budget=99)


if __name__=='__main__':unittest.main()
