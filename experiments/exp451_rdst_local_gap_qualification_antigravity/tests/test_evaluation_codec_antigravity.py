import json, tempfile, unittest
from pathlib import Path

class EvaluationCodecAntigravityTest(unittest.TestCase):
    def test_codec_is_pure_and_exposes_reconstruction_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity import evaluation_codec_antigravity as codec
        import inspect
        self.assertTrue(callable(codec.reconstruct_and_validate_antigravity))
        for forbidden in ("Path","os","sqlite3","open_ordered_label_session_antigravity"):
            self.assertFalse(hasattr(codec,forbidden))
        source=inspect.getsource(codec)
        for forbidden in (".replay_antigravity",".evaluation_antigravity",".dashboard_export_antigravity",".sources_antigravity"):
            self.assertNotIn(forbidden,source)

    def test_hand_computed_four_transition_derivation_and_bootstrap_parity_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity import evaluation_codec_antigravity as codec
        from experiments.exp449_residual_fn_method_qualification_antigravity.bootstrap_antigravity import paired_delta_antigravity
        from experiments.exp449_residual_fn_method_qualification_antigravity.contracts_antigravity import ConfusionAntigravity
        from experiments.exp451_rdst_local_gap_qualification_antigravity.predecessor_antigravity import verify_exp448_predecessor_antigravity
        predecessor=verify_exp448_predecessor_antigravity(); families=codec.FAMILIES_ANTIGRAVITY; rows=[]
        self.assertEqual(sum(predecessor.target_test_lengths),72300)
        for i in range(48):
            length=predecessor.target_test_lengths[i]
            cases=((ConfusionAntigravity(0,0,1,length-1),ConfusionAntigravity(1,0,0,length-1),(1,0,0,0)),(ConfusionAntigravity(0,0,0,length),ConfusionAntigravity(0,1,0,length-1),(0,1,0,0)),(ConfusionAntigravity(1,0,0,length-1),ConfusionAntigravity(0,0,1,length-1),(0,0,1,0)),(ConfusionAntigravity(0,1,0,length-1),ConfusionAntigravity(0,0,0,length),(0,0,0,1)))
            baseline,candidate,transitions=cases[i] if i<4 else (ConfusionAntigravity(0,0,0,length),ConfusionAntigravity(0,0,0,length),(0,0,0,0))
            order=predecessor.target_dataset_order[i]
            rows.append(codec.DatasetEvaluationAntigravity(i,(order,families[i%8],f"row{i}"),length,baseline,candidate,*transitions,0,"0"*64))
        result=codec.derive_evaluation_antigravity(tuple(rows),"a"*40,"b"*64,("c"*64,"d"*64,"e"*64))
        self.assertEqual(result.transition_totals,(1,1,1,1)); self.assertEqual(result.transition_dataset_counts,(1,1,1,1))
        self.assertEqual(result.target_fn_delta,0); self.assertEqual(result.candidate_global,result.baseline_global)
        self.assertEqual(result.bootstrap,paired_delta_antigravity(result.baseline_matrix,result.candidate_matrix))
        forged=rows.copy(); original=rows[0]; forged[0]=codec.DatasetEvaluationAntigravity(0,original.identity,original.target_length,original.baseline,original.candidate,0,0,0,0,0,"0"*64)
        with self.assertRaises(ValueError): codec.derive_evaluation_antigravity(tuple(forged),"a"*40,"b"*64,("c"*64,"d"*64,"e"*64))
        for delta in (-1,1):
            malformed=rows.copy(); original=malformed[-1]; length=original.target_length+delta
            malformed[-1]=codec.DatasetEvaluationAntigravity(original.sequence,original.identity,length,ConfusionAntigravity(0,0,0,length),ConfusionAntigravity(0,0,0,length),0,0,0,0,0,"0"*64)
            with self.assertRaises(ValueError): codec.derive_evaluation_antigravity(tuple(malformed),"a"*40,"b"*64,("c"*64,"d"*64,"e"*64))

    def test_codec_owned_reconstruction_rejects_nested_exact_type_mutations_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity import evaluation_codec_antigravity as codec
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from .test_evaluation_antigravity import SHA, fixture_antigravity
        canonical=lambda value:json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()+b"\n"
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); expected=_evaluate_with_dependencies_antigravity(authority,SHA,deps)
            names=("context_antigravity.json","datasets_antigravity.json","result_antigravity.json","dashboard_antigravity.json"); raw=[(root/name).read_bytes() for name in names]
            self.assertEqual(codec.reconstruct_and_validate_antigravity(*raw,expected_prediction=authority,expected_predecessor=deps.predecessor_loader()),expected)
            mutations=((1,lambda x:x["rows"][0].__setitem__("target_length",True)),(2,lambda x:x["baseline_matrix"][0].__setitem__(0,True)),(2,lambda x:x.__setitem__("extra",1)))
            for index,mutate in mutations:
                changed=list(raw); value=json.loads(changed[index]); mutate(value); changed[index]=canonical(value)
                with self.assertRaises(ValueError): codec.reconstruct_and_validate_antigravity(*changed,expected_prediction=authority,expected_predecessor=deps.predecessor_loader())

if __name__=="__main__": unittest.main()
