import unittest

class EvaluationCodecAntigravityTest(unittest.TestCase):
    def test_codec_is_pure_and_exposes_reconstruction_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity import evaluation_codec_antigravity as codec
        self.assertTrue(callable(codec.reconstruct_and_validate_antigravity))
        for forbidden in ("Path","os","sqlite3","open_ordered_label_session_antigravity"):
            self.assertFalse(hasattr(codec,forbidden))

    def test_hand_computed_four_transition_derivation_and_bootstrap_parity_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity import evaluation_codec_antigravity as codec
        from experiments.exp449_residual_fn_method_qualification_antigravity.bootstrap_antigravity import paired_delta_antigravity
        from experiments.exp449_residual_fn_method_qualification_antigravity.contracts_antigravity import ConfusionAntigravity
        families=codec.FAMILIES_ANTIGRAVITY; rows=[]
        cases=((ConfusionAntigravity(0,0,1,9),ConfusionAntigravity(1,0,0,9),(1,0,0,0)),(ConfusionAntigravity(0,0,0,10),ConfusionAntigravity(0,1,0,9),(0,1,0,0)),(ConfusionAntigravity(1,0,0,9),ConfusionAntigravity(0,0,1,9),(0,0,1,0)),(ConfusionAntigravity(0,1,0,9),ConfusionAntigravity(0,0,0,10),(0,0,0,1)))
        for i in range(48):
            baseline,candidate,transitions=cases[i] if i<4 else (ConfusionAntigravity(0,0,0,10),ConfusionAntigravity(0,0,0,10),(0,0,0,0))
            rows.append(codec.DatasetEvaluationAntigravity(i,(i,families[i%8],f"row{i}"),10,baseline,candidate,*transitions,0,"0"*64))
        result=codec.derive_evaluation_antigravity(tuple(rows),"a"*40,"b"*64,("c"*64,"d"*64,"e"*64))
        self.assertEqual(result.transition_totals,(1,1,1,1)); self.assertEqual(result.transition_dataset_counts,(1,1,1,1))
        self.assertEqual(result.target_fn_delta,0); self.assertEqual(result.candidate_global,result.baseline_global)
        self.assertEqual(result.bootstrap,paired_delta_antigravity(result.baseline_matrix,result.candidate_matrix))
        forged=rows.copy(); forged[0]=codec.DatasetEvaluationAntigravity(0,(0,families[0],"row0"),10,cases[0][0],cases[0][1],0,0,0,0,0,"0"*64)
        with self.assertRaises(ValueError): codec.derive_evaluation_antigravity(tuple(forged),"a"*40,"b"*64,("c"*64,"d"*64,"e"*64))

if __name__=="__main__": unittest.main()
