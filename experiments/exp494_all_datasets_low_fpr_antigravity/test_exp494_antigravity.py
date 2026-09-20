"""Boundary checks for the additive evaluation, without training score sources."""
import unittest
import numpy as np
from .evaluation_antigravity import metric,predict_queries,summarize
from experiments.exp493_stable_meta_routing_antigravity.meta_antigravity import predict_antigravity

class EvaluationTests(unittest.TestCase):
    def test_ties_and_integer_false_positive_budget(self):
        y=np.array([0]*1000+[1,1]);s=np.array(list(range(1000))+[1000,998.5])
        m=metric(y,s)
        self.assertEqual(m['tp'],[1,2,2,2]);self.assertEqual(m['fp'],[0,1,5,10])
        self.assertEqual(metric(np.array([0,0,1]),np.array([1.,0.,1.]))['tp'],[0]*4)
    def test_family_weighting_is_not_dataset_weighting(self):
        def item(f,t):return dict(source_family=f,metrics=dict(tpr=[t]*4,tp=[int(t)]*4,n_anomaly=1,n_normal=10,zf=int(t),auc=t))
        s=summarize([item('a',1),item('a',1),item('a',1),item('b',0)])
        self.assertEqual(s['family_macro_tpr'],[.5]*4);self.assertEqual(s['dataset_macro_tpr'],[.75]*4)
    def test_frozen_query_fit_replays_and_excludes_whole_family(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(12,4));x[0,0]=np.nan;y=rng.normal(size=(12,6));f=np.repeat(['a','b','c','d'],3)
        expected,_=predict_antigravity(x,y,f)
        actual=predict_queries(x,y,f,x,f);np.testing.assert_allclose(actual,expected,rtol=0,atol=0)
        changed=y.copy();changed[f=='a']+=10000
        np.testing.assert_array_equal(predict_queries(x,changed,f,x[f=='a'],f[f=='a']),actual[f=='a'])
        new=predict_queries(x,y,f,x[:1],['new_family']);self.assertTrue(np.isfinite(new).all())
    def test_invalid_or_single_class_metrics_are_not_silently_included(self):
        for y,s in [(np.array([0,0]),np.array([1.,2.])),(np.array([0,1]),np.array([np.nan,2.]))]:
            with self.assertRaises(ValueError):metric(y,s)
if __name__=='__main__':unittest.main()
