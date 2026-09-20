import sqlite3
import numpy as np
from .pilot_antigravity import load_normal_train,perturbations

def test_loader_ignores_malformed_test_and_non_normal_training_rows():
    con=sqlite3.connect(':memory:')
    con.executescript('CREATE TABLE datasets(id INTEGER,name TEXT); CREATE TABLE instances(dataset_id INTEGER,split TEXT,label TEXT,instance_index INTEGER,values_blob BLOB); INSERT INTO datasets VALUES(1,"tiny");')
    valid=np.arange(16,dtype='<f4').tobytes()
    con.executemany('INSERT INTO instances VALUES(?,?,?,?,?)',[(1,'TRAIN','0',0,valid),(1,'TRAIN','0',1,valid),(1,'TRAIN','1',2,b'x'),(1,'TEST','0',0,b'x')])
    x=load_normal_train(con,'tiny');assert x.shape==(2,16) and np.isfinite(x).all()
    con.close()

def test_proxy_corruptions_are_deterministic_finite_and_do_not_modify_input():
    x=np.random.default_rng(10).normal(size=(5,31));original=x.copy()
    a=perturbations(x,41);b=perturbations(x,41)
    assert len(a)==4
    for name,v in a.items():
        np.testing.assert_array_equal(v,b[name]);assert v.shape==x.shape and np.isfinite(v).all()
    np.testing.assert_array_equal(x,original)
