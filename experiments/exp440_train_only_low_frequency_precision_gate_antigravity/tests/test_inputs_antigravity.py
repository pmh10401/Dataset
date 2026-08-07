from __future__ import annotations

from hashlib import sha256
import numpy as np
import unittest

from experiments.exp439_scale_aware_reconstruction_repair_antigravity.contracts_antigravity import RawValueSeries
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.inputs_antigravity import Exp439RawDataset
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.inputs_antigravity import (
    partition_train_rows_antigravity,
    test_row_abstention_reason_antigravity as abstention_reason,
)


def _row(index:int,values:list[float],split:str)->RawValueSeries:
    raw=np.asarray(values,dtype='<f4').tobytes()
    return RawValueSeries('A',index,split,np.frombuffer(raw,dtype='<f4'),sha256(raw).hexdigest())


class Exp440InputsAntigravityTests(unittest.TestCase):
    def test_nonfinite_train_row_is_excluded_whole_without_harming_finite_siblings(self)->None:
        rows=tuple(_row(i,[float(i+j) for j in range(16)],'TRAIN') for i in range(4))+(_row(4,[1.0,float('nan')]+[2.0]*14,'TRAIN'),)
        prepared=partition_train_rows_antigravity(Exp439RawDataset('A','TRAIN',rows))
        self.assertEqual(len(prepared.groups),1);self.assertEqual(len(prepared.groups[0].rows),4);self.assertEqual(tuple(x.instance_index for x in prepared.exclusions),(4,))

    def test_exact_lengths_never_mix_and_nonfinite_test_abstains_per_row(self)->None:
        rows=tuple(_row(i,[float(i)]*16,'TRAIN') for i in range(4))+tuple(_row(i+4,[float(i)]*32,'TRAIN') for i in range(4))
        prepared=partition_train_rows_antigravity(Exp439RawDataset('A','TRAIN',rows))
        self.assertEqual(tuple(g.raw_length for g in prepared.groups),(16,32))
        self.assertIsNone(abstention_reason(_row(0,[1.0]*16,'TEST')))
        self.assertEqual(abstention_reason(_row(1,[1.0,float('inf')],'TEST')),'nonfinite_test')

    def test_groups_over_compute_cap_abstain_without_subsampling(self)->None:
        rows=tuple(_row(i,[float(i+j) for j in range(16)],'TRAIN') for i in range(257))
        prepared=partition_train_rows_antigravity(Exp439RawDataset('A','TRAIN',rows))
        self.assertEqual(prepared.groups,());self.assertEqual(prepared.insufficient_lengths,(16,))


if __name__=='__main__': unittest.main()
