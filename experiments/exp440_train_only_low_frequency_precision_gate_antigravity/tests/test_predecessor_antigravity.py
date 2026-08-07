from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.predecessor_antigravity import (
    parse_hard_coordinates_antigravity,
    validate_prediction_source_antigravity,
)


class Exp440PredecessorAntigravityTests(unittest.TestCase):
    def test_prediction_source_rejects_evaluation_paths_and_label_fields(self) -> None:
        with self.assertRaises(ValueError):
            validate_prediction_source_antigravity(Path('/tmp/evaluation_antigravity/x.json'), {}, 'a'*64, 'a'*64)
        with self.assertRaises(ValueError):
            validate_prediction_source_antigravity(Path('/tmp/prediction_antigravity.json'), {'test_label': 1}, 'a'*64, 'a'*64)

    def test_prediction_source_requires_exact_sha_and_population_order(self) -> None:
        raw=b'{"datasets":["A","B"],"contains_test_labels":false}\n'
        payload=json.loads(raw)
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'prediction_antigravity.json';path.write_bytes(raw)
            self.assertEqual(validate_prediction_source_antigravity(path,payload,sha256(raw).hexdigest(),sha256(raw).hexdigest(),expected_population=('A','B')),('A','B'))
            with self.assertRaises(ValueError): validate_prediction_source_antigravity(path,payload,'0'*64,sha256(raw).hexdigest(),expected_population=('A','B'))
            with self.assertRaises(ValueError): validate_prediction_source_antigravity(path,payload,sha256(raw).hexdigest(),sha256(raw).hexdigest(),expected_population=('B','A'))

    def test_hard_parser_is_label_free_unique_and_hash_bound(self) -> None:
        raw=b'{"dataset":"A","candidate_index":2,"contains_test_labels":false,"contains_evaluation_metrics":false}\n'
        self.assertEqual(parse_hard_coordinates_antigravity(raw,sha256(raw).hexdigest()),frozenset({('A',2)}))
        with self.assertRaises(ValueError): parse_hard_coordinates_antigravity(raw,'0'*64)
        with self.assertRaises(ValueError): parse_hard_coordinates_antigravity(raw+raw,sha256(raw+raw).hexdigest())


if __name__=='__main__': unittest.main()
