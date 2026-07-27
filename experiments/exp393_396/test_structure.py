import inspect
import unittest

from experiments.exp393_396.artifact_io import load_prediction_inputs
from experiments.exp393_396.run_experiment_393_396_pipeline import predict_dataset


class StructureTests(unittest.TestCase):
    def test_predict_dataset_has_no_test_label_parameter(self):
        parameters = inspect.signature(predict_dataset).parameters

        self.assertNotIn("y_test", parameters)
        self.assertNotIn("labels", parameters)

    def test_prediction_loader_query_does_not_select_label(self):
        source = inspect.getsource(load_prediction_inputs).lower()

        self.assertNotIn("i.label", source)
        self.assertNotIn("labels_blob", source)

    def test_prediction_function_does_not_load_evaluation_labels(self):
        source = inspect.getsource(predict_dataset)

        self.assertNotIn("load_test_labels", source)
        self.assertNotIn("y_test", source)


if __name__ == "__main__":
    unittest.main()
