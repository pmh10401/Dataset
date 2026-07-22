import unittest


class EventSemanticsTest(unittest.TestCase):
    def test_adjacent_instance_indices_are_not_time_points(self) -> None:
        candidate_index_unit = "complete_test_series_instance_virtual_run"
        self.assertNotEqual(candidate_index_unit, "within_series_time_point")


if __name__ == "__main__":
    unittest.main()
