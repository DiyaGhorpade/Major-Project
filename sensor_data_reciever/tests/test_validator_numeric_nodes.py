"""Numeric node IDs and the shared CSV column order."""
import unittest

from validator import ValidationError, validate_record


class TestNumericNodes(unittest.TestCase):
    def record(self, node_id):
        return ["2026-08-28T07:00:00Z", "10", node_id, "5", "14",
                "45.2", "23.1", "60.5", "512.0"]

    def test_accepts_both_nodes_in_configured_column_order(self):
        for node_id in ("1", "2"):
            with self.subTest(node_id=node_id):
                result = validate_record(self.record(node_id))
                self.assertIsInstance(result, dict)
                self.assertEqual(result["node_id"], node_id)
                self.assertEqual(result["session_id"], 10)
                self.assertEqual(result["sampling_point"], 5)
                self.assertEqual(result["plant_id"], 14)

    def test_rejects_other_node_identifiers(self):
        for node_id in ("0", "3", "-1", "1.0", "01", "NODE_01", "abc"):
            with self.subTest(node_id=node_id):
                errors = validate_record(self.record(node_id))
                self.assertIsInstance(errors[0], ValidationError)
                self.assertEqual(errors[0].sensor, "node_id")
                self.assertEqual(errors[0].bad_value, node_id)
                self.assertEqual(errors[0].plant_id, "14")

    def test_reports_missing_node(self):
        for node_id in ("", "   ", None):
            with self.subTest(node_id=node_id):
                errors = validate_record(self.record(node_id))
                self.assertEqual(errors[0].sensor, "node_id")
                self.assertEqual(errors[0].reason, "Missing node_id")

    def test_keeps_field_count_validation(self):
        errors = validate_record(self.record("1")[:-1])
        self.assertEqual(errors[0].sensor, "record")

    def test_sensor_id_uses_sensor_range_instead_of_scan_count(self):
        fields = self.record("1")
        fields[3] = "12"
        self.assertIsInstance(validate_record(fields), dict)
        fields[3] = "13"
        errors = validate_record(fields)
        self.assertEqual(errors[0].sensor, "sampling_point")
        self.assertEqual(errors[0].reason, "Expected sensor ID 1-12")
