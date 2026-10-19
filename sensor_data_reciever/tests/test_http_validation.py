"""HTTP uploads use the standalone validator's decisions and errors."""
import csv
from datetime import datetime
from uuid import UUID
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import receiver
import validator


class TestHttpValidation(unittest.TestCase):
    def test_upload_matches_validator(self):
        base = ["123456", "1", "1", "12", "12", "45.2", "23.1", "60.5", "512"]
        cases = [base, base[:-1]]
        for index, value in [(2, "3"), (3, "13"), (5, "bad"),
                             (5, "101"), (0, ""), (5, "nan")]:
            fields = base.copy()
            fields[index] = value
            cases.append(fields)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "readings.csv"
            with patch.object(receiver, "CSV_FILE", str(output)):
                receiver._init_csv()
                client = receiver.app.test_client()
                accepted = []
                for fields in cases:
                    with self.subTest(fields=fields):
                        result = validator.validate_record(fields)
                        with patch.object(receiver.validator, "validate_record",
                                          wraps=validator.validate_record) as validate:
                            response = client.post("/upload", data=",".join(fields))
                            validate.assert_called_once_with(fields)
                        if isinstance(result, dict):
                            self.assertEqual(response.status_code, 200)
                            accepted.append(fields)
                        else:
                            self.assertEqual(response.status_code, 400)
                            self.assertIn(result[0].reason, response.get_data(as_text=True))
                with output.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.reader(stream))
                self.assertEqual(rows[0], receiver.CSV_HEADER)
                self.assertEqual([row[3:] for row in rows[1:]], accepted)
                ids = [row[0] for row in rows[1:]]
                self.assertEqual(len(ids), len(set(ids)))
                for row in rows[1:]:
                    self.assertEqual(UUID(row[0]).version, 4)
                    received = datetime.fromisoformat(row[1])
                    accepted_at = datetime.fromisoformat(row[2])
                    self.assertEqual(received.utcoffset().total_seconds(), 0)
                    self.assertEqual(accepted_at.utcoffset().total_seconds(), 0)
                    self.assertLessEqual(received, accepted_at)

    def test_old_header_is_not_appended_to(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "old.csv"
            original = ",".join(receiver.config.CSV_HEADER) + "\n"
            output.write_text(original, encoding="utf-8")
            with patch.object(receiver, "CSV_FILE", str(output)):
                response = receiver.app.test_client().post(
                    "/upload", data="123456,1,1,1,1,45,23,60,512")
                self.assertEqual(response.status_code, 500)
            self.assertEqual(output.read_text(encoding="utf-8"), original)
