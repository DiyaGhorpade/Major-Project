"""
test_module_boundaries.py — Task 8.2

Verifies module responsibility boundaries by inspecting source text.
Each test reads the source file and asserts that forbidden constructs
are absent, per Requirements 7.1–7.5.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).parent.parent


def _src(filename: str) -> str:
    """Return the source text of a module in the workspace root."""
    return (ROOT / filename).read_text(encoding="utf-8")


# ============================================================
# Requirement 7.1 — Validator boundaries
# ============================================================

class TestValidatorBoundaries:
    """
    THE Validator SHALL contain no calls to open(), no use of the os module,
    no serial I/O, no record_counter variable, and no generate_record_id function.
    """

    def setup_method(self):
        self.src = _src("validator.py")

    def test_no_open_call(self):
        """validator.py must not call open()."""
        assert "open(" not in self.src, \
            "validator.py contains open() — file I/O belongs in storage.py"

    def test_no_os_import(self):
        """validator.py must not import the os module."""
        assert "import os" not in self.src, \
            "validator.py imports 'os' — filesystem access belongs in storage.py"

    def test_no_serial_import(self):
        """validator.py must not import serial (pyserial)."""
        assert "import serial" not in self.src, \
            "validator.py imports 'serial' — serial I/O belongs in receiver.py"

    def test_no_record_counter(self):
        """validator.py must not reference record_counter."""
        assert "record_counter" not in self.src, \
            "validator.py references record_counter — that state belongs in receiver.py"

    def test_no_generate_record_id(self):
        """validator.py must not define or call generate_record_id."""
        assert "generate_record_id" not in self.src, \
            "validator.py references generate_record_id — that belongs in receiver.py"


# ============================================================
# Requirement 7.2 — Storage boundaries
# ============================================================

class TestStorageBoundaries:
    """
    THE Storage SHALL contain no calls to validate_record, no conditional logic
    based on field values, no serial I/O, no record_counter variable, and no
    generate_record_id function.
    """

    def setup_method(self):
        self.src = _src("storage.py")

    def test_no_validate_record(self):
        """storage.py must not call or reference validate_record."""
        assert "validate_record" not in self.src, \
            "storage.py references validate_record — validation belongs in validator.py"

    def test_no_serial_import(self):
        """storage.py must not import serial (pyserial)."""
        assert "import serial" not in self.src, \
            "storage.py imports 'serial' — serial I/O belongs in receiver.py"

    def test_no_record_counter(self):
        """storage.py must not reference record_counter."""
        assert "record_counter" not in self.src, \
            "storage.py references record_counter — that state belongs in receiver.py"

    def test_no_generate_record_id(self):
        """storage.py must not define or call generate_record_id."""
        assert "generate_record_id" not in self.src, \
            "storage.py references generate_record_id — that belongs in receiver.py"

    def test_no_conditional_field_logic(self):
        """
        storage.py must not contain conditional branching on sensor field names.
        The only 'if' statements permitted are file-existence checks.
        Forbidden patterns: 'if soil', 'if temperature', 'if humidity',
        'if light', 'if plant_id', 'if session_id', 'if sampling_point'.
        """
        forbidden_field_conditions = [
            r"\bif\s+soil\b",
            r"\bif\s+temperature\b",
            r"\bif\s+humidity\b",
            r"\bif\s+light\b",
            r"\bif\s+plant_id\b",
            r"\bif\s+session_id\b",
            r"\bif\s+sampling_point\b",
        ]
        for pattern in forbidden_field_conditions:
            assert not re.search(pattern, self.src), \
                f"storage.py contains conditional field logic matching '{pattern}' — " \
                "field-value branching belongs in validator.py"


# ============================================================
# Requirement 7.3 — Receiver boundaries
# ============================================================

class TestReceiverBoundaries:
    """
    THE Receiver SHALL NOT contain any range-check conditions on sensor field
    values and SHALL NOT contain any direct calls to open() for output file writing.
    """

    def setup_method(self):
        self.src = _src("receiver.py")

    def test_no_range_check_less_than_0(self):
        """receiver.py must not hard-code '< 0' range checks."""
        assert "< 0" not in self.src, \
            "receiver.py contains '< 0' range check — range checks belong in validator.py"

    def test_no_range_check_greater_than_100(self):
        """receiver.py must not hard-code '> 100' range checks."""
        assert "> 100" not in self.src, \
            "receiver.py contains '> 100' range check — range checks belong in validator.py"

    def test_no_range_check_less_than_neg40(self):
        """receiver.py must not hard-code '< -40' range checks."""
        assert "< -40" not in self.src, \
            "receiver.py contains '< -40' range check — range checks belong in validator.py"

    def test_no_range_check_greater_than_80(self):
        """receiver.py must not hard-code '> 80' range checks."""
        assert "> 80" not in self.src, \
            "receiver.py contains '> 80' range check — range checks belong in validator.py"

    def test_no_range_check_less_than_1(self):
        """receiver.py must not hard-code '< 1' range checks."""
        assert "< 1" not in self.src, \
            "receiver.py contains '< 1' range check — range checks belong in validator.py"

    def test_no_range_check_greater_than_6(self):
        """receiver.py must not hard-code '> 6' range checks."""
        assert "> 6" not in self.src, \
            "receiver.py contains '> 6' range check — range checks belong in validator.py"

    def test_no_range_check_greater_than_16(self):
        """receiver.py must not hard-code '> 16' range checks."""
        assert "> 16" not in self.src, \
            "receiver.py contains '> 16' range check — range checks belong in validator.py"

    def test_no_range_check_greater_than_1000(self):
        """receiver.py must not hard-code '> 1000' range checks."""
        assert "> 1000" not in self.src, \
            "receiver.py contains '> 1000' range check — range checks belong in validator.py"

    def test_no_direct_open_sensor_data_file(self):
        """receiver.py must not open sensor_data.csv directly."""
        assert "open(config.SENSOR_DATA_FILE" not in self.src, \
            "receiver.py opens SENSOR_DATA_FILE directly — file I/O belongs in storage.py"

    def test_no_direct_open_error_log_file(self):
        """receiver.py must not open error_log.csv directly."""
        assert "open(config.ERROR_LOG_FILE" not in self.src, \
            "receiver.py opens ERROR_LOG_FILE directly — file I/O belongs in storage.py"


# ============================================================
# Requirement 7.4 — Config boundaries
# ============================================================

class TestConfigBoundaries:
    """
    THE Config SHALL contain no functions, no classes, no property accessors,
    and no imports beyond the typing module.
    """

    def setup_method(self):
        self.src = _src("config.py")

    def test_no_function_definitions(self):
        """config.py must not define any functions."""
        assert "def " not in self.src, \
            "config.py contains a function definition — Config must be constants only"

    def test_no_class_definitions(self):
        """config.py must not define any classes."""
        assert "class " not in self.src, \
            "config.py contains a class definition — Config must be constants only"

    def test_no_non_typing_imports(self):
        """
        config.py must not import anything other than (optionally) typing.
        Check that there are no 'import X' or 'from X import' lines where X
        is not 'typing'.
        """
        import_lines = [
            line.strip()
            for line in self.src.splitlines()
            if re.match(r"^\s*(import|from)\s+", line)
        ]
        for line in import_lines:
            # Allow: 'import typing' or 'from typing import ...'
            assert re.match(r"^(import typing|from typing\s+import)\b", line), \
                f"config.py has a non-typing import: '{line}' — Config must be constants only"
