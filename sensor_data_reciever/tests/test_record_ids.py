"""Repeated accepted uploads receive distinct receiver UUIDs; no deduplication."""
from uuid import UUID
from hypothesis import given, settings, strategies as st
import tempfile
from pathlib import Path
from unittest.mock import patch
import receiver
from .helpers import read_rows


@given(st.integers(min_value=1, max_value=25))
@settings(max_examples=15, deadline=None)
def test_accepted_uploads_have_unique_uuid4_ids(count):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "readings.csv"
        with patch.object(receiver, "CSV_FILE", str(path)):
            client = receiver.app.test_client()
            for _ in range(count):
                assert client.post("/upload", data="0,7,1,1,1,50,23,60,512").status_code == 200
        rows = read_rows(path)
    assert rows[0] == receiver.CSV_HEADER
    identities = [row[0] for row in rows[1:]]
    assert len(identities) == len(set(identities)) == count
    assert all(UUID(value).version == 4 for value in identities)
