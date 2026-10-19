"""Jetson receipt and approval metadata must precede storage."""
from datetime import datetime, timezone
from unittest.mock import patch
import receiver
from .helpers import read_rows


def test_receipt_validation_approval_write_flush_sync_response_order(
        client, output, valid_fields):
    events = []
    receipt = datetime(2026, 10, 19, 0, 0, 0, tzinfo=timezone.utc)
    approval = datetime(2026, 10, 19, 0, 0, 1, tzinfo=timezone.utc)
    original_validate = receiver.validator.validate_record
    real_open = open
    real_fsync = receiver.os.fsync

    class Clock:
        @classmethod
        def now(cls, zone=None):
            if zone is None:  # Console logging is unrelated to record metadata.
                return approval
            assert zone == timezone.utc
            event, value = ("received_at", receipt) if not events else ("accepted_at", approval)
            events.append(event)
            return value

    def validate(fields):
        events.append("validate")
        return original_validate(fields)

    def identity():
        events.append("record_id")
        return "fixed-id"

    class ObservedFile:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            self.stream.__enter__()
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def write(self, value):
            events.append("write")
            assert value.startswith("fixed-id," + receipt.isoformat(timespec="microseconds")
                                    + "," + approval.isoformat(timespec="microseconds") + ",")
            return self.stream.write(value)
        def flush(self):
            events.append("flush")
            return self.stream.flush()
        def fileno(self):
            return self.stream.fileno()

    def observed_open(path, mode="r", **kwargs):
        stream = real_open(path, mode, **kwargs)
        return ObservedFile(stream) if mode == "a" else stream

    def sync(fd):
        events.append("fsync")
        return real_fsync(fd)

    with patch.object(receiver, "datetime", Clock), \
         patch.object(receiver.validator, "validate_record", side_effect=validate), \
         patch.object(receiver, "uuid4", side_effect=identity), \
         patch.object(receiver, "open", side_effect=observed_open, create=True), \
         patch.object(receiver.os, "fsync", side_effect=sync):
        response = client.post("/upload", data=",".join(valid_fields))
        events.append("response")
    assert response.status_code == 200
    assert events == ["received_at", "validate", "record_id", "accepted_at",
                      "write", "flush", "fsync", "response"]
    assert read_rows(output)[1][3:] == valid_fields


def test_invalid_record_gets_no_approval_or_id(client, output):
    with patch.object(receiver, "datetime") as clock, patch.object(receiver, "uuid4") as identity:
        assert client.post("/upload", data="0,7,2,1,1,nan,23,60,512").status_code == 400
        identity.assert_not_called()
        assert clock.now.call_args_list[0].args == (timezone.utc,)
        # The second now() call is console logging, with no UTC metadata argument.
        assert sum(call.args == (timezone.utc,) for call in clock.now.call_args_list) == 1
    assert len(read_rows(output)) == 1
