import numpy as np
import pandas as pd
import pytest

from conftest import make_plant_frame
from plant_ml.data_loader import DataLoadError, load_csv, normalise_frame, parse_timestamps
from plant_ml.validation import ValidationError, validate_frame


def labelled(n=20):
    df = make_plant_frame(n)
    df.insert(1, "session_id", 1)
    df.insert(2, "node_id", "1")
    df.insert(3, "sampling_point", 1)
    df["stress_level"] = "healthy"
    df["stress_type"] = "none"
    return df[["timestamp", "session_id", "node_id", "sampling_point", "plant_id", "soil", "temperature", "humidity",
               "light", "stress_level", "stress_type"]]


def test_valid_frame_passes_all_modes():
    for mode in ("fail", "warn", "quarantine"):
        res = validate_frame(labelled(), mode=mode)
        assert len(res.clean) == 20 and res.report["rows_quarantined"] == 0


def test_out_of_range_values_are_masked_not_dropped_in_warn_mode():
    df = labelled()
    df.loc[3, "temperature"] = 55.0
    df.loc[4, "humidity"] = 5.0
    res = validate_frame(df, mode="warn")
    assert len(res.clean) == 20                         # nothing deleted
    assert np.isnan(res.clean.loc[3, "temperature"])    # masked, and counted:
    assert res.report["sensor_range_violations"]["temperature"] == 1
    assert res.report["sensor_range_violations"]["humidity"] == 1


def test_fail_mode_raises_with_report():
    df = labelled()
    df.loc[0, "soil"] = 101
    with pytest.raises(ValidationError) as e:
        validate_frame(df, mode="fail")
    assert e.value.report["sensor_range_violations"]["soil"] == 1


def test_quarantine_mode_moves_bad_rows_and_keeps_reason():
    df = labelled()
    df.loc[2, "light"] = -5
    res = validate_frame(df, mode="quarantine")
    assert len(res.clean) == 19 and len(res.quarantined) == 1
    assert "out_of_range_light" in res.quarantined["_issues"].iloc[0]


def test_duplicates_are_detected_and_quarantined():
    df = pd.concat([labelled(), labelled().iloc[[5]]], ignore_index=True)
    res = validate_frame(df, mode="warn")
    assert res.report["duplicate_plant_timestamp"] == 1
    assert len(res.clean) == 20 and len(res.quarantined) == 1


def test_bad_plant_id_and_bad_timestamp_are_quarantined():
    df = labelled()
    df.loc[1, "plant_id"] = 17
    df.loc[2, "timestamp"] = pd.NaT
    res = validate_frame(df, mode="warn")
    assert len(res.quarantined) == 2
    assert res.report["other_issue_counts"]["bad_plant_id"] == 1 and res.report["other_issue_counts"]["bad_timestamp"] == 1


def test_missing_sensor_values_are_kept_as_nan_and_reported():
    df = labelled()
    df.loc[[1, 2], "soil"] = np.nan
    res = validate_frame(df, mode="fail")
    assert res.clean["soil"].isna().sum() == 2
    assert res.report["missingness"]["soil"]["count"] == 2


def test_invalid_labels_are_set_to_nan_in_warn_mode_not_guessed():
    df = labelled()
    df.loc[0, "stress_level"] = "catastrophic"
    res = validate_frame(df, mode="warn")
    assert pd.isna(res.clean.loc[0, "stress_level"]) and len(res.clean) == 20


def test_report_contents():
    r = validate_frame(labelled(), mode="warn").report
    for key in ("rows", "plants", "time_range", "missingness", "duplicate_plant_timestamp", "sensor_range_violations",
                "stress_level_counts", "experimental_group_counts"):
        assert key in r
    assert r["experimental_group_source"].startswith("derived")


def test_inference_frame_without_labels_validates():
    res = validate_frame(make_plant_frame(10), mode="fail", require_labels=False)
    assert len(res.clean) == 10


def test_missing_required_column_raises():
    with pytest.raises(ValidationError):
        validate_frame(make_plant_frame(5).drop(columns=["humidity"]), mode="warn", require_labels=False)


def test_timestamp_parsing_normalises_timezones():
    s = pd.Series(["2026-10-01T10:00:00+05:30", "2026-10-01T04:30:00Z", "2026-10-01T10:00:00"])
    p = parse_timestamps(s)
    assert str(p.dt.tz) == "Asia/Kolkata"
    assert p.nunique() == 1                      # same instant; naive assumed Asia/Kolkata
    assert parse_timestamps(pd.Series(["garbage"])).isna().all()


def test_loader_errors_are_clear(tmp_path):
    with pytest.raises(DataLoadError, match="not found"):
        load_csv(tmp_path / "nope.csv")
    p = tmp_path / "bad.csv"
    p.write_text("timestamp,plant_id\n2026-10-01T00:00:00+05:30,1\n")
    with pytest.raises(DataLoadError, match="missing required column"):
        load_csv(p)


def test_loader_sorting_contract(tmp_path, small_df):
    # build_features sorts; loader must not drop or reorder-validate rows silently
    p = tmp_path / "x.csv"
    shuffled = small_df.sample(frac=1.0, random_state=1)
    shuffled.to_csv(p, index=False)
    assert len(load_csv(p)) == len(small_df)
