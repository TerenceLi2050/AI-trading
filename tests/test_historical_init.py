import json
from pathlib import Path

import pandas as pd
import pytest

from quant_history import pilot
from quant_history.pilot import PILOT, apply_industries, quality_at, reconstruct_membership
from quant_screen.quality_study import load_quality_history


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence():
    return json.loads((ROOT / PILOT / "evidence.json").read_bytes())


def test_official_change_and_snapshot_disagreement(evidence):
    sources = evidence["sources"]
    samples = pd.read_csv(ROOT / sources["observations"]["path"], dtype={"index": str, "code": str})
    changes = pd.read_csv(ROOT / sources["changes"]["path"], dtype={"index": str, "code": str})
    members, opening, closing = reconstruct_membership(samples, changes, evidence["scope"])
    assert len(opening) == len(closing) == 800
    assert len(opening - closing) == len(closing - opening) == 49
    lookup = members.set_index("code")
    assert lookup.at["600167", "end_date"] == pd.Timestamp("2024-06-17")
    assert lookup.at["688475", "start_date"] == pd.Timestamp("2024-06-17")
    assert lookup.at["002202", "start_date"] == pd.Timestamp("2024-06-14")

    bad = samples.copy()
    mask = bad["requested_date"].eq("2024-07-01") & bad["code"].eq("sz.000001")
    bad.loc[mask, "code"] = "sz.009999"
    with pytest.raises(ValueError, match="reconciliation"):
        reconstruct_membership(bad, changes, evidence["scope"])


def test_industry_dates_and_missing_segments():
    start, middle, end = pd.to_datetime(["2024-06-14", "2024-06-17", "2024-06-20"])
    members = pd.DataFrame([{"code": "000001", "start_date": start, "end_date": end,
                             "ann_date": pd.Timestamp("2024-05-27"), "industry": pd.NA, "source": "test anchor"}])
    dated = pd.DataFrame([{"code": "000001", "start_date": middle, "end_date": end,
                           "ann_date": start, "industry": "test industry", "source": "test reference"}])
    rows = apply_industries(members, dated)
    assert rows["start_date"].tolist() == [start, middle]
    assert rows["end_date"].tolist() == [middle, end]
    assert pd.isna(rows["industry"].iat[0])
    assert rows["industry"].iat[1] == "test industry"
    with pytest.raises(ValueError, match="dated source"):
        apply_industries(members, dated.assign(ann_date=middle))


def test_actual_filing_versions_match_future_truncation(evidence):
    path = ROOT / evidence["sources"]["filings"]["path"]
    filings = load_quality_history(path)
    checks = pd.DataFrame(evidence["filing_checks"])
    for name in ("date", "report_date"):
        checks[name] = pd.to_datetime(checks[name])
    checks["code"] = checks["code"].astype(filings["code"].dtype)
    full = quality_at(checks, filings)
    assert full["version"].fillna("").tolist() == full["expected_version"].tolist()
    original_cash = full.loc[full["date"].eq("2024-05-08"), "cash_from_sales_ytd"].iat[0]
    corrected_cash = full.loc[full["date"].eq("2024-05-09"), "cash_from_sales_ytd"].iat[0]
    assert original_cash == pytest.approx(5759491131.39)
    assert corrected_cash == pytest.approx(7922364984.92)
    deadline = pd.Timestamp("2024-05-07")
    past = quality_at(checks.loc[checks["date"].le(deadline)], filings.loc[filings["version_known_date"].le(deadline)])
    pd.testing.assert_frame_equal(full.loc[full["date"].le(deadline)].reset_index(drop=True), past)


def test_prior_provider_refusal_stops_history_download(monkeypatch, tmp_path):
    monkeypatch.setattr(pilot, "utc_now", lambda: "2026-11-21T08:00:00+00:00")
    monkeypatch.setattr(pilot, "_collect_prices", lambda *args: pytest.fail("Provider contacted during backoff"))
    result = pilot.collect_missing_prices({"000001"}, pd.DatetimeIndex(["2024-06-14"]),
                                          tmp_path / "archive", tmp_path / "cache", "2026-11-22T05:45:58+00:00")
    assert result["status"] == "prior_provider_refusal_backoff"
    assert result["pending"] == 1
    assert not (tmp_path / "cache").exists()
