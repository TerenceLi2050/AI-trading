from datetime import datetime
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_product import collect
from quant_product.cli import main
from quant_product.evidence import extract_regular_notice
from quant_screen.sessions import CST


PROJECT = Path(__file__).resolve().parents[1]
SOURCES = PROJECT / "data" / "historical_pilot" / "2024-06" / "sources"


def _last_state(root):
    paths = list(root.glob("reports/product/*/status.json"))
    return json.loads(paths[0].read_bytes())


def test_new_research_reports_gaps_in_an_empty_directory(tmp_path):
    assert main(["--root", str(tmp_path), "research", "--start", "2024-06-14", "--end", "2024-07-01",
                 "--oos-start", "2024-06-17"]) == 2
    status = _last_state(tmp_path)
    assert status["state"] == "BLOCKED"
    assert "缺历史输入 membership.csv" in status["data_audit"]["reasons"]
    assert not (tmp_path / "data/cache").exists()
    assert not list(tmp_path.rglob("candidate.json"))


def test_bao_known_refusal_prevents_new_connection(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "utc_now", lambda: "2026-11-21T09:00:00+00:00")
    monkeypatch.setattr(collect, "download_sessions", lambda *_: pytest.fail("BaoStock called during backoff"))
    result = collect.collect_baostock(tmp_path, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05"))
    assert result["state"] == "BACKOFF"


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="optional CSI PDF extraction tool")
def test_official_pdf_extracted_with_effective_trading_date(tmp_path):
    destination = tmp_path / "raw/csi/notices/15267"
    destination.mkdir(parents=True)
    shutil.copyfile(SOURCES / "csi_notice_15267.json", destination / "notice.json")
    shutil.copyfile(SOURCES / "csi_20240531_adjustments.pdf", destination / "4450.pdf")
    shutil.copyfile(PROJECT / "data/input/sessions.csv", tmp_path / "sessions.csv")
    extracted = extract_regular_notice(tmp_path, 15267)
    changes = pd.read_csv(tmp_path / extracted["file"], dtype={"index": str, "code": str})
    assert len(changes) == 124
    assert changes["effective_date"].eq("2024-06-17").all()
    assert changes.groupby(["index", "action"]).size().tolist() == [12, 12, 50, 50]
    assert "001965" in set(changes.loc[changes["action"].eq("in"), "code"])


def test_model_scores_use_the_exact_training_features_and_peer_pool():
    from sklearn.linear_model import Ridge

    from quant_product.model import FEATURE_COLUMNS, blend_quality, score_ridge
    from quant_screen.models import design

    day = pd.Timestamp("2024-06-17")
    frame = pd.DataFrame(np.arange(3 * len(FEATURE_COLUMNS), dtype=np.float32).reshape(3, -1) / 60,
                         columns=FEATURE_COLUMNS).assign(date=day, code=["000001", "000002", "000003"])
    frame.loc[0, FEATURE_COLUMNS[0]] = np.nan
    values = design(frame, FEATURE_COLUMNS)
    fitted = Ridge(alpha=1, solver="cholesky").fit(values, np.array([0.0, 0.5, 1.0], dtype=np.float32))
    model = {"columns": FEATURE_COLUMNS, "coefficients": fitted.coef_.tolist(), "intercept": float(fitted.intercept_)}
    scored = score_ridge(frame, model, explanations=True)
    np.testing.assert_array_equal(scored["ridge"].to_numpy(), fitted.predict(values))
    qualified = frame.loc[[1], ["date", "code"]].assign(quality_only=.8)
    ranked = blend_quality(qualified, scored, .65)
    assert ranked["model_rank"].iat[0] == pytest.approx(2 / 3)
    assert ranked["model"].iat[0] == pytest.approx(.65 * .8 + .35 * 2 / 3)
    assert scored["model_factors"].str.contains(" / ").all()


def test_candidate_cannot_use_days_seen_at_freeze(tmp_path):
    from quant_product.model import policy_id
    from quant_product.research import validate_candidate
    from quant_screen.config import load_config
    from quant_screen.io import sha256_file, write_json

    config = load_config(PROJECT / "configs/default.toml")
    root = tmp_path / "reports/product/fixture"
    model = root / "candidate.json"
    now = datetime(2024, 6, 18, 16, tzinfo=CST).isoformat()
    write_json(model, {"state": "FROZEN", "policy_sha256": policy_id(config), "created_utc": now,
                       "validation_start": "2024-06-18", "training": {"last_label_exit": "2024-06-17"}})
    write_json(root / "status.json", {"state": "FROZEN", "candidate_sha256": sha256_file(model),
                                    "development_accepted": True})
    with pytest.raises(ValueError, match="冻结日期"):
        validate_candidate(None, config, model, tmp_path, tmp_path)
    assert not (tmp_path / "models/active.json").exists()


def test_comparison_uses_same_schedule_and_both_cost_levels(tmp_path, monkeypatch):
    from quant_product import research
    from quant_screen.config import load_config

    config = load_config(PROJECT / "configs/default.toml")
    config["portfolio"]["top_k"] = 1
    schedule = pd.DatetimeIndex(["2024-06-17", "2024-07-01"])
    grid = pd.MultiIndex.from_product([schedule, ["000001", "000002"]], names=["date", "code"]).to_frame(index=False)
    grid = grid.assign(baseline=[.9, .6, .8, .5], industry=["银行", "地产"] * 2)
    qualified = grid.assign(quality_only=[.7, .8, .85, .65], rules_same_pool=grid["baseline"])
    qualified["quality_mix"] = .65 * qualified["quality_only"] + .35 * qualified["baseline"]
    predictions = grid[["date", "code"]].assign(ridge=[.9, .7, .6, .8])
    calls = []

    def check_schedule(quotes, market, picks, panel, name, settings, limits, cost_multiplier):
        assert pd.DatetimeIndex(picks["date"]).equals(schedule)
        assert picks["code"].str.len().eq(6).all()
        calls.append((name, cost_multiplier))
        return pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(research, "simulate", check_schedule)
    history = SimpleNamespace(quotes=None, market=None)
    features = SimpleNamespace(panel=None)
    result = research._replay_comparison(history, features, (grid, qualified), predictions, schedule, config, tmp_path)
    assert result["state"] == "COMPARED"
    assert set(calls) == {(name, costs) for name in [*research.COHORTS, "model"] for costs in (1, 2)}
    predictions = predictions.loc[predictions["date"].eq(schedule[0])]
    failure = research._replay_comparison(history, features, (grid, qualified), predictions, schedule, config, tmp_path)
    assert failure["state"] == "BLOCKED"
    assert failure["missing_signal_dates"]["model"] == ["2024-07-01"]
