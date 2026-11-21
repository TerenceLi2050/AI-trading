from pathlib import Path

import pandas as pd
import pytest
import requests

from quant_product.akshare_prices import SinaTransport, _assemble_prices_locked, _normalize, pair_prices


def _bars(days, *, missing_high=False):
    return pd.DataFrame({
        "date": days, "open": 10.0, "high": 9.0 if missing_high else 10.5, "low": 9.5, "close": 10.0,
        "volume": 100.0, "amount": 1000.0, "turnover": .001, "outstanding_share": 100000.0,
    })


def test_pair_keeps_equal_price_days_and_marks_missing_or_invalid():
    start, end = pd.Timestamp("2024-06-14"), pd.Timestamp("2024-06-19")
    raw = _normalize(_bars(["2024-06-14", "2024-06-17", "2024-06-18"]), start, end)
    tail = _bars(["2024-06-19"], missing_high=True)
    adjusted = _normalize(pd.concat([_bars(["2024-06-14", "2024-06-17"]), tail], ignore_index=True), start, end)
    result = pair_prices(raw, adjusted, "000001")
    assert result["price_pair"].tolist() == [True, True, False, False]
    assert result["pair_issue"].tolist() == ["", "", "missing_hfq", "missing_raw"]
    bad = pair_prices(raw, _normalize(_bars(["2024-06-14"], missing_high=True), start, end), "000001")
    assert bad["pair_issue"].iat[0] == "invalid_price_or_volume"


def test_sina_transport_reuses_only_success_and_surfaces_refusal(monkeypatch):
    monkeypatch.setattr("quant_product.akshare_prices.time.sleep", lambda _: None)
    calls = []

    def get(url):
        calls.append(url)
        response = requests.Response()
        response.status_code = 429 if url.endswith("/blocked") else 200
        response.url = url
        return response

    transport = SinaTransport(get)
    first = transport.request("https://example.test/valid")
    assert transport.request("https://example.test/valid") is first
    with pytest.raises(requests.HTTPError) as refused:
        transport.request("https://example.test/blocked")
    assert refused.value.response.status_code == 429
    assert transport.requests == len(calls) == 2


def test_quote_without_verified_state_never_becomes_canonical(tmp_path: Path):
    day = pd.Timestamp("2024-06-14")
    output = tmp_path / "output"
    bars = _normalize(_bars([str(day.date())]), day, day)
    pair_prices(bars, bars, "000001").to_parquet(tmp_path / "akshare_prices.parquet", index=False)
    pd.DataFrame([{"date": day, "is_session": 1, "source": "test-calendar"}]).to_csv(tmp_path / "sessions.csv", index=False)
    trade = {"date": day, "code": "000002", "raw_open": 10, "raw_high": 10.5, "raw_low": 9.5, "raw_close": 10,
             "raw_preclose": 10, "trade_status": 1, "is_st": 0}
    pd.DataFrame([trade]).to_parquet(tmp_path / "trade_state.parquet", index=False)
    report = _assemble_prices_locked(tmp_path, output, day, day)
    assert report["state"] == "BLOCKED"
    assert pd.read_parquet(output / "akshare_unresolved.parquet")["status"].tolist() == ["STATUS_MISSING"]
    assert not (tmp_path / "quotes.parquet").exists()
