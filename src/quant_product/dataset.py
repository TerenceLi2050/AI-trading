"""Audit normalized historical inputs before they reach features and labels."""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from quant_screen.data import PRICE_FIELDS, benchmark_for
from quant_screen.features import build_features
from quant_screen.io import dates, numeric, read_table, require, sha256_file, stock_codes, write_csv, write_json
from quant_screen.market import trade_flags
from quant_screen.pit import membership
from quant_screen.quality_study import load_quality_history
from quant_screen.sessions import CST, load_sessions


INPUTS = {
    "sessions": "sessions.csv",
    "market": "calendar.csv",
    "total_return": "csi800_total_return.csv",
    "membership": "membership.csv",
    "quotes": "quotes.parquet",
    "quality": "quality_history.csv",
}
BAR_FIELDS = [*PRICE_FIELDS, "raw_preclose", "trade_status", "is_st", "limit_up", "limit_down", "vendor_pct"]
KEYS = ["date", "code"]


@dataclass
class History:
    quotes: pd.DataFrame
    market: pd.DataFrame
    members: pd.Series
    industries: pd.Series
    coverage: pd.DataFrame
    financials: pd.DataFrame
    fingerprints: dict

    def features(self, config):
        return build_features(self.quotes, self.market, self.members, self.industries, self.coverage, None, config)


def date_window(start, end):
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if (pd.isna(first) or pd.isna(last) or first.tz is not None or last.tz is not None
            or first.normalize() != first or last.normalize() != last or first > last):
        raise ValueError("历史窗口需要按顺序给定两个 YYYY-MM-DD 日期")
    return first, last


def load_quotes(path):
    quotes = read_table(path)
    required = [*KEYS, *BAR_FIELDS, "price_source"]
    require(quotes, required, KEYS, "标准历史行情")
    quotes["date"] = dates(quotes["date"])
    quotes["code"] = stock_codes(quotes["code"])
    numeric(quotes, BAR_FIELDS)
    require(quotes, required, KEYS, "标准化后的历史行情")
    if (not quotes["price_source"].astype("string").str.strip().fillna("").isin(
            {"baostock_raw_and_hfq", "verified_raw_and_adjusted"}).all()
            or not quotes[["trade_status", "is_st"]].isin([0, 1]).all().all()):
        raise ValueError("历史行情须有可追溯的原始及复权价格和明确的交易状态")
    quotes["snapshot_industry"] = pd.NA
    return quotes.sort_values(KEYS, ignore_index=True)


def load_market(root, start, end):
    schedule = load_sessions(root / INPUTS["sessions"])
    if start not in schedule.index or end not in schedule.index:
        raise ValueError("独立交易日历没有覆盖所请求的自然日区间")
    days = schedule.loc[start:end].query("is_session == 1").index
    if days.empty:
        raise ValueError("历史区间没有交易日")

    def series(name, fields):
        frame = read_table(root / INPUTS[name])
        require(frame, ["date", *fields, "source"], ["date"], name)
        frame["date"] = dates(frame["date"])
        require(frame, ["date", *fields, "source"], ["date"], name)
        frame = frame.set_index("date").reindex(days)
        numeric(frame, fields)
        if not frame[fields].gt(0).all().all() or frame["source"].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} 未完整覆盖独立交易日")
        return frame[fields]

    market = series("market", ["market_open", "market_close"]).join(series("total_return", ["total_return_close"]))
    return benchmark_for(market)


def snapshot_fingerprints(root, paths):
    return {key: sha256_file(root / path) for key, path in paths.items()}


def verify_source_manifest(root, filenames):
    path = root / "source_manifest.json"
    if not path.is_file():
        return ["缺原始来源校验清单 source_manifest.json"]
    record = json.loads(path.read_bytes())
    reasons = []
    entries = record["sources"]
    for key, relative in filenames.items():
        entry = entries.get(key)
        if not entry or entry["file"] != relative or sha256_file(root / relative) != entry["sha256"]:
            reasons.append(f"{relative} 与来源清单不一致")
            continue
        if not entry.get("source"):
            reasons.append(f"{relative} 缺数据源")
        evidence = entry.get("evidence", {})
        if key in {"membership", "quality"} and not evidence:
            reasons.append(f"{relative} 缺公告或供应商原始档案")
        for name, digest in evidence.items():
            original = (root / name).resolve()
            if not original.is_relative_to(root.resolve()) or not original.is_file() or sha256_file(original) != digest:
                reasons.append(f"{relative} 的来源原件无法复核: {name}")
    return reasons


def audit_history(root: Path, output: Path, start, end, config) -> tuple[dict, History | None]:
    start, end = date_window(start, end)
    if end >= pd.Timestamp(datetime.now(CST).date()):
        raise ValueError("历史验收截至日须早于北京时间今天")
    status = {"state": "BLOCKED", "start": str(start.date()), "end": str(end.date()), "reasons": []}
    missing = [path for path in INPUTS.values() if not (root / path).is_file()]
    if missing:
        status["reasons"].extend(f"缺历史输入 {path}" for path in missing)
        write_json(output / "data_audit.json", status)
        return status, None
    status["reasons"].extend(verify_source_manifest(root, INPUTS))
    market = load_market(root, start, end)
    quotes = load_quotes(root / INPUTS["quotes"])
    quotes = quotes.loc[quotes["date"].between(start, end)].reset_index(drop=True)
    if quotes.empty or not quotes["date"].isin(market.index).all():
        raise ValueError("历史行情为空或含独立交易日历之外的日期")
    active, industries, coverage = membership(quotes, market.index, root / INPUTS["membership"])
    industries = industries.str.strip().replace("", pd.NA)
    flags = trade_flags(quotes, market.index)
    normal = flags["bar_ok"] & flags["open_ok"] & ~flags["missing_open_limit"] & flags["is_st_known"]
    suspended = quotes["trade_status"].eq(0)
    n = quotes["date"]
    fields = {
        "valid": active & (normal | suspended),
        "suspended": active & suspended,
        "unusable": active & ~(normal | suspended),
        "industry_known": active & industries.notna(),
    }
    for name, mask in fields.items():
        coverage[name] = n.loc[mask].value_counts().reindex(market.index, fill_value=0)
    coverage["not_observed"] = coverage["expected"] - coverage["observed"]
    coverage["usable_coverage"] = coverage["valid"].div(coverage["expected"].where(coverage["expected"].gt(0)))
    coverage["industry_coverage"] = coverage["industry_known"].div(coverage["expected"].where(coverage["expected"].gt(0)))
    write_csv(output / "daily_input_coverage.csv", coverage.rename_axis("date").reset_index())
    status["sessions"] = len(market)
    status["members_missing_price"] = int(coverage["not_observed"].sum() + coverage["unusable"].sum())
    status["industry_unknown"] = int((coverage["expected"] - coverage["industry_known"]).sum())
    settings = config["data"]
    for mask, reason in [
        (coverage["expected"].ne(800), "逐日有效成分未能核对到 800 只"),
        (~coverage["usable_coverage"].ge(settings["minimum_member_coverage"]), "可用复权价、原始价或交易状态覆盖不足"),
        (~coverage["industry_coverage"].ge(settings["minimum_member_coverage"]), "时点行业覆盖不足"),
    ]:
        if mask.any():
            bad = coverage.index[mask]
            status["reasons"].append(f"{reason}，{len(bad)} 个交易日，{bad[0].date()} 至 {bad[-1].date()}")
    financials = load_quality_history(root / INPUTS["quality"])
    original = financials.groupby(["code", "report_date"])
    if (not original["ann_date"].nunique().eq(1).all()
            or not original["ann_date"].first().eq(original["version_known_date"].min()).all()):
        status["reasons"].append("部分财报版本缺初报或首次公告日不一致")
    status["financial_versions"] = len(financials)
    status["financial_securities"] = financials["code"].nunique()
    known = financials.loc[financials["version_known_date"].lt(market.index[-1]), "code"]
    intervals = read_table(root / INPUTS["membership"])
    first_day, last_day = dates(intervals["start_date"]), dates(intervals["end_date"])
    during = first_day.le(market.index[-1]) & (last_day.isna() | last_day.gt(market.index[0]))
    known_members = known.loc[known.isin(stock_codes(intervals.loc[during, "code"]))].nunique()
    status["members_with_visible_financials_upper_bound"] = known_members
    if known_members < settings["minimum_cross_section"]:
        status["reasons"].append("窗口内有可见财报的成员上限低于最小横截面")
    fingerprints = snapshot_fingerprints(root, INPUTS)
    if (root / "source_manifest.json").is_file():
        fingerprints["sources"] = sha256_file(root / "source_manifest.json")
    status["input_sha256"] = fingerprints
    status["state"] = "READY" if not status["reasons"] else "BLOCKED"
    write_json(output / "data_audit.json", status)
    if status["reasons"]:
        return status, None
    return status, History(quotes, market, active, industries, coverage, financials, fingerprints)
