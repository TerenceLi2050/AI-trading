"""Rebuild and reconcile a bounded historical pilot without the legacy panel."""

from pathlib import Path
import json

import pandas as pd

from quant_screen.history import RAW_HISTORY_FIELDS
from quant_screen.io import dates, file_lock, numeric, read_table, require, sha256_file, stock_codes, utc_now, write_csv, write_parquet
from quant_screen.market import trade_flags
from quant_screen.pit import membership
from quant_screen.providers import BaoStockAccessError, _error
from quant_screen.quality_study import QUALITY_FIELDS, load_quality_history
from quant_screen.sessions import load_sessions
from quant_screen.universe_audit import _checked as checked_shard, _collect_prices


PILOT = Path("data/historical_pilot/2024-06")
COUNTS = {"000300": 300, "000905": 500}
KEYS = ["date", "code"]
PERFORMANCE = "https://www.csindex.com.cn/csindex-home/perf/index-perf"


def _json(path):
    return json.loads(path.read_bytes())


def _checked(root, spec):
    path = root / spec["path"]
    if sha256_file(path) != spec["sha256"]:
        raise ValueError(f"Source checksum mismatch: {spec['path']}")
    return path


def _snapshot(samples, requested):
    rows = samples.loc[samples["requested_date"].eq(requested)].copy()
    rows["code"] = stock_codes(rows["code"].str.split(".").str[-1])
    require(rows, ["index", "updateDate", "code"], ["code"], f"snapshot {requested}")
    stamps = dates(rows["updateDate"])
    if not stamps.le(requested).all() or rows.groupby("index").size().to_dict() != COUNTS:
        raise ValueError(f"Incomplete or future-dated sample at {requested}")
    return {index: set(group["code"]) for index, group in rows.groupby("index")}, stamps.max()


def reconstruct_membership(samples, changes, pilot):
    """Replay the official delta and require exact agreement at both later checkpoints."""
    keys = ["index", "action", "code"]
    require(changes, keys, keys, "CSI notice attachment")
    changes = changes.assign(code=stock_codes(changes["code"]))
    require(changes, keys, keys, "normalized CSI notice attachment")
    expected = {(index, action): count for index, count in {"000300": 12, "000905": 50}.items()
                for action in ("in", "out")}
    if changes.groupby(["index", "action"]).size().to_dict() != expected:
        raise ValueError("Official entries or exits do not match the announced counts")
    base, base_stamp = _snapshot(samples, pilot["anchor"])
    start, effective, end = (pd.Timestamp(pilot[key]) for key in ("start", "effective", "end"))
    announcement = pd.Timestamp(pilot["announcement"])
    if not (base_stamp <= pd.Timestamp(pilot["anchor"]) < start < effective <= end and announcement < effective):
        raise ValueError("Invalid historical anchor or effective date")
    updated = {}
    for index, members in base.items():
        subset = changes.loc[changes["index"].eq(index)]
        entries = set(subset.loc[subset["action"].eq("in"), "code"])
        exits = set(subset.loc[subset["action"].eq("out"), "code"])
        if not exits <= members or entries & members:
            raise ValueError(f"Official delta conflicts with the opening snapshot for {index}")
        updated[index] = (members - exits) | entries
    if len(set.union(*updated.values())) != sum(COUNTS.values()):
        raise ValueError("Official delta creates overlapping sub-index membership")
    for day in (pilot["effective"], pilot["end"]):
        observed, _ = _snapshot(samples, day)
        if updated != observed:
            raise ValueError(f"Official delta fails constituent reconciliation at {day}")

    opening, closing = set.union(*base.values()), set.union(*updated.values())
    records = []
    for members, begin, stop, known, source in (
        (opening - closing, start, effective, base_stamp, pilot["anchor_source"]),
        (opening & closing, start, end + pd.Timedelta(days=1), base_stamp, pilot["anchor_source"]),
        (closing - opening, effective, end + pd.Timedelta(days=1), announcement, pilot["notice_source"]),
    ):
        records.extend(dict(code=code, start_date=begin, end_date=stop, ann_date=known, industry=pd.NA,
                            source=source) for code in sorted(members))
    return pd.DataFrame(records), opening, closing


def apply_industries(members, industries):
    """Split membership at independently documented industry boundaries."""
    if industries.empty:
        return members
    fields = ["code", "start_date", "end_date", "ann_date", "industry", "source"]
    keys = ["code", "start_date"]
    require(industries, fields, keys, "industry intervals")
    industries = industries.copy()
    industries["code"] = stock_codes(industries["code"])
    for column in ("start_date", "end_date", "ann_date"):
        industries[column] = dates(industries[column])
    require(industries, fields, keys, "normalized industry intervals")
    industries = industries.sort_values(keys)
    prior = industries.groupby("code")["end_date"].shift()
    if (industries["start_date"].lt(prior).any()
            or not industries["end_date"].gt(industries["start_date"]).all()
            or not industries["ann_date"].lt(industries["start_date"]).all()
            or industries[["source", "industry"]].fillna("").apply(lambda c: c.str.strip().eq("")).any().any()):
        raise ValueError("Industry intervals overlap or lack dated source evidence")
    classifications = {code: rows for code, rows in industries.groupby("code")}
    segments = []
    for row in members.to_dict("records"):
        cursor = row["start_date"]
        history = classifications.get(row["code"], industries.iloc[:0])
        for item in history.itertuples(index=False):
            begin, end = max(cursor, item.start_date), min(row["end_date"], item.end_date)
            if begin >= end:
                continue
            if begin > cursor:
                segments.append({**row, "start_date": cursor, "end_date": begin})
            segments.append({**row, "start_date": begin, "end_date": end,
                             "ann_date": max(row["ann_date"], item.ann_date), "industry": item.industry,
                             "source": f"{row['source']} | {item.source}"})
            cursor = end
        if cursor < row["end_date"]:
            segments.append({**row, "start_date": cursor})
    return pd.DataFrame(segments, columns=members.columns)


def quality_at(requests, filings):
    """Select the newest available version of each requested period, after its known date."""
    requests = requests.assign(date=requests["date"].astype("datetime64[ns]"),
                               report_date=requests["report_date"].astype("datetime64[ns]"))
    return pd.merge_asof(
        requests.sort_values("date"), filings.sort_values("version_known_date"),
        left_on="date", right_on="version_known_date", by=["code", "report_date"],
        direction="backward", allow_exact_matches=False,
    )


def collect_missing_prices(codes, days, archives, cache, retry_after):
    pending = [code for code in sorted(codes) if not any(checked_shard(folder / f"{code}.parquet") for folder in (archives, cache))]
    if not pending:
        return {"status": "up_to_date", "pending": 0}
    if pd.Timestamp(utc_now()) < pd.Timestamp(retry_after):
        return {"status": "prior_provider_refusal_backoff", "next_allowed_utc": retry_after, "pending": len(pending)}
    with file_lock(cache / ".collection.lock", wait_for_collection=True):
        pending = [code for code in pending if not checked_shard(cache / f"{code}.parquet")]
        try:
            problems = _collect_prices(pending, days[0], days[-1], cache) if pending else {}
        except BaoStockAccessError as exc:
            problems = {"provider": _error(exc)}
        completed = sum(checked_shard(cache / f"{code}.parquet") for code in pending)
    return {"status": "complete" if completed == len(pending) else "partial", "requested": len(pending),
            "completed": completed, "failures": problems}


def _quotes(raw_path, folder, days, codes, files, price_cache):
    raw = pd.read_parquet(raw_path, columns=[*KEYS, *RAW_HISTORY_FIELDS],
                          filters=[("date", ">=", days[0]), ("date", "<=", days[-1]), ("code", "in", sorted(codes))])
    raw["code"] = stock_codes(raw["code"])
    require(raw, [*KEYS, *RAW_HISTORY_FIELDS], KEYS, "independent unadjusted bars")
    frames = []
    for code in sorted(codes):
        path = folder / f"{code}.parquet"
        if not path.exists() and price_cache is not None and checked_shard(price_cache / path.name):
            path = price_cache / path.name
        if not path.exists():
            continue
        digest = sha256_file(path)
        if digest != path.with_suffix(".sha256").read_text(encoding="ascii").strip():
            raise ValueError(f"Historical bar checksum mismatch: {code}")
        rows = pd.read_parquet(path)
        if not rows["code"].eq(code).all() or not rows["price_source"].eq("baostock_raw_and_hfq").all():
            raise ValueError(f"Historical bar provenance mismatch: {code}")
        files[path.name] = digest
        frames.append(rows.loc[rows["date"].isin(days)])
    if not frames:
        raise ValueError("No matching raw-and-adjusted bar archives")
    complete = pd.concat(frames, ignore_index=True)
    complete["code"] = stock_codes(complete["code"])
    require(complete, [*KEYS, *RAW_HISTORY_FIELDS], KEYS, "independent raw and adjusted bars")
    compare = raw.merge(complete[KEYS + RAW_HISTORY_FIELDS], on=KEYS, suffixes=("_raw", "_full"), validate="one_to_one")
    for field in RAW_HISTORY_FIELDS:
        a, b = compare[f"{field}_raw"], compare[f"{field}_full"]
        tolerance = 0 if field in {"trade_status", "is_st"} else 0.011
        if not ((a.isna() & b.isna()) | a.sub(b).abs().le(tolerance)).all():
            raise ValueError(f"Archived raw vs adjusted-source history disagrees for {field}")
    all_raw = (pd.concat([raw, complete[raw.columns]], ignore_index=True)
               .drop_duplicates(KEYS, keep="last").sort_values(KEYS, ignore_index=True))
    if not all_raw["date"].isin(days).all() or not all_raw[["trade_status", "is_st"]].isin([0, 1]).all().all():
        raise ValueError("Historical bars contain invalid dates or trading status")
    return all_raw, complete.sort_values(KEYS, ignore_index=True)


def _benchmarks(price_path, total_path, days):
    def series(path, code, cols):
        rows = pd.DataFrame(_json(path)["data"]) if path.suffix == ".json" else read_table(path)
        require(rows, ["tradeDate", "indexCode", *cols], ["tradeDate"], f"official index {code}")
        if not rows["indexCode"].astype("string").str.zfill(6).eq(code).all():
            raise ValueError(f"Unrecognized benchmark source for {code}")
        rows["date"] = pd.to_datetime(rows["tradeDate"].astype(str), format="%Y%m%d", errors="raise")
        rows = rows.set_index("date").reindex(days).rename_axis("date")
        numeric(rows, cols)
        if not rows[cols].gt(0).all().all():
            raise ValueError(f"Official benchmark is missing sessions for {code}")
        return rows[cols]

    price = series(price_path, "000906", ["open", "close"]).set_axis(["market_open", "market_close"], axis=1)
    total = series(total_path, "H00906", ["close"]).set_axis(["total_return_close"], axis=1)
    return price.join(total).assign(source=PERFORMANCE).reset_index()


def _check_notice(source, specs, scope):
    notice = _json(source["official_notice"])["data"]
    catalog = _json(source["rebalance_search"])
    entries = catalog["data"]
    if (not catalog["success"] or catalog["total"] != len(entries)
            or sorted(item["id"] for item in entries) != specs["rebalance_search"]["reviewed_ids"]
            or notice["publishDate"] != scope["announcement"]
            or len([item for item in entries if item["id"] == notice["id"]]) != 1
            or specs["adjustment_attachment"]["url"] not in {item["fileUrl"] for item in notice["enclosureList"]}):
        raise ValueError("Official notice, attachment or reviewed announcement inventory differs from pilot evidence")


def rebuild(root, output, *, price_cache=None, collect_prices=False):
    evidence = _json(root / PILOT / "evidence.json")
    specs = evidence["sources"]
    source = {key: _checked(root, spec) for key, spec in specs.items()}
    args = evidence["scope"]
    _check_notice(source, specs, args)
    calendar = load_sessions(source["calendar"])
    start, end = pd.Timestamp(args["start"]), pd.Timestamp(args["end"])
    if start not in calendar.index or end not in calendar.index:
        raise ValueError("Source calendar does not span the declared pilot window")
    days = calendar.loc[start:end].query("is_session == 1").index
    if not len(days) or pd.Timestamp(args["effective"]) not in days:
        raise ValueError("Official effective date is not a trading session")
    samples = pd.read_csv(source["observations"], dtype={"index": str, "code": str})
    changes = pd.read_csv(source["changes"], dtype={"index": str, "code": str})
    members, opening, closing = reconstruct_membership(samples, changes, args)
    industries = read_table(source["industries"]) if "industries" in source else pd.DataFrame()
    members = apply_industries(members, industries)
    codes = opening | closing
    archived_bars = root / evidence["adjusted_folder"]
    if price_cache is not None:
        price_cache /= f"{days[0]:%Y%m%d}-{days[-1]:%Y%m%d}"
        if price_cache.resolve() == archived_bars.resolve():
            raise ValueError("Price cache must be separate from the historical archive")
    collection = (collect_missing_prices(codes, days, archived_bars, price_cache, evidence["baostock_retry_after_utc"])
                  if collect_prices else {"status": "offline"})
    bar_hashes = {}
    raw, quotes = _quotes(source["raw"], archived_bars, days, codes, bar_hashes, price_cache)
    benchmark = _benchmarks(source["price_index"], source["total_return"], days)
    filings = load_quality_history(source["filings"])
    by_period = filings.groupby(["code", "report_date"])
    if (not by_period["ann_date"].nunique().eq(1).all()
            or not by_period["ann_date"].first().eq(by_period["version_known_date"].min()).all()):
        raise ValueError("Financial version chains must include and retain their original filing date")
    versions = {spec["version"]: spec for spec in specs.values() if "version" in spec}
    if set(filings["version"]) != set(versions) or not filings["version"].is_unique:
        raise ValueError("Every financial version must refer to a distinct archived filing")
    for item in filings.itertuples(index=False):
        doc = versions[item.version]
        if item.source != doc["url"] or item.version_known_date != pd.Timestamp(doc["known_date"]):
            raise ValueError(f"Financial version date or source does not match its archive: {item.version}")
    filing_index = filings[["code", "report_date", "ann_date", "version_known_date", "version", "source"]].copy()
    for field in ("path", "sha256"):
        filing_index[field] = filing_index["version"].map({key: spec[field] for key, spec in versions.items()})
    window = pd.DataFrame(evidence["filing_checks"])
    window["code"] = stock_codes(window["code"])
    for field in ("date", "report_date"):
        window[field] = dates(window[field])
    if not window["date"].isin(calendar.query("is_session == 1").index).all():
        raise ValueError("Financial checks must use actual trading sessions")
    filing_checks = quality_at(window, filings)
    if not filing_checks["version"].fillna("").eq(filing_checks.pop("expected_version")).all():
        raise ValueError("Historical filing availability failed the publication or correction boundary")

    input_dir, reports = output / "data/input", output / "reports"
    member_path = input_dir / "membership.csv"
    write_csv(member_path, members.sort_values(["code", "start_date"]))
    grid = pd.MultiIndex.from_product([days, sorted(codes)], names=KEYS).to_frame(index=False)
    grid["code"] = stock_codes(grid["code"])
    live, industry, counts = membership(grid, days, member_path)
    if not counts["expected"].eq(sum(COUNTS.values())).all():
        raise ValueError("Reconstructed daily membership count differs from CSI 800")
    daily = grid.loc[live].assign(industry=industry[live])
    full = quotes[KEYS].assign(adjusted_valid=trade_flags(quotes)["bar_ok"])
    daily = (daily.merge(raw[KEYS + ["trade_status", "is_st"]], on=KEYS, how="left", validate="one_to_one")
             .merge(full, on=KEYS, how="left", validate="one_to_one"))
    daily["price_status"] = "raw_only_or_invalid"
    daily.loc[daily["adjusted_valid"].eq(True), "price_status"] = "raw_and_adjusted"
    daily.loc[daily["trade_status"].eq(0), "price_status"] = "suspended"
    daily.loc[daily["trade_status"].isna(), "price_status"] = "missing_or_unverified"
    quality = quality_at(daily[KEYS].assign(report_date=pd.Timestamp(args["report_date"])), filings)
    daily = daily.merge(quality[KEYS].assign(
        filing_visible=quality["version"].notna(), quality_complete=quality[QUALITY_FIELDS].notna().all(axis=1)
    ), on=KEYS, validate="one_to_one")
    coverage = daily.groupby("date").agg(members=("code", "size"), industry_known=("industry", "count"),
                                         filing_visible=("filing_visible", "sum"),
                                         quality_complete=("quality_complete", "sum"))
    price_status = daily.groupby(["date", "price_status"]).size().unstack(fill_value=0)
    columns = ["raw_and_adjusted", "raw_only_or_invalid", "suspended", "missing_or_unverified"]
    coverage = coverage.join(price_status.reindex(columns=columns, fill_value=0))
    coverage["industry_unknown"] = coverage["members"] - coverage["industry_known"]
    write_csv(input_dir / "calendar.csv", benchmark.drop(columns="total_return_close"))
    write_csv(input_dir / "csi800_total_return.csv", benchmark[["date", "total_return_close", "source"]])
    write_csv(input_dir / "sessions.csv", calendar.loc[start:end].reset_index())
    write_csv(input_dir / "quality_history.csv", filings)
    write_parquet(input_dir / "quotes.parquet", quotes)
    write_parquet(reports / "unadjusted_bars.parquet", raw)
    write_csv(reports / "daily_membership.csv", daily.drop(columns="adjusted_valid"))
    write_csv(reports / "daily_coverage.csv", coverage.reset_index())
    write_csv(reports / "filing_versions.csv", filing_index)
    write_csv(reports / "filing_boundary_checks.csv", filing_checks)
    write_csv(reports / "announcement_changes.csv", changes.assign(
        ann_date=args["announcement"], effective_date=args["effective"], source=args["notice_source"],
        attachment_sha256=specs["adjustment_attachment"]["sha256"],
    ))

    blockers = {}
    for key, values in {
        "historical_industry": coverage["industry_unknown"],
        "tradeable_raw_and_adjusted_bars": coverage["raw_only_or_invalid"] + coverage["missing_or_unverified"],
        "complete_ytd_financials": coverage["members"] - coverage["quality_complete"],
    }.items():
        affected = values[values.gt(0)]
        if not affected.empty:
            blockers[key] = {"first": str(affected.index[0].date()), "last": str(affected.index[-1].date()),
                             "member_sessions": int(affected.sum())}
    blockers["evaluation_window"] = {"available_sessions": len(days),
                                     "scope": "P0 pilot without factor warmup or a complete holding window"}
    return {"state": "BLOCKED", "reconstruction": "PASSED", "created_utc": utc_now(), "scope": args, "sessions": len(days),
            "coverage_totals": {key: int(value) for key, value in coverage.sum().items()},
            "price_collection": collection,
            "constituents_in": len(closing - opening), "constituents_out": len(opening - closing),
            "filing_versions": len(filings), "filing_boundary_checks": len(filing_checks),
            "delisting_and_corporate_actions": "not_verified",
            "source_files": specs, "adjusted_shard_hashes": bar_hashes, "blockers": blockers,
            "outputs_sha256": {path.relative_to(output).as_posix(): sha256_file(path)
                               for folder in (input_dir, reports) for path in sorted(folder.iterdir())}}
