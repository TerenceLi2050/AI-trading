"""Archive Sina daily quotes through AKShare and join independently verified status."""

from datetime import datetime
from importlib import metadata
import json
from pathlib import Path
import time
from unittest.mock import patch
import warnings

import pandas as pd
import requests

from quant_screen.history import collect_raw_history
from quant_screen.io import dates, file_lock, numeric, read_table, require, sha256_file, stock_codes
from quant_screen.io import utc_now, write_csv, write_json, write_parquet
from quant_screen.sessions import CST, load_sessions

from .collect import _bao_pending, _register
from .dataset import INPUTS, KEYS, date_window, load_quotes, verify_source_manifest


AKSHARE_VERSION = "1.20.65"
LEGS = ("open", "high", "low", "close")
RAW = [f"raw_{leg}" for leg in LEGS]
ADJUSTED = [f"adj_{leg}" for leg in LEGS]
BAR_FIELDS = ["date", *LEGS, "volume", "amount", "turnover", "outstanding_share"]
PRICE_FILE = "akshare_prices.parquet"
STATE_FILE = "trade_state.parquet"
PROVIDER = "AKShare stock_zh_a_hist_sina"


def _require_akshare():
    try:
        version = metadata.version("akshare")
    except metadata.PackageNotFoundError as error:
        raise ValueError(f"安装 AKShare {AKSHARE_VERSION} 隔离环境") from error
    if version != AKSHARE_VERSION:
        raise ValueError(f"使用经过样本验证的 AKShare {AKSHARE_VERSION} 隔离环境")
    import akshare as ak

    ak.set_timeout((8, 15))
    return ak


def _read_json(path):
    return json.loads(path.read_bytes())


def _sina_symbol(code):
    return ("sh" if code.startswith("6") else "sz") + code


class SinaTransport:
    def __init__(self, get):
        self.get = get
        self.next_request = 0.0
        self.responses = {}
        self.requests = 0

    def request(self, url, **kwargs):
        if not kwargs and url in self.responses:
            return self.responses[url]
        time.sleep(max(0.0, self.next_request - time.monotonic()))
        self.next_request = time.monotonic() + 0.9
        self.requests += 1
        response = self.get(url, **kwargs)
        if response.status_code == 404:
            return response
        response.raise_for_status()
        if not kwargs:
            self.responses[url] = response
        return response


def _codes(root, start, end, explicit):
    if explicit:
        values = stock_codes(pd.Series(explicit))
        return sorted(values.unique()), {"kind": "explicit_codes"}
    name = INPUTS["membership"]
    if not (root / name).is_file():
        raise ValueError("批量采集需要有来源清单的历史成分表，抽样可直接指定 --code")
    issues = verify_source_manifest(root, {"membership": name})
    if issues:
        raise ValueError("；".join(issues))
    members = read_table(root / name)
    require(members, ["code", "start_date", "end_date"], ["code", "start_date"], "历史成分")
    begins, stops = dates(members["start_date"]), dates(members["end_date"])
    eligible = begins.le(end) & (stops.isna() | stops.gt(start))
    members = members.loc[eligible]
    codes = sorted(stock_codes(members["code"]).unique())
    if not codes:
        raise ValueError("请求区间与已核验成分范围不相交")
    upper = stops.loc[eligible].max() if not stops.loc[eligible].isna().any() else pd.NaT
    return codes, {"kind": "verified_membership_union", "first_interval_start": str(begins.loc[eligible].min().date()),
                   "end_exclusive": str(upper.date()) if pd.notna(upper) else None,
                   "membership_sha256": sha256_file(root / name)}


def _normalize(frame, start, end):
    if frame.empty:
        if not set(frame.columns).issubset([*BAR_FIELDS, "bar_valid"]):
            raise ValueError("空行情返回了未知字段")
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                             **{name: pd.Series(dtype="float64") for name in BAR_FIELDS[1:]},
                             "bar_valid": pd.Series(dtype="bool")})
    require(frame, BAR_FIELDS, ["date"], PROVIDER)
    frame = frame[BAR_FIELDS].copy()
    frame["date"] = dates(frame["date"])
    numeric(frame, BAR_FIELDS[1:])
    require(frame, BAR_FIELDS, ["date"], f"normalized {PROVIDER}")
    bounds = frame["high"].ge(frame[["open", "close", "low"]].max(axis=1))
    bounds &= frame["low"].le(frame[["open", "close"]].min(axis=1))
    if not frame["date"].between(start, end).all():
        raise ValueError("日线包含请求窗口之外的日期")
    frame["bar_valid"] = frame[list(LEGS)].gt(0).all(axis=1) & frame[["volume", "amount"]].ge(0).all(axis=1) & bounds
    return frame.sort_values("date", ignore_index=True)


def pair_prices(raw, adjusted, code):
    left = raw.rename(columns={**dict(zip(LEGS, RAW)), "bar_valid": "raw_valid"})
    right = adjusted[["date", *LEGS, "volume", "amount", "bar_valid"]].rename(columns={
        **dict(zip(LEGS, ADJUSTED)), "volume": "adj_volume", "amount": "adj_amount", "bar_valid": "hfq_valid",
    })
    rows = left.merge(right, on="date", how="outer", indicator=True, validate="one_to_one")
    same_volume = rows["volume"].eq(rows["adj_volume"])
    same_amount = rows["amount"].sub(rows["adj_amount"]).abs().le(.01)
    both = rows["_merge"].eq("both")
    valid = rows["raw_valid"].eq(True) & rows["hfq_valid"].eq(True)
    rows["price_pair"] = both & valid & same_volume & same_amount
    rows["pair_issue"] = rows["_merge"].astype("string").map({
        "left_only": "missing_hfq", "right_only": "missing_raw", "both": "quantity_mismatch",
    }).where(~rows["price_pair"], "")
    rows.loc[both & ~valid, "pair_issue"] = "invalid_price_or_volume"
    rows["code"] = code
    rows["price_source"] = "akshare_sina_observed"
    return rows.drop(columns=["_merge", "adj_volume", "adj_amount"])[
        [*KEYS, *RAW, *ADJUSTED, "volume", "amount", "turnover", "outstanding_share", "price_pair", "pair_issue", "price_source"]
    ]


def _receipt(folder, code, start, end):
    receipt = folder / code / "receipt.json"
    if not receipt.is_file():
        return None
    record = _read_json(receipt)
    if (record.get("version") != AKSHARE_VERSION or record.get("code") != code
            or record.get("start") != str(start.date()) or record.get("end") != str(end.date())):
        return None
    for name in ("raw", "hfq"):
        path = receipt.parent / f"{name}.parquet"
        if not path.is_file() or sha256_file(path) != record["files"].get(name):
            return None
    return receipt.parent


def _raw_pair(folder, code, start, end):
    source = _receipt(folder, code, start, end)
    if source is None:
        return None
    return [_normalize(pd.read_parquet(source / f"{name}.parquet"), start, end) for name in ("raw", "hfq")]


def _download_prices(root, folder, codes, start, end, max_new):
    previous = root / "raw/akshare/access.json"
    if previous.is_file():
        backoff = _read_json(previous)
        if pd.Timestamp(utc_now()) < pd.Timestamp(backoff["retry_after_utc"]):
            return {"state": "BACKOFF", **backoff}
    ak = _require_akshare()
    from akshare.exceptions import InvalidParameterError
    from akshare.utils import http

    transport = SinaTransport(http.get)
    failures = {}
    with patch.object(http, "get", transport.request):
        selected = codes[:max_new]
        for number, code in enumerate(selected, 1):
            transport.responses.clear()
            report = {}
            try:
                with warnings.catch_warnings(record=True) as notices:
                    warnings.simplefilter("always")
                    for label, adjust in (("raw", ""), ("hfq", "hfq")):
                        frame = ak.stock_zh_a_hist_sina(symbol=_sina_symbol(code), start_date=start.strftime("%Y%m%d"),
                                                       end_date=end.strftime("%Y%m%d"), adjust=adjust)
                        frame = _normalize(frame, start, end)
                        path = folder / code / f"{label}.parquet"
                        write_parquet(path, frame)
                        report[label] = sha256_file(path)
                write_json(folder / code / "receipt.json",
                           {"version": AKSHARE_VERSION, "provider": PROVIDER, "code": code,
                            "start": str(start.date()), "end": str(end.date()), "created_utc": utc_now(),
                            "files": report, "warnings": [str(note.message) for note in notices][:10]})
            except InvalidParameterError as error:
                failures[code] = str(error)[:250]
            except requests.HTTPError as error:
                status = error.response.status_code if error.response is not None else None
                if status in {401, 403, 418, 429}:
                    return {**_refusal(previous, status), "http_requests": transport.requests, "failures": failures}
                failures[code] = f"HTTP {status}"
                break
            except (OSError, ValueError, TypeError, SyntaxError, IndexError, KeyError) as error:
                failures[code] = f"{type(error).__name__}: {str(error)[:250]}"
                break
            if number % 25 == 0:
                print(f"AKShare {number}/{len(selected)} codes checked, {transport.requests} HTTP calls", flush=True)
    return {"state": "ATTEMPTED", "http_requests": transport.requests, "failures": failures}


def collect_calendar(root, start, end):
    start, end = date_window(start, end)
    previous = root / "raw/akshare/access.json"
    with file_lock(root / ".history_sources.lock", wait_for_collection=True):
        if previous.is_file():
            backoff = _read_json(previous)
            if pd.Timestamp(utc_now()) < pd.Timestamp(backoff["retry_after_utc"]):
                return {"state": "BACKOFF", **backoff}
        ak = _require_akshare()
        from akshare.utils import http

        transport = SinaTransport(http.get)
        try:
            with patch.object(http, "get", transport.request):
                raw = ak.calendar_trade_date_hist_sina()
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else None
            if status not in {401, 403, 418, 429}:
                raise
            return _refusal(previous, status)
        require(raw, ["trade_date"], ["trade_date"], "Sina trading calendar")
        dates_from_source = pd.DatetimeIndex(dates(raw["trade_date"])).sort_values()
        local = dates_from_source[(dates_from_source >= start) & (dates_from_source <= end)]
        if (dates_from_source[0] > start or dates_from_source[-1] < end
                or dates_from_source.has_duplicates or not (local.dayofweek < 5).all()):
            raise ValueError("新浪交易日历的窗口、日期或唯一性未通过")
        frame = pd.DataFrame({"date": pd.date_range(start, end)})
        frame["is_session"] = frame["date"].isin(dates_from_source).astype(int)
        if not frame["is_session"].any():
            raise ValueError("指定窗口没有已知交易日")
        target = root / INPUTS["sessions"]
        if target.is_file():
            known = load_sessions(target)
            proposed = frame.set_index("date")["is_session"]
            scope = known.index.intersection(proposed.index)
            if not known.loc[scope, "is_session"].eq(proposed.loc[scope]).all():
                raise ValueError("新浪与已核验的交易日历在重叠日期不一致")
            if known.index.min() < start or known.index.max() > end:
                if start in known.index and end in known.index:
                    return {"state": "COLLECTED", "sessions": int(proposed.sum()),
                            "canonical": "unchanged_wider_calendar", "compared_civil_dates": len(scope)}
                raise ValueError("日历扩展须覆盖既有规范日历的整个范围")
        stamp = root / "raw/akshare" / f"calendar-{utc_now().replace(':', '').replace('+', 'Z')}.csv"
        write_csv(stamp, raw)
        source = "AKShare calendar_trade_date_hist_sina"
        write_csv(target, frame.assign(source=source))
        _register(root, "sessions", target, source, {stamp.relative_to(root).as_posix(): sha256_file(stamp)})
        return {"state": "COLLECTED", "sessions": int(frame["is_session"].sum()), "civil_dates": len(frame),
                "start": str(start.date()), "end": str(end.date())}


def _refusal(path, status):
    now = pd.Timestamp(utc_now())
    retry = now + pd.Timedelta(days=1)
    notice = {"http_status": status, "observed_utc": now.isoformat(), "retry_after_utc": retry.isoformat()}
    write_json(path, notice)
    return {"state": "BACKOFF", **notice}


def collect_prices(root: Path, start, end, *, codes=(), max_new=None, offline=False):
    start, end = date_window(start, end)
    if end >= pd.Timestamp(datetime.now(CST).date()):
        raise ValueError("历史采集截至日须早于北京时间今天")
    with file_lock(root / ".history_sources.lock", wait_for_collection=True):
        return _collect_prices_locked(root, start, end, codes, max_new, offline)


def _collect_prices_locked(root, start, end, explicit, max_new, offline):
    codes, scope = _codes(root, start, end, explicit)
    folder = root / "raw/akshare/bars" / f"{AKSHARE_VERSION}-{start:%Y%m%d}-{end:%Y%m%d}"
    pending = [code for code in codes if _receipt(folder, code, start, end) is None]
    access = (_download_prices(root, folder, pending, start, end, max_new)
              if pending and not offline else {"state": "OFFLINE" if offline else "UP_TO_DATE"})
    pieces, receipts, per_code = [], {}, []
    for code in codes:
        pair = _raw_pair(folder, code, start, end)
        if pair is None:
            per_code.append({"code": code, "state": "PENDING", "price_pairs": 0})
            continue
        joined = pair_prices(*pair, code)
        count = int(joined["price_pair"].sum())
        per_code.append({"code": code, "state": "PAIRED" if count else "NO_PAIRED_HISTORY",
                         "raw_rows": len(pair[0]), "hfq_rows": len(pair[1]),
                         "price_pairs": count, "unpaired_rows": len(joined) - count})
        pieces.append(joined)
        for key in ("receipt.json", "raw.parquet", "hfq.parquet"):
            file = folder / code / key
            receipts[file.relative_to(root).as_posix()] = sha256_file(file)
    stats = pd.DataFrame(per_code)
    write_csv(folder / "coverage.csv", stats)
    if pieces and sum(map(len, pieces)):
        path = (root if not explicit else folder) / PRICE_FILE
        write_parquet(path, pd.concat(pieces, ignore_index=True).sort_values(KEYS))
        if not explicit:
            _register(root, "akshare_prices", path, PROVIDER, receipts)
    report = {"state": "PARTIAL", "scope": scope, "start": str(start.date()), "end": str(end.date()),
              "requested_codes": len(codes), "codes_with_pairs": int(stats["state"].eq("PAIRED").sum()),
              "price_pairs": int(stats["price_pairs"].sum()), "pending": int(stats["state"].eq("PENDING").sum()),
              "access": access, "coverage": (folder / "coverage.csv").relative_to(root).as_posix(),
              "trade_status": "INDEPENDENT_EVIDENCE_REQUIRED"}
    if report["codes_with_pairs"] == len(codes) and int(stats["unpaired_rows"].sum()) == 0:
        report["state"] = "COLLECTED"
    write_json(folder / "collection.json", report)
    return report


def register_trade_state(root, source, raw_name, manifest_name):
    """Import a hashed BaoStock unadjusted history archive into the product store."""
    source = Path(source).resolve()
    raw = source / raw_name
    manifest = source / manifest_name
    record = _read_json(manifest)
    if record.get("source") != "BaoStock unadjusted historical bars" or sha256_file(raw) != record.get("sha256"):
        raise ValueError("交易状态来源清单与 BaoStock 原始行情摘要不匹配")
    state = read_table(raw)
    required = [*KEYS, *RAW, "raw_preclose", "trade_status", "is_st"]
    require(state, required, KEYS, "独立交易状态")
    state["date"] = dates(state["date"])
    state["code"] = stock_codes(state["code"])
    numeric(state, required[2:])
    require(state, required, KEYS, "标准化独立交易状态")
    if not state[["trade_status", "is_st"]].isin([0, 1]).all().all():
        raise ValueError("交易及 ST 状态必须是来源明确的二元值")
    target = root / STATE_FILE
    with file_lock(root / ".history_sources.lock", wait_for_collection=True):
        custody = root / "raw/verified_trade_state"
        copy = custody / f"{record['sha256']}.parquet"
        from .evidence import _copy_verified

        _copy_verified(raw, copy, record["sha256"])
        manifest_copy = copy.with_suffix(".json")
        _copy_verified(manifest, manifest_copy, sha256_file(manifest))
        write_parquet(target, state[required])
        _register(root, "trade_state", target, record["source"],
                  {item.relative_to(root).as_posix(): sha256_file(item) for item in (copy, manifest_copy)})
    return {"state": "IMPORTED", "rows": len(state), "securities": state["code"].nunique()}


def collect_trade_state(root, start, end):
    start, end = date_window(start, end)
    retry_after = _bao_pending(root)
    if retry_after is not None:
        return {"state": "BACKOFF", "retry_after_utc": retry_after.isoformat()}
    codes, scope = _codes(root, start, end, ())
    folder = root / "raw/bao_trade_state" / f"{start:%Y%m%d}-{end:%Y%m%d}"
    name = "raw_history.parquet"
    with file_lock(folder / ".state_collection.lock", wait_for_collection=True):
        report = collect_raw_history(codes, start, end, folder / name)
        if report["status"] != "published":
            return {"state": "BLOCKED", "scope": scope, "collection": report}
        return {**register_trade_state(root, folder, name, "raw_history.json"), "scope": scope, "collection": report}


def assemble_prices(root, output, start, end):
    start, end = date_window(start, end)
    with file_lock(root / ".history_sources.lock", wait_for_collection=True):
        absent = [name for name in (PRICE_FILE, STATE_FILE, INPUTS["sessions"]) if not (root / name).is_file()]
        if absent:
            return {"state": "BLOCKED", "reasons": [f"缺独立报价或状态输入 {name}" for name in absent]}
        sources = {"akshare_prices": PRICE_FILE, "trade_state": STATE_FILE, "sessions": INPUTS["sessions"]}
        if (root / INPUTS["membership"]).is_file():
            sources["membership"] = INPUTS["membership"]
        issues = verify_source_manifest(root, sources)
        if issues:
            return {"state": "BLOCKED", "reasons": issues}
        return _assemble_prices_locked(root, output, start, end)


def _assemble_prices_locked(root, output, start, end):
    schedule = load_sessions(root / INPUTS["sessions"])
    if start not in schedule.index or end not in schedule.index:
        raise ValueError("交易日历未覆盖组装窗口")
    days = schedule.loc[start:end].query("is_session == 1").index
    quotes = read_table(root / PRICE_FILE)
    quotes = quotes.loc[quotes["date"].between(start, end)]
    trades = read_table(root / STATE_FILE)
    trades = trades.loc[trades["date"].isin(days) & trades["code"].isin(quotes["code"])]
    require(quotes, KEYS + RAW + ADJUSTED, KEYS, "AKShare 报价")
    off_calendar = quotes.loc[~quotes["date"].isin(days), [*KEYS, "pair_issue"]].assign(status="OUTSIDE_CALENDAR")
    quotes = quotes.loc[quotes["date"].isin(days)]
    joined = quotes.merge(trades, on=KEYS, how="outer", suffixes=("", "_verified"), validate="one_to_one")
    comparable = joined[RAW].sub(joined[[f"{name}_verified" for name in RAW]].set_axis(RAW, axis=1)).abs().le(.011).all(axis=1)
    observed = joined["price_pair"].eq(True)
    active = joined["trade_status"].eq(1)
    suspended = joined["trade_status"].eq(0)
    traded = joined["volume"].gt(0) & joined["amount"].gt(0)
    accepted_active = observed & comparable & active & traded & joined["raw_preclose"].gt(0) & joined["is_st"].isin([0, 1])
    accepted_suspension = suspended & ~joined["volume"].gt(0) & joined["is_st"].isin([0, 1])
    joined["status"] = "STATUS_MISSING"
    joined.loc[active & ~observed, "status"] = "PAIRED_PRICE_MISSING"
    joined.loc[active & observed & ~accepted_active, "status"] = "PRICE_OR_STATE_MISMATCH"
    joined.loc[suspended & ~accepted_suspension, "status"] = "SUSPENSION_CONFLICT"
    joined.loc[accepted_active, "status"] = "VERIFIED_TRADING"
    joined.loc[accepted_suspension, "status"] = "VERIFIED_SUSPENSION"
    for name in RAW:
        joined.loc[accepted_suspension, name] = joined.loc[accepted_suspension, f"{name}_verified"]
    valid = joined.loc[accepted_active | accepted_suspension, [
        *KEYS, *RAW, *ADJUSTED, "volume", "amount", "turnover", "outstanding_share", "raw_preclose", "trade_status", "is_st",
    ]].sort_values(KEYS)
    fields = [*KEYS, "status", "pair_issue"]
    assessed = pd.concat([joined[fields], off_calendar[fields]], ignore_index=True)
    accepted = {"VERIFIED_TRADING", "VERIFIED_SUSPENSION"}
    missing_members = _report_members(root, days, assessed, output)
    write_csv(output / "akshare_reconciliation.csv", assessed.groupby(["date", "status"], sort=True).size().rename("rows").reset_index())
    write_parquet(output / "akshare_unresolved.parquet", assessed.loc[~assessed["status"].isin(accepted)])
    if valid.empty:
        return {"state": "BLOCKED", "reasons": ["没有行情与交易状态交叉核对合格的记录"]}
    for key in ("total_shares", "limit_down", "limit_up", "vendor_pct"):
        valid[key] = float("nan")
    valid["price_source"] = "verified_raw_and_adjusted"
    target = root / INPUTS["quotes"]
    staged = output / "verified_akshare_quotes.parquet"
    write_parquet(staged, valid)
    load_quotes(staged)
    canonical = "REGISTERED"
    if target.is_file() and sha256_file(target) != sha256_file(staged):
        canonical = "EXISTING_CANONICAL_PRESERVED"
    else:
        if not target.is_file():
            write_parquet(target, valid)
        _register(root, "quotes", target, f"{PROVIDER} + verified BaoStock raw and venue state", _preserve_quote_evidence(root))
    counts = {str(key): int(value) for key, value in assessed["status"].value_counts().items()}
    unverified = len(assessed) - len(valid)
    return {"state": "PARTIAL" if unverified or missing_members else "BUILT", "verified_rows": len(valid),
            "unresolved_rows": unverified, "status_counts": counts,
            "member_sessions_without_paired_input": missing_members, "canonical": canonical}


def _preserve_quote_evidence(root):
    from .evidence import _copy_verified

    entries = _read_json(root / "source_manifest.json")["sources"]
    proof = {}
    for name in ("akshare_prices", "trade_state", "sessions"):
        entry = entries[name]
        rel = Path("raw/verified_quotes") / entry["sha256"] / entry["file"]
        _copy_verified(root / entry["file"], root / rel, entry["sha256"])
        proof[rel.as_posix()] = entry["sha256"]
        proof.update(entry.get("evidence", {}))
    return proof


def _report_members(root, days, assessed, output):
    path = root / INPUTS["membership"]
    if not path.is_file():
        return 0
    members = read_table(path)
    members["code"] = stock_codes(members["code"])
    first, last = dates(members["start_date"]), dates(members["end_date"])
    positions = days.searchsorted(first)
    stops = days.searchsorted(last.fillna(pd.Timestamp.max.normalize()))
    rows = [pd.DataFrame({"date": days[i:j], "code": code})
            for code, i, j in zip(members["code"], positions, stops) if i < j]
    if not rows:
        return 0
    frame = pd.concat(rows, ignore_index=True)
    require(frame, KEYS, KEYS, "逐日成分")
    frame = frame.merge(assessed[KEYS + ["status", "pair_issue"]], on=KEYS, how="left", validate="one_to_one")
    frame["status"] = frame["status"].fillna("NO_PRICE_OR_STATE")
    frame["pair_issue"] = frame["pair_issue"].fillna("")
    out = frame.loc[~frame["status"].isin(["VERIFIED_TRADING", "VERIFIED_SUSPENSION"])]
    write_parquet(output / "akshare_member_gaps.parquet", out)
    write_csv(output / "akshare_member_daily_coverage.csv",
              frame.groupby(["date", "status"]).size().rename("rows").reset_index())
    return len(out)
