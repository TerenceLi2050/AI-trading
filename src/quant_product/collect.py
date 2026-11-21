"""Archive public historical market data and source catalogues."""

from datetime import datetime
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

import pandas as pd
import requests

from quant_screen.history import download_sessions
from quant_screen.io import atomic_path, dates, file_lock, numeric, read_table, require, sha256_file, stock_codes
from quant_screen.io import utc_now, write_csv, write_json, write_parquet
from quant_screen.providers import BaoStockAccessError, _bars, _baostock, _normalize, _rows
from quant_screen.sessions import CST, load_sessions

from .dataset import INPUTS, date_window, load_quotes


CSI = "https://www.csindex.com.cn/csindex-home"
PRICE_URL = f"{CSI}/perf/index-perf"
NOTICES_URL = f"{CSI}/announcement/queryAnnouncementByVonew"
CNINFO = "https://www.cninfo.com.cn"
CNINFO_FILES = "https://static.cninfo.com.cn"
BAO_RETRY = "2026-11-22T05:45:58+00:00"


def _years(start, end):
    for year in range(start.year, end.year + 1):
        yield max(start, pd.Timestamp(year, 1, 1)), min(end, pd.Timestamp(year, 12, 31))


def _json_response(session, path, method, url, **kwargs):
    if path.is_file():
        return json.loads(path.read_bytes())
    time.sleep(0.25)
    response = session.request(method, url, timeout=20, **kwargs)
    response.raise_for_status()
    return response.json()


def _accepted_json(path, payload):
    if not path.is_file():
        write_json(path, payload)


def collect_benchmarks(session, root, start, end):
    raw = root / "raw/csi"
    coverage = {}
    for code, key, fields in (
        ("000906", "market", {"open": "market_open", "close": "market_close"}),
        ("H00906", "total_return", {"close": "total_return_close"}),
    ):
        blocks, archived = [], {}
        for first, last in _years(start, end):
            path = raw / f"{code}-{first:%Y%m%d}-{last:%Y%m%d}.json"
            payload = _json_response(session, path, "GET", PRICE_URL, params={
                "indexCode": code, "startDate": first.strftime("%Y%m%d"), "endDate": last.strftime("%Y%m%d"),
            })
            if not payload.get("success"):
                raise ValueError(f"中证指数拒绝返回 {code} 的历史行情")
            frame = pd.DataFrame(payload["data"])
            required = ["tradeDate", "indexCode", *fields]
            require(frame, required, ["tradeDate"], f"中证指数 {code}")
            frame["date"] = pd.to_datetime(frame["tradeDate"].astype(str), format="%Y%m%d", errors="raise").astype("datetime64[ns]")
            numeric(frame, fields)
            if not frame["indexCode"].astype("string").str.zfill(6).eq(code).all() or not frame["date"].between(first, last).all():
                raise ValueError(f"中证指数 {code} 返回了非请求序列")
            _accepted_json(path, payload)
            archived[path.relative_to(root).as_posix()] = sha256_file(path)
            blocks.append(frame[["date", *fields]].rename(columns=fields))
        rows = pd.concat(blocks, ignore_index=True).sort_values("date")
        require(rows, ["date", *fields.values()], ["date"], f"合并基准 {code}")
        target = root / INPUTS[key]
        write_csv(target, rows.assign(source=PRICE_URL))
        _register(root, key, target, PRICE_URL, archived)
        coverage[code] = {"rows": len(rows), "first": rows["date"].min().date().isoformat(),
                          "last": rows["date"].max().date().isoformat()}
    return coverage


def _register(root, key, path, provider, evidence=None):
    manifest = root / "source_manifest.json"
    sources = json.loads(manifest.read_bytes())["sources"] if manifest.is_file() else {}
    sources[key] = {"file": path.relative_to(root).as_posix(), "source": provider, "sha256": sha256_file(path)}
    if evidence:
        sources[key]["evidence"] = evidence
    write_json(manifest, {"sources": sources})


def collect_notice_catalog(session, root, start, end):
    """Keep every page of the CSI rebalance catalogue and its attachment index."""
    records = []
    for first, last in _years(start, end):
        page, count, total, seen = 1, 0, None, set()
        while total is None or count < total:
            path = root / "raw/csi" / f"notices-id-{first:%Y%m%d}-{last:%Y%m%d}-{page}.json"
            form = {
                "lang": "cn", "startDate": str(first.date()), "endDate": str(last.date()),
                "page": {"key": "", "page": page, "rows": 100, "sortBy": "id", "order": "desc"},
                "classList": [], "indexList": [],
                "relatedTopics": ["index_rebalance"], "typeList": [],
            }
            payload = _json_response(session, path, "POST", NOTICES_URL, json=form)
            items = payload["data"]
            found = {item["id"] for item in items}
            if (not payload.get("success") or len(found) != len(items) or seen & found
                    or total not in (None, payload["total"]) or not items and payload["total"]
                    or not all(first <= pd.Timestamp(item["publishDate"]) <= last for item in items)):
                raise ValueError(f"中证调样目录分页或返回日期不一致：{first.date()} 至 {last.date()} 第 {page} 页")
            total = payload["total"]
            count += len(items)
            if count > total:
                raise ValueError("中证调样目录条数不一致")
            _accepted_json(path, payload)
            records.extend(items)
            seen.update(found)
            page += 1
    frame = pd.DataFrame(records, columns=["id", "title", "publishDate", "theme", "noticeType", "fileUrl", "fileName"])
    if frame["id"].duplicated().any():
        raise ValueError("中证调样目录跨期重复")
    frame["source"] = frame["id"].map(lambda value: f"{CSI}/announcement/queryAnnouncementById?id={value}")
    write_csv(root / "announcements.csv", frame)
    return len(frame)


class _NoticeLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        href = dict(attrs).get("href")
        if tag == "a" and href:
            self.urls.append(href)


def notice_attachments(data):
    entries = {entry["fileUrl"]: str(int(entry["id"])) for entry in (data.get("enclosureList") or [])}
    html = _NoticeLinks()
    html.feed(data.get("content") or "")
    for url in html.urls:
        entries.setdefault(url, f"legacy-{sha256(url.encode('utf-8')).hexdigest()[:16]}")
    for url, identifier in entries.items():
        parts = urlsplit(url)
        suffix = Path(parts.path).suffix.lower()
        if (parts.scheme == "https" and parts.hostname in {"oss-ch.csindex.com.cn", "www.csindex.com.cn"}
                and suffix in {".pdf", ".xls", ".xlsx", ".csv"}):
            yield url, f"{identifier}{suffix}"


def collect_notice_documents(root, ids):
    pause = root / "raw/csi/notice_access.json"
    with file_lock(root / ".history_sources.lock", wait_for_collection=True), requests.Session() as session:
        if pause.is_file():
            prior = json.loads(pause.read_bytes())
            if pd.Timestamp(utc_now()) < pd.Timestamp(prior["retry_after_utc"]):
                return {"state": "BACKOFF", **prior}
        try:
            return _collect_notice_documents(session, root, ids)
        except requests.HTTPError as error:
            response = error.response
            if response is None or response.status_code not in {401, 403, 418, 429}:
                raise
            now = pd.Timestamp(utc_now())
            record = {"http_status": response.status_code, "request_url": response.url,
                      "observed_utc": now.isoformat(), "retry_after_utc": (now + pd.Timedelta(days=1)).isoformat()}
            write_json(pause, record)
            return {"state": "BACKOFF", **record}


def _collect_notice_documents(session, root, ids):
    details = []
    missing = []
    archived = []
    signatures = {".pdf": b"%PDF-", ".xls": bytes.fromhex("D0CF11E0A1B11AE1"), ".xlsx": b"PK\x03\x04"}
    for identifier in sorted(set(ids)):
        if identifier < 1:
            raise ValueError("中证公告编号必须为正数")
        folder = root / "raw/csi/notices" / str(identifier)
        path = folder / "notice.json"
        url = f"{CSI}/announcement/queryAnnouncementById"
        payload = _json_response(session, path, "GET", url, params={"id": identifier})
        if not payload.get("success") or payload["data"]["id"] != identifier:
            raise ValueError(f"中证公告 {identifier} 的返回编号不符")
        _accepted_json(path, payload)
        archived.append(identifier)
        for file_url, filename in notice_attachments(payload["data"]):
            target = folder / filename
            if not target.is_file():
                time.sleep(.25)
                response = session.get(file_url, timeout=25)
                if response.status_code == 404:
                    missing.append({"id": identifier, "source": file_url, "http_status": 404})
                    continue
                response.raise_for_status()
                signature = signatures.get(target.suffix)
                if not response.content or (signature and not response.content.startswith(signature)):
                    missing.append({"id": identifier, "source": file_url, "reason": "unexpected_file_format"})
                    continue
                with atomic_path(target) as temp:
                    temp.write_bytes(response.content)
            details.append({"id": identifier, "ann_date": payload["data"]["publishDate"], "source": file_url,
                            "path": target.relative_to(root).as_posix(), "sha256": sha256_file(target)})
    return {"state": "PARTIAL" if missing else "COLLECTED", "announcements": archived, "evidence": details, "missing": missing}


def _bao_pending(root):
    retry_after = pd.Timestamp(BAO_RETRY)
    record = root / "raw/baostock/access.json"
    if record.is_file():
        retry_after = max(retry_after, pd.Timestamp(json.loads(record.read_bytes())["retry_after"]))
    return retry_after if pd.Timestamp(utc_now()) < retry_after else None


def collect_baostock(root, start, end, observations=()):
    retry_after = _bao_pending(root)
    if retry_after is not None:
        return {"state": "BACKOFF", "retry_after_utc": retry_after.isoformat()}
    try:
        calendar_path = root / INPUTS["sessions"]
        if not calendar_path.is_file() or not {start, end} <= set(load_sessions(calendar_path).index):
            download_sessions(calendar_path, start, end)
            _register(root, "sessions", calendar_path, "BaoStock query_trade_dates")
        observed = _archive_observations(root, start, end, observations) if observations else []
        member_file = root / INPUTS["membership"]
        if not member_file.is_file():
            return {"state": "MEMBERSHIP_REQUIRED", "message": "需要核验的历史成分决定行情采集范围", "observations": observed}
        members = read_table(member_file)
        require(members, ["code", "start_date", "end_date", "source"], ["code", "start_date"], "采集股票池")
        begins, stops = dates(members["start_date"]), dates(members["end_date"])
        selected = members.loc[begins.le(end) & (stops.isna() | stops.gt(start)), "code"]
        codes = sorted(stock_codes(selected).unique())
        if not codes:
            raise ValueError("采集区间没有可核验成员")
        return {**_collect_shards(root, codes, start, end), "observations": observed}
    except BaoStockAccessError as error:
        retry_after = (pd.Timestamp(utc_now()) + pd.Timedelta(days=1)).isoformat()
        write_json(root / "raw/baostock/access.json", {"code": error.code, "retry_after": retry_after})
        return {"state": "BACKOFF", "retry_after_utc": retry_after, "code": error.code}


def _archive_observations(root, start, end, observations):
    dates_to_query = sorted({str(date_window(day, day)[0].date()) for day in observations})
    if any(not str(start.date()) <= day <= str(end.date()) for day in dates_to_query):
        raise ValueError("历史观察日期须落在采集窗口内")
    folder = root / "raw/baostock/observations"
    outcomes = []
    with _baostock() as source:
        getters = {
            "000300": source.query_hs300_stocks, "000905": source.query_zz500_stocks,
            "industry": source.query_stock_industry,
        }
        for day in dates_to_query:
            for key, getter in getters.items():
                target = folder / f"{key}-{day}.csv"
                rows = read_table(target) if target.is_file() else _rows(getter(date=day))
                require(rows, ["code", "updateDate"], ["code"], f"BaoStock {key} observation")
                valid = dates(rows["updateDate"]).le(pd.Timestamp(day)).all()
                if not target.is_file():
                    write_csv(target, rows)
                outcomes.append({"date": day, "kind": key, "rows": len(rows), "no_future_stamp": bool(valid),
                                 "file": target.relative_to(root).as_posix(), "sha256": sha256_file(target)})
    return outcomes


def _collect_shards(root, codes, start, end):
    folder = root / "raw/baostock/bars" / f"{start:%Y%m%d}-{end:%Y%m%d}"
    folder.mkdir(parents=True, exist_ok=True)

    def verified(code):
        path = folder / f"{code}.parquet"
        receipt = path.with_suffix(".json")
        return path.is_file() and receipt.is_file() and sha256_file(path) == json.loads(receipt.read_bytes())["sha256"]

    pending = [code for code in codes if not verified(code)]
    failures = {}
    if pending:
        with _baostock() as source:
            for code in pending:
                try:
                    raw, adjusted = _bars(source, code, start, end)
                    frame = _normalize(raw, adjusted, pd.DataFrame(), code)
                    if not frame["date"].between(start, end).all():
                        raise ValueError("用户请求范围外的历史行情")
                    path = folder / f"{code}.parquet"
                    write_parquet(path, frame)
                    write_json(path.with_suffix(".json"), {"sha256": sha256_file(path), "acquired_utc": utc_now(),
                                                          "source": "BaoStock query_history_k_data_plus 3/1"})
                except OSError as error:
                    failures[code] = str(error)[:240]
                    break
                except ValueError as error:
                    failures[code] = str(error)[:240]
    files = [folder / f"{code}.parquet" for code in codes if verified(code)]
    if files:
        quotes = pd.concat([load_quotes(path) for path in files], ignore_index=True).sort_values(["date", "code"])
        target = root / INPUTS["quotes"]
        write_parquet(target, quotes)
        _register(root, "quotes", target, "BaoStock query_history_k_data_plus 3/1",
                  {path.relative_to(root).as_posix(): sha256_file(path) for path in files})
    return {"state": "COLLECTED" if len(files) == len(codes) else "PARTIAL", "requested": len(codes),
            "collected": len(files), "failures": failures}


def collect_filings(session, root, codes, start, end):
    stocks_path = root / "raw/cninfo/stocks.json"
    cargo = _json_response(session, stocks_path, "GET", f"{CNINFO}/new/data/szse_stock.json")
    companies = {item["code"]: item["orgId"] for item in cargo["stockList"] if item.get("code") and item.get("orgId")}
    if not companies:
        raise ValueError("巨潮资讯的证券目录为空")
    _accepted_json(stocks_path, cargo)
    result, unresolved = {}, []
    categories = "category_ndbg_szsh;category_bndbg_szsh;category_yjdbg_szsh;category_sjdbg_szsh;"
    url = f"{CNINFO}/new/hisAnnouncement/query"
    for code in codes:
        if code not in companies:
            unresolved.append(code)
            continue
        for first, last in _years(start, end):
            for kind, search, category in (("periodic", "", categories), ("correction", "更正", "")):
                query = {"pageSize": 30, "column": "szse", "tabName": "fulltext", "plate": "",
                         "stock": f"{code},{companies[code]}", "searchkey": search, "category": category,
                         "seDate": f"{first.date()}~{last.date()}", "isHLtitle": "false"}
                prefix = root / "raw/cninfo" / f"{kind}-{code}-{first:%Y%m%d}-{last:%Y%m%d}"
                for row in _filing_pages(session, url, query, prefix, code, first, last):
                    if kind == "correction" and not any(tag in row["announcementTitle"] for tag in
                                                         ("年度报告", "季度报告", "半年报", "季度报", "定期报告", "会计差错")):
                        continue
                    identifier = row["announcementId"]
                    if identifier not in result:
                        result[identifier] = _archive_pdf(session, root, row)
    fields = ["code", "title", "announcement_date", "announcement_id", "source", "path", "sha256"]
    frame = pd.DataFrame(result.values(), columns=fields)
    write_csv(root / "filings_index.csv", frame)
    return {"documents": len(result), "unresolved_security_codes": unresolved, "normalized_financials": "FIELD_REVIEW_REQUIRED"}


def _filing_pages(session, url, query, prefix, code, start, end):
    page, found, total = 1, set(), None
    while total is None or len(found) < total:
        path = prefix.with_name(f"{prefix.name}-{page}.json")
        payload = _json_response(session, path, "POST", url, data={**query, "pageNum": page})
        rows = payload["announcements"] or []
        current = {row["announcementId"] for row in rows}
        if (len(current) != len(rows) or found & current or not rows and payload["totalAnnouncement"]
                or total not in (None, payload["totalAnnouncement"]) or any(row["secCode"] != code for row in rows)
                or any(not start.date() <= _filing_date(row) <= end.date() for row in rows)):
            raise ValueError(f"巨潮资讯的 {code} 公告分页、日期或证券代码不一致")
        found.update(current)
        total = payload["totalAnnouncement"]
        if len(found) > total:
            raise ValueError(f"巨潮资讯的 {code} 公告条数不一致")
        _accepted_json(path, payload)
        yield from rows
        page += 1


def _filing_date(row):
    return datetime.fromtimestamp(int(row["announcementTime"]) / 1000, CST).date()


def _archive_pdf(session, root, row):
    relative = row["adjunctUrl"]
    identifier = str(row["announcementId"])
    if not identifier.isdecimal() or not relative.startswith("finalpage/") or not relative.lower().endswith(".pdf"):
        raise ValueError("公告原件地址或编号不符合巨潮资讯文档格式")
    url = f"{CNINFO_FILES}/{relative}"
    path = root / "raw/cninfo/pdf" / f"{identifier}.pdf"
    if not path.is_file():
        time.sleep(0.25)
        response = session.get(url, timeout=25)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF-"):
            raise ValueError(f"公告 {identifier} 返回的内容不是 PDF")
        with atomic_path(path) as tmp:
            tmp.write_bytes(response.content)
    ann_date = _filing_date(row).isoformat()
    return {"code": row["secCode"], "title": row["announcementTitle"], "announcement_date": ann_date,
            "announcement_id": identifier, "source": url, "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path)}


def collect_history(root: Path, start, end, *, market=True, notices=False, bars=False, filing_codes=(), observations=()):
    start, end = date_window(start, end)
    if end >= pd.Timestamp(datetime.now(CST).date()):
        raise ValueError("历史采集截至日须早于北京时间今天，实时数据使用日常入口")
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / ".history_sources.lock", wait_for_collection=True), requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": f"{CNINFO}/"})
        report = {"created_utc": utc_now(), "start": str(start.date()), "end": str(end.date())}
        jobs = {}
        if market:
            jobs["benchmarks"] = lambda: collect_benchmarks(session, root, start, end)
        if notices:
            jobs["announcements"] = lambda: collect_notice_catalog(session, root, start, end)
        if bars or observations:
            jobs["baostock"] = lambda: collect_baostock(root, start, end, observations)
        if filing_codes:
            codes = sorted(stock_codes(pd.Series(filing_codes)).unique())
            jobs["filings"] = lambda: collect_filings(session, root, codes, start, end)
        for name, job in jobs.items():
            try:
                report[name] = job()
            except (OSError, ValueError, KeyError, requests.RequestException) as error:
                report[name] = {"state": "SOURCE_UNAVAILABLE", "error": str(error)[:400]}
        write_json(root / "collection.json", report)
    return report
