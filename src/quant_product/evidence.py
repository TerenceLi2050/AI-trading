"""Build auditable research tables from reviewed index and filing evidence."""

import json
from pathlib import Path
import re
import shutil
import subprocess
from html import unescape

import pandas as pd

from quant_history.pilot import apply_industries
from quant_screen.io import atomic_path, dates, file_lock, read_table, require, sha256_file, stock_codes, write_csv
from quant_screen.quality_study import QUALITY_FIELDS, load_quality_history
from quant_screen.sessions import load_sessions

from .collect import _register, notice_attachments
from .dataset import INPUTS, date_window, verify_source_manifest


COMPONENTS = {"000300": 300, "000905": 500}


def import_sources(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError("导入来源与产品数据目录须分开")
    entries = json.loads((source / "source_manifest.json").read_bytes())["sources"]
    files = {key: name for key, name in INPUTS.items() if key in entries}
    if not files:
        raise ValueError("来源清单没有认识的历史输入")
    reasons = verify_source_manifest(source, files)
    if reasons:
        raise ValueError("；".join(reasons))
    with file_lock(target / ".history_sources.lock", wait_for_collection=True):
        for key, name in files.items():
            entry = entries[key]
            copied = {}
            for relative, digest in entry.get("evidence", {}).items():
                original = (source / relative).resolve()
                if not original.is_relative_to(source):
                    raise ValueError("原件路径超出来源目录")
                relative_output = Path("raw/imported") / digest / original.name
                _copy_verified(original, target / relative_output, digest)
                copied[relative_output.as_posix()] = digest
            _copy_verified(source / name, target / name, entry["sha256"])
            _register(target, key, target / name, entry["source"], copied)
    return sorted(files)


def _copy_verified(source, target, digest):
    if target.is_file() and sha256_file(target) == digest:
        return
    with atomic_path(target) as temp:
        shutil.copyfile(source, temp)
        if sha256_file(temp) != digest:
            raise ValueError(f"来源文件在读取时发生改变：{source.name}")


def _proof(root, entries):
    proof = {}
    for relative in entries:
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"缺来源原件 {relative}")
        proof[path.relative_to(root.resolve()).as_posix()] = sha256_file(path)
    return proof


def rebuild_membership(root, anchor_path, changes_paths, start, end, evidence, industries_path=None, checks_path=None):
    """Apply reviewed parent-index deltas on their effective trading dates."""
    start, end = date_window(start, end)
    schedule = load_sessions(root / INPUTS["sessions"])
    if schedule is None or start not in schedule.index or end not in schedule.index:
        raise ValueError("成分窗口须有独立交易日历")
    sessions = schedule.loc[start:end].query("is_session == 1").index
    if sessions.empty:
        raise ValueError("成分窗口没有交易日")
    proof = _proof(root, [anchor_path, *changes_paths, *evidence])
    if not evidence:
        raise ValueError("须核验锚点和本区间调样公告原件")
    anchor = pd.read_csv(root / anchor_path, dtype={"index": str, "code": str}, encoding="utf-8-sig")
    require(anchor, ["index", "code", "ann_date", "source"], ["code"], "成分锚点")
    anchor["code"] = stock_codes(anchor["code"])
    anchor["ann_date"] = dates(anchor["ann_date"])
    require(anchor, ["code", "ann_date"], ["code"], "标准成分锚点")
    if (anchor.groupby("index").size().to_dict() != COMPONENTS
            or not anchor["ann_date"].lt(sessions[0]).all() or anchor["source"].fillna("").str.strip().eq("").any()):
        raise ValueError("期初成分必须齐全、互斥并先于首个交易日已知")

    fields = ["effective_date", "ann_date", "index", "action", "code", "source"]
    pieces = [pd.read_csv(root / path, dtype={"index": str, "code": str}, encoding="utf-8-sig") for path in changes_paths]
    if any(set(fields) - set(piece) for piece in pieces):
        raise ValueError("调样流水字段不完整")
    changes = pd.concat(pieces, ignore_index=True)
    if not changes.empty:
        changes["code"] = stock_codes(changes["code"])
        for column in ("effective_date", "ann_date"):
            changes[column] = dates(changes[column])
        require(changes, fields, ["effective_date", "index", "action", "code"], "调样流水")
        if (not changes["effective_date"].isin(sessions[1:]).all()
                or not changes["ann_date"].lt(changes["effective_date"]).all()
                or not changes["index"].isin(COMPONENTS).all() or not changes["action"].isin(["in", "out"]).all()
                or changes["source"].fillna("").str.strip().eq("").any()):
            raise ValueError("调样流水含不合法的生效日、公告日或来源")
    pools = {name: set(rows["code"]) for name, rows in anchor.groupby("index")}
    active = {row.code: (sessions[0], row.ann_date, row.source) for row in anchor.itertuples(index=False)}
    checkpoints = {sessions[0]: pools.copy()}
    rows = []
    for effective, day in changes.groupby("effective_date", sort=True):
        before = set.union(*pools.values())
        for index, actions in day.groupby("index"):
            entries = set(actions.loc[actions["action"].eq("in"), "code"])
            exits = set(actions.loc[actions["action"].eq("out"), "code"])
            if not exits <= pools[index] or entries & pools[index]:
                raise ValueError(f"{effective.date()} 调样与 {index} 原成分不一致")
            pools[index] = pools[index] - exits | entries
        after = set.union(*pools.values())
        if len(after) != sum(COMPONENTS.values()) or {k: len(v) for k, v in pools.items()} != COMPONENTS:
            raise ValueError(f"{effective.date()} 调样后父指数数量或重叠异常")
        for code in sorted(before - after):
            rows.append(_interval(code, active.pop(code), effective))
        for item in day.loc[day["action"].eq("in") & day["code"].isin(after - before)].itertuples(index=False):
            active[item.code] = (effective, item.ann_date, item.source)
        checkpoints[effective] = pools.copy()
    rows.extend(_interval(code, state, end + pd.Timedelta(days=1)) for code, state in sorted(active.items()))
    result = pd.DataFrame(rows)
    if checks_path:
        proof.update(_proof(root, [checks_path]))
        _check_membership_snapshots(root / checks_path, checkpoints, sessions)
    if industries_path:
        proof.update(_proof(root, [industries_path]))
        result = apply_industries(result, read_table(root / industries_path))
    with file_lock(root / ".history_sources.lock", wait_for_collection=True):
        output = root / INPUTS["membership"]
        write_csv(output, result.sort_values(["code", "start_date"]))
        _register(root, "membership", output, "reviewed CSI 000300/000905 anchor and effective notices", proof)
    return len(result)


def _check_membership_snapshots(path, checkpoints, sessions):
    observed = pd.read_csv(path, dtype={"index": str, "code": str}, encoding="utf-8-sig")
    fields = ["date", "index", "code", "source"]
    require(observed, fields, ["date", "code"], "核对成分快照")
    observed["code"] = stock_codes(observed["code"])
    observed["date"] = dates(observed["date"])
    require(observed, fields, ["date", "code"], "标准核对成分")
    if not observed["date"].isin(sessions).all() or observed["source"].fillna("").str.strip().eq("").any():
        raise ValueError("核对截面须为实际交易日且有来源")
    event_days = pd.DatetimeIndex(sorted(checkpoints))
    for day, part in observed.groupby("date", sort=True):
        pos = event_days.searchsorted(day, side="right") - 1
        actual = {index: set(rows["code"]) for index, rows in part.groupby("index")}
        if checkpoints[event_days[pos]] != actual:
            raise ValueError(f"{day.date()} 的独立截面与公告重放成分不一致")


def _interval(code, state, end):
    start, known, source = state
    return dict(code=code, start_date=start, end_date=end, ann_date=known, industry=pd.NA, source=source)


def extract_regular_notice(root, identifier, *, effective_date=None, effective_known=None, effective_evidence=None):
    """Extract CSI PDF or exchange-anchored spreadsheet changes."""
    folder = root / "raw/csi/notices" / str(identifier)
    receipt = json.loads((folder / "notice.json").read_bytes())["data"]
    content = unescape(re.sub(r"<[^>]+>", " ", receipt["content"]))
    matching = re.findall(r"(20\d\d)年\s*(\d{1,2})月\s*(\d{1,2})日\s*(收市后|收盘后)?\s*(?:正式)?生效", content)
    announced = pd.Timestamp(receipt["publishDate"])
    evidence = []
    if len(matching) == 1 and effective_date is None:
        year, month, day, after_close = matching[0]
        effective = pd.Timestamp(int(year), int(month), int(day))
    elif not matching and effective_date and effective_known and effective_evidence:
        effective, _ = date_window(effective_date, effective_date)
        observed, _ = date_window(effective_known, effective_known)
        proof = _proof(root, [effective_evidence])
        raw = (root / effective_evidence).read_text(encoding="utf-8-sig")
        confirmed = unescape(re.sub(r"<[^>]+>", " ", raw))
        chinese_date = f"{effective.year}年{effective.month}月{effective.day}日"
        if (not any(value in confirmed for value in (chinese_date, str(effective.date())))
                or str(observed.date()) not in raw or "终止上市" not in confirmed
                or "退市日" not in content):
            raise ValueError("独立交易所公告无法确认调样引用的退市日期")
        announced = max(announced, observed)
        after_close = False
        evidence.extend(proof)
    else:
        raise ValueError("调样公告没有可唯一定位的生效日期；事件相对日期须提交生效公告原件")
    schedule = load_sessions(root / INPUTS["sessions"])
    if schedule is None or effective not in schedule.index:
        raise ValueError("调样公告的生效日缺独立交易日历")
    sessions = schedule.index[schedule["is_session"].eq(1)]
    pos = sessions.searchsorted(effective, side="right" if after_close else "left")
    if pos >= len(sessions) or (not after_close and effective != sessions[pos]):
        raise ValueError("无法定位调样生效后的首个交易日")
    effective = sessions[pos]
    rows = []
    heading = re.compile(r"(?m)^\s*(沪深\s*300|中证\s*\d{3,4})\s*指数样本调整名单[：:]")
    for file_url, filename in notice_attachments(receipt):
        path = folder / filename
        ext = path.suffix
        if not path.is_file():
            continue
        if ext in {".xls", ".xlsx"}:
            pairs = _sheet_pairs(path)
            rows.extend(_notice_row(effective, announced, index, action, code, file_url)
                        for index, action, code in pairs)
            evidence.append(path.relative_to(root).as_posix())
            continue
        if ext != ".pdf":
            continue
        text = _pdf_text(path)
        if text is None:
            continue
        matches = list(heading.finditer(text))
        for i, match in enumerate(matches):
            title = re.sub(r"\s", "", match[1])
            index = {"沪深300": "000300", "中证500": "000905"}.get(title)
            if index is None:
                continue
            stop = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section = text[match.end():stop]
            section = re.split(r"(?:备选|备用)名单[：:]", section, maxsplit=1)[0]
            expected = re.search(rf"{title[:2]}\s*{title[2:]}\s*指数更换\s*(\d+)\s*只样本", content)
            pairs = [re.findall(r"(?<!\d)\d{6}(?!\d)", line) for line in section.splitlines()]
            pairs = [pair for pair in pairs if pair]
            if expected is None or len(pairs) != int(expected[1]) or any(len(pair) != 2 for pair in pairs):
                raise ValueError(f"中证公告 {identifier} 的 {index} 附件条数与正文不符")
            for removed, added in pairs:
                for action, code in (("out", removed), ("in", added)):
                    rows.append(_notice_row(effective, announced, index, action, code, file_url))
            evidence.append(path.relative_to(root).as_posix())
    frame = pd.DataFrame(rows)
    require(frame, ["index", "action", "code"], ["index", "action", "code"], "中证定期调样附件")
    if not set(frame["index"]) <= set(COMPONENTS) or not announced < effective:
        raise ValueError("调样附件父指数异常或公告晚于生效")
    if not matching and not any(code in confirmed and code in content
                                for code in frame.loc[frame["action"].eq("out"), "code"]):
        raise ValueError("交易所退市公告未对应到被调出的证券")
    target = root / "normalized/notices" / f"{identifier}.csv"
    write_csv(target, frame)
    return {"rows": len(frame), "file": target.relative_to(root).as_posix(),
            "evidence": [*sorted(set(evidence)), (folder / "notice.json").relative_to(root).as_posix()]}


def _notice_row(effective, announced, index, action, code, source):
    return {"effective_date": effective, "ann_date": announced, "index": index,
            "action": action, "code": code, "source": source}


def _sheet_pairs(path):
    try:
        table = pd.read_excel(path, header=None, dtype=str)
    except ImportError as error:
        raise ValueError("解析 XLSX 原件需要安装 openpyxl") from error
    if table.shape[1] != 6 or len(table) < 3:
        raise ValueError("调样电子表格布局未识别")
    if table.iloc[0, [0, 2, 4]].tolist() != ["指数代码", "调出", "调入"] or table.iloc[1, [2, 4]].tolist() != ["证券代码", "证券代码"]:
        raise ValueError("调样电子表格表头未识别")
    pairs = []
    for index, _, removed, _, added, _ in table.iloc[2:].itertuples(index=False, name=None):
        if index not in COMPONENTS:
            continue
        for action, symbol in (("out", removed), ("in", added)):
            if pd.isna(symbol) or symbol.strip() in {"", "-"}:
                raise ValueError(f"母指数 {index} 的临时调样缺调入或调出代码")
            code = stock_codes(pd.Series([symbol])).iat[0]
            pairs.append((index, action, code))
    return pairs


def build_financial_versions(root, verified_fields=None):
    """Normalize reviewed fields or a supported filing layout, preserving each published version."""
    index = read_table(root / "filings_index.csv")
    archive_fields = ["code", "title", "announcement_date", "announcement_id", "path", "sha256", "source"]
    require(index, archive_fields, ["announcement_id"], "财报原件目录")
    index["code"] = stock_codes(index["code"])
    index["announcement_date"] = dates(index["announcement_date"])
    index["announcement_id"] = index["announcement_id"].astype(str)
    index["report_period"] = index["title"].map(_period)
    full_revision = (index["title"].str.contains(r"（更正后）|\(更正后\)|（修订版）|\(修订版\)", regex=True, na=False)
                     & ~index["title"].str.contains("摘要|公告", regex=True, na=False))
    revision_keys = set(zip(index.loc[full_revision, "code"], index.loc[full_revision, "report_period"],
                            index.loc[full_revision, "announcement_date"]))
    known = {}
    failures = {}
    versions = []
    overrides = {}
    if verified_fields:
        normalized = pd.read_csv(root / verified_fields, dtype={"announcement_id": str}, encoding="utf-8-sig")
        require(normalized, ["announcement_id", "report_date", *QUALITY_FIELDS], ["announcement_id"], "复核后财务字段")
        overrides = {row["announcement_id"]: row for row in normalized.to_dict("records")}
        if set(overrides) - set(index["announcement_id"]):
            raise ValueError("复核财务字段包含未归档的公告编号")
    for item in index.sort_values(["announcement_date", "announcement_id"]).itertuples(index=False):
        report_date = item.report_period
        if "摘要" in item.title or "英文" in item.title:
            continue
        unclear = pd.isna(report_date)
        correction_note = bool(re.search(r"关于.*更正|更正公告|会计差错", item.title))
        if correction_note:
            if (item.code, report_date, item.announcement_date) not in revision_keys:
                failures[item.announcement_id] = "更正事项缺可定位的同日修订报表"
            continue
        if unclear:
            continue
        identity = (item.code, report_date)
        if item.announcement_date <= report_date:
            failures[item.announcement_id] = "公告日期早于报告期或日期不足"
            continue
        corrections = bool(re.search(r"更正|修订|更新|补充", item.title))
        if corrections and identity not in known:
            failures[item.announcement_id] = "更正公告缺已保存的原始披露"
            continue
        if not corrections:
            known.setdefault(identity, item.announcement_date)
        proof = _proof(root, [item.path])
        if next(iter(proof.values())) != item.sha256:
            raise ValueError("归档财报原件摘要不一致")
        fields = overrides.get(item.announcement_id)
        if fields is None:
            fields = _bank_pdf(root / item.path)
        elif pd.Timestamp(fields["report_date"]) != report_date:
            raise ValueError("复核财务数据报告期与公告标题不一致")
        if fields is None:
            failures[item.announcement_id] = "未识别该报表格式或字段不足"
            continue
        versions.append({"code": item.code, "report_date": report_date, "ann_date": known[identity],
                         "version_known_date": item.announcement_date, "version": item.announcement_id,
                         "source": item.source, "path": item.path, "sha256": item.sha256,
                         **{field: fields[field] for field in QUALITY_FIELDS}})
    if not versions:
        return {"state": "FIELD_REVIEW_REQUIRED", "versions": 0, "unresolved": failures}
    frame = pd.DataFrame(versions)
    key = ["code", "report_date", "version_known_date"]
    ambiguous = frame.duplicated(key, keep=False)
    first_available = frame.groupby(["code", "report_date"])["version_known_date"].transform("min")
    blocked = ambiguous | first_available.ne(frame["ann_date"])
    # A missing intermediate correction invalidates later availability of the older version.
    bad_versions = index.loc[index["announcement_id"].isin(failures)]
    blocked_periods = set(zip(bad_versions["code"], bad_versions["report_period"]))
    unknown_codes = set(bad_versions.loc[bad_versions["report_period"].isna(), "code"])
    blocked |= pd.Series(list(zip(frame["code"], frame["report_date"])), index=frame.index).isin(blocked_periods)
    blocked |= frame["code"].isin(unknown_codes)
    for version in frame.loc[blocked, "version"]:
        failures[version] = "该报告期存在同日版本冲突或未完成的版本链"
    frame = frame.loc[~blocked]
    if frame.empty:
        return {"state": "FIELD_REVIEW_REQUIRED", "versions": 0, "unresolved": failures}
    with file_lock(root / ".history_sources.lock", wait_for_collection=True):
        path = root / INPUTS["quality"]
        with atomic_path(path) as temp:
            frame.to_csv(temp, index=False, encoding="utf-8-sig")
            load_quality_history(temp)
        files = ["filings_index.csv", *frame["path"].unique()]
        if verified_fields:
            files.append(verified_fields)
        _register(root, "quality", path, "CNINFO archived periodic reports", _proof(root, files))
    return {"state": "PARTIAL" if failures else "NORMALIZED", "versions": len(frame), "unresolved": failures}


def _period(title):
    match = re.search(r"(20\d\d)年\s*(?:第)?(一季度|三季度|半年度|半年报|中期报告|年度)", title)
    if match is None:
        return None
    year, kind = match.groups()
    month_day = {"一季度": "03-31", "三季度": "09-30", "半年度": "06-30",
                 "半年报": "06-30", "中期报告": "06-30", "年度": "12-31"}[kind]
    return pd.Timestamp(f"{year}-{month_day}")


def _pdf_text(path, page_limit=None):
    executable = shutil.which("pdftotext")
    if executable is None:
        return None
    span = ["-f", "1", "-l", str(page_limit)] if page_limit is not None else []
    result = subprocess.run([executable, *span, "-layout", "-enc", "UTF-8", str(path), "-"],
                            capture_output=True, timeout=15, check=False)
    if result.returncode:
        return None
    return result.stdout.decode("utf-8")


def _bank_pdf(path):
    text = _pdf_text(path, page_limit=5)
    if text is None:
        return None
    match = re.search(r"1\.1\s+主要会计数据和财务指标(.*?)1\.2\s", text, re.S)
    if not match or "人民币百万元" not in match[1]:
        return None
    block = match[1]
    labels = {
        "roe_ytd_pct": ("加权平均净资产收益率（未年化）", 0),
        "book_per_share": ("归属于本行普通股股东的每股净资产（元/股）", 0),
        "profit_ytd": ("归属于本行股东的净利润", 0),
        "eps_ytd": ("基本/稀释每股收益（元/股）", 0),
        "operating_cash_per_share_ytd": ("每股经营活动产生的现金流量净额（元/股）", 0),
        "profit_growth_pct": ("归属于本行股东的净利润", 2),
        "revenue_growth_pct": ("营业收入", 2),
    }
    values = {}
    for field, (label, pos) in labels.items():
        rows = re.findall(rf"(?m)^\s*{re.escape(label)}[ \t]+([^\r\n]+)", block)
        if len(rows) != 1:
            return None
        numbers = re.findall(r"\(?-?\d[\d,]*(?:\.\d+)?%?\)?", rows[0])
        if len(numbers) != 3 or (field.endswith("_pct") and "%" not in numbers[pos]):
            return None
        value = numbers[pos].replace(",", "").replace("%", "").replace("(", "-").replace(")", "")
        values[field] = float(value) * (1_000_000 if field == "profit_ytd" else 1)
    return values
