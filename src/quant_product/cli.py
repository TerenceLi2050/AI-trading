"""Command-line orchestration for the independent product."""

import argparse
from pathlib import Path
from uuid import uuid4

import pandas as pd

from quant_screen.config import load_config
from quant_screen.io import file_lock, utc_now, write_json

from .dataset import audit_history


def main(argv=None):
    parser = argparse.ArgumentParser(description="中证800独立量化选股产品")
    parser.add_argument("--root", type=Path, required=True, help="产品数据目录")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/default.toml")
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect-history", help="独立采集官方基准、调样目录、行情或财报原件")
    _window(collect)
    collect.add_argument("--skip-benchmarks", action="store_true")
    collect.add_argument("--notices", action="store_true")
    collect.add_argument("--bars", action="store_true")
    collect.add_argument("--observe-date", action="append", default=[], help="归档指定日期的成分与行业观察")
    collect.add_argument("--filing-code", action="append", default=[])
    collect.add_argument("--filings", action="store_true", help="按已核验历史成分收集全部公告")
    prices = sub.add_parser("collect-akshare-prices", help="按历史成分或指定代码归档新浪原始和复权日线")
    _window(prices)
    prices.add_argument("--code", action="append", default=[], help="抽样证券代码，可重复指定")
    prices.add_argument("--max-new", type=int, help="本轮最多新采证券数")
    prices.add_argument("--offline", action="store_true", help="只校验及汇总已有分片")
    _window(sub.add_parser("collect-akshare-calendar", help="独立采集并校验沪深交易日历"))
    trade = sub.add_parser("import-akshare-trade-state", help="导入已校验的 BaoStock 原始交易状态")
    trade.add_argument("source", type=Path)
    trade.add_argument("--raw", default="raw_history.parquet")
    trade.add_argument("--manifest", default="raw_history.json")
    _window(sub.add_parser("collect-akshare-trade-state", help="按已核验成分使用 BaoStock 补采前收、交易及 ST 状态"))
    _window(sub.add_parser("assemble-akshare-prices", help="按独立状态核对 AKShare 行情并生成研究标准表"))
    imported = sub.add_parser("import-history", help="导入带来源清单的标准数据及原件")
    imported.add_argument("source", type=Path)
    members = sub.add_parser("build-membership", help="把已复核的母指数锚点和调样流水转为逐日有效区间")
    _window(members)
    members.add_argument("--anchor", required=True, help="相对历史目录的期初锚点")
    members.add_argument("--changes", required=True, action="append", help="相对历史目录的调样流水，可重复指定")
    members.add_argument("--evidence", required=True, action="append", help="相对历史目录的来源原件，可重复指定")
    members.add_argument("--industries", help="相对历史目录的时点行业区间")
    members.add_argument("--checkpoints", help="相对历史目录的独立成分核对截面")
    notices = sub.add_parser("collect-notice", help="下载中证指定调样公告及原始附件")
    notices.add_argument("id", type=int, nargs="+")
    normalize = sub.add_parser("extract-notice", help="从标准调样附件提取调入和调出")
    normalize.add_argument("id", type=int)
    normalize.add_argument("--effective", help="公告引用退市日等外部事件时的已核验生效交易日")
    normalize.add_argument("--effective-known", help="生效日正式公告的日期")
    normalize.add_argument("--effective-evidence", help="相对历史目录的交易所生效公告原件")
    filings = sub.add_parser("build-financials", help="重建逐公告财报字段及发布时间")
    filings.add_argument("--verified-fields", help="相对历史目录的复核字段表")
    _window(sub.add_parser("audit-history", help="检查独立历史输入及逐日覆盖"))
    research = sub.add_parser("research", help="以合格历史数据训练、回测、冻结达标候选")
    _research_window(research)
    bootstrap = sub.add_parser("bootstrap", help="采集或导入、验收并启动独立研究")
    _research_window(bootstrap)
    bootstrap.add_argument("--source", type=Path, help="可选的原始证据与标准表来源包")
    validate = sub.add_parser("validate-model", help="用冻结之后的独立窗口评价并启用达标模型")
    _window(validate)
    validate.add_argument("--candidate", type=Path, required=True)
    daily = sub.add_parser("daily", help="独立日常更新与发布")
    daily.add_argument("--import-release", type=Path, help="导入带摘要的日常行情归档")
    cycle = sub.add_parser("cycle", help="推进模型研究或独立验收，并运行日常更新")
    _cycle_args(cycle)
    cycle.add_argument("--import-release", type=Path)
    watch = sub.add_parser("watch", help="按交易日历持续运行日常更新")
    _cycle_args(watch, required=False)
    args = parser.parse_args(argv)
    if args.command == "collect-akshare-prices" and args.max_new is not None and args.max_new < 1:
        parser.error("--max-new 须为正整数")
    root = args.root.resolve()
    history = root / "data/history"
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.command == "watch":
        from quant_screen.watch import watch_daily

        data = config["data"]
        if bool(args.start) != bool(args.oos_start) or args.source and not args.start:
            parser.error("持续推进研究需要同时提供 --start 与 --oos-start")
        options = ["daily"] if not args.start else ["cycle", "--start", args.start, "--oos-start", args.oos_start]
        if args.source:
            options.extend(["--source", str(args.source.resolve())])
        return watch_daily(lambda: main(["--root", str(root), "--config", str(config_path), *options]),
                           root / data["session_calendar"], data["settled_after_cst"])
    output = root / "reports/product" / f"{pd.Timestamp(utc_now()):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    try:
        state = _run(args, root, history, output, config, config_path)
    except (OSError, ValueError, KeyError) as error:
        state = {"state": "INVALID_INPUT", "error": str(error), "created_utc": utc_now()}
    write_json(output / "status.json", state)
    print(output / "status.json", flush=True)
    return 0 if state["state"] in {"COLLECTED", "READY", "IMPORTED", "BUILT", "NORMALIZED", "FROZEN", "APPROVED", "PUBLISHED"} else 2


def _window(parser):
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)


def _research_window(parser):
    _window(parser)
    parser.add_argument("--oos-start", required=True, help="开发样本外回测起点")


def _cycle_args(parser, required=True):
    parser.add_argument("--start", required=required, help="独立历史窗口起点")
    parser.add_argument("--oos-start", required=required, help="开发样本外起点")
    parser.add_argument("--source", type=Path, help="可选的已核验历史来源包")


def _run(args, root, history, output, config, config_path):
    if args.command == "daily":
        from .publish import daily

        return daily(root, config_path, config, output, args.import_release)
    if args.command == "cycle":
        from .operate import advance
        from .publish import daily

        try:
            research = advance(root, history, output / "research", config, args.start, args.oos_start, args.source)
        except Exception as error:
            research = {"state": "BLOCKED", "error_type": type(error).__name__, "reasons": [str(error)]}
        published = daily(root, config_path, config, output / "daily", args.import_release)
        return {**published, "research": research}
    if args.command == "collect-history":
        from .collect import collect_history
        from .dataset import INPUTS
        from quant_screen.io import read_table
        codes = args.filing_code
        if args.filings:
            codes = [*codes, *read_table(history / INPUTS["membership"])["code"].unique()]
        collected = collect_history(history, args.start, args.end, market=not args.skip_benchmarks,
                                    notices=args.notices, bars=args.bars, filing_codes=codes, observations=args.observe_date)
        issues = {"SOURCE_UNAVAILABLE", "BACKOFF", "MEMBERSHIP_REQUIRED", "PARTIAL"}
        blocked = any(isinstance(item, dict) and item.get("state") in issues for item in collected.values())
        return {"state": "BLOCKED" if blocked else "COLLECTED", **collected}
    if args.command == "collect-akshare-prices":
        from .akshare_prices import collect_prices
        return collect_prices(history, args.start, args.end, codes=args.code, max_new=args.max_new, offline=args.offline)
    if args.command == "collect-akshare-calendar":
        from .akshare_prices import collect_calendar
        return collect_calendar(history, args.start, args.end)
    if args.command == "import-akshare-trade-state":
        from .akshare_prices import register_trade_state
        return register_trade_state(history, args.source, args.raw, args.manifest)
    if args.command == "collect-akshare-trade-state":
        from .akshare_prices import collect_trade_state
        return collect_trade_state(history, args.start, args.end)
    if args.command == "assemble-akshare-prices":
        from .akshare_prices import assemble_prices
        return assemble_prices(history, output, args.start, args.end)
    if args.command == "import-history":
        from .evidence import import_sources
        return {"state": "IMPORTED", "tables": import_sources(args.source, history)}
    if args.command == "build-membership":
        from .evidence import rebuild_membership
        count = rebuild_membership(history, args.anchor, args.changes, args.start, args.end, args.evidence,
                                   args.industries, args.checkpoints)
        return {"state": "BUILT", "rows": count}
    if args.command == "collect-notice":
        from .collect import collect_notice_documents
        return collect_notice_documents(history, args.id)
    if args.command == "extract-notice":
        from .evidence import extract_regular_notice
        return {"state": "BUILT", **extract_regular_notice(history, args.id, effective_date=args.effective,
                                                           effective_known=args.effective_known,
                                                           effective_evidence=args.effective_evidence)}
    if args.command == "build-financials":
        from .evidence import build_financial_versions
        return build_financial_versions(history, args.verified_fields)
    acquisition = None
    if args.command == "bootstrap":
        acquisition = _bootstrap(args, history)
    with file_lock(history / ".history_sources.lock", wait_for_collection=True):
        status, dataset = audit_history(history, output, args.start, args.end, config)
    if args.command == "audit-history":
        return status
    context = {"data_audit": status}
    if acquisition is not None:
        context["acquisition"] = acquisition
    if dataset is None:
        return {"state": "BLOCKED", **context}
    if args.command == "validate-model":
        from .research import validate_candidate

        return {**validate_candidate(dataset, config, args.candidate.resolve(), output, root), **context}
    from .research import develop

    return {**develop(dataset, config, output, args.oos_start), **context}


def _bootstrap(args, history):
    from quant_screen.io import read_table

    from .collect import collect_history
    from .dataset import INPUTS
    from .evidence import build_financial_versions, import_sources

    imported = import_sources(args.source, history) if args.source else []
    has = {key for key, name in INPUTS.items() if (history / name).is_file()}
    codes = ()
    if "membership" in has and "quality" not in has:
        codes = read_table(history / INPUTS["membership"])["code"].unique().tolist()
    result = {"imported": imported}
    if has != set(INPUTS):
        result.update(collect_history(history, args.start, args.end,
                                     market=not {"market", "total_return"} <= has,
                                     notices="membership" not in has, bars=not {"sessions", "quotes"} <= has,
                                     filing_codes=codes))
    if codes and (history / "filings_index.csv").is_file():
        result["financial_versions"] = build_financial_versions(history)
    return result
