"""Run the independent daily release and publish only an approved model signal."""

from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from quant_screen.cli import main as daily_main
from quant_screen.current import apply_current_membership, apply_current_quality, load_current
from quant_screen.daily import available_releases, verify_release
from quant_screen.features import build_features
from quant_screen.io import dates, file_lock, read_table, sha256_file, utc_now, write_csv, write_json
from quant_screen.market import trade_flags
from quant_screen.protocol import record_prospective
from quant_screen.screen import generate
from quant_screen.sessions import CST, load_sessions, quote_completion

from .collect import _bao_pending
from .model import blend_quality, policy_id, score_ridge


def daily(root, config_path, config, output, source=None):
    output_root = root / config["data"]["output"]
    with file_lock(root / ".product_daily.lock", wait_for_collection=True):
        return _locked_daily(root, config_path, config, output_root, output, source)


def _locked_daily(root, config_path, config, reports, output, source):
    cmd = ["--root", str(root), "--config", str(config_path), "--mode", "daily"]
    if source is not None:
        cmd.extend(["--import-release", str(source.resolve())])
    elif _bao_pending(root / "data/history") is not None:
        # During a known refusal, the existing daily runner can use a verified archived batch offline.
        day = pd.Timestamp(datetime.now(CST).date())
        release, _ = available_releases(root / config["data"]["daily_updates"], day)
        if release is None:
            return {"state": "BLOCKED", "reasons": ["BaoStock 处于退避且尚无可核验日常行情批次"]}
        cmd.extend(["--import-release", str(release.parent)])
    status = daily_main(cmd)
    pointer = reports / "latest_run.json"
    if not pointer.is_file():
        return {"state": "BLOCKED", "reasons": ["日常运行缺状态报告"], "exit_code": status}
    baseline = json.loads(pointer.read_bytes())
    if status != 0 or baseline["state"] != "FUNDAMENTAL_SCREEN":
        return {"state": "BLOCKED", "daily": baseline, "exit_code": status}
    model_state = _model_for_day(root, reports, config, baseline, output)
    result = {"state": "PUBLISHED", "daily": baseline, "model": model_state, "created_utc": utc_now()}
    write_json(output / "status.json", result)
    write_json(reports / "latest_product.json", {"run": str(output), "state": result["state"],
                                               "signal_date": baseline["signal_date"], "model": model_state["state"]})
    return result


def _verified_local(root, filename, digest):
    path = (root / filename).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256_file(path) != digest:
        raise ValueError("日常发布所需的文件不在产品运行目录内或摘要不符")
    return path


def _model_for_day(root, reports, config, pointer, output):
    active = root / "models" / "active.json"
    if not active.is_file():
        return {"state": "RULES_ONLY", "reason": "没有通过独立验收的模型"}
    try:
        record = json.loads(active.read_bytes())
        evidence = _verified_local(root, record["validation"], record["validation_sha256"])
        model_file = _verified_local(root, record["candidate"], record["candidate_sha256"])
        approved = json.loads(evidence.read_bytes())
        model = json.loads(model_file.read_bytes())
        day = pd.Timestamp(pointer["signal_date"])
        if (approved["state"] != "APPROVED" or approved["candidate_sha256"] != record["candidate_sha256"]
                or model["policy_sha256"] != policy_id(config)
                or day <= pd.Timestamp(approved["window_end"])):
            raise ValueError("模型尚未在适用版本和日期通过独立验收")
        screen_run = reports / "runs" / Path(pointer["run"]).name
        original = json.loads((screen_run / "audit.json").read_bytes())
        release = _verified_local(root, original["release_path"], original["manifest_sha256"])
        meta = json.loads(release.read_bytes())
        verify_release(release, meta)
        if not quote_completion([str(release)], day, config["data"]["settled_after_cst"]):
            raise ValueError("当期批次无法确认收盘")
        snapshot, statements = load_current([str(release)], day)
        if snapshot is None or statements is None:
            raise ValueError("当期批次无法确认成分和财务")
        quotes = pd.read_parquet(release.parent / "quotes.parquet")
        market = read_table(release.parent / "benchmark.csv")
        market["date"] = dates(market["date"])
        market = market.set_index("date").sort_index()
        calendar = load_sessions(release.parent / "sessions.csv")
        active_codes = pd.Series(False, index=quotes.index)
        industries = pd.Series(pd.NA, index=quotes.index, dtype="string")
        coverage = pd.DataFrame(index=market.index, columns=["expected", "observed", "coverage"], dtype=float)
        active_codes, industries, coverage = apply_current_membership(quotes, day, active_codes, industries, coverage, snapshot, statements)
        features = build_features(quotes, market, active_codes, industries, coverage, None, config)
        panel = features.panel
        rules = config["eligibility"]
        apply_current_quality(panel, day, snapshot, statements, rules["minimum_cash_conversion"], rules["fundamental_max_age_days"])
        panel["closed_bar_verified"] = active_codes & (trade_flags(quotes)["bar_ok"] | quotes["trade_status"].eq(0))
        session = json.loads((screen_run / "screen.json").read_bytes())["session"]
        _, state, decisions = generate(panel, features.market, day, day, coverage, config, None, session)
        if state["state"] != "FUNDAMENTAL_SCREEN":
            raise ValueError("模型打分前的当期数据门槛未通过")
        eligible = panel.loc[panel["date"].eq(day) & panel["eligible"] & panel["is_st"].eq(0)]
        predicted = score_ridge(eligible, model, explanations=True)
        valid = decisions.loc[decisions["quality_qualified"], ["date", "code", "name", "industry", "quality_score", "baseline"]]
        scored = blend_quality(valid.rename(columns={"quality_score": "quality_only"}), predicted, rules["quality_weight"])
        scored = scored.merge(predicted[["date", "code", "model_factors"]], on=["date", "code"], validate="one_to_one")
        k = config["portfolio"]["top_k"]
        if len(scored) < k:
            raise ValueError("模型可打分候选数量不足")
        picks = scored.sort_values(["model", "code"], ascending=[False, True]).head(k).copy()
        picks.insert(0, "rank", range(1, len(picks) + 1))
        observation = record_prospective(reports, screen_run, state, picks, {
            **original, "source": f"approved_ridge:{record['candidate_sha256']}",
        }, horizon=config["research"]["horizon"], sessions=calendar, variants=picks.assign(variant="model"))
        table = output / "model_picks.csv"
        write_csv(table, picks)
        return {"state": "MODEL_PUBLISHED", "picks": str(table), "sha256": sha256_file(table),
                "candidate_sha256": record["candidate_sha256"], "prospective": observation["status"]}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"state": "RULES_ONLY", "reason": str(error)}
