"""Common-date rule comparisons, rolling training and frozen Ridge inference."""

from datetime import datetime
import json
from math import ceil

import numpy as np
import pandas as pd

from quant_screen.backtest import performance, simulate
from quant_screen.io import file_lock, sha256_file, utc_now, write_csv, write_json
from quant_screen.labels import executable_labels
from quant_screen.models import samples, train, training_sample, walk_forward
from quant_screen.quality_study import historical_quality_signals
from quant_screen.screen import diversified_screen
from quant_screen.sessions import CST
from quant_screen.validation import deflated_sharpe_and_pbo, rank_ic

from .model import blend_quality, policy_id, score_ridge


COHORTS = ["quality_mix", "quality_mix_diversified", "quality_only", "price_quality_pool", "price_all"]
VALIDATION_SESSIONS = 126


def scheduled_signals(days, start, settings):
    position = days.searchsorted(start)
    stop = len(days) - settings["horizon"] - 1
    return days[position:stop:settings["rebalance_every"]]


def _rank(rows, name, top_k):
    ranked = rows[["date", "code", name]].dropna(subset=[name]).sort_values(
        ["date", name, "code"], ascending=[True, False, True], kind="stable")
    return ranked.groupby("date", sort=False).head(top_k).copy()


def _rule_inputs(history, features, dates, config, output):
    panel = features.panel.loc[features.panel["date"].isin(dates)]
    eligible = panel.loc[panel["eligible"] & panel["is_st"].eq(0)]
    ranks, coverage = historical_quality_signals(panel, history.financials, config)
    write_csv(output / "rule_coverage.csv", coverage)
    counts = ranks.groupby("date").size() if not ranks.empty else pd.Series(dtype=int)
    missing = counts.reindex(dates, fill_value=0).lt(config["portfolio"]["top_k"])
    if missing.any() or eligible.empty:
        return None, {"state": "BLOCKED", "reasons": ["计划调仓日缺足够的时点财务或量价截面"],
                      "missing_signal_dates": [str(day.date()) for day in dates[missing]]}
    return (eligible, ranks), None


def _replay_comparison(history, features, rule_inputs, predictions, dates, config, output):
    settings, portfolio = config["research"], config["portfolio"]
    top_k = portfolio["top_k"]
    eligible, ranks = rule_inputs
    weight = config["eligibility"]["quality_weight"]
    plain = ranks.rename(columns={"rules_same_pool": "price_quality_pool"})
    picks = {name: _rank(plain, name, top_k) for name in ("quality_mix", "quality_only", "price_quality_pool")}
    picks["price_all"] = _rank(eligible.rename(columns={"baseline": "price_all"}), "price_all", top_k)
    diversified = []
    for _, section in plain.groupby("date", sort=True):
        decisions = section.rename(columns={"quality_only": "quality_score"}).assign(name=section["code"], quality_qualified=True)
        ranked = diversified_screen(decisions, config).rename(columns={"final_score": "quality_mix_diversified"})
        diversified.append(ranked[["date", "code", "quality_mix_diversified"]])
    picks["quality_mix_diversified"] = pd.concat(diversified, ignore_index=True)
    picks["model"] = _rank(blend_quality(plain, predictions, weight), "model", top_k)
    if "ensemble" in predictions:
        trial = blend_quality(plain, predictions[["date", "code", "ensemble"]].rename(columns={"ensemble": "ridge"}), weight)
        picks["ensemble_model"] = _rank(trial.rename(columns={"model": "ensemble_model"}), "ensemble_model", top_k)
    missing = {}
    for name, frame in picks.items():
        counts = frame.groupby("date").size().reindex(dates, fill_value=0)
        missing[name] = [str(day.date()) for day in counts.index[counts.lt(top_k)]]
    if any(missing.values()):
        return {"state": "BLOCKED", "reasons": ["计划调仓日缺合格名单，禁止仅挑可评分日期回测"],
                "missing_signal_dates": {name: gaps for name, gaps in missing.items() if gaps}}

    paths, ledgers, stats = {}, [], {}
    for name in [*COHORTS, *(key for key in picks if key not in COHORTS)]:
        picks_for_strategy = picks[name]
        for multiplier in (1, 2):
            variant = name if multiplier == 1 else f"{name}_2x"
            limits = {**portfolio, "keep_rank": top_k,
                      "max_industry_names": portfolio["max_industry_names"] if name == "quality_mix_diversified" else top_k}
            nav, fills = simulate(history.quotes, history.market, picks_for_strategy, features.panel, name, settings,
                                  limits, cost_multiplier=multiplier)
            stats[variant] = performance(nav)
            paths[variant] = nav
            ledgers.append(fills.assign(strategy=variant))
    write_csv(output / "portfolio_nav.csv", pd.concat([nav.assign(strategy=name) for name, nav in paths.items()], ignore_index=True))
    write_csv(output / "portfolio_fills.csv", pd.concat(ledgers, ignore_index=True))
    write_csv(output / "signal_picks.csv", pd.concat(
        [pick.rename(columns={name: "score"}).assign(strategy=name) for name, pick in picks.items()], ignore_index=True))
    tests = {}
    if "ensemble_model" in picks:
        # The shared diagnostic uses 'ensemble' as its focal-strategy key.
        trials = {"ensemble": paths["model"], "quality_mix": paths["quality_mix"], "ensemble_model": paths["ensemble_model"]}
        tests = deflated_sharpe_and_pbo(trials)
        tests["dsr_ridge"] = tests.pop("dsr_ensemble")
        tests["focal_strategy"] = "model"
    return {"state": "COMPARED", "signals": len(dates), "performance": stats, "robustness": tests}


def _risk_checks(performance):
    regular, costly = performance.get("model", {}), performance.get("model_2x", {})
    control, control_2x = performance.get("quality_mix", {}), performance.get("quality_mix_2x", {})
    return (regular.get("annual_excess", -np.inf) > max(0, control.get("annual_excess", np.inf))
            and costly.get("annual_excess", -np.inf) > max(0, control_2x.get("annual_excess", np.inf))
            and regular.get("max_unpriced_weight", 1) <= 0.05
            and costly.get("max_unpriced_weight", 1) <= 0.05
            and regular.get("max_drawdown", -1) >= control.get("max_drawdown", 0)
            and regular.get("sessions", 0) >= VALIDATION_SESSIONS)


def develop(history, config, output, oos_start):
    features = history.features(config)
    market = history.market.index
    first = pd.Timestamp(oos_start)
    if pd.isna(first) or first.tz is not None or first.normalize() != first:
        raise ValueError("样本外起点需要 YYYY-MM-DD")
    if market.searchsorted(first) < config["data"]["warmup_sessions"]:
        return {"state": "BLOCKED", "reasons": ["样本外起点之前缺充足的历史预热行情"]}
    settings = {**config["research"], "oos_start": str(first.date()),
                "lockbox_start": str((market[-1] + pd.Timedelta(days=1)).date())}
    days = scheduled_signals(market, first, settings)
    if len(days) == 0:
        return {"state": "BLOCKED", "reasons": ["开发区间没有足够持有期的调仓日"]}
    rules, blocked = _rule_inputs(history, features, days, config, output)
    if blocked:
        return blocked
    labels = executable_labels(history.quotes, history.market, settings["horizon"])
    original = samples(features.panel, labels, features.columns, settings)
    # Keep sample cadence and scheduled portfolio signals even when the strides do not divide one another.
    extra = features.panel["eligible"] & features.panel["date"].isin(days) & ~features.panel.index.isin(original.index)
    extras = features.panel.loc[extra, ["date", "code", "baseline", *features.columns]].join(
        labels.loc[extra, ["excess_return", "entry_date", "exit_date", "session"]])
    train_frame = pd.concat([original, extras])
    predictions, fitted = walk_forward(train_frame, market, features.columns, settings)
    if predictions.empty:
        return {"state": "BLOCKED", "reasons": ["没有达到隔离期、训练样本和样本外日期要求的滚动模型"]}
    common = predictions.merge(labels[["date", "code", "excess_return"]], on=["date", "code"], validate="one_to_one")
    model_names = ["baseline", "ridge", "lgbm_rank", "lgbm_huber", "catboost", "ensemble"]
    diagnostics, _ = rank_ic(common, model_names, config["data"]["minimum_cross_section"],
                            ceil(settings["horizon"] / settings["sample_every"]))
    write_csv(output / "ranking_diagnostics.csv", diagnostics)
    compared = _replay_comparison(history, features, rules, predictions, days, config, output)
    if compared["state"] != "COMPARED":
        return compared
    robust = compared["robustness"]
    ridge_ic = diagnostics.loc[diagnostics["signal"].eq("ridge"), "rank_ic"].iat[0]
    passed = (_risk_checks(compared["performance"]) and pd.notna(ridge_ic) and ridge_ic > 0
              and robust.get("dsr_ridge") is not None and robust["dsr_ridge"] >= 0.95
              and robust.get("pbo") is not None and robust["pbo"] < 0.50)
    state = {**compared, "state": "DEVELOPMENT_FAILED", "walk_forward_fits": [item.train_stats for _, item in fitted],
             "ridge_rank_ic": ridge_ic, "development_accepted": bool(passed)}
    if not passed:
        return state
    fit_day = market[-1]
    training = training_sample(original, market, fit_day, settings)
    if training["date"].nunique() < settings["min_train_dates"]:
        return {**state, "state": "BLOCKED", "reasons": ["冻结拟合日缺足够已到期标签"]}
    model = train(training, features.columns, fit_day, settings)
    ridge = model.models["ridge"]
    candidate = {
        "state": "FROZEN", "created_utc": utc_now(), "policy_sha256": policy_id(config),
        "validation_start": str((pd.Timestamp(datetime.now(CST).date()) + pd.Timedelta(days=1)).date()),
        "columns": model.columns, "coefficients": ridge.coef_.tolist(), "intercept": float(ridge.intercept_),
        "training": model.train_stats, "history_sha256": history.fingerprints,
    }
    path = output / "candidate.json"
    write_json(path, candidate)
    return {**state, "state": "FROZEN", "candidate": str(path), "candidate_sha256": sha256_file(path)}


def validate_candidate(history, config, candidate_path, output, root):
    if not candidate_path.resolve().is_relative_to((root / "reports/product").resolve()):
        raise ValueError("候选模型必须来自本产品根目录下的冻结运行")
    candidate = json.loads(candidate_path.read_bytes())
    digest = sha256_file(candidate_path)
    origin = json.loads((candidate_path.parent / "status.json").read_bytes())
    created = pd.Timestamp(candidate["created_utc"]).tz_convert(CST).date()
    start = pd.Timestamp(candidate["validation_start"])
    if (candidate["state"] != "FROZEN" or candidate["policy_sha256"] != policy_id(config)
            or origin["state"] != "FROZEN" or origin["candidate_sha256"] != digest
            or not origin["development_accepted"]
            or start <= pd.Timestamp(created) or pd.Timestamp(candidate["training"]["last_label_exit"]) >= start):
        raise ValueError("候选模型或冻结日期未通过独立验证要求")
    days = history.market.index
    first = days.searchsorted(start)
    horizon = config["research"]["horizon"]
    stop = first + VALIDATION_SESSIONS + horizon + 1
    if first < config["data"]["warmup_sessions"] or stop > len(days) or days[0] >= start:
        return {"state": "BLOCKED", "reasons": ["冻结之后的独立验证窗口或其预热行情尚未完整"]}
    end_day = days[stop - 1]
    signals = scheduled_signals(days[:stop], start, config["research"])
    record = root / "models" / "validation" / f"{digest}.json"
    with file_lock(record.with_suffix(".lock"), wait_for_collection=True):
        if record.is_file():
            result = json.loads(record.read_bytes())
        else:
            features = history.features(config)
            rules, blocked = _rule_inputs(history, features, signals, config, output)
            if blocked:
                return blocked
            eligible, _ = rules
            predictions = score_ridge(eligible, candidate)
            history.market = history.market.loc[:end_day]
            compared = _replay_comparison(history, features, rules, predictions, signals, config, output)
            if compared["state"] != "COMPARED":
                return compared
            passed = _risk_checks(compared["performance"])
            result = {**compared, "state": "APPROVED" if passed else "VALIDATION_FAILED", "validated_utc": utc_now(),
                      "window_start": str(days[first].date()), "window_end": str(end_day.date()),
                      "candidate_sha256": digest, "input_sha256": history.fingerprints}
            write_json(record, result)
        if result["state"] == "APPROVED":
            activate = root / "models" / "active.json"
            with file_lock(activate.with_suffix(".lock")):
                current = json.loads(activate.read_bytes()) if activate.is_file() else {}
                if result["validated_utc"] >= current.get("validated_utc", ""):
                    write_json(activate, {"candidate": candidate_path.relative_to(root).as_posix(), "candidate_sha256": digest,
                               "validation": record.relative_to(root).as_posix(), "validation_sha256": sha256_file(record),
                               "validated_utc": result["validated_utc"]})
    return result
