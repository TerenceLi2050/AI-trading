from dataclasses import dataclass
from typing import Any

from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


STRATEGIES = ["baseline", "ridge", "lgbm_rank", "lgbm_huber", "catboost", "ensemble"]


@dataclass
class ModelSet:
    columns: list[str]
    models: dict[str, Any]
    train_stats: dict


def design(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    # All columns are already same-date percentiles. Neutral imputation is invariant to future rows.
    return frame[columns].fillna(0.5).to_numpy(dtype=np.float32)


def training_sample(frame: pd.DataFrame, sessions: pd.DatetimeIndex, fit_date: pd.Timestamp, config: dict):
    position = sessions.searchsorted(fit_date)
    end = position - config["embargo"]
    if end < 1:
        return frame.iloc[:0]
    start = max(0, position - config["train_sessions"])
    return frame.loc[frame["session"].ge(start) & frame["date"].lt(fit_date)
                     & frame["exit_date"].lt(sessions[end]) & frame["excess_return"].notna()]


def train(frame: pd.DataFrame, columns: list[str], fit_date: pd.Timestamp, config: dict) -> ModelSet:
    frame = frame.sort_values(["date", "code"])
    targets = frame.groupby("date")["excess_return"].rank(pct=True)
    x = design(frame, columns)
    y = targets.to_numpy(dtype=np.float32)
    base = dict(n_estimators=180, learning_rate=0.035, num_leaves=15, max_depth=4,
                min_child_samples=250, reg_alpha=0.1, reg_lambda=5.0, colsample_bytree=0.8,
                subsample=0.8, subsample_freq=1, verbosity=-1, force_col_wise=True,
                random_state=config["seed"], n_jobs=config["threads"])
    ridge = Ridge(alpha=50, solver="cholesky").fit(x, y)
    ranker = lgb.LGBMRanker(objective="lambdarank", **base).fit(
        x, np.minimum((y * 5).astype(int), 4), group=frame.groupby("date", sort=False).size().to_numpy())
    huber = lgb.LGBMRegressor(objective="huber", **base).fit(x, y)
    cat = CatBoostRegressor(iterations=180, depth=4, learning_rate=0.04, l2_leaf_reg=12,
                            thread_count=config["threads"], random_seed=config["seed"],
                            allow_writing_files=False, verbose=False).fit(x, y)
    stats = {"fit_date": str(fit_date.date()), "rows": len(frame), "days": frame["date"].nunique(),
             "first_signal_date": str(frame["date"].min().date()), "last_signal_date": str(frame["date"].max().date()),
             "last_label_exit": str(frame["exit_date"].max().date())}
    return ModelSet(columns, {"ridge": ridge, "lgbm_rank": ranker, "lgbm_huber": huber, "catboost": cat}, stats)


def score(frame: pd.DataFrame, bundle: ModelSet) -> pd.DataFrame:
    x = design(frame, bundle.columns)
    predictions = frame[["date", "code", "baseline"]].copy()
    for name, model in bundle.models.items():
        predictions[name] = model.predict(x) if name in ("ridge", "catboost") else model.booster_.predict(x, num_threads=model.n_jobs)
    rank = predictions.groupby("date")[list(bundle.models)].rank(pct=True)
    predictions["ensemble"] = rank.mean(axis=1)
    return predictions


def explain(frame: pd.DataFrame, bundle: ModelSet, limit: int = 3) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="string", index=frame.index)
    model = bundle.models["lgbm_rank"]
    contributions = model.booster_.predict(design(frame, bundle.columns), pred_contrib=True, num_threads=model.n_jobs)
    explanations = []
    for row in contributions[:, :-1]:
        largest = np.argsort(-np.abs(row), kind="stable")[:limit]
        explanations.append(" / ".join(f"{bundle.columns[i].removeprefix('rank_')} {row[i]:+.4f}" for i in largest))
    return pd.Series(explanations, index=frame.index, dtype="string")


def samples(panel: pd.DataFrame, labels: pd.DataFrame, columns: list[str], research: dict) -> pd.DataFrame:
    keep = panel["eligible"] & labels["session"].mod(research["sample_every"]).eq(0)
    fields = ["date", "code", "baseline", *columns]
    return panel.loc[keep, fields].join(labels.loc[keep, ["excess_return", "entry_date", "exit_date", "session"]])


def walk_forward(sample: pd.DataFrame, sessions: pd.DatetimeIndex, columns: list[str], config: dict):
    oos = sample.loc[sample["date"].ge(config["oos_start"]) & sample["date"].lt(config["lockbox_start"])]
    dates = pd.DatetimeIndex(oos["date"].unique()).sort_values()
    anchor = pd.Timestamp(config["oos_start"])
    blocks = ((dates.year - anchor.year) * 12 + dates.month - anchor.month) // config["refit_months"]
    frames, fitted = [], []
    for block in np.unique(blocks):
        block_dates = dates[blocks == block]
        fit_date = block_dates[0]
        rows = training_sample(sample, sessions, fit_date, config)
        if rows["date"].nunique() < config["min_train_dates"]:
            continue
        bundle = train(rows, columns, fit_date, config)
        predictions = score(oos.loc[oos["date"].isin(block_dates)], bundle)
        frames.append(predictions)
        fitted.append((block_dates, bundle))
        print(f"walk-forward {fit_date.date()} → {block_dates[-1].date()} ({len(rows):,} train rows)", flush=True)
    if not frames:
        return pd.DataFrame(columns=["date", "code", *STRATEGIES]), fitted
    return pd.concat(frames).sort_values(["date", "code"]), fitted
