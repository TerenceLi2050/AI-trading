"""The model file format and scoring shared by research and daily publication."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from quant_screen.features import PRICE_FACTORS
from quant_screen.io import sha256_file


FEATURE_COLUMNS = [f"rank_{name}" for name in PRICE_FACTORS]
FACTOR_NAMES = list(PRICE_FACTORS)


def policy_id(config):
    modules = Path(__file__).resolve().parents[1] / "quant_screen"
    fingerprints = {name: sha256_file(modules / name)
                    for name in ("features.py", "current.py", "quality_study.py", "market.py", "backtest.py", "screen.py", "labels.py")}
    fingerprints.update({name: sha256_file(Path(__file__).with_name(name))
                         for name in ("model.py", "research.py", "dataset.py", "publish.py")})
    options = {name: config[name] for name in ("eligibility", "portfolio")}
    options["data"] = {name: config["data"][name] for name in (
        "minimum_cross_section", "minimum_member_count", "minimum_member_coverage", "minimum_fundamental_coverage", "warmup_sessions",
    )}
    options["research"] = {name: config["research"][name] for name in ("horizon", "rebalance_every")}
    return sha256(json.dumps([fingerprints, options], sort_keys=True).encode()).hexdigest()


def score_ridge(frame, candidate, *, explanations=False):
    if candidate["columns"] != FEATURE_COLUMNS or len(candidate["coefficients"]) != len(FEATURE_COLUMNS):
        raise ValueError("候选模型因子列与当期特征引擎不一致")
    weights = np.asarray(candidate["coefficients"], dtype=np.float32)
    intercept = np.float32(candidate["intercept"])
    if not np.isfinite(weights).all() or not np.isfinite(intercept):
        raise ValueError("候选模型包含非有限权重")
    x = frame[FEATURE_COLUMNS].fillna(0.5).to_numpy(dtype=np.float32)
    result = frame[["date", "code"]].copy()
    result["ridge"] = x @ weights + intercept
    if explanations:
        contributions = (x - 0.5) * weights
        indices = np.argsort(-np.abs(contributions), axis=1, kind="stable")[:, :3]
        result["model_factors"] = [
            " / ".join(f"{FACTOR_NAMES[i]} {row[i]:+.5f}" for i in leaders)
            for row, leaders in zip(contributions, indices)
        ]
    return result


def blend_quality(qualified, predictions, weight):
    scored = predictions[["date", "code", "ridge"]].copy()
    scored["model_rank"] = scored.groupby("date")["ridge"].rank(pct=True)
    result = qualified.merge(scored, on=["date", "code"], validate="one_to_one")
    result["model"] = weight * result["quality_only"] + (1 - weight) * result["model_rank"]
    return result
