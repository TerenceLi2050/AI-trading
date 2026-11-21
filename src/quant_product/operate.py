"""Advance approved inputs through the model lifecycle alongside daily publication."""

from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from quant_screen.io import dates, file_lock, read_table, sha256_file, utc_now, write_json
from quant_screen.sessions import CST

from .dataset import INPUTS, audit_history
from .model import policy_id


def advance(root, history_root, output, config, start, oos_start, source=None):
    state_path = root / "models/research_cycle.json"
    with file_lock(state_path.with_suffix(".lock"), wait_for_collection=True):
        if source is not None:
            from .evidence import import_sources

            import_sources(source, history_root)
        missing = [path for path in INPUTS.values() if not (history_root / path).is_file()]
        if missing:
            return {"state": "BLOCKED", "reasons": [f"缺历史输入 {path}" for path in missing]}
        return _advance(root, history_root, output, config, start, oos_start, state_path)


def _advance(root, history_root, output, config, start, oos_start, state_path):
    from .research import VALIDATION_SESSIONS, develop, validate_candidate

    now = pd.Timestamp(datetime.now(CST).date())
    available_end = min(
        dates(read_table(history_root / INPUTS[key])["date"]).max()
        for key in ("market", "total_return", "sessions")
    )
    end = min(available_end, now - pd.Timedelta(days=1))
    current_policy = policy_id(config)
    prior = json.loads(state_path.read_bytes()) if state_path.is_file() else {}
    fingerprints = {key: sha256_file(history_root / filename) for key, filename in INPUTS.items()}
    manifest = history_root / "source_manifest.json"
    fingerprints["sources"] = sha256_file(manifest) if manifest.is_file() else None
    unchanged = (prior.get("inputs") == fingerprints and prior.get("policy") == current_policy
                 and prior.get("start") == start and prior.get("oos_start") == oos_start)
    if unchanged:
        return _current(prior)
    pending = _pending_candidate(root, current_policy)
    if pending is not None:
        file, candidate = pending
        valid_after = pd.Timestamp(candidate["validation_start"])
        required = VALIDATION_SESSIONS + config["research"]["horizon"] + 1
        # The exact window is evaluated against the exchange calendar in validate_candidate.
        if (end - valid_after).days < required:
            return {"state": "WAITING_INDEPENDENT", "candidate": str(file), "validation_start": str(valid_after.date())}
    elif (prior.get("policy") == current_policy and prior.get("last_development")
          and prior.get("result") in {"DEVELOPMENT_FAILED", "APPROVED", "VALIDATION_FAILED"}
          and now < pd.Timestamp(prior["last_development"]) + pd.DateOffset(months=config["research"]["refit_months"])):
        return _current(prior)
    with file_lock(history_root / ".history_sources.lock", wait_for_collection=True):
        audit, history = audit_history(history_root, output, start, end, config)
    if history is None:
        state = audit
    elif pending is not None:
        state = validate_candidate(history, config, pending[0], output, root)
    else:
        path = output / "development"
        state = develop(history, config, path, oos_start)
        write_json(path / "status.json", state)
    attempt = {"inputs": fingerprints, "policy": current_policy, "start": start, "oos_start": oos_start,
               "result": state["state"], "last_attempt": utc_now(),
               "audit": (output / "data_audit.json").relative_to(root).as_posix(),
               "last_development": prior.get("last_development") if pending else str(now.date())}
    write_json(state_path, attempt)
    return state


def _current(prior):
    return {"state": "CURRENT", "last_attempt": prior.get("last_attempt"), "result": prior["result"],
            "audit": prior.get("audit")}


def _pending_candidate(root: Path, policy):
    pending = []
    for file in (root / "reports/product").rglob("candidate.json"):
        digest = sha256_file(file)
        if (root / "models/validation" / f"{digest}.json").is_file():
            continue
        status_path = file.parent / "status.json"
        if not status_path.is_file():
            continue
        state = json.loads(status_path.read_bytes())
        candidate = json.loads(file.read_bytes())
        if state.get("candidate_sha256") == digest and candidate["policy_sha256"] == policy:
            pending.append((file, candidate))
    return max(pending, key=lambda item: item[1]["created_utc"]) if pending else None
