"""Rebuild the historical pilot in an empty data root."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from quant_history.pilot import rebuild
from quant_screen.io import file_lock, write_json


def main():
    parser = argparse.ArgumentParser(description="从已归档的原始证据重建中证800历史试点")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path, required=True, help="新的历史数据目录")
    parser.add_argument("--price-cache", type=Path, help="试点补采行情缓存目录")
    parser.add_argument("--collect-missing-prices", action="store_true", help="允许从 BaoStock 补采缺少的日线")
    args = parser.parse_args()
    source, output = args.source_root.resolve(), args.output_root.resolve()
    cache = args.price_cache.resolve() if args.price_cache else None
    if output == source or source.is_relative_to(output):
        parser.error("--output-root must be separate from the source checkout")
    if args.collect_missing_prices and cache is None:
        parser.error("--collect-missing-prices requires --price-cache")
    if cache is not None and cache.is_relative_to(output):
        parser.error("--price-cache must be outside the output root")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".historical_init.lock"
    with file_lock(lock):
        if any(path != lock for path in output.iterdir()):
            parser.error("--output-root must be empty")
        try:
            state = rebuild(source, output, price_cache=cache, collect_prices=args.collect_missing_prices)
        except (OSError, ValueError, KeyError) as exc:
            state = {"state": "INVALID_EVIDENCE", "error": str(exc)}
        report = output / "reports" / "historical_init.json"
        write_json(report, state)
    print(report, flush=True)
    return 0 if state.get("reconstruction") == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
