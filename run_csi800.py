"""Standalone entry point for the CSI 800 product."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from quant_product.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
