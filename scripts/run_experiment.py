#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vtrace_lab.experiments import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the V-trace lag verification experiment")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--episodes-per-seed", type=int, default=200)
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    result, path = run_experiment(
        seeds=args.seeds,
        episodes_per_seed=args.episodes_per_seed,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "output": str(path),
        "counterexample": result["counterexample_summary"],
        "c_fixed_point_and_convergence": result["c_fixed_point_and_convergence"],
        "batch_reference_equivalence": result["batch_reference_equivalence"],
    }, indent=2))


if __name__ == "__main__":
    main()
