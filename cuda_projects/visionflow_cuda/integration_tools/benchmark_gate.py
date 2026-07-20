from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare_p95(current: dict, baseline: dict, max_regression: float = 0.15) -> list[dict]:
    baseline_rows = {
        row["operation"]: float(row["gpu_including_transfer"]["p95_ms"])
        for row in baseline.get("benchmark", {}).get("measurements", [])
    }
    failures = []
    for row in current.get("benchmark", {}).get("measurements", []):
        operation = row["operation"]
        if operation not in baseline_rows:
            failures.append({"operation": operation, "reason": "missing_baseline"})
            continue
        baseline_p95 = baseline_rows[operation]
        current_p95 = float(row["gpu_including_transfer"]["p95_ms"])
        limit = baseline_p95 * (1.0 + max_regression)
        if current_p95 > limit:
            failures.append({
                "operation": operation, "baseline_p95_ms": baseline_p95,
                "current_p95_ms": current_p95, "limit_p95_ms": round(limit, 6),
                "regression_ratio": round(current_p95 / baseline_p95 - 1.0, 6),
            })
    if not baseline_rows:
        failures.append({"reason": "baseline_has_no_benchmark_measurements"})
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail when RTX benchmark P95 regresses from baseline.")
    parser.add_argument("--current", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--max-regression", type=float, default=0.15)
    args = parser.parse_args()
    current = json.loads(Path(args.current).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    failures = compare_p95(current, baseline, args.max_regression)
    print(json.dumps({"max_regression": args.max_regression, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
