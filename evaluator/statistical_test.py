#!/usr/bin/env python
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from scipy import stats


def mean_var_std_ci(values: List[float], alpha: float = 0.05) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n < 2:
        mean = float(arr.mean()) if n == 1 else float("nan")
        return {
            "mean": mean,
            "variance": float("nan"),
            "std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }

    mean = float(arr.mean())
    var = float(arr.var(ddof=1))
    std = float(arr.std(ddof=1))
    se = std / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_low = mean - t_crit * se
    ci_high = mean + t_crit * se
    return {
        "mean": mean,
        "variance": var,
        "std": std,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def load_scores_jsonl(path: Path, field_a: str, field_b: str):
    ids = []
    a = []
    b = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ids.append(rec.get("id"))
            a.append(float(rec[field_a]))
            b.append(float(rec[field_b]))
    return ids, a, b


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Statistical tests over per-example scores for two systems "
            "(paired t-test, Wilcoxon, descriptive stats with 95%% CI)."
        )
    )
    parser.add_argument(
        "jsonl_file",
        type=Path,
        help="JSONL file with aligned per-example scores.",
    )
    parser.add_argument(
        "--field-a",
        required=True,
        help="JSON field name for system A scores (e.g., 'em_modelA').",
    )
    parser.add_argument(
        "--field-b",
        required=True,
        help="JSON field name for system B scores (e.g., 'em_modelB').",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for confidence intervals (default 0.05 for 95%% CI).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON results; prints to stdout if omitted.",
    )
    args = parser.parse_args()

    ids, scores_a, scores_b = load_scores_jsonl(args.jsonl_file, args.field_a, args.field_b)

    stats_a = mean_var_std_ci(scores_a, alpha=args.alpha)
    stats_b = mean_var_std_ci(scores_b, alpha=args.alpha)

    t_stat, t_p = stats.ttest_rel(scores_a, scores_b, nan_policy="omit")


    try:
        w_stat, w_p = stats.wilcoxon(scores_a, scores_b)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    result: Dict[str, Any] = {
        "n_examples": len(ids),
        "alpha": args.alpha,
        "system_a": {
            "field": args.field_a,
            **stats_a,
        },
        "system_b": {
            "field": args.field_b,
            **stats_b,
        },
        "paired_t_test": {
            "t_stat": float(t_stat),
            "p_value": float(t_p),
        },
        "wilcoxon_signed_rank": {
            "w_stat": float(w_stat),
            "p_value": float(w_p),
        },
    }

    out_str = json.dumps(result, indent=2)
    if args.output is None:
        print(out_str)
    else:
        args.output.write_text(out_str, encoding="utf-8")


if __name__ == "__main__":
    main()
