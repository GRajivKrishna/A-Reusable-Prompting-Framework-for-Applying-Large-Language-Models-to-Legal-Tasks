#!/usr/bin/env python
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from rouge_score import rouge_scorer  # pip install rouge-score
from scipy import stats  # for confidence intervals


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


def normalize_label(x: str) -> str:
    x = (x or "").strip().lower()
    if x in ["yes", "y", "true"]:
        return "yes"
    if x in ["no", "n", "false"]:
        return "no"
    # Fallback: return raw lowercased token
    return x


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Extract labels and texts
    y_true = []
    y_pred = []
    refs = []
    preds = []

    for rec in records:
        gold = normalize_label(rec.get("target_text", ""))
        pred = normalize_label(rec.get("model_response", ""))
        y_true.append(gold)
        y_pred.append(pred)
        refs.append(str(rec.get("target_text", "")))
        preds.append(str(rec.get("model_response", "")))

    # Exact match (per-example)
    exact_matches = [1.0 if t == p else 0.0 for t, p in zip(y_true, y_pred)]

    # ROUGE-L per-example
    rougeL_p = []
    rougeL_r = []
    rougeL_f = []
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    for ref, pred in zip(refs, preds):
        s = scorer.score(ref, pred)["rougeL"]
        rougeL_p.append(s.precision)
        rougeL_r.append(s.recall)
        rougeL_f.append(s.fmeasure)

    # Classification metrics (global)
    labels = sorted(set(y_true) | set(y_pred))
    precision_macro = precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)

    precision_micro = precision_score(y_true, y_pred, average="micro", labels=labels, zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average="micro", labels=labels, zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", labels=labels, zero_division=0)

    precision_per_class = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    metrics = {
        "n_examples": len(records),
        "exact_match": mean_var_std_ci(exact_matches),
        "rougeL_precision": mean_var_std_ci(rougeL_p),
        "rougeL_recall": mean_var_std_ci(rougeL_r),
        "rougeL_f1": mean_var_std_ci(rougeL_f),
        "classification": {
            "labels": labels,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
            "precision_per_class": {
                label: float(v) for label, v in zip(labels, precision_per_class)
            },
            "recall_per_class": {
                label: float(v) for label, v in zip(labels, recall_per_class)
            },
            "f1_per_class": {
                label: float(v) for label, v in zip(labels, f1_per_class)
            },
        },
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compute standard metrics (Exact Match, ROUGE-L, F1, etc.) for JSONL datasets."
    )
    parser.add_argument(
        "jsonl_files",
        nargs="+",
        type=Path,
        help="One or more JSONL files with fields 'model_response' and 'target_text'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write metrics JSON; prints to stdout if omitted.",
    )
    args = parser.parse_args()

    all_records: List[Dict[str, Any]] = []
    for path in args.jsonl_files:
        all_records.extend(load_jsonl(path))

    metrics = compute_metrics(all_records)

    out_str = json.dumps(metrics, indent=2, ensure_ascii=False)
    if args.output is None:
        print(out_str)
    else:
        args.output.write_text(out_str, encoding="utf-8")


if __name__ == "__main__":
    main()
