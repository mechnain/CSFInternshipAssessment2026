"""Evaluate ranked ReID results against ground-truth labels.

Reports closed-set (Rank-1, Rank-5, mAP), thresholded hard-match
precision / recall / F1, open-set rejection accuracy, and
threshold-independent open-set AUROC (separation of known vs. unknown
queries by top-1 similarity).

A query whose `true_dog_id` is the literal string ``unknown`` is treated
as an open-set query: the correct decision is ``unknown``.

Single-threshold mode:
  python src/evaluate.py \\
      --results results/ranked_results.csv \\
      --labels  data/sample/labels.csv \\
      --threshold 0.70 \\
      --output  results/metrics.json

Sweep mode (scans tau_match, reports best operating point):
  python src/evaluate.py \\
      --results results/ranked_results.csv \\
      --labels  data/sample/labels.csv \\
      --sweep \\
      --output  results/metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import DEFAULT_TAU_POSSIBLE, OpenSetThresholds

logger = logging.getLogger("eval")

UNKNOWN_LABEL = "unknown"
_DECISION_KEY = {"match": "match", "possible_match": "possible", "unknown": "unknown"}
_REQUIRED_RESULT_COLUMNS = {"query", "rank", "gallery_id", "similarity"}

# Accept both the assessment-spec schema (query_image,true_id) and the
# earlier internal schema (query_filename,true_dog_id). The two get
# normalised to the spec names before evaluation.
_LABEL_COLUMN_ALIASES = {
    "query_filename": "query_image",
    "true_dog_id": "true_id",
}


def _normalize_labels(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.rename(columns=_LABEL_COLUMN_ALIASES).copy()
    missing = {"query_image", "true_id"} - set(labels.columns)
    if missing:
        raise ValueError(
            f"labels.csv is missing required columns: {sorted(missing)}. "
            f"Expected columns: query_image,true_id."
        )
    labels["query_image"] = labels["query_image"].astype(str)
    duplicated = labels["query_image"][labels["query_image"].duplicated()].unique()
    if len(duplicated):
        raise ValueError(
            "labels.csv contains duplicate query_image rows: "
            f"{sorted(duplicated)[:10]}"
        )
    return labels


def _normalize_results(results: pd.DataFrame) -> pd.DataFrame:
    missing = _REQUIRED_RESULT_COLUMNS - set(results.columns)
    if missing:
        raise ValueError(
            f"results CSV is missing required columns: {sorted(missing)}. "
            f"Expected at least: {sorted(_REQUIRED_RESULT_COLUMNS)}."
        )

    results = results.copy()
    results["query"] = results["query"].astype(str)
    results["rank"] = pd.to_numeric(results["rank"], errors="raise").astype(int)
    results["similarity"] = pd.to_numeric(results["similarity"], errors="raise")
    if (results["rank"] < 1).any():
        raise ValueError("results CSV contains ranks below 1.")

    dupes = results.duplicated(subset=["query", "rank"])
    if dupes.any():
        bad = results.loc[dupes, ["query", "rank"]].head(10).to_dict("records")
        raise ValueError(f"results CSV contains duplicate query/rank rows: {bad}")

    missing_rank1 = sorted(
        set(results["query"]) - set(results.loc[results["rank"] == 1, "query"])
    )
    if missing_rank1:
        raise ValueError(
            "results CSV is missing a rank-1 row for queries: "
            f"{missing_rank1[:10]}"
        )
    return results.sort_values(["query", "rank"]).reset_index(drop=True)


def _average_precision(matches: np.ndarray) -> float:
    """AP over a binary relevance vector ranked top-1..top-k."""
    if matches.size == 0 or matches.sum() == 0:
        return 0.0
    cum_hits = np.cumsum(matches)
    ranks = np.arange(1, matches.size + 1)
    precision_at_k = cum_hits / ranks
    return float((precision_at_k * matches).sum() / matches.sum())


def _open_set_auroc(known_sims: list[float], unknown_sims: list[float]) -> float | None:
    """AUROC of separating known (label=1) from unknown (label=0) queries
    using their top-1 similarity as the score. Threshold-independent
    measure of how well the embedding space supports open-set rejection.
    Returns None if either group is empty.
    """
    if not known_sims or not unknown_sims:
        return None
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:  # pragma: no cover
        return None
    y_true = [1] * len(known_sims) + [0] * len(unknown_sims)
    y_score = list(known_sims) + list(unknown_sims)
    return float(roc_auc_score(y_true, y_score))


def evaluate(
    results: pd.DataFrame,
    labels: pd.DataFrame,
    thresholds: OpenSetThresholds,
) -> dict:
    labels = _normalize_labels(labels)
    labels["true_id"] = labels["true_id"].fillna(UNKNOWN_LABEL).astype(str)
    labels = labels.set_index("query_image")

    results = _normalize_results(results)

    rank1 = rank5 = 0
    aps: list[float] = []
    tp = fp = fn = tn = 0
    abstain_known = abstain_unknown = 0
    closed_total = open_total = 0
    open_correct_unknown = 0
    known_top1_sims: list[float] = []
    unknown_top1_sims: list[float] = []

    decision_counts = {"match": 0, "possible_match": 0, "unknown": 0}
    confusion = {
        f"true_{t}_pred_{p}": 0
        for t in ("known", "unknown")
        for p in ("match", "possible", "unknown")
    }

    skipped = 0
    for query, group in results.groupby("query", sort=False):
        if query not in labels.index:
            logger.warning("Query %s missing from labels.csv, skipping.", query)
            skipped += 1
            continue

        truth = str(labels.loc[query, "true_id"])
        ranked_ids = group["gallery_id"].tolist()
        ranked_sims = group["similarity"].to_numpy(dtype=float)
        top1_id = ranked_ids[0]
        top1_sim = float(ranked_sims[0])
        decision = thresholds.decide(top1_sim)
        decision_counts[decision] += 1

        is_open = truth == UNKNOWN_LABEL
        pred_key = _DECISION_KEY[decision]
        truth_key = "unknown" if is_open else "known"
        confusion[f"true_{truth_key}_pred_{pred_key}"] += 1

        if is_open:
            open_total += 1
            unknown_top1_sims.append(top1_sim)
            if decision == "unknown":
                open_correct_unknown += 1
                tn += 1
            elif decision == "match":
                # Confidently claimed a wrong identity for an absent dog.
                fp += 1
            else:
                abstain_unknown += 1
        else:
            closed_total += 1
            known_top1_sims.append(top1_sim)
            relevance = np.array([rid == truth for rid in ranked_ids], dtype=int)
            if relevance.size > 0 and relevance[0] == 1:
                rank1 += 1
            if relevance[:5].sum() > 0:
                rank5 += 1
            aps.append(_average_precision(relevance))

            if decision == "match" and top1_id == truth:
                tp += 1
            elif decision == "match":
                fp += 1
                fn += 1
            elif decision == "unknown":
                fn += 1
            else:
                abstain_known += 1
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None:
        f1 = None
    elif (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    auroc = _open_set_auroc(known_top1_sims, unknown_top1_sims)

    return {
        "thresholds": {"match": thresholds.match, "possible": thresholds.possible},
        "n_queries_total": closed_total + open_total,
        "n_queries_skipped": skipped,
        "closed_set": {
            "n_queries": closed_total,
            "rank_1_accuracy": (rank1 / closed_total) if closed_total else None,
            "rank_5_accuracy": (rank5 / closed_total) if closed_total else None,
            "mAP": float(np.mean(aps)) if aps else None,
        },
        "open_set": {
            "n_queries": open_total,
            "unknown_accuracy": (open_correct_unknown / open_total) if open_total else None,
            "non_match_rejection_rate": (
                (open_correct_unknown + abstain_unknown) / open_total
                if open_total
                else None
            ),
            "auroc_known_vs_unknown": auroc,
        },
        "thresholded": {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "abstain_known": abstain_known,
            "abstain_unknown": abstain_unknown,
            "abstain_total": abstain_known + abstain_unknown,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "decision_counts": decision_counts,
        "confusion_matrix": confusion,
    }


def sweep_thresholds(
    results: pd.DataFrame,
    labels: pd.DataFrame,
    taus: np.ndarray | None = None,
    possible_offset: float = 0.15,
) -> pd.DataFrame:
    """Scan tau_match over a range; tau_possible follows at a fixed offset.

    For each tau we recompute the threshold-based decisions and report
    the closed-set thresholded accuracy, the open-set unknown accuracy,
    and a balanced score = mean of the two. Useful to pick an operating
    point on validation data without hard-coding test labels.
    """
    if taus is None:
        taus = np.round(np.arange(0.50, 1.001, 0.02), 4)
    rows = []
    for tau in taus:
        thr = OpenSetThresholds(
            match=float(tau),
            possible=max(float(tau) - possible_offset, 0.0),
        )
        m = evaluate(results, labels, thr)
        closed_acc = (
            m["thresholded"]["TP"] / m["closed_set"]["n_queries"]
            if m["closed_set"]["n_queries"]
            else None
        )
        unknown_acc = m["open_set"]["unknown_accuracy"]
        balanced = (
            (closed_acc + unknown_acc) / 2
            if closed_acc is not None and unknown_acc is not None
            else None
        )
        rows.append(
            {
                "tau_match": float(tau),
                "tau_possible": thr.possible,
                "closed_set_rank_1": m["closed_set"]["rank_1_accuracy"],
                "thresholded_correct_rate": closed_acc,
                "unknown_accuracy": unknown_acc,
                "unknown_non_match_rejection_rate": m["open_set"][
                    "non_match_rejection_rate"
                ],
                "balanced_accuracy": balanced,
                "precision": m["thresholded"]["precision"],
                "recall": m["thresholded"]["recall"],
                "f1": m["thresholded"]["f1"],
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ranked ReID results.")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="tau_match: top-1 similarity at or above which we declare 'match'.",
    )
    p.add_argument(
        "--possible-threshold",
        type=float,
        default=DEFAULT_TAU_POSSIBLE,
        help="tau_possible: top-1 similarity at or above which we declare 'possible_match'.",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "Also scan tau_match across [0.50, 1.00] step 0.02; write a sweep "
            "CSV next to --output and pick the tau with the best balanced "
            "(closed + open) accuracy."
        ),
    )
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    results = pd.read_csv(args.results)
    labels = pd.read_csv(args.labels)
    thresholds = OpenSetThresholds(match=args.threshold, possible=args.possible_threshold)
    metrics = evaluate(results, labels, thresholds)

    if args.sweep:
        logger.warning(
            "Threshold sweeps choose an operating point on the provided labels. "
            "Use this on a validation split, then report final metrics on a "
            "separate test split."
        )
        sweep_df = sweep_thresholds(results, labels)
        sweep_path = args.output.with_name(args.output.stem + "_sweep.csv")
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        sweep_df.to_csv(sweep_path, index=False)
        best_idx = sweep_df["balanced_accuracy"].idxmax() if sweep_df["balanced_accuracy"].notna().any() else None
        if best_idx is not None:
            best = sweep_df.loc[best_idx].to_dict()
            metrics["sweep"] = {
                "csv": str(sweep_path),
                "recommended_tau_match": best["tau_match"],
                "recommended_tau_possible": best["tau_possible"],
                "balanced_accuracy_at_best": best["balanced_accuracy"],
                "selection_warning": (
                    "Use recommended thresholds only if this sweep was run on "
                    "validation data, not the final test labels."
                ),
            }
            logger.info(
                "Sweep: best tau_match=%.2f (balanced acc=%.3f). Sweep CSV -> %s",
                best["tau_match"],
                best["balanced_accuracy"],
                sweep_path,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics written to %s\n%s", args.output, json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
