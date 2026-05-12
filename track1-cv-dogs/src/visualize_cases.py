"""Render success and failure cases from ranked ReID results.

Reads `ranked_results.csv` + `labels.csv` and produces two figures:

  * success_cases.png  -- queries the system got right (correct hard
    match on a known dog, OR correctly flagged 'unknown').
  * failure_cases.png  -- queries the system got wrong, labeled with the
    failure type (wrong_identity, false_alarm, missed_known, ambiguous).

Each case row shows: query | top-1 ref | top-2 ref | top-3 ref, with
similarity scores. The leftmost cell is annotated with truth + decision.

Usage:
  python src/visualize_cases.py \\
      --results   results/ranked_results.csv \\
      --labels    data/sample/labels.csv \\
      --reference data/sample/reference \\
      --query     data/sample/query \\
      --success-output results/success_cases.png \\
      --failure-output results/failure_cases.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import discover_gallery, load_image

logger = logging.getLogger("cases")

UNKNOWN_LABEL = "unknown"
SUCCESS_COLOR = "#2ca02c"
FAILURE_COLOR = "#d62728"
NEUTRAL_COLOR = "#7f7f7f"
MAX_PER_FIGURE = 8
TOP_K = 3

# Accept the spec schema (query_image,true_id) and the earlier internal
# schema (query_filename,true_dog_id).
_LABEL_COLUMN_ALIASES = {
    "query_filename": "query_image",
    "true_dog_id": "true_id",
}


def _categorize(truth: str, top1_id: str, decision: str) -> tuple[str, str]:
    """Return (category, group) where group is 'success' or 'failure'."""
    is_open = truth == UNKNOWN_LABEL
    if is_open:
        if decision == "unknown":
            return ("true_negative", "success")
        if decision == "match":
            return ("false_alarm", "failure")
        return ("ambiguous_unknown", "failure")
    # known query
    if decision == "match" and top1_id == truth:
        return ("true_positive", "success")
    if decision == "match" and top1_id != truth:
        return ("wrong_identity", "failure")
    if decision == "unknown":
        return ("missed_known", "failure")
    return ("ambiguous_known", "failure")


def _style_axis(ax, color: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(2)
        spine.set_visible(True)


def _build_cases(results: pd.DataFrame, labels: pd.DataFrame) -> list[dict]:
    """Annotate each query with truth, decision, category, group."""
    labels = labels.rename(columns=_LABEL_COLUMN_ALIASES).copy()
    missing = {"query_image", "true_id"} - set(labels.columns)
    if missing:
        raise ValueError(
            f"labels.csv is missing required columns: {sorted(missing)}. "
            f"Expected columns: query_image,true_id."
        )
    labels["true_id"] = labels["true_id"].fillna(UNKNOWN_LABEL).astype(str)
    labels.set_index("query_image", inplace=True)

    cases: list[dict] = []
    for query, group in results.sort_values(["query", "rank"]).groupby("query", sort=False):
        if query not in labels.index:
            continue
        truth = str(labels.loc[query, "true_id"])
        top1 = group.iloc[0]
        decision = str(top1.get("decision") or "")
        if not decision:
            # Some pipeline outputs only carry decision on rank-1; fall back
            # to the literal value here (which will be the empty string).
            decision = ""
        category, group_label = _categorize(truth, str(top1["gallery_id"]), decision)
        cases.append(
            {
                "query": query,
                "truth": truth,
                "decision": decision,
                "category": category,
                "group": group_label,
                "rows": group.head(TOP_K).reset_index(drop=True),
            }
        )
    return cases


def _render(
    cases: list[dict],
    reference_dir: Path,
    query_dir: Path,
    output_path: Path,
    title: str,
    border_color: str,
) -> Path | None:
    if not cases:
        logger.info("No cases for %s; skipping.", output_path.name)
        return None
    cases = cases[:MAX_PER_FIGURE]
    gallery = discover_gallery(reference_dir)

    n_rows = len(cases)
    n_cols = 1 + TOP_K
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.4 * n_cols, 2.8 * n_rows),
        squeeze=False,
    )

    for r, case in enumerate(cases):
        q_path = query_dir / case["query"]
        ax = axes[r][0]
        if q_path.exists():
            ax.imshow(load_image(q_path))
        ax.set_title(
            f"query: {case['query']}\n"
            f"truth: {case['truth']}\n"
            f"decision: {case['decision'] or '-'}\n"
            f"[{case['category']}]",
            fontsize=8,
            color=border_color,
            pad=8,
        )
        _style_axis(ax, border_color)

        rows = case["rows"]
        for i in range(TOP_K):
            ax = axes[r][1 + i]
            if i < len(rows):
                row = rows.iloc[i]
                ref_id = str(row["gallery_id"])
                ref_imgs = gallery.get(ref_id, [])
                if ref_imgs:
                    ax.imshow(load_image(ref_imgs[0]))
                ax.set_title(
                    f"#{int(row['rank'])} {ref_id}\nsim={float(row['similarity']):.3f}",
                    fontsize=8,
                    pad=8,
                )
                _style_axis(ax, NEUTRAL_COLOR)
            else:
                _style_axis(ax, NEUTRAL_COLOR)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render success + failure case visualizations.")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--query", type=Path, required=True)
    p.add_argument(
        "--success-output",
        type=Path,
        default=Path("results/success_cases.png"),
    )
    p.add_argument(
        "--failure-output",
        type=Path,
        default=Path("results/failure_cases.png"),
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    results = pd.read_csv(args.results)
    labels = pd.read_csv(args.labels)

    cases = _build_cases(results, labels)
    successes = [c for c in cases if c["group"] == "success"]
    failures = [c for c in cases if c["group"] == "failure"]
    logger.info("Identified %d successes and %d failures.", len(successes), len(failures))

    if not successes and not failures:
        logger.warning("No cases to render (results/labels misaligned?).")
        return

    out_s = _render(
        successes,
        args.reference,
        args.query,
        args.success_output,
        title="Success cases (correct match on known; correct 'unknown' on absent)",
        border_color=SUCCESS_COLOR,
    )
    out_f = _render(
        failures,
        args.reference,
        args.query,
        args.failure_output,
        title="Failure cases (wrong identity, false alarm, missed known, ambiguous)",
        border_color=FAILURE_COLOR,
    )
    if out_s:
        logger.info("Wrote %s", out_s)
    if out_f:
        logger.info("Wrote %s", out_f)


if __name__ == "__main__":
    main()
