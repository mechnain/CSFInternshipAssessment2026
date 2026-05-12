"""Render a top-k retrieval grid for qualitative review.

For each query we show the query image on the left and the top-k matched
gallery identities to the right, each annotated with cosine similarity
and the open-set decision (color-coded).

Usage:
  python src/visualize_results.py \
      --results   results/ranked_results.csv \
      --reference data/sample/reference \
      --query     data/sample/query \
      --output    results/top_matches.png \
      --top-k 5
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

logger = logging.getLogger("viz")

DECISION_COLOR = {
    "match": "#2ca02c",
    "possible_match": "#ff7f0e",
    "unknown": "#d62728",
    "": "#7f7f7f",
}


def _style_axis(ax, color: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(2)
        spine.set_visible(True)


def render_grid(
    results: pd.DataFrame,
    reference_dir: Path,
    query_dir: Path,
    output_path: Path,
    top_k: int = 5,
    max_queries: int = 12,
) -> Path:
    gallery = discover_gallery(reference_dir)
    shown_k = min(top_k, len(gallery))

    queries = list(results["query"].drop_duplicates())
    if not queries:
        raise ValueError("Results CSV contains no queries.")
    if max_queries and max_queries > 0:
        queries = queries[:max_queries]

    n_rows = len(queries)
    n_cols = 1 + shown_k
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2 * n_cols, 2 * n_rows),
        squeeze=False,
    )

    for r, query_name in enumerate(queries):
        q_path = query_dir / query_name
        ax = axes[r][0]
        if q_path.exists():
            ax.imshow(load_image(q_path))
        ax.set_title(f"query\n{query_name}", fontsize=8)
        _style_axis(ax, "#000000")

        rows = (
            results[results["query"] == query_name]
            .sort_values("rank")
            .head(shown_k)
            .reset_index(drop=True)
        )
        # Decision is carried only on the rank-1 row of the pipeline output.
        top_decision = ""
        if len(rows) > 0 and isinstance(rows.loc[0, "decision"], str):
            top_decision = rows.loc[0, "decision"] or ""

        for i in range(shown_k):
            ax = axes[r][1 + i]
            if i < len(rows):
                row = rows.iloc[i]
                ref_id = str(row["gallery_id"])
                ref_imgs = gallery.get(ref_id, [])
                if ref_imgs:
                    ax.imshow(load_image(ref_imgs[0]))
                sim = float(row["similarity"])
                rank = int(row["rank"])
                # Only the rank-1 cell is colored by the decision; later
                # ranks use a neutral border so the viewer focuses on the
                # actual verdict.
                color = DECISION_COLOR.get(top_decision, DECISION_COLOR[""]) if i == 0 \
                    else DECISION_COLOR[""]
                title = f"#{rank} {ref_id}\nsim={sim:.3f}"
                if i == 0 and top_decision:
                    title += f"\n[{top_decision}]"
                ax.set_title(title, fontsize=8, color=color)
                _style_axis(ax, color)
            else:
                _style_axis(ax, DECISION_COLOR[""])

    fig.suptitle(
        "Dog ReID top-k retrievals\n"
        "rank-1 border: green=match, orange=possible_match, red=unknown",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize top-k ReID results.")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--query", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--max-queries",
        type=int,
        default=12,
        help="Cap on how many queries to draw (0 = no cap; rendering a "
        "large evaluation set in one figure is extremely slow).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    results = pd.read_csv(args.results)
    out = render_grid(
        results,
        args.reference,
        args.query,
        args.output,
        top_k=args.top_k,
        max_queries=args.max_queries,
    )
    logger.info("Saved %s", out)


if __name__ == "__main__":
    main()
