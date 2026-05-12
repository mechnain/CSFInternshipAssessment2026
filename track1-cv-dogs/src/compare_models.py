"""Compare pretrained backbones on the same ReID evaluation.

For each model we run the full pipeline (gallery prototypes -> query
embeddings -> cosine ranking -> open-set decisions), evaluate against
the provided labels, and record metrics + per-image latency. Results
are written to a single CSV so reviewers can compare backbones at a
glance.

Models compared by default: DINOv2 ViT-S/14, ResNet50 (ImageNet),
EfficientNet-B0 (ImageNet). Each backbone is loaded fresh, so the
first run downloads weights for any model not yet cached.

Usage:
  python src/compare_models.py \\
      --reference data/sample/reference \\
      --query     data/sample/query \\
      --labels    data/sample/labels.csv \\
      --output    results/model_comparison.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import evaluate
from src.utils import (
    DEFAULT_TAU_MATCH,
    DEFAULT_TAU_POSSIBLE,
    OpenSetThresholds,
    build_gallery_prototypes,
    discover_gallery,
    discover_images,
    embed_images,
    load_embedder,
    rank_gallery,
    set_seed,
)

logger = logging.getLogger("compare")

DEFAULT_MODELS = ["dinov2", "resnet50", "efficientnet_b0"]


def _run_one(
    model_name: str,
    reference: Path,
    query: Path,
    labels: pd.DataFrame,
    thresholds: OpenSetThresholds,
    device: str,
    batch_size: int,
    top_k: int,
) -> dict:
    """Run one backbone end-to-end. Returns a metrics row (or an error row)."""
    out: dict = {"model": model_name}
    try:
        t = time.perf_counter()
        model, transform = load_embedder(model_name, device=device)
        out["model_load_s"] = round(time.perf_counter() - t, 3)

        gallery = discover_gallery(reference)
        t = time.perf_counter()
        gallery_ids, prototypes = build_gallery_prototypes(
            gallery, model, transform, device=device, batch_size=batch_size
        )
        out["gallery_embed_s"] = round(time.perf_counter() - t, 3)

        query_paths = discover_images(query)
        if not query_paths:
            raise FileNotFoundError(f"No query images under {query}.")
        t = time.perf_counter()
        q_feats = embed_images(
            query_paths, model, transform, device=device, batch_size=batch_size
        )
        query_embed_s = time.perf_counter() - t
        out["query_embed_s"] = round(query_embed_s, 3)
        out["sec_per_query_image"] = round(query_embed_s / len(query_paths), 4)

        results_df = rank_gallery(
            query_paths,
            q_feats,
            gallery_ids,
            prototypes,
            thresholds,
            top_k,
            round_similarity=None,
        )
        metrics = evaluate(results_df, labels, thresholds)

        out.update(
            {
                "n_gallery_identities": len(gallery_ids),
                "n_query_images": len(query_paths),
                "rank_1": metrics["closed_set"]["rank_1_accuracy"],
                "rank_5": metrics["closed_set"]["rank_5_accuracy"],
                "mAP": metrics["closed_set"]["mAP"],
                "precision_at_tau": metrics["thresholded"]["precision"],
                "recall_at_tau": metrics["thresholded"]["recall"],
                "f1_at_tau": metrics["thresholded"]["f1"],
                "unknown_accuracy": metrics["open_set"]["unknown_accuracy"],
                "auroc_known_vs_unknown": metrics["open_set"]["auroc_known_vs_unknown"],
                "error": "",
            }
        )
    except Exception as exc:  # noqa: BLE001 -- we want to report any failure
        logger.exception("Model %s failed: %s", model_name, exc)
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare pretrained backbones for dog ReID.")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--query", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Backbones to compare. Default: {DEFAULT_MODELS}",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--tau-match", type=float, default=DEFAULT_TAU_MATCH)
    p.add_argument("--tau-possible", type=float, default=DEFAULT_TAU_POSSIBLE)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    set_seed(args.seed)

    labels = pd.read_csv(args.labels)
    thresholds = OpenSetThresholds(match=args.tau_match, possible=args.tau_possible)

    rows = []
    for model_name in args.models:
        logger.info("=== Evaluating %s ===", model_name)
        rows.append(
            _run_one(
                model_name,
                args.reference,
                args.query,
                labels,
                thresholds,
                args.device,
                args.batch_size,
                args.top_k,
            )
        )

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Wrote %s", args.output)

    # Print a compact summary so terminals are immediately useful.
    summary_cols = [
        c
        for c in [
            "model",
            "rank_1",
            "mAP",
            "f1_at_tau",
            "unknown_accuracy",
            "auroc_known_vs_unknown",
            "sec_per_query_image",
            "error",
        ]
        if c in df.columns
    ]
    with pd.option_context("display.max_colwidth", 80, "display.width", 200):
        logger.info("Comparison:\n%s", df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
