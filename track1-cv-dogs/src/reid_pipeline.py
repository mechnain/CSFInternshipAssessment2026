"""End-to-end dog re-identification pipeline.

Steps:
  1. Build gallery prototypes from `--reference/<identity>/...`.
  2. Embed every image under `--query`.
  3. Rank gallery identities per query by cosine similarity.
  4. Apply the open-set threshold rule on top-1 similarity.
  5. Write a per-query top-k CSV to `--output`.

Usage:
  python src/reid_pipeline.py \
      --reference data/sample/reference \
      --query     data/sample/query \
      --output    results/ranked_results.csv \
      --top-k 10
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow `python src/reid_pipeline.py` from the track1 root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

logger = logging.getLogger("reid")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dog re-identification pipeline.")
    p.add_argument("--reference", type=Path, required=True, help="Reference gallery dir.")
    p.add_argument("--query", type=Path, required=True, help="Query images dir.")
    p.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--model",
        choices=["dinov2", "resnet50", "efficientnet_b0"],
        default="dinov2",
        help=(
            "Pretrained backbone. dinov2 is the default; resnet50 and "
            "efficientnet_b0 are CPU-friendly baselines."
        ),
    )
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--tau-match", type=float, default=DEFAULT_TAU_MATCH)
    p.add_argument("--tau-possible", type=float, default=DEFAULT_TAU_POSSIBLE)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def run(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    thresholds = OpenSetThresholds(match=args.tau_match, possible=args.tau_possible)

    logger.info("Loading %s on %s ...", args.model, args.device)
    t0 = time.perf_counter()
    model, transform = load_embedder(args.model, device=args.device)
    logger.info("Model ready in %.1fs.", time.perf_counter() - t0)

    gallery = discover_gallery(args.reference)
    gallery_ids, prototypes = build_gallery_prototypes(
        gallery, model, transform, device=args.device, batch_size=args.batch_size
    )
    logger.info(
        "Gallery: %d identities, %d total reference images.",
        len(gallery_ids),
        sum(len(v) for v in gallery.values()),
    )

    query_paths = discover_images(args.query)
    if not query_paths:
        raise FileNotFoundError(f"No query images under {args.query}.")
    logger.info("Embedding %d query images ...", len(query_paths))
    q_feats = embed_images(
        query_paths, model, transform, device=args.device, batch_size=args.batch_size
    )

    df = rank_gallery(query_paths, q_feats, gallery_ids, prototypes, thresholds, args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Wrote %s (%d rows).", args.output, len(df))
    return args.output


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(parse_args())


if __name__ == "__main__":
    main()
