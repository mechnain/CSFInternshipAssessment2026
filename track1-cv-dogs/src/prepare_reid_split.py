"""Create an identity-disjoint dog ReID split from identity folders.

Expected input layout:

  source/<identity>/*.jpg

Output layout:

  output/reference/<identity>/*.jpg
  output/query/*.jpg
  output/labels.csv

Unknown identities are held out of the gallery entirely and appear only as
query rows with true_id="unknown".
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import IMAGE_SUFFIXES


def _identity_images(source: Path) -> dict[str, list[Path]]:
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    identities: dict[str, list[Path]] = {}
    for identity_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        images = sorted(
            p
            for p in identity_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if images:
            identities[identity_dir.name] = images
    if not identities:
        raise RuntimeError(f"No identity folders with images found under {source}")
    return identities


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def create_split(
    source: Path,
    output: Path,
    refs_per_identity: int,
    open_set_fraction: float,
    seed: int,
    overwrite: bool,
    queries_per_identity: int = 0,
) -> Path:
    if refs_per_identity < 1:
        raise ValueError("refs_per_identity must be at least 1.")
    if not 0.0 < open_set_fraction < 1.0:
        raise ValueError("open_set_fraction must be between 0 and 1.")
    if queries_per_identity < 0:
        raise ValueError("queries_per_identity must be >= 0 (0 = no cap).")
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output} already exists and is not empty. Pass --overwrite to replace it."
            )
        shutil.rmtree(output)

    identities = _identity_images(source)
    usable = {
        identity: images
        for identity, images in identities.items()
        if len(images) > refs_per_identity
    }
    if len(usable) < 2:
        raise RuntimeError(
            "Need at least two identities with more images than refs_per_identity."
        )

    rng = random.Random(seed)
    identity_names = sorted(usable)
    rng.shuffle(identity_names)

    n_open = max(1, round(len(identity_names) * open_set_fraction))
    n_open = min(n_open, len(identity_names) - 1)
    open_ids = set(identity_names[:n_open])
    known_ids = identity_names[n_open:]

    ref_dir = output / "reference"
    query_dir = output / "query"
    labels_path = output / "labels.csv"
    rows: list[tuple[str, str]] = [("query_image", "true_id")]

    n_known_queries = 0
    for identity in sorted(known_ids):
        images = usable[identity][:]
        rng.shuffle(images)
        refs = images[:refs_per_identity]
        queries = images[refs_per_identity:]
        if queries_per_identity:
            queries = queries[:queries_per_identity]
        for idx, src in enumerate(refs, start=1):
            dst = ref_dir / identity / f"{identity}_ref{idx}{src.suffix.lower()}"
            _copy(src, dst)
        for idx, src in enumerate(queries, start=1):
            name = f"{identity}_query{idx}{src.suffix.lower()}"
            _copy(src, query_dir / name)
            rows.append((name, identity))
            n_known_queries += 1

    n_open_queries = 0
    for identity in sorted(open_ids):
        images = usable[identity][:]
        rng.shuffle(images)
        if queries_per_identity:
            images = images[:queries_per_identity]
        for idx, src in enumerate(images, start=1):
            name = f"{identity}_unknown{idx}{src.suffix.lower()}"
            _copy(src, query_dir / name)
            rows.append((name, "unknown"))
            n_open_queries += 1

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)

    manifest = output / "split_manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                f"source={source.resolve()}",
                f"seed={seed}",
                f"refs_per_identity={refs_per_identity}",
                f"queries_per_identity={queries_per_identity}",
                f"open_set_fraction={open_set_fraction}",
                f"known_identities={len(known_ids)}",
                f"open_set_identities={len(open_ids)}",
                f"known_query_images={n_known_queries}",
                f"open_set_query_images={n_open_queries}",
                f"query_images={n_known_queries + n_open_queries}",
            ]
        )
        + "\n"
    )
    return labels_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare an identity-disjoint ReID split.")
    p.add_argument("--source", type=Path, required=True, help="Input identity-folder root.")
    p.add_argument("--output", type=Path, required=True, help="Output split directory.")
    p.add_argument("--refs-per-identity", type=int, default=2)
    p.add_argument(
        "--queries-per-identity",
        type=int,
        default=0,
        help=(
            "Cap on query images per identity (applies to known AND "
            "open-set identities). 0 = no cap, take all remaining images."
        ),
    )
    p.add_argument("--open-set-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels = create_split(
        source=args.source,
        output=args.output,
        refs_per_identity=args.refs_per_identity,
        open_set_fraction=args.open_set_fraction,
        seed=args.seed,
        overwrite=args.overwrite,
        queries_per_identity=args.queries_per_identity,
    )
    print(f"Wrote split labels -> {labels}")


if __name__ == "__main__":
    main()
