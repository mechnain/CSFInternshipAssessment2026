"""Materialise the HuggingFace DogFaceNet_224resize parquet into an
identity-folder layout that prepare_reid_split.py can consume.

The HuggingFace mirror at:

  https://huggingface.co/datasets/dimidagd/DogFaceNet_224resize

stores rows of (image_bytes, label) in parquet. This script:

  1. Downloads the parquet shard(s) into the local HF cache.
  2. Loads them with pandas / pyarrow.
  3. Keeps only the top ``--max-identities`` labels by image count (so a
     CPU evaluation finishes in a reasonable time). Pass 0 to keep all.
  4. Drops identities with fewer than ``--min-images-per-identity``
     images so the downstream split has refs + queries to work with.
  5. Writes each image to ``<output>/dog_<label>/img_<idx>.jpg``.

Output is intended to live under ``data/dogfacenet/source/`` which is
gitignored.

Usage:
  python src/fetch_dogfacenet.py \\
      --output data/dogfacenet/source \\
      --max-identities 100 \\
      --min-images-per-identity 3 \\
      --seed 0
"""

from __future__ import annotations

import argparse
import io
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

HF_REPO_ID = "dimidagd/DogFaceNet_224resize"
HF_REPO_TYPE = "dataset"
DEFAULT_PARQUET_FILENAME = "data/train-00000-of-00001.parquet"

logger = logging.getLogger("fetch")


def _download_parquet(filename: str, cache_dir: Path | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required. It ships with timm, but you can "
            "also `pip install huggingface_hub` explicitly."
        ) from exc

    kwargs = {
        "repo_id": HF_REPO_ID,
        "repo_type": HF_REPO_TYPE,
        "filename": filename,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    local_path = Path(hf_hub_download(**kwargs))
    return local_path


def _decode_image(value) -> Image.Image:
    """The HF Image feature serialises to a dict with key 'bytes' or a
    raw bytes payload depending on writer version. Handle both."""
    if isinstance(value, dict):
        payload = value.get("bytes") or value.get("path")
        if not payload:
            raise ValueError(f"Unrecognised image cell: {value!r}")
    else:
        payload = value
    return Image.open(io.BytesIO(payload)).convert("RGB")


def extract(
    output: Path,
    max_identities: int,
    min_images_per_identity: int,
    parquet_filename: str,
    cache_dir: Path | None,
    overwrite: bool,
    seed: int,
) -> dict:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output} already exists and is not empty. Pass --overwrite."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s/%s ...", HF_REPO_ID, parquet_filename)
    parquet_path = _download_parquet(parquet_filename, cache_dir)
    logger.info("Cached at %s (%.1f MB).", parquet_path, parquet_path.stat().st_size / 1e6)

    logger.info("Loading parquet (this may take a few seconds) ...")
    df = pd.read_parquet(parquet_path, columns=["image", "label"])
    logger.info("Loaded %d rows with %d unique labels.", len(df), df["label"].nunique())

    # Count images per label and sort by count desc, then label asc for
    # deterministic tie-breaking.
    counts = df["label"].value_counts().rename_axis("label").reset_index(name="n")
    counts = counts.sort_values(["n", "label"], ascending=[False, True]).reset_index(drop=True)

    eligible = counts[counts["n"] >= min_images_per_identity]
    if max_identities and max_identities > 0:
        eligible = eligible.head(max_identities)
    keep_labels = set(eligible["label"].tolist())
    logger.info(
        "Keeping %d identities (>= %d images each). Total images to write: %d.",
        len(keep_labels),
        min_images_per_identity,
        int(eligible["n"].sum()),
    )

    df = df[df["label"].isin(keep_labels)].copy()
    # Deterministic within-identity ordering.
    df = df.sort_values(["label"]).reset_index(drop=True)

    n_written = 0
    last_label = None
    idx_in_id = 0
    for _, row in df.iterrows():
        label = str(row["label"])
        if label != last_label:
            idx_in_id = 1
            last_label = label
        else:
            idx_in_id += 1

        ident_dir = output / f"dog_{label}"
        ident_dir.mkdir(parents=True, exist_ok=True)
        img = _decode_image(row["image"])
        img.save(ident_dir / f"img_{idx_in_id:03d}.jpg", format="JPEG", quality=92)
        n_written += 1

    manifest = output / "_fetch_manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                f"source=hf://{HF_REPO_ID}/{parquet_filename}",
                f"seed={seed}",
                f"max_identities={max_identities}",
                f"min_images_per_identity={min_images_per_identity}",
                f"identities_written={len(keep_labels)}",
                f"images_written={n_written}",
            ]
        )
        + "\n"
    )

    return {
        "identities": len(keep_labels),
        "images": n_written,
        "output": str(output),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download + extract DogFaceNet_224resize from HuggingFace.")
    p.add_argument("--output", type=Path, required=True, help="Identity-folder root, e.g. data/dogfacenet/source")
    p.add_argument(
        "--max-identities",
        type=int,
        default=0,
        help="Keep only the top-N identities by image count (0 = no cap).",
    )
    p.add_argument(
        "--min-images-per-identity",
        type=int,
        default=3,
        help="Drop identities with fewer than this many images.",
    )
    p.add_argument(
        "--parquet-filename",
        default=DEFAULT_PARQUET_FILENAME,
        help="Path within the HF repo to the parquet shard.",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional override for the HF cache directory.",
    )
    p.add_argument("--seed", type=int, default=0, help="Recorded in the fetch manifest only.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    summary = extract(
        output=args.output,
        max_identities=args.max_identities,
        min_images_per_identity=args.min_images_per_identity,
        parquet_filename=args.parquet_filename,
        cache_dir=args.cache_dir,
        overwrite=args.overwrite,
        seed=args.seed,
    )
    print(
        f"OK -> {summary['identities']} identities, "
        f"{summary['images']} images at {summary['output']}"
    )


if __name__ == "__main__":
    main()
