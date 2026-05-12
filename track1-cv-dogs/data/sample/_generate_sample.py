"""Generate a tiny deterministic synthetic sample for smoke testing.

These images are NOT real dogs. They are stylised colored blobs that let
the pipeline run end-to-end without requiring any dataset download. They
intentionally contain enough visual structure that a pretrained backbone
can still produce distinguishable embeddings per "identity", but they do
NOT constitute a real ReID evaluation. Real numbers must come from a
genuine dataset like DogFaceNet.

Directory and naming convention (matches the assessment spec):

  data/sample/reference/<dog_id>/<dog_id>_refN.jpg
  data/sample/query/<dog_id>_queryN.jpg            # known dogs
  data/sample/query/<dog_id>_unknownN.jpg          # open-set absent dogs
  data/sample/labels.csv  (columns: query_image,true_id)

Run from the repo:
  python track1-cv-dogs/data/sample/_generate_sample.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent
REF_DIR = ROOT / "reference"
QRY_DIR = ROOT / "query"
LABELS = ROOT / "labels.csv"

IMG_SIZE = 224
JPEG_QUALITY = 90
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Identities that appear in BOTH the gallery and the query set (closed-set).
KNOWN_IDENTITIES = ["dog_001", "dog_002", "dog_003", "dog_004"]
N_REFS_PER_KNOWN = 2
N_QUERIES_PER_KNOWN = 2

# Identities that appear ONLY in the query set (open-set unknowns).
UNKNOWN_IDENTITIES = ["dog_005", "dog_006"]
N_QUERIES_PER_UNKNOWN = 1


def _palette(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "bg_color": tuple(int(c) for c in rng.integers(40, 200, size=3)),
        "blob_color": tuple(int(c) for c in rng.integers(40, 230, size=3)),
        "blob_xy": [(int(rng.integers(40, 184)), int(rng.integers(40, 184))) for _ in range(4)],
        "blob_r": [int(rng.integers(20, 55)) for _ in range(4)],
    }


def _render(identity_seed: int, instance_seed: int) -> Image.Image:
    palette = _palette(identity_seed)
    rng = np.random.default_rng(instance_seed)

    bg = tuple(
        int(np.clip(c + rng.integers(-12, 13), 0, 255)) for c in palette["bg_color"]
    )
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg)
    draw = ImageDraw.Draw(img)

    for (x, y), r in zip(palette["blob_xy"], palette["blob_r"]):
        dx, dy = (int(v) for v in rng.integers(-8, 9, size=2))
        color = tuple(
            int(np.clip(c + rng.integers(-20, 21), 0, 255))
            for c in palette["blob_color"]
        )
        draw.ellipse(
            (x + dx - r, y + dy - r, x + dx + r, y + dy + r),
            fill=color,
        )

    return img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 1.2))))


def _save_jpg(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=JPEG_QUALITY, optimize=True)


def _clear(dir_path: Path) -> None:
    if not dir_path.exists():
        return
    for f in dir_path.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES:
            f.unlink()


def main() -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    QRY_DIR.mkdir(parents=True, exist_ok=True)
    _clear(REF_DIR)
    _clear(QRY_DIR)

    rows: list[tuple[str, str]] = [("query_image", "true_id")]

    # Reference images for each known identity: <dog_id>/<dog_id>_refN.jpg.
    for i, ident in enumerate(KNOWN_IDENTITIES):
        ident_dir = REF_DIR / ident
        ident_dir.mkdir(exist_ok=True)
        for k in range(N_REFS_PER_KNOWN):
            img = _render(identity_seed=100 + i, instance_seed=200 + 10 * i + k)
            _save_jpg(img, ident_dir / f"{ident}_ref{k + 1}.jpg")

    # Closed-set query images: <dog_id>_queryN.jpg with true_id=<dog_id>.
    for i, ident in enumerate(KNOWN_IDENTITIES):
        for k in range(N_QUERIES_PER_KNOWN):
            name = f"{ident}_query{k + 1}.jpg"
            img = _render(identity_seed=100 + i, instance_seed=500 + 10 * i + k)
            _save_jpg(img, QRY_DIR / name)
            rows.append((name, ident))

    # Open-set query images: <dog_id>_unknownN.jpg with true_id="unknown".
    for j, ident in enumerate(UNKNOWN_IDENTITIES):
        for k in range(N_QUERIES_PER_UNKNOWN):
            name = f"{ident}_unknown{k + 1}.jpg"
            img = _render(identity_seed=900 + j, instance_seed=950 + j * 10 + k)
            _save_jpg(img, QRY_DIR / name)
            rows.append((name, "unknown"))

    with LABELS.open("w", newline="") as f:
        csv.writer(f).writerows(rows)

    n_ref = len(KNOWN_IDENTITIES) * N_REFS_PER_KNOWN
    n_qry_known = len(KNOWN_IDENTITIES) * N_QUERIES_PER_KNOWN
    n_qry_unknown = len(UNKNOWN_IDENTITIES) * N_QUERIES_PER_UNKNOWN
    print(
        f"Generated {n_ref} reference images across {len(KNOWN_IDENTITIES)} identities."
    )
    print(
        f"Generated {n_qry_known + n_qry_unknown} query images "
        f"({n_qry_known} closed-set, {n_qry_unknown} open-set)."
    )
    print(f"Wrote labels -> {LABELS}")


if __name__ == "__main__":
    main()
