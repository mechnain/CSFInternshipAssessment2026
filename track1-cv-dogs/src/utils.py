"""Shared helpers for the dog ReID pipeline.

Deliberately small: image IO, deterministic transforms, embedding model
loading, gallery prototype building, cosine similarity, and the open-set
decision rule. All downstream scripts compose these primitives.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Conservative starting defaults; reviewers should tune them on a validation
# split from the real target distribution before reporting test metrics.
DEFAULT_TAU_MATCH = 0.70
DEFAULT_TAU_POSSIBLE = 0.55


@dataclass(frozen=True)
class OpenSetThresholds:
    """Two thresholds on top-1 cosine similarity drive the decision."""

    match: float = DEFAULT_TAU_MATCH
    possible: float = DEFAULT_TAU_POSSIBLE

    def __post_init__(self) -> None:
        if not 0.0 <= self.possible <= self.match <= 1.0:
            raise ValueError(
                "Expected thresholds to satisfy 0 <= possible <= match <= 1. "
                f"Got possible={self.possible}, match={self.match}."
            )

    def decide(self, top1_similarity: float) -> str:
        if top1_similarity >= self.match:
            return "match"
        if top1_similarity >= self.possible:
            return "possible_match"
        return "unknown"


# ---------- File discovery ----------

def discover_images(root: Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def discover_gallery(root: Path) -> dict[str, list[Path]]:
    """Discover the gallery layout.

    Two conventions are accepted:
      * reference/<identity>/<image>  -> one identity per subfolder (preferred).
      * reference/<image>             -> each image is its own identity.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Reference directory not found: {root}")
    subdirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if subdirs:
        gallery = {d.name: discover_images(d) for d in subdirs}
        return {k: v for k, v in gallery.items() if v}
    return {p.stem: [p] for p in discover_images(root)}


def load_image(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ---------- Embedding backbones ----------

def _build_transform(img_size: int) -> Callable[[Image.Image], "Tensor"]:  # noqa: F821
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_embedder(model_name: str, device: str = "cpu"):
    """Return (model, transform) for the chosen pretrained backbone.

    Only DINOv2 ViT-S/14 (primary) and ResNet50 (lightweight fallback) are
    supported in this minimal build. No fine-tuning.
    """
    import torch
    from torch import nn

    model_name = model_name.lower()
    device_t = torch.device(device)

    if model_name == "dinov2":
        try:
            import timm
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "timm is required for the dinov2 backbone. "
                "Install it via `pip install timm`."
            ) from exc
        model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            img_size=224,
            dynamic_img_size=True,
        )
        transform = _build_transform(224)
    elif model_name == "resnet50":
        from torchvision import models

        weights = models.ResNet50_Weights.IMAGENET1K_V2
        net = models.resnet50(weights=weights)
        # Drop the classifier; keep the 2048-d penultimate features.
        net.fc = nn.Identity()
        model = net
        transform = _build_transform(224)
    elif model_name == "efficientnet_b0":
        try:
            import timm
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "timm is required for the efficientnet_b0 backbone. "
                "Install it via `pip install timm`."
            ) from exc
        model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        transform = _build_transform(224)
    else:
        raise ValueError(
            f"Unknown model {model_name!r}. "
            f"Supported: 'dinov2', 'resnet50', 'efficientnet_b0'."
        )

    model.eval()
    model.to(device_t)
    return model, transform


def embed_images(
    paths: Iterable[Path],
    model,
    transform,
    device: str = "cpu",
    batch_size: int = 16,
) -> np.ndarray:
    """Return L2-normalised embeddings, shape (N, D)."""
    import torch

    paths = list(paths)
    if not paths:
        return np.zeros((0, 0), dtype=np.float32)

    device_t = torch.device(device)
    feats: list[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            tensors = torch.stack(
                [transform(load_image(p)) for p in batch_paths]
            ).to(device_t)
            out = model(tensors)
            if isinstance(out, (tuple, list)):
                out = out[0]
            feats.append(out.detach().float().cpu().numpy())

    arr = np.concatenate(feats, axis=0)
    return l2_normalize(arr)


# ---------- Linear algebra ----------

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_similarity(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Cosine similarity for already-L2-normalised vectors = dot product."""
    return query @ gallery.T


def rank_gallery(
    query_paths: list[Path],
    query_features: np.ndarray,
    gallery_ids: list[str],
    gallery_features: np.ndarray,
    thresholds: OpenSetThresholds,
    top_k: int,
    round_similarity: int | None = 6,
) -> "pd.DataFrame":
    """Build the ranked per-query result table used by all entry points."""
    import pandas as pd

    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}.")
    if len(query_paths) != len(query_features):
        raise ValueError(
            f"Query path/feature count mismatch: {len(query_paths)} paths, "
            f"{len(query_features)} feature rows."
        )
    if len(gallery_ids) != len(gallery_features):
        raise ValueError(
            f"Gallery id/feature count mismatch: {len(gallery_ids)} ids, "
            f"{len(gallery_features)} feature rows."
        )

    sims = cosine_similarity(query_features, gallery_features)
    top_k = min(top_k, len(gallery_ids))
    order = (-sims).argsort(axis=1)[:, :top_k]

    rows: list[dict] = []
    for qi, qpath in enumerate(query_paths):
        top1_sim = float(sims[qi, order[qi, 0]])
        decision = thresholds.decide(top1_sim)
        for rank_idx in range(top_k):
            gi = int(order[qi, rank_idx])
            sim = float(sims[qi, gi])
            rows.append(
                {
                    "query": qpath.name,
                    "rank": rank_idx + 1,
                    "gallery_id": gallery_ids[gi],
                    "similarity": round(sim, round_similarity)
                    if round_similarity is not None
                    else sim,
                    "decision": decision if rank_idx == 0 else "",
                }
            )

    return pd.DataFrame(
        rows,
        columns=["query", "rank", "gallery_id", "similarity", "decision"],
    )


def build_gallery_prototypes(
    gallery: dict[str, list[Path]],
    model,
    transform,
    device: str = "cpu",
    batch_size: int = 16,
) -> tuple[list[str], np.ndarray]:
    """Mean of per-image embeddings, re-normalised. Standard ReID baseline."""
    ids: list[str] = []
    protos: list[np.ndarray] = []
    for identity, paths in gallery.items():
        feats = embed_images(paths, model, transform, device=device, batch_size=batch_size)
        if feats.size == 0:
            logger.warning("No usable images for identity %s; skipping.", identity)
            continue
        proto = feats.mean(axis=0, keepdims=True)
        protos.append(l2_normalize(proto))
        ids.append(identity)
    if not ids:
        raise RuntimeError("Gallery contains no usable images.")
    return ids, np.vstack(protos)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
