# Model Card

This card covers the **system** assembled in this prototype, not any
single neural network. The system is a thin wrapper around pretrained
image encoders.

## Summary

| Field | Value |
|---|---|
| Task | Individual dog re-identification (1:N gallery retrieval with open-set rejection). |
| Inputs | RGB images (any size; resized + center-cropped to 224x224). |
| Outputs | Per-query top-k ranked gallery identities with cosine similarity, plus an open-set decision in `{match, possible_match, unknown}`. |
| Training | **None.** All backbones are frozen pretrained models. |
| Fine-tuning | None in this build. |
| Primary backbone | DINOv2 ViT-S/14 (`vit_small_patch14_dinov2.lvd142m` via `timm`). |
| Alternative backbones | ResNet50 (ImageNet-1k V2 via `torchvision`), EfficientNet-B0 (ImageNet via `timm`). |
| Similarity metric | Cosine similarity on L2-normalised global features. |
| Gallery representation | Mean of per-image L2-normalised embeddings, re-normalised (prototype). |
| Open-set rule | Two-threshold rule on top-1 similarity: `>= tau_match -> match`, `>= tau_possible -> possible_match`, else `unknown`. Defaults `tau_match=0.70`, `tau_possible=0.55`. |

## Intended use

- Prototyping and offline analysis of dog re-identification on small to
  medium galleries.
- Triage tool: surface a ranked shortlist for a human reviewer.
- Pedagogical baseline for the CSF Summer 2026 hiring assessment.

## Out-of-scope use

- Any production deployment without further validation.
- Legal or compensatory decisions about animal ownership.
- Species other than dogs (sheep, cattle, etc.) without re-evaluation.
- Forensic identification claims; the system is a similarity ranker, not
  an identity oracle.

## Why these backbones

| Backbone | Rationale |
|---|---|
| DINOv2 ViT-S/14 | Self-supervised on a large diverse image corpus; widely reported as strong on fine-grained retrieval, which is what individual ReID needs. Small enough for CPU. Primary default. |
| ResNet50 | Long-standing ImageNet baseline. Cheap on CPU. Useful sanity check: if a backbone barely beats ResNet50, the lift from a fancier model is questionable. |
| EfficientNet-B0 | Compact CNN with strong throughput on CPU. Lets us see whether the gap to DINOv2 is mostly architecture (ViT vs CNN) or mostly training objective (supervised vs self-supervised). |

CLIP was deliberately **not** added in this build to avoid an extra
heavy dependency. It is an obvious next backbone to compare and is
listed in "Next steps" below.

## Evaluation

The same `evaluate.py` script reports closed-set Rank-1 / Rank-5 / mAP,
hard-match precision / recall / F1, open-set unknown accuracy,
open-set non-match rejection rate, and threshold-independent open-set
AUROC (separation of known vs. unknown queries by top-1 similarity).
`possible_match` is treated as an abstention, not as a true negative.

Real numbers should be produced against an identity-disjoint DogFaceNet
split (see `DATASET_CARD.md`). The synthetic sample committed to this
repo is for smoke testing only and its numbers are not informative.

## Known limitations and failure modes

- **No fine-tuning.** Frozen pretrained features ignore the cost surface
  of "what makes two dogs the same individual." A small metric-learning
  fine-tune (triplet / ArcFace) on identity pairs would typically lift
  Rank-1 by several points.
- **Threshold sensitivity.** The decision rule reduces to two scalars on
  top-1 similarity. Optimal thresholds shift with backbone, image
  quality, gallery composition, and the prior probability of an open-set
  query. The `--sweep` flag on `evaluate.py` is useful on validation data
  only; final test metrics should use thresholds chosen beforehand.
- **Global features only.** No body / face fusion, no part-based
  matching, no re-ranking. Same-breed near-duplicates and occluded
  faces are the expected failure modes.
- **Prototype gallery.** We collapse multiple reference images per
  identity into a single mean embedding. This is cheap but loses
  multi-view information; per-image gallery retrieval with a max-over-
  gallery-images rule is a straightforward upgrade.
- **No quality gating.** Blurred, low-light, or heavily occluded query
  images are not detected or rejected.
- **No calibration.** Cosine similarity is not a probability and is not
  calibrated; the decision rule treats it as a score for thresholding.

## Bias considerations

- The pretrained backbones inherit the biases of their training data
  (LVD-142M for DINOv2; ImageNet-1k for ResNet50 and EfficientNet-B0).
  Breeds, lighting conditions, and capture styles over-represented
  there will yield better embeddings than under-represented ones.
- The gallery prototype strategy assumes reference images are
  representative. If a dog's gallery shots are all frontal close-ups,
  the system will be biased against side-on field shots of the same
  animal.

## Next steps (informing follow-up work)

1. Add CLIP ViT-B/32 (`open_clip_torch`) to `compare_models.py`.
2. Run on DogFaceNet with an identity-disjoint split; sweep thresholds.
3. Add per-image gallery retrieval (max over gallery images) alongside
   the prototype strategy and compare.
4. Add a small metric-learning fine-tune (triplet or ArcFace) on a few
   thousand identity pairs.
5. Add image-quality gating to drop unusable queries before retrieval.
