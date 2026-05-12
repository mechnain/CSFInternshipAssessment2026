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

### Real evaluation results

Dataset: DogFaceNet HuggingFace mirror (`dimidagd/DogFaceNet_224resize`),
top 100 identities, identity-disjoint split (`seed=0`,
`refs_per_identity=2`, `queries_per_identity=3`,
`open_set_fraction=0.10`). 90 known identities (180 reference images,
270 closed-set queries) and 10 open-set identities (30 unknown
queries). Full split parameters: see `DATASET_CARD.md` and
`split_manifest.txt`.

DINOv2 ViT-S/14 at default thresholds (`tau_match=0.70`,
`tau_possible=0.55`):

| metric | value |
|---|---|
| Closed-set Rank-1 | 0.893 |
| Closed-set Rank-5 | 0.985 |
| Closed-set mAP | 0.935 |
| Open-set unknown accuracy (hard reject) | 0.100 |
| Open-set non-match rejection rate | 0.800 |
| Open-set known-vs-unknown AUROC | 0.871 |
| Hard-match precision / recall / F1 | 0.908 / 0.733 / 0.811 |
| CPU latency | 0.078 s / query image |

Decision distribution at default thresholds: 218 `match`, 78
`possible_match`, 4 `unknown` (out of 300 queries). The thresholds are
deliberately conservative: most uncertain queries are routed to
`possible_match` (human review) rather than committed to a hard
`unknown`.

### Backbone comparison (same split, default thresholds)

| backbone | Rank-1 | Rank-5 | mAP | F1 @ tau=0.70 | unknown_acc | AUROC | sec / img |
|---|---|---|---|---|---|---|---|
| DINOv2 ViT-S/14 | **0.893** | **0.985** | **0.935** | **0.811** | 0.100 | **0.871** | 0.078 |
| EfficientNet-B0 | 0.856 | 0.978 | 0.910 | 0.758 | **0.167** | 0.847 | **0.022** |
| ResNet50 | 0.826 | 0.959 | 0.879 | 0.792 | 0.033 | 0.866 | 0.067 |

DINOv2 leads on retrieval quality and on the threshold-independent
open-set signal (AUROC). EfficientNet-B0 is ~3.5x faster on CPU at a
cost of ~4 Rank-1 points and is the better choice if throughput
dominates. ResNet50 trails on retrieval but stays close on AUROC,
which is consistent with its features still separating known from
unknown reasonably even when fine-grained ranking is weaker.

### Threshold sweep summary

`evaluate.py --sweep` (results in `results/metrics_sweep.csv`) on
DINOv2:

| tau_match | F1 | unknown_accuracy | non_match_rejection |
|---|---|---|---|
| 0.62 | 0.859 | 0.033 | 0.433 |
| 0.70 (default) | 0.811 | 0.100 | 0.800 |
| 0.78 | 0.522 | 0.367 | 0.933 |

The sweep is a diagnostic, not a recommendation. Final thresholds
should be chosen on a validation fold and frozen before reporting;
`evaluate.py` writes a `selection_warning` into `metrics.json` when
sweep results are produced.

The committed `data/sample/` synthetic data is for smoke testing only
and contributes no numbers in this section.

## Known limitations and failure modes

- **No fine-tuning.** Frozen pretrained features ignore the cost surface
  of "what makes two dogs the same individual." A small metric-learning
  fine-tune (triplet / ArcFace) on identity pairs would typically lift
  Rank-1 by several points.
- **100-identity evaluation cap.** Numbers above are for the top 100
  most-photographed DogFaceNet identities. Scaling to the full 1,393
  identities will introduce more impostor identities and almost
  certainly lower Rank-1; that experiment is the obvious next step.
- **Threshold sensitivity.** The decision rule reduces to two scalars on
  top-1 similarity. Optimal thresholds shift with backbone, image
  quality, gallery composition, and the prior probability of an open-set
  query. The `--sweep` flag on `evaluate.py` is useful on validation data
  only; final test metrics should use thresholds chosen beforehand.
- **Same-coat confusion (observed).** In `failure_cases.png`,
  dark-coated dogs are confidently mis-matched to other dark-coated
  identities at similarities 0.71-0.75. The true identity is usually
  within the top-5 at similarity within 0.02 of rank 1; this drives
  most of the Rank-1 / Rank-5 gap (0.893 vs. 0.985).
- **Open-set abstention bias (observed).** Most unknown-query failures
  land between `tau_possible=0.55` and `tau_match=0.70`, producing
  `possible_match` rather than a confident `unknown`. Hard
  `unknown_accuracy` is 0.100 even though non-match rejection is 0.800
  and AUROC is 0.871. The separating signal is in the embedding space;
  threshold choice is leaving it on the table.
- **Global features only.** No body / face fusion, no part-based
  matching, no re-ranking. Same-breed near-duplicates and occluded
  faces are the expected failure modes.
- **Prototype gallery.** Multiple reference images per identity are
  collapsed into a single mean embedding. This is cheap but loses
  multi-view information; per-image gallery retrieval with a
  max-over-gallery-images rule is a straightforward upgrade.
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
