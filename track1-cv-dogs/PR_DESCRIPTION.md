# AI Programmer Assessment - Dog Re-Identification Pipeline

**Full Name:** Hasnain Shaikh
**Email:** hasnainshaikhwork@gmail.com
**Selected Track:** AI Programmer / Computer Vision Track

## Summary

This PR adds a runnable dog re-identification prototype for the Canadian Sheep Federation AI Programmer assessment. The system compares a reference dog image or small reference gallery against query images, returns ranked candidate matches, and supports open-set decisions when the queried dog may not be present in the gallery.

The project is framed as individual animal ReID, not breed classification. The core question is:

> Is this query image the same individual dog as the reference?

It is evaluated on an identity-disjoint split of DogFaceNet (HuggingFace mirror, 100 most-photographed identities, 300 queries, 10 open-set unknowns). Headline numbers, default thresholds, DINOv2 ViT-S/14: Rank-1 0.893, Rank-5 0.985, mAP 0.935, open-set AUROC 0.871, F1 0.811, non-match rejection 0.800.

## What I Built

- Pretrained embedding-based ReID pipeline.
- Cosine-similarity ranking against gallery prototypes.
- Open-set decisions: `match`, `possible_match`, and `unknown`, with a configurable two-threshold rule.
- Evaluation script for retrieval and threshold metrics, including AUROC and an optional threshold sweep.
- Success/failure visualization scripts.
- Backbone comparison across DINOv2 ViT-S/14, ResNet50, and EfficientNet-B0.
- Deterministic synthetic sample for offline smoke testing without any data download.
- Real-data ingestion: `src/fetch_dogfacenet.py` pulls the HuggingFace mirror of DogFaceNet, caps to top-N identities, and writes identity-folder format.
- Identity-disjoint split builder (`src/prepare_reid_split.py`) with `--queries-per-identity` for balanced evaluation and a reproducibility manifest.
- Dataset card, model card, report, and evaluator tests.

## Key Technical Decisions

I used frozen pretrained visual encoders instead of training from scratch. The assessment window and limited identity-labeled animal data make a pretrained retrieval baseline more reliable, inspectable, and reproducible than rushed fine-tuning.

Each image is embedded, L2-normalized, and compared with cosine similarity. Multiple reference images for one identity are averaged into a normalized prototype. This keeps the retrieval logic simple and easy to audit.

Open-set recognition is included because real animal identification systems should not force every query to match a known gallery animal. `possible_match` is treated as a human-review abstention rather than a true negative.

DINOv2 ViT-S/14 is the default backbone because self-supervised features are commonly strong on fine-grained retrieval. ResNet50 and EfficientNet-B0 are kept as cheap baselines so reviewers can see the cost/quality trade-off explicitly.

## How To Run

From `track1-cv-dogs/`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Smoke test on the committed synthetic sample (no download required):

```bash
python src/reid_pipeline.py ^
  --reference data/sample/reference ^
  --query data/sample/query ^
  --output results/ranked_results.csv ^
  --top-k 10

python src/evaluate.py ^
  --results results/ranked_results.csv ^
  --labels data/sample/labels.csv ^
  --threshold 0.70 ^
  --output results/metrics.json
```

Real evaluation on DogFaceNet (top 100 identities, fully reproducible):

```bash
python src/fetch_dogfacenet.py ^
  --output data/dogfacenet/source ^
  --max-identities 100 ^
  --min-images-per-identity 3

python src/prepare_reid_split.py ^
  --source data/dogfacenet/source ^
  --output data/dogfacenet/split ^
  --refs-per-identity 2 ^
  --queries-per-identity 3 ^
  --open-set-fraction 0.10 ^
  --seed 0 ^
  --overwrite

python src/reid_pipeline.py ^
  --reference data/dogfacenet/split/reference ^
  --query data/dogfacenet/split/query ^
  --output results/ranked_results.csv ^
  --top-k 10 ^
  --model dinov2

python src/evaluate.py ^
  --results results/ranked_results.csv ^
  --labels data/dogfacenet/split/labels.csv ^
  --threshold 0.70 ^
  --possible-threshold 0.55 ^
  --output results/metrics.json ^
  --sweep

python src/visualize_cases.py ^
  --results results/ranked_results.csv ^
  --labels data/dogfacenet/split/labels.csv ^
  --reference data/dogfacenet/split/reference ^
  --query data/dogfacenet/split/query ^
  --success-output results/success_cases.png ^
  --failure-output results/failure_cases.png

python src/compare_models.py ^
  --reference data/dogfacenet/split/reference ^
  --query data/dogfacenet/split/query ^
  --labels data/dogfacenet/split/labels.csv ^
  --output results/model_comparison.csv
```

Tests:

```bash
python -m unittest discover -s tests
python -m compileall src tests
```

## Evaluation Results

DogFaceNet (HuggingFace mirror `dimidagd/DogFaceNet_224resize`), 100 identities (top-N by image count), `seed=0`, `refs_per_identity=2`, `queries_per_identity=3`, `open_set_fraction=0.10`. Split: 90 known identities (180 reference images, 270 closed-set queries), 10 open-set identities (30 unknown queries). All thresholds at defaults (`tau_match=0.70`, `tau_possible=0.55`).

| metric | DINOv2 ViT-S/14 | ResNet50 | EfficientNet-B0 |
|---|---|---|---|
| Closed-set Rank-1 | **0.893** | 0.826 | 0.856 |
| Closed-set Rank-5 | **0.985** | 0.959 | 0.978 |
| Closed-set mAP | **0.935** | 0.879 | 0.910 |
| Hard-match F1 | **0.811** | 0.792 | 0.758 |
| Open-set unknown accuracy | 0.100 | 0.033 | **0.167** |
| Open-set non-match rejection | **0.800** | n/a | n/a |
| Open-set AUROC | **0.871** | 0.866 | 0.847 |
| CPU latency (sec/query image) | 0.078 | 0.067 | **0.022** |

Notes:
- The Rank-1 / Rank-5 gap (0.89 -> 0.99) means the correct identity is almost always in the top 5; most of the failure mass is rank confusion within visually similar coats, not retrieval misses.
- `unknown_accuracy=0.100` is the hard-reject rate at the default threshold; `non_match_rejection=0.800` is the fraction of unknown queries the system declines to confidently match (`unknown` or `possible_match`). AUROC=0.871 says the separation signal is in the embedding space; threshold choice controls how much of it is captured.
- The threshold sweep (`results/metrics_sweep.csv`) shows F1 peaks near `tau_match=0.62` at 0.859 but with `unknown_accuracy=0.033`. Default 0.70 is the operating compromise; final thresholds should be picked on a validation fold.

The evaluator reports:

- Rank-1, Rank-5, mAP
- precision, recall, and F1 for hard `match` decisions
- unknown accuracy and non-match rejection rate
- known-vs-unknown AUROC
- abstention counts and confusion-style decision summary

## Screenshots / Output

Local artifacts generated by the commands above:

- `results/ranked_results.csv` (300 rows, top-10 each)
- `results/metrics.json`
- `results/metrics_sweep.csv`
- `results/model_comparison.csv`
- `results/top_matches.png`
- `results/success_cases.png`
- `results/failure_cases.png`

These are gitignored. The visualization PNGs are attached to the PR for review.

## What Did Not Work

Closed-set ranking alone is not enough. At default thresholds the model confidently mis-matches two distinct kinds of queries:

1. **Same-coat near-duplicates.** Dark-coated identities can score 0.71-0.75 against the wrong individual; the true identity sits at rank 2 or 3 within 0.02 similarity. This is the main driver of the Rank-1 / Rank-5 gap.
2. **Open-set queries in the abstention band.** Unknown dogs frequently land between `tau_possible=0.55` and `tau_match=0.70`, producing `possible_match` instead of a confident `unknown`. Hard `unknown_accuracy` is therefore low (0.100) even though non-match rejection is high (0.800).

The threshold sweep makes the trade-off explicit but does not magically fix it; calibration would need to happen on validation data with the cost of false alarms weighed against the cost of human review.

## Known Limitations

- Only the top-100 DogFaceNet identities were evaluated; full 1,393-identity scaling has not been measured.
- No dog-specific fine-tuning. Embeddings are frozen.
- No detector or segmentation model; global image embeddings can encode background and pose.
- Prototype averaging for gallery identities loses view-specific information.
- No blur, occlusion, or low-resolution quality gating.
- Threshold defaults are starting points; for production they must be tuned on a held-out validation set and frozen before final reporting.

## Sheep Generalization

Sheep ReID would be harder than dog ReID because sheep in one flock can be visually similar by design, appearance changes with wool growth, shearing, mud, lighting, pose, and season, and public identity-labeled sheep data is limited.

A practical sheep system should start as retrieval assistance, not fully automated identity assignment. It should detect or segment the animal before embedding, fuse region-specific features (face / ear-tag / body), return calibrated top candidates, route `possible_match` cases to human review, and use confirmed corrections for active learning. Ear tags, capture time, pen, and farm records could provide useful auxiliary signals.

## Next Steps

1. Re-run on the full 1,393-identity DogFaceNet to characterise scaling.
2. Tune open-set thresholds on a held-out validation fold; freeze them before test reporting.
3. Compare prototype averaging with max-over-reference-image retrieval.
4. Add dog detection or segmentation before embedding extraction.
5. Add image-quality checks for blur, occlusion, and low resolution.
6. Add a small triplet/ArcFace fine-tune on identity pairs once enough data is available.
