# AI Programmer Assessment - Dog Re-Identification Pipeline

**Full Name:** Hasnain Shaikh
**Email:** hasnainshaikhwork@gmail.com
**Selected Track:** AI Programmer / Computer Vision Track

## Summary

This PR adds a runnable dog re-identification prototype for the Canadian Sheep Federation AI Programmer assessment. The system compares a reference dog image or small reference gallery against query images, returns ranked candidate matches, and supports open-set decisions when the queried dog may not be present in the gallery.

The project is framed as individual animal ReID, not breed classification. The core question is:

> Is this query image the same individual dog as the reference?

## What I Built

- Pretrained embedding-based ReID pipeline.
- Cosine-similarity ranking against gallery prototypes.
- Open-set decisions: `match`, `possible_match`, and `unknown`.
- Evaluation script for retrieval and threshold metrics.
- Success/failure visualization scripts.
- Backbone comparison across DINOv2, ResNet50, and EfficientNet-B0.
- Deterministic synthetic sample for smoke testing.
- Split-preparation utility for identity-folder datasets such as DogFaceNet.
- Dataset card, model card, report, and evaluator tests.

## Key Technical Decisions

I used frozen pretrained visual encoders instead of training from scratch. The assessment window and limited identity-labeled animal data make a pretrained retrieval baseline more reliable, inspectable, and reproducible than rushed fine-tuning.

Each image is embedded, L2-normalized, and compared with cosine similarity. Multiple reference images for one identity are averaged into a normalized prototype. This keeps the retrieval logic simple and easy to audit.

Open-set recognition is included because real animal identification systems should not force every query to match a known gallery animal. `possible_match` is treated as a human-review abstention rather than a true negative.

## How To Run

From `track1-cv-dogs/`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the smoke-test pipeline:

```bash
python src/reid_pipeline.py ^
  --reference data/sample/reference ^
  --query data/sample/query ^
  --output results/ranked_results.csv ^
  --top-k 10
```

Evaluate:

```bash
python src/evaluate.py ^
  --results results/ranked_results.csv ^
  --labels data/sample/labels.csv ^
  --threshold 0.70 ^
  --output results/metrics.json
```

Generate success/failure visualizations:

```bash
python src/visualize_cases.py ^
  --results results/ranked_results.csv ^
  --labels data/sample/labels.csv ^
  --reference data/sample/reference ^
  --query data/sample/query ^
  --success-output results/success_cases.png ^
  --failure-output results/failure_cases.png
```

Run tests:

```bash
python -m unittest discover -s tests
python -m compileall src tests
```

## Evaluation

The committed sample dataset is synthetic and exists only so the full pipeline can be run without downloading data. It should not be interpreted as real dog ReID performance.

The evaluator reports:

- Rank-1 accuracy
- Rank-5 accuracy
- mAP
- precision, recall, and F1 for hard `match` decisions
- unknown accuracy
- non-match rejection rate
- known-vs-unknown AUROC
- abstention counts
- confusion-style decision summary

For real evaluation, the repo includes `src/prepare_reid_split.py`, which creates identity-disjoint gallery/query splits from identity-folder data such as DogFaceNet.

## Screenshots / Output

Local generated outputs:

- `results/ranked_results.csv`
- `results/metrics.json`
- `results/model_comparison.csv`
- `results/top_matches.png`
- `results/success_cases.png`
- `results/failure_cases.png`

These are gitignored run artifacts. The PR should attach or show the visualization outputs when submitted.

## What Did Not Work

Closed-set ranking alone is not sufficient. The smoke run shows that a model can rank known synthetic identities correctly while still failing to reject unknown queries at the default threshold.

Thresholds are dataset-dependent. The defaults are only starting points and must be tuned on a validation split before test reporting.

Global embeddings can also encode background, crop style, pose, and lighting. A real deployment would need detection or segmentation, better gallery coverage, and image-quality checks.

## Known Limitations

- Synthetic committed sample only.
- No real dog benchmark committed.
- No dog-specific fine-tuning.
- No detector or segmentation model.
- Global image embeddings only.
- Simple prototype averaging for gallery identities.
- No blur, occlusion, or low-resolution quality gate.

## Sheep Generalization

Sheep ReID would be harder than dog ReID because sheep in one flock may be visually similar and public identity-labeled sheep data is limited. Appearance can change with wool growth, shearing, mud, lighting, pose, and seasonal conditions.

A practical sheep system should start as retrieval assistance, not fully automated identity assignment. It should return top candidates and uncertainty scores, route `possible_match` cases to human review, and use confirmed corrections for active learning. Ear tags, time, location, pen grouping, and farm records could provide useful auxiliary signals.

## Next Steps

1. Run a real identity-disjoint DogFaceNet or curated dog evaluation.
2. Attach real positive, negative, and unknown-query visualizations.
3. Tune thresholds on validation data and freeze them before final test metrics.
4. Compare prototype retrieval with max-over-reference-image retrieval.
5. Add dog detection or segmentation before embedding extraction.
6. Add image-quality checks for blur, occlusion, and low resolution.
