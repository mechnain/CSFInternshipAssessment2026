# Dog Re-Identification Pipeline

Canadian Sheep Federation Summer 2026 AI Programmer Assessment - Computer Vision Track

This repository contains a runnable prototype for **individual dog re-identification**. It is not a breed classifier. Given a reference dog image or small reference gallery, the system ranks query images by visual similarity and can abstain when the queried dog is likely absent from the gallery.

## Quick Review

- Approach: frozen pretrained image embeddings + cosine similarity.
- Default model: DINOv2 ViT-S/14.
- Baselines: ResNet50 and EfficientNet-B0.
- Outputs: ranked CSV, metrics JSON, top-match visualization, success/failure visualizations, model comparison CSV.
- Open-set decisions: `match`, `possible_match`, `unknown`.
- Important limitation: committed sample data is synthetic and only proves the pipeline runs.

The current repository includes a deterministic synthetic sample under `data/sample/`. It is useful for smoke testing, but it is **not real ReID evidence**. For final benchmark claims, run the same commands on an identity-disjoint DogFaceNet or curated real-dog split.

## Why This Approach

Dog ReID asks whether two images show the same individual animal. A breed classifier is the wrong tool because it learns category-level differences, not individual identity.

This prototype uses frozen pretrained vision backbones because the assessment timeline and limited identity-labeled dog data make training from scratch risky. Each image is converted into an L2-normalized embedding. For each gallery identity, reference embeddings are averaged into one prototype and normalized again. Query embeddings are compared to gallery prototypes with cosine similarity.

Open-set recognition is included because real deployments should not force every query to match a gallery animal. A two-threshold rule maps the top-1 similarity to:

| Decision | Meaning |
|---|---|
| `match` | confident gallery match |
| `possible_match` | uncertain case for human review |
| `unknown` | likely absent from the gallery |

## Repository Layout

```text
track1-cv-dogs/
  src/
    reid_pipeline.py         # gallery + queries -> ranked CSV
    evaluate.py              # metrics JSON and optional validation sweep
    visualize_results.py     # top-k retrieval grid
    visualize_cases.py       # success/failure visualizations
    compare_models.py        # backbone comparison
    prepare_reid_split.py    # identity-disjoint split builder
    utils.py                 # image IO, embeddings, ranking, thresholds
  tests/
    test_evaluate.py
  data/
    sample/                  # synthetic smoke-test data
  results/
    .gitkeep                 # generated outputs are gitignored
  README.md
  REPORT.md
  DATASET_CARD.md
  MODEL_CARD.md
  PR_DESCRIPTION.md
  requirements.txt
```

## Setup

```bash
cd track1-cv-dogs
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

The first model run downloads pretrained weights to the local Torch/Hugging Face cache.

## Smoke-Test Commands

Run from `track1-cv-dogs/`.

```bash
python src/reid_pipeline.py ^
  --reference data/sample/reference ^
  --query data/sample/query ^
  --output results/ranked_results.csv ^
  --top-k 10
```

```bash
python src/evaluate.py ^
  --results results/ranked_results.csv ^
  --labels data/sample/labels.csv ^
  --threshold 0.70 ^
  --output results/metrics.json
```

```bash
python src/visualize_results.py ^
  --results results/ranked_results.csv ^
  --reference data/sample/reference ^
  --query data/sample/query ^
  --output results/top_matches.png ^
  --top-k 5
```

```bash
python src/visualize_cases.py ^
  --results results/ranked_results.csv ^
  --labels data/sample/labels.csv ^
  --reference data/sample/reference ^
  --query data/sample/query ^
  --success-output results/success_cases.png ^
  --failure-output results/failure_cases.png
```

```bash
python src/compare_models.py ^
  --reference data/sample/reference ^
  --query data/sample/query ^
  --labels data/sample/labels.csv ^
  --output results/model_comparison.csv
```

Generated files are local artifacts and are ignored by git:

```text
results/ranked_results.csv
results/metrics.json
results/model_comparison.csv
results/top_matches.png
results/success_cases.png
results/failure_cases.png
```

## Real Evaluation Workflow

Use an identity-folder dataset such as DogFaceNet:

```text
data/dogfacenet/images/
  dog_001/*.jpg
  dog_002/*.jpg
  ...
```

Create an identity-disjoint split:

```bash
python src/prepare_reid_split.py ^
  --source data/dogfacenet/images ^
  --output data/processed/dogfacenet_seed0 ^
  --refs-per-identity 2 ^
  --open-set-fraction 0.10 ^
  --seed 0 ^
  --overwrite
```

Then run the same pipeline:

```bash
python src/reid_pipeline.py ^
  --reference data/processed/dogfacenet_seed0/reference ^
  --query data/processed/dogfacenet_seed0/query ^
  --output results/ranked_results_dogfacenet.csv ^
  --top-k 10 ^
  --model dinov2
```

```bash
python src/evaluate.py ^
  --results results/ranked_results_dogfacenet.csv ^
  --labels data/processed/dogfacenet_seed0/labels.csv ^
  --sweep ^
  --output results/metrics_dogfacenet.json
```

Use `--sweep` only on validation data. Freeze selected thresholds before reporting final test metrics.

## Metrics

`evaluate.py` reports:

- closed-set Rank-1 accuracy
- Rank-5 accuracy
- mAP
- hard-match precision, recall, and F1
- unknown accuracy
- non-match rejection rate
- known-vs-unknown AUROC
- decision counts
- confusion-style summary
- abstention counts for `possible_match`

`possible_match` is treated as abstention, not as a true negative.

## Dataset Notes

`data/sample/` contains deterministic colored-blob JPEGs. They are not real dogs. They exist so reviewers can run every command without downloading data.

For meaningful evaluation, use an identity-labeled dog dataset with:

- known identities present in both gallery and query sets
- unknown identities present only in the query set
- multiple reference images per known identity
- a fixed seed and split manifest

See `DATASET_CARD.md` for details.

## Limitations

- No dog-specific fine-tuning.
- No detection or segmentation before embedding.
- Global image embeddings can learn background or pose artifacts.
- Prototype averaging can lose view-specific information.
- Thresholds are dataset-dependent.
- Synthetic sample metrics are not evidence of real-world performance.
- Real deployment would need quality checks, human review, and farm-specific validation.

## Tests

```bash
python -m unittest discover -s tests
python -m compileall src tests
```

## Next Steps

Highest-value improvements:

1. Run and report a real DogFaceNet or curated real-dog evaluation.
2. Add real positive, negative, and unknown-query artifacts to the PR.
3. Tune open-set thresholds on validation data and freeze them for test metrics.
4. Compare prototype averaging with max-over-reference-image retrieval.
5. Add dog detection or segmentation before embedding extraction.
6. Add image-quality checks for blur, occlusion, and low resolution.
