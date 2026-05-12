# Dog Re-Identification Pipeline

Canadian Sheep Federation Summer 2026 AI Programmer Assessment - Computer Vision Track

This repository contains a runnable prototype for **individual dog re-identification**. It is not a breed classifier. Given a reference dog image or small reference gallery, the system ranks query images by visual similarity and can abstain when the queried dog is likely absent from the gallery.

## Quick Review

- Approach: frozen pretrained image embeddings + cosine similarity.
- Default model: DINOv2 ViT-S/14.
- Baselines: ResNet50 and EfficientNet-B0.
- Outputs: ranked CSV, metrics JSON, top-match visualization, success/failure visualizations, model comparison CSV.
- Open-set decisions: `match`, `possible_match`, `unknown`.
- Evaluated on a 100-identity DogFaceNet split (HuggingFace mirror, identity-disjoint, 300 queries with 30 open-set unknowns). Headline numbers below.

The committed sample under `data/sample/` is deterministic synthetic data for smoke testing only; numbers in the **Real Results** section below come from DogFaceNet, not the sample.

## Real Results

DogFaceNet (`dimidagd/DogFaceNet_224resize` on HuggingFace), 100 identities, `seed=0`, `refs_per_identity=2`, `queries_per_identity=3`, `open_set_fraction=0.10`. Default thresholds `tau_match=0.70`, `tau_possible=0.55`.

| metric | DINOv2 ViT-S/14 | ResNet50 | EfficientNet-B0 |
|---|---|---|---|
| Rank-1 | **0.893** | 0.826 | 0.856 |
| Rank-5 | **0.985** | 0.959 | 0.978 |
| mAP | **0.935** | 0.879 | 0.910 |
| Hard-match F1 | **0.811** | 0.792 | 0.758 |
| Open-set AUROC | **0.871** | 0.866 | 0.847 |
| Open-set unknown accuracy | 0.100 | 0.033 | **0.167** |
| Non-match rejection rate (DINOv2) | 0.800 | - | - |
| CPU latency (sec/query image) | 0.078 | 0.067 | **0.022** |

Reproduce these numbers with the commands in [Real Evaluation Workflow](#real-evaluation-workflow).

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
    fetch_dogfacenet.py      # pulls DogFaceNet from HuggingFace into identity folders
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

The repo ships `src/fetch_dogfacenet.py` to pull DogFaceNet from its HuggingFace mirror (`dimidagd/DogFaceNet_224resize`) and write it as identity folders. Dataset images are gitignored; only outputs in `results/` and the manifest in `data/dogfacenet/split/split_manifest.txt` are reproducible artifacts.

Fetch the top-N most-photographed identities (capped at 100 here to keep CPU runtime reasonable):

```bash
python src/fetch_dogfacenet.py ^
  --output data/dogfacenet/source ^
  --max-identities 100 ^
  --min-images-per-identity 3
```

Create an identity-disjoint split:

```bash
python src/prepare_reid_split.py ^
  --source data/dogfacenet/source ^
  --output data/dogfacenet/split ^
  --refs-per-identity 2 ^
  --queries-per-identity 3 ^
  --open-set-fraction 0.10 ^
  --seed 0 ^
  --overwrite
```

`--queries-per-identity` caps query images per identity so the open-set and closed-set query counts stay balanced (here: 270 known + 30 unknown queries). All parameters are recorded in `split_manifest.txt`.

Run the pipeline, evaluate with a threshold sweep, and produce visualizations:

```bash
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

python src/visualize_results.py ^
  --results results/ranked_results.csv ^
  --reference data/dogfacenet/split/reference ^
  --query data/dogfacenet/split/query ^
  --output results/top_matches.png ^
  --top-k 5 ^
  --max-queries 12

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

`--sweep` writes `results/metrics_sweep.csv` with `tau_match` over `[0.50, 1.00]` step 0.02. Use it only on validation data; freeze the chosen thresholds before final test reporting. `--max-queries` on the top-matches visualization caps the rendered grid so it stays a manageable size on large splits.

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

- Real evaluation is capped at the top 100 DogFaceNet identities (CPU runtime). Full 1,393-identity scaling has not been measured and would likely lower Rank-1.
- No dog-specific fine-tuning; embeddings are frozen.
- No detection or segmentation before embedding; global features can encode background or pose.
- Prototype averaging can lose view-specific information.
- Thresholds are dataset-dependent. Defaults are starting points; production use requires tuning on a validation fold.
- The committed `data/sample/` is synthetic and is for smoke testing only - its numbers are not informative.
- Real deployment would need quality checks, human review, and farm-specific validation.

## Tests

```bash
python -m unittest discover -s tests
python -m compileall src tests
```

## Next Steps

Highest-value improvements:

1. Re-run on the full 1,393-identity DogFaceNet to characterise scaling behaviour.
2. Tune open-set thresholds on a held-out validation fold and freeze them before final test reporting.
3. Compare prototype averaging with max-over-reference-image retrieval.
4. Add dog detection or segmentation before embedding extraction.
5. Add image-quality checks for blur, occlusion, and low resolution.
6. Add a small triplet/ArcFace fine-tune on identity pairs once enough data is available.
