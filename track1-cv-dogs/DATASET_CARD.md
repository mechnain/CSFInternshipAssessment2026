# Dataset Card

This card covers the two datasets referenced by this prototype:

1. The tiny **synthetic sample** committed under `data/sample/` (smoke test only).
2. The **DogFaceNet** dataset, recommended for real evaluation (not committed).

## 1. Synthetic sample (committed)

| Field | Value |
|---|---|
| Location | `data/sample/` |
| Size | ~40 KB total, 18 JPEG images |
| Identities (gallery) | 4 known (`dog_001`..`dog_004`), 2 reference images each |
| Open-set identities | 2 (`dog_005`, `dog_006`) -- query-only, never in gallery |
| Query images | 10 (8 closed-set positives, 2 open-set unknowns) |
| Image size | 224x224 |
| Image format | JPEG, quality 90 |
| Generator | `data/sample/_generate_sample.py` (deterministic) |
| License | None required (procedurally generated) |

### Folder layout

```text
data/sample/reference/
  dog_001/dog_001_ref1.jpg, dog_001_ref2.jpg
  dog_002/dog_002_ref1.jpg, dog_002_ref2.jpg
  dog_003/dog_003_ref1.jpg, dog_003_ref2.jpg
  dog_004/dog_004_ref1.jpg, dog_004_ref2.jpg

data/sample/query/
  dog_001_query1.jpg, dog_001_query2.jpg       # true_id = dog_001
  dog_002_query1.jpg, dog_002_query2.jpg       # true_id = dog_002
  dog_003_query1.jpg, dog_003_query2.jpg       # true_id = dog_003
  dog_004_query1.jpg, dog_004_query2.jpg       # true_id = dog_004
  dog_005_unknown1.jpg, dog_006_unknown1.jpg   # true_id = unknown
```

### Labels schema

`labels.csv` has exactly two columns:

```csv
query_image,true_id
dog_001_query1.jpg,dog_001
dog_001_query2.jpg,dog_001
dog_002_query1.jpg,dog_002
dog_005_unknown1.jpg,unknown
```

`true_id = "unknown"` marks an open-set query whose dog is absent from
the gallery. The evaluator also accepts the older
`query_filename,true_dog_id` column names for backward compatibility,
but new label files should use the schema above.

These images are **not real dogs**. They are colored blobs on a colored
background, produced by a deterministic generator. They exist solely so
that the four CLI commands in `README.md` run end-to-end without any
data download. Numbers obtained on this sample are **not** a real ReID
evaluation: identities are visually trivial to separate, and the
similarity distribution is far tighter than for natural images.

To regenerate (or extend):

```bash
python data/sample/_generate_sample.py
```

## 2. DogFaceNet (recommended for real evaluation, not committed)

| Field | Value |
|---|---|
| Source | <https://github.com/GuillaumeMougeot/DogFaceNet> |
| Description | Dog face images annotated with individual identity. |
| Approx. scale | ~8,000 images across ~1,400 individual dogs (varies by release). |
| Image content | Cropped, roughly frontal dog faces. |
| License | See upstream repository; we do not redistribute. |
| Intended task | Individual dog face re-identification. |

### Why DogFaceNet rather than Stanford Dogs or similar

Stanford Dogs and OpenImages provide **breed**-level labels (and sometimes
not even that). ReID requires **per-individual** identity labels:
multiple images of the same specific dog. DogFaceNet is one of the few
public datasets that supplies this for dogs.

### Splits used in this prototype

We do not commit DogFaceNet or a real split file. Reviewers running on
real data should construct one with `src/prepare_reid_split.py`, which
implements the following protocol:

1. Shuffle identities (not images) with a fixed seed.
2. Reserve ~10% of identities as **open-set unknowns** -- they appear in
   the query set but never in the reference gallery.
3. From the remaining ~90% of identities, take 1-2 images per identity
   as the gallery reference (`reference/<identity>/...`).
4. Take the remaining images of those identities as closed-set queries.
5. Mix in the open-set identities' images as additional queries, with
   `true_id = "unknown"` in `labels.csv`.

Example:

```bash
python src/prepare_reid_split.py \
    --source data/dogfacenet/images \
    --output data/processed/dogfacenet_seed0 \
    --refs-per-identity 2 \
    --open-set-fraction 0.10 \
    --seed 0
```

Identities must be **disjoint** between gallery and "unknown" queries.
If the same individual appears in both, the open-set evaluation is
invalid.

### Required folder layout (same as the sample)

```text
reference/<identity>/<identity>_refN.jpg     # gallery, multiple refs per identity
query/<filename>.jpg                          # any filename; mapped to truth via labels.csv
labels.csv                                    # columns: query_image,true_id
```

`true_id` may be `unknown` for open-set queries.

## Known biases and limitations

- **Pose bias.** DogFaceNet images are roughly frontal face crops; a
  pipeline tuned on this distribution will struggle on side-on body
  shots, occluded faces, or low-light field images.
- **Breed skew.** Some breeds are over-represented. Same-breed near-
  duplicate confusion is a known failure mode.
- **Image quality.** Resolution and lighting vary; we do not currently
  filter or quality-gate inputs.
- **No metadata.** Age, sex, coat colour, and other attributes are not
  provided in our pipeline. They could help with negative mining if
  available.
- **No identity-disjoint test set is shipped.** Reviewers must
  construct one as above for an honest evaluation.

## Out-of-scope datasets

- Sheep, cattle, or other livestock: this pipeline is not trained or
  evaluated on those species. See the sheep generalization discussion
  in the report (when produced) for what would need to change.
