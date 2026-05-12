# Data

This repository commits only the small synthetic sample in `data/sample/`.

Do not commit full raw datasets, private images, or downloaded model weights. For real evaluation, place external identity-folder datasets under ignored paths such as:

```text
data/dogfacenet/source/    # populated by src/fetch_dogfacenet.py
data/dogfacenet/split/     # populated by src/prepare_reid_split.py
data/raw/
data/processed/
```

`src/fetch_dogfacenet.py` pulls the HuggingFace mirror of DogFaceNet (`dimidagd/DogFaceNet_224resize`) into identity folders. `src/prepare_reid_split.py` then builds an identity-disjoint split with known gallery identities and query-only unknown identities. See `DATASET_CARD.md` for the protocol and the manifest fields.
