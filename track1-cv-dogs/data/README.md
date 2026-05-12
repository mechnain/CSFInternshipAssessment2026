# Data

This repository commits only the small synthetic sample in `data/sample/`.

Do not commit full raw datasets, private images, or downloaded model weights. For real evaluation, place external identity-folder datasets under ignored paths such as:

```text
data/dogfacenet/
data/raw/
data/processed/
```

Use `src/prepare_reid_split.py` to create an identity-disjoint split with known gallery identities and query-only unknown identities.
