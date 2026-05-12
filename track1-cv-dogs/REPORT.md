# Report

## 1. Approach

This prototype treats dog re-identification as an image retrieval problem, not as breed classification. The goal is to decide whether a query image shows the same individual dog as a reference image or reference gallery.

The system uses frozen pretrained image encoders to convert images into L2-normalized embeddings. For each gallery identity, embeddings from available reference images are averaged and normalized again to form one prototype. Each query embedding is compared against all gallery prototypes using cosine similarity. Higher cosine similarity means the query is closer to that gallery identity in the learned feature space.

I selected pretrained embeddings because the assessment time window and limited identity-labeled data make training from scratch unreliable. A rushed supervised model would likely overfit and would be harder to evaluate honestly. DINOv2 ViT-S/14 is the default backbone because self-supervised features are often strong for fine-grained retrieval. ResNet50 and EfficientNet-B0 are included as simpler baselines for comparison.

The system also supports open-set recognition. Instead of forcing every query to match a gallery dog, the top-1 similarity is mapped to `match`, `possible_match`, or `unknown`. `possible_match` is treated as an abstention for human review, not as an automatic identification.

## 2. Evaluation

The evaluation reports both retrieval quality and decision quality. Retrieval is measured with Rank-1 accuracy, Rank-5 accuracy, and mAP. Threshold behavior is measured with precision, recall, F1, unknown accuracy, non-match rejection rate, abstention counts, and AUROC for separating known from unknown queries.

### Dataset and split

Real evaluation uses the HuggingFace mirror of DogFaceNet (`dimidagd/DogFaceNet_224resize`, 8,363 images, 1,393 identities, pre-resized to 224x224). To keep CPU runtime tractable, the prototype was evaluated on the top 100 identities by image count using `src/fetch_dogfacenet.py`, then split with `src/prepare_reid_split.py` (`seed=0`, `refs_per_identity=2`, `queries_per_identity=3`, `open_set_fraction=0.10`). The resulting split is identity-disjoint: 90 known identities (180 reference images, 270 closed-set queries) and 10 open-set identities held out of the gallery entirely (30 queries labeled `unknown`). `split_manifest.txt` records every parameter so the split is reproducible.

The committed `data/sample/` dataset is separate and synthetic; it exists only to let reviewers run every command without a data download. No metric in this report comes from synthetic data.

### Headline results (DINOv2 ViT-S/14, default thresholds tau_match=0.70 / tau_possible=0.55)

| metric | value |
|---|---|
| Closed-set Rank-1 | 0.893 |
| Closed-set Rank-5 | 0.985 |
| Closed-set mAP | 0.935 |
| Open-set unknown accuracy (hard reject) | 0.100 |
| Open-set non-match rejection rate (unknown + possible_match) | 0.800 |
| Open-set known-vs-unknown AUROC | 0.871 |
| Hard-match precision / recall / F1 | 0.908 / 0.733 / 0.811 |
| CPU latency, DINOv2 | 0.078 s/query image |

The Rank-1 / Rank-5 gap (0.893 -> 0.985) shows the embedding space is doing most of the work: the correct identity is almost always in the top 5 even when it loses rank 1. Open-set AUROC of 0.871 means known and unknown queries are well separated by top-1 similarity in principle. The hard-reject rate of 0.100 reflects the chosen default threshold, not the embedding space's limit. Non-match rejection of 0.800 means the system declines to commit to a confident match on 80% of unknown queries (it either abstains as `possible_match` or rejects as `unknown`).

### Threshold behaviour

`evaluate.py --sweep` scans `tau_match` over `[0.50, 1.00]` step 0.02 (results in `results/metrics_sweep.csv`). The sweep shows the trade-off explicitly:

| tau_match | F1 | unknown_accuracy | non_match_rejection |
|---|---|---|---|
| 0.62 | 0.859 | 0.033 | 0.433 |
| 0.70 (default) | 0.811 | 0.100 | 0.800 |
| 0.78 | 0.637 | 0.467 | 0.967 |

F1 peaks around 0.62 but at the cost of accepting almost all unknown queries. Default 0.70 is the working compromise. Operationally I would not use the sweep recommendation directly; thresholds must be picked on a validation fold and frozen before final reporting, which is reflected in the `selection_warning` field that `evaluate.py` writes into `metrics.json` when the sweep runs.

### Backbone comparison (same split)

| model | Rank-1 | mAP | F1 @ tau=0.70 | unknown_acc | AUROC | sec/img |
|---|---|---|---|---|---|---|
| dinov2 | 0.893 | 0.935 | 0.811 | 0.100 | 0.871 | 0.078 |
| efficientnet_b0 | 0.856 | 0.910 | 0.758 | 0.167 | 0.847 | 0.022 |
| resnet50 | 0.826 | 0.879 | 0.792 | 0.033 | 0.866 | 0.067 |

DINOv2 leads on retrieval and on AUROC, which is consistent with the self-supervised feature literature for fine-grained matching. EfficientNet-B0 is ~3.5x faster on CPU for ~4 points of Rank-1 - useful for any throughput-constrained deployment. ResNet50 trails in retrieval but stays close in AUROC, suggesting its features still separate known from unknown well even when ranking is weaker.

## 3. Failure Modes (observed in `results/failure_cases.png`)

Out of 300 queries, 99 (33%) land in the failure visualization. Breakdown: 14 `wrong_match` (closed-set, confident but wrong identity), 57 `ambiguous_known` (abstention on a known query), 1 `false_negative` (known query rejected as `unknown`), 6 `false_alarm` (unknown query confidently matched to a gallery identity), 21 `ambiguous_unknown` (unknown query in the abstention band). Two distinct mechanisms dominate.

**Failure mode 1: recurring identity-pair confusion at high similarity.** The 14 `wrong_match` cases are not random noise; the same gallery identity attracts multiple queries from the same wrong querier. Both `dog_585_query2.jpg` and `dog_585_query3.jpg` rank-1 to `dog_456` (similarities 0.74 and 0.826). Both `dog_599_query1.jpg` and `dog_599_query3.jpg` rank-1 to `dog_788` (0.78 and 0.72). Gallery identity `dog_497` attracts wrong queries from two distinct individuals (`dog_653` and `dog_723`). The full wrong-match similarity range is 0.700-0.826, well above the default `tau_match=0.70`, so these are confident wrong decisions, not edge cases. The true identity is often present at rank 2-5 within ~0.02 of rank 1, which is what the Rank-1 / Rank-5 gap (0.893 vs 0.985) reflects. Mitigations: (a) per-image gallery retrieval (max over reference images) instead of a single prototype, which would preserve a high-similarity view that prototype averaging smooths away, (b) re-ranking with a more expensive pairwise comparison on the top-k candidates, (c) a small metric-learning fine-tune that targets the recurring confusion pairs directly.

**Failure mode 2: open-set queries land in the abstention band, not in `unknown`.** Of 30 unknown queries, only 3 receive a hard `unknown` decision; 21 land between `tau_possible=0.55` and `tau_match=0.70` (similarities 0.561-0.697) and become `possible_match`, and 6 cross `tau_match` and become confident `false_alarm`. One open-set identity in particular is poorly handled: all 3 queries from `dog_598` rank-1 to gallery `dog_13` at similarities 0.703, 0.740, and 0.812. This pattern is consistent with the metric picture: unknown_accuracy=0.100 is low, non_match_rejection=0.800 is high (most unknowns are at least abstained on), and AUROC=0.871 says the embedding space already contains most of the separating signal - threshold choice is leaving it on the table. Mitigations: (a) calibrate `tau_possible` on a validation fold against the operating cost of human review vs false alarms, (b) add an explicit known-vs-unknown classifier on top of the top-1 similarity to exploit the AUROC signal, (c) use per-identity prototype variance as an extra feature, since well-represented identities should have tighter similarity distributions than under-represented ones.

A third pattern that I cannot quantify without per-image inspection is global-embedding leakage of background and pose into identity. Detection or segmentation before embedding would let that be measured rather than guessed at.

## 4. Generalization To Sheep

Sheep ReID is materially harder than dog ReID for two reasons. First, public identity-labeled sheep data does not exist at DogFaceNet's scale, so pretrained features have nothing comparable to lean on. Second, sheep within one flock often look visually similar by design (same breed, similar coat, similar body type), and appearance shifts with wool growth, shearing, mud, lighting, pose, and season.

I would change four things relative to this prototype.

1. **Data collection comes first.** A meaningful sheep system needs multiple reference images per animal, an explicit unknown set, controlled lighting/distance, and a documented capture protocol. The split tooling here (`prepare_reid_split.py` + `split_manifest.txt`) is the same shape regardless of species.
2. **Detect or segment first, then embed.** Sheep are typically photographed in crowded scenes; whole-image embeddings will encode pen, background, and neighbouring animals. A detector or segmenter is non-optional.
3. **Region-specific embeddings.** Faces, ear markings, and ear tags are the highest-signal regions for sheep ID. The right approach is region-conditioned embeddings (face + ear-tag + body) fused at the score level, not a single global embedding.
4. **Calibrate against the deployment cost matrix.** False identity assignment is more expensive than abstention in a farm setting. The default `tau_match` would be raised, `possible_match` would route to a human, and the AUROC signal would be exposed directly as a confidence score rather than collapsed into a hard decision.

I would not start with fully automated sheep identity decisions. I would start with retrieval assistance: top candidates, calibrated confidence, and an abstention label. Confirmed corrections feed back into active learning and eventually into a metric-learning fine-tune once enough labeled identity pairs accumulate.

## 5. Practical Deployment Considerations

For deployment, false identity assignment is riskier than abstention. The system should prioritize calibrated uncertainty, log human corrections, and use confirmed corrections for future active learning. Auxiliary signals such as ear tags, time, location, pen grouping, or farm records could help resolve visually ambiguous cases. The CPU latency numbers (DINOv2 0.078 s/image; EfficientNet-B0 0.022 s/image) put either backbone comfortably inside an interactive triage workflow.

## 6. Limitations And Next Steps

This is a prototype, not a production model. Specific known limitations:

- **Frozen features.** No metric-learning fine-tune; the embedding space was not optimized for "same individual vs different individual" specifically.
- **100-identity cap.** The reported numbers are for the top-100 most-photographed identities. Running on the full 1,393-identity DogFaceNet would almost certainly lower Rank-1 (more impostor identities to confuse with) and is the obvious next experiment.
- **Single similarity strategy.** Cosine over global features only; no body/face fusion, no part-based matching, no re-ranking.
- **Prototype averaging.** Multiple reference images are collapsed into one mean embedding. Per-image gallery retrieval would preserve view-specific evidence.
- **No image-quality gating.** Blur, occlusion, and low-resolution queries are not detected or rejected.
- **No detector or segmenter.** Background and pose leak into embeddings.

Highest-value next steps:

1. Re-run on the full 1,393-identity DogFaceNet to characterise scaling behaviour.
2. Tune thresholds on a held-out validation fold and freeze them before final reporting.
3. Add per-image gallery retrieval (max over reference images) and compare to prototype averaging.
4. Add a dog detector or segmenter before embedding to remove background influence.
5. Try a small triplet/ArcFace fine-tune on a few thousand identity pairs.
6. Add image-quality gating for blur, occlusion, and low resolution.
