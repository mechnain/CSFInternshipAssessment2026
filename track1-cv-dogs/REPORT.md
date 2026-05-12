# Report

## 1. Approach

This prototype treats dog re-identification as an image retrieval problem, not as breed classification. The goal is to decide whether a query image shows the same individual dog as a reference image or reference gallery.

The system uses frozen pretrained image encoders to convert images into L2-normalized embeddings. For each gallery identity, embeddings from available reference images are averaged and normalized again to form one prototype. Each query embedding is compared against all gallery prototypes using cosine similarity. Higher cosine similarity means the query is closer to that gallery identity in the learned feature space.

I selected pretrained embeddings because the assessment time window and limited identity-labeled data make training from scratch unreliable. A rushed supervised model would likely overfit and would be harder to evaluate honestly. DINOv2 ViT-S/14 is the default backbone because self-supervised features are often strong for fine-grained retrieval. ResNet50 and EfficientNet-B0 are included as simpler baselines for comparison.

The system also supports open-set recognition. Instead of forcing every query to match a gallery dog, the top-1 similarity is mapped to `match`, `possible_match`, or `unknown`. `possible_match` is treated as an abstention for human review, not as an automatic identification.

## 2. Evaluation

The evaluation reports both retrieval quality and decision quality. Retrieval is measured with Rank-1 accuracy, Rank-5 accuracy, and mAP. Threshold behavior is measured with precision, recall, F1, unknown accuracy, non-match rejection rate, abstention counts, and AUROC for separating known from unknown queries.

The committed `data/sample/` dataset is synthetic and exists only for smoke testing. It verifies that the scripts, labels, outputs, and visualizations work end to end, but it is not evidence of real dog ReID performance. A meaningful evaluation should use an identity-disjoint DogFaceNet or curated real-dog split prepared with `src/prepare_reid_split.py`. Thresholds should be selected on validation data and frozen before final test reporting.

On the current synthetic smoke run, closed-set retrieval is trivial, while the default open-set threshold rejects none of the unknown queries. That result is useful because it shows why Rank-1 accuracy alone is not enough: open-set calibration must be evaluated separately.

## 3. Failure Modes

First, threshold calibration can fail even when ranking looks good. Unknown dogs may still receive high similarity to the nearest gallery prototype, producing false alarms. The mitigation is to tune thresholds on validation unknowns, report false alarms explicitly, and route uncertain cases to human review.

Second, global embeddings can confuse identity with context. Background, lighting, crop style, pose, or camera setup may influence similarity. Detection or segmentation before embedding would reduce background leakage.

Third, prototype averaging loses view-specific evidence. A single mean embedding can blur together frontal, side, and occluded views. A useful follow-up is max-over-reference-image retrieval alongside prototype retrieval.

## 4. Generalization To Sheep

Sheep ReID is harder than dog ReID because public identity-labeled sheep data is limited and many animals in a flock can look visually similar. Appearance changes with wool growth, shearing, mud, lighting, pose, and seasonal conditions. A sheep system would need farm-specific galleries, multiple images per animal, explicit unknown examples, and careful validation under the target camera setup.

I would not start with fully automated sheep identity decisions. I would start with retrieval assistance: return top candidates, confidence scores, and an uncertainty label, then let a human confirm difficult cases.

## 5. Practical Deployment Considerations

For deployment, false identity assignment is riskier than abstention. The system should prioritize calibrated uncertainty, log human corrections, and use confirmed corrections for future active learning. Auxiliary signals such as ear tags, time, location, pen grouping, or farm records could help resolve visually ambiguous cases.

## 6. Limitations And Next Steps

This is a prototype, not a production model. It uses frozen features, global image embeddings, simple prototype averaging, and no image-quality gating. The highest-value next steps are real DogFaceNet evaluation, validation-only threshold tuning, per-reference-image retrieval, dog detection or segmentation, image-quality checks, and eventual metric-learning fine-tuning if enough identity-labeled data is available.
