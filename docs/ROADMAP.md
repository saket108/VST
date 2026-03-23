# VSTDet Roadmap

## Phase 1: Pure Detection Baseline

Goal:
- establish a stable, image-only detector baseline
- keep the architecture simple enough to debug and reproduce
- produce one official baseline checkpoint and result table

Config:
- `configs/vstdet_pure_baseline.yaml`

Characteristics:
- `mobilenet_v3_large` backbone
- `bifusion` neck
- `fcos` assignment
- moderate augmentation
- practical runtime on Kaggle T4

Success criteria:
- reproducible training
- clean resume/checkpoint/eval workflow
- one official per-class validation table

## Phase 2: Competitive Pure Detector

Goal:
- push the image-only detector as far as practical before adding multimodal inputs
- benchmark against YOLO baselines

Config:
- `configs/vstdet_competitive_pure.yaml`

Characteristics:
- `convnext_tiny` backbone
- `cafpn` neck
- `atss` assignment
- stronger augmentation
- slower but stronger than the baseline path

Success criteria:
- best custom pure-detector result
- direct comparison against YOLO26n and YOLO11n

## Phase 3: Multimodal Detector

Only start this after Phase 2 is defensible.

Reason:
- without a strong pure detector, multimodal gains are hard to interpret
- the unimodal detector must be credible on its own first
