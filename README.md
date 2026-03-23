# VSTDet Research Starter

This project uses a cleaner research-style layout instead of a flat script package.

## Layout

- `configs/`: experiment YAML files
- `data/`: dataset loading
- `model/`: detector architecture and building blocks
- `training/`: losses and training engine
- `utils/`: box ops, point generation, and evaluation
- `tools/`: executable entrypoints

## What it is

`VSTDet` is a custom one-stage anchor-free detector built directly in PyTorch:

- pretrained `torchvision` backbones or the custom `ContextBridge` backbone
- bidirectional weighted feature fusion neck
- optional `CAFPN` neck for a stronger context-aware pyramid
- decoupled classification, box, and centerness head
- FCOS-style or ATSS-style point assignment and GIoU-based regression loss
- direct support for datasets in YOLO annotation format

This is a research baseline. The code is custom, but a paper contribution still comes from experiments, ablations, and evidence.

## Main files

- `tools/train.py`: training entrypoint
- `model/detector.py`: architecture definition
- `training/losses.py`: target assignment and losses
- `training/engine.py`: training and validation loop
- `data/dataset.py`: YOLO-format dataset loader
- `utils/evaluator.py`: lightweight `mAP50` and `mAP50-95`

## Recommended Path

The repo now has two explicit pure-detection phases:

- baseline: `configs/vstdet_pure_baseline.yaml`
- competitive: `configs/vstdet_competitive_pure.yaml`

The short roadmap is in `docs/ROADMAP.md`.

## Train

Config-driven run:

```powershell
python tools/train.py --config configs/vstdet_small.yaml --data path\to\data.yaml
```

CLI-driven run:

```powershell
python tools/train.py --data path\to\data.yaml --variant small --backbone efficientnet_v2_s --imgsz 896 --epochs 300 --batch-size 8
```

Stronger pretrained backbone:

```powershell
python tools/train.py --data path\to\data.yaml --variant small --backbone convnext_tiny --imgsz 896 --epochs 300 --batch-size 6
```

Stronger detector variant:

```powershell
python tools/train.py --data path\to\data.yaml --variant small --backbone convnext_tiny --neck cafpn --head-depth 3 --assigner atss --imgsz 640 --epochs 300 --batch-size 4
```

Pure detection baseline:

```powershell
python tools/train.py --config configs/vstdet_pure_baseline.yaml --data path\to\data.yaml
```

Competitive pure detector:

```powershell
python tools/train.py --config configs/vstdet_competitive_pure.yaml --data path\to\data.yaml
```

Lighter pretrained backbone:

```powershell
python tools/train.py --data path\to\data.yaml --variant tiny --backbone mobilenet_v3_large --imgsz 896 --epochs 300 --batch-size 12
```

Custom backbone ablation:

```powershell
python tools/train.py --data path\to\data.yaml --variant small --backbone custom --no-pretrained-backbone --imgsz 896 --epochs 300 --batch-size 8
```

## Colab quickstart

If you want a short Colab verification run, use the Colab config:

```bash
python tools/train.py --config configs/vstdet_colab_10ep.yaml --data /content/drive/MyDrive/Aircraft_dataset/data.yaml
```

Recommended Colab setup:

1. Enable GPU in Colab.
2. Clone the repo and install requirements:

```bash
git clone https://github.com/saket108/VST.git
cd VST
pip install -r requirements.txt
```

3. Mount Google Drive and point `--data` to a YOLO-style dataset YAML stored there.

If your dataset root lives on Drive at `/content/drive/MyDrive/Aircraft_dataset`, a working YAML is:

```yaml
path: /content/drive/MyDrive/Aircraft_dataset
train: images/train
val: images/val
test: images/test
nc: 6
names:
  - crack
  - dent
  - corrosion
  - scratch
  - missing-head
  - paint-peel-off
```
