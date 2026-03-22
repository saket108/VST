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
- decoupled classification, box, and centerness head
- FCOS-style point assignment and GIoU-based regression loss
- direct support for datasets in YOLO annotation format

This is a research baseline. The code is custom, but a paper contribution still comes from experiments, ablations, and evidence.

## Main files

- `tools/train.py`: training entrypoint
- `model/detector.py`: architecture definition
- `training/losses.py`: target assignment and losses
- `training/engine.py`: training and validation loop
- `data/dataset.py`: YOLO-format dataset loader
- `utils/evaluator.py`: lightweight `mAP50` and `mAP50-95`

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

Lighter pretrained backbone:

```powershell
python tools/train.py --data path\to\data.yaml --variant tiny --backbone mobilenet_v3_large --imgsz 896 --epochs 300 --batch-size 12
```

Custom backbone ablation:

```powershell
python tools/train.py --data path\to\data.yaml --variant small --backbone custom --no-pretrained-backbone --imgsz 896 --epochs 300 --batch-size 8
```
