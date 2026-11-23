# Linear probing

Frozen features are probed with a 1x1 convolution: semantic segmentation on VOC,
COCO-Stuff, ADE20K and Cityscapes, and depth on COCO with Depth Anything V2
pseudo-labels (`task=depth`).

```bash
# Train a head (20 epochs, 5 on COCO), then evaluate it.
python -m vit_up.eval_kits.probing_toolkit.run_probing model=dinov3_splus/vit_up \
  schedule/mode=train schedule/dataset=voc dataset_evaluation=voc

# Evaluate the pretrained head of the native-resolution DINOv3-S+ backbone on upsampled features.
python -m vit_up.eval_kits.probing_toolkit.run_probing model=dinov3_splus/vit_up \
  schedule/mode=eval schedule/dataset=voc dataset_evaluation=voc
```

Depth uses `task=depth schedule/dataset=coco dataset_evaluation=coco`. Datasets are
read from `data_root` (default `data/`, see `scripts/download_datasets.py`).
Metrics are written to `<run dir>/<mode>_metrics.json` and the trained head to
`<run dir>/checkpoints/`.
