# vit-up

ViT-Up: Faithful Feature Upsampling for Vision Transformers

Run all commands from the repository root.

## Training

Train a ViT-Up model with PyTorch Lightning using one of the run configs:

```bash
python main.py fit --config configs/runs/dinov3_splus.yaml
```

For the DINOv3 base variant:

```bash
python main.py fit --config configs/runs/dinov3_base.yaml
```

The run config defines the backbone, query embedding, ViT-Up blocks, optimizer,
data paths, logging, and checkpointing.

### Ablations

```bash
ABLATION_COMMON="--config configs/runs/dinov3_splus.yaml --config configs/runs/ablations/schedule.yaml"
```

```bash
python main.py fit ${ABLATION_COMMON} --config configs/runs/ablations/...
```

## Evaluation

The evaluation kits use Hydra configs under `vit_up/eval_kits/config`.
The examples below evaluate the DINOv3 S+ ViT-Up model. Replace
`dinov3/splus/vit_up` with `dinov3/base/vit_up` for the base model.

Outputs are written under the configured `mnt_dir` output folder. Override it
from the command line if your datasets or output root live elsewhere:

```bash
python <eval_script>.py model=dinov3/splus/vit_up mnt_dir=/path/to/eval_root
```

### Download Datasets

```bash
python scripts/download_datasets.py
```

### Linear Probing

Train a segmentation probing head on VOC:

```bash
python vit_up/eval_kits/probing_toolkit/run_probing.py schedule/mode=train schedule/dataset=voc model=dinov3/splus/vit_up
```

Evaluate a trained or configured probing head:

```bash
python vit_up/eval_kits/probing_toolkit/run_probing.py schedule/mode=eval schedule/dataset=voc model=dinov3/splus/vit_up
```

### Semantic Correspondence

Run the 2D semantic correspondence benchmark:

```bash
python vit_up/eval_kits/correspondence_2d_toolkit/run_correspondence_2d.py model=dinov3/splus/vit_up
```

### Geometric Correspondence

Run NAVI geometric correspondence:

```bash
python vit_up/eval_kits/geometric_correspondence_toolkit/evaluate_navi_correspondence.py model=dinov3/splus/vit_up
```

### Runtime

Benchmark runtime and memory over the configured output resolutions:

```bash
python vit_up/eval_kits/runtime_toolkit/run_runtime_bench.py model=dinov3/splus/vit_up
```

To only print model parameter counts:

```bash
python vit_up/eval_kits/runtime_toolkit/run_runtime_bench.py model=dinov3/splus/vit_up print_model_params_only=true
```

## Monitoring

```bash
watch -n 1 nvidia-smi
pkill -f main.py
```
