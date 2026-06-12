## Probing Toolkit

This toolkit runs probing experiments using Hydra configs. The recommended entrypoint is
`run_probing.py` (Hydra root `probing.yaml`), which selects configs by `mode` and
`dataset` and applies schedule defaults (epochs/lr) automatically.

### Quick Start

Run eval (requires a probe head checkpoint):

```bash
python nf_dino/eval_kits/probing_toolkit/run_probing.py schedule/mode=eval \
    schedule/dataset=voc task=seg model=dinov3/vit_up_112_probe \
    head_ckpt=assets/pretrained_heads/seg/voc/dinov3/backbone_probe.pth
```

Train or finetune:

```bash
python nf_dino/eval_kits/probing_toolkit/run_probing.py mode=train dataset=coco model=dinov3/uplift_probe
python nf_dino/eval_kits/probing_toolkit/run_probing.py \
	schedule/mode=finetune schedule/dataset=coco model=dinov3/uplift_probe \
	head_ckpt=/path/to/seed_head.pth
```

### Config Layout

Configs live under `nf_dino/eval_kits/config/probing`.

- `probing.yaml`: top-level Hydra root (contains project defaults, `mode`, `dataset`, and
  shared settings)
- `schedule/`: default `num_epochs` and `optimizer.lr` by `mode` and `dataset`
- `dataset_evaluation/`: dataset-specific settings (including `tag` and `root`)
- `model/`, `dataloader/`, etc.: other config groups used by the toolkit

The run directory is set to:

```
output/probing/${mode}/${eval.task}/${dataset_evaluation.tag}/${hydra:runtime.choices.model}/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

### Dataset Roots

Dataset roots are defined in `dataset_evaluation/*.yaml` and use `data_root` from
`probing.yaml` (defaults to `${project_root}/data`). Override as needed:

```bash
python nf_dino/eval_kits/probing_toolkit/run_probing.py \
	schedule/mode=eval schedule/dataset=cityscapes model=dinov3/uplift_probe \
	data_root=/path/to/datasets
```

### Schedule Defaults

Defaults for `num_epochs` and `optimizer.lr` come from `schedule/mode/*.yaml` and
`schedule/dataset/*.yaml`. For example, `dataset=coco` sets `num_epochs_override=5`,
which is applied for `train` and `finetune` modes.

### Head Checkpoint Shortcut

Use `head_ckpt=...` as a shorthand for both eval and training initialization.
`mode=eval` evaluates the checkpointed head, while `mode=train` and `mode=finetune`
use it to initialize the probe head before training.

### WandB defaults

If `logger.wandb.run_name` or `logger.wandb.tags` are not set, the launcher builds
reasonable defaults similar to `scripts/probing.py`:

- `logger.wandb.run_name`: `{method}-{YYYY-MM-DD}` (method is taken from the
  `model` spec, e.g. `uplift_probe`)
- `logger.wandb.tags`: `[mode, backbone, task, dataset, method]` (filtered)

Override any value directly at the CLI:

```bash
python nf_dino/eval_kits/probing_toolkit/run_probing.py schedule/mode=train schedule/dataset=voc \
	num_epochs=10 optimizer.lr=1e-3
```
