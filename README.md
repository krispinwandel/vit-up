# ViT-Up: Faithful Feature Upsampling for Vision Transformers

<p align="center">
  <a href="https://vitup.papers.discuna.com/">Project Page</a> |
  <a href="https://arxiv.org/abs/2606.14024">arXiv</a> |
  <a href="https://colab.research.google.com/github/krispinwandel/vit-up/blob/main/inference_example_colab.ipynb">Google Colab</a> |
  <a href="https://app.discuna.com/invite/krispinwandel">Discuna Forum</a> |
  <a href="https://huggingface.co/papers/2606.14024">Hugging Face (paper)</a>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/Krispin/vit-up"><b>Try Hugging Face Demo (no setup required!)</b></a>
</p>

ViT-Up is an implicit feature upsampler for Vision Transformers that predicts backbone-aligned features at arbitrary continuous image coordinates. Pretrained through self-supervised feature distillation on ImageNet-1K, it supports data-constrained dense prediction and fine-grained correspondence by letting downstream heads operate directly on dense DINOv3 features.

<p align="center">
  <img src="assets/readme/model_overview.jpg" alt="ViT-Up model overview" width="900">
</p>

<p align="center">
  <a href="#installation">Installation</a> |
  <a href="#models">Models</a> |
  <a href="#inference">Inference</a> |
  <a href="#training">Training</a> |
  <a href="#evaluation">Evaluation</a> |
  <a href="#citation">Citation</a>
</p>

## Installation

```bash
git clone https://github.com/krispinwandel/vit-up.git && cd vit-up
uv sync            # or: pip install -e . --find-links https://whl.natten.org
hf auth login      # DINOv3 is gated: request access on its Hugging Face model page first
```

The pinned environment uses Python 3.11, PyTorch 2.9 and CUDA 12.8.

## Models

All checkpoints are hosted at [Krispin/vit-up](https://huggingface.co/Krispin/vit-up) and are downloaded on first use.

| Release | Torch Hub entry point | Backbone | Training |
|---|---|---|---|
| `dinov3_splus` | `vit_up_dinov3_splus` | DINOv3-S+ | Stage 2 (main model) |
| `dinov3_splus_stage1` | `vit_up_dinov3_splus_stage1` | DINOv3-S+ | Stage 1 |
| `dinov3_splus_g28_56` | `vit_up_dinov3_splus_g28_56` | DINOv3-S+ | Stage 2 at 448 and 896 px (28x28 and 56x56 backbone grids); runs at 896 px by default |
| `dinov3_base` | `vit_up_dinov3_base` | DINOv3-B | Stage 2 |
| `dinov3_base_stage1` | – | DINOv3-B | Stage 1 |
| `dinov2_small` | `vit_up_dinov2_small` | DINOv2-S | Stage 2 |

Stage 1 distills the frozen backbone into ViT-Up while training LoRA adapters of the backbone. Stage 2 refines the
model by self-distillation from the frozen Stage 1 model, with the adapters merged into the backbone.

## Inference

```python
import torch
from PIL import Image

from vit_up.inference import ViTUpWrapper

model = ViTUpWrapper("dinov3_splus")  # a release name or a checkpoint directory

image = Image.open("assets/fruit_store.png").convert("RGB")
dense = model.dense(image, output_size=448)        # (1, 448, 448, C) features at the cell centers

queries = torch.rand(1, 100, 2)                    # normalized (x, y) in [0, 1]
features = model(query_coords=queries)             # (1, 100, C), reusing the encoded image
layers = model(query_coords=queries, return_all_layers=True)  # one (1, 100, C) tensor per selected layer
```

Images are given as PIL images or ImageNet-normalized `(B, 3, H, W)` tensors and are resized to the backbone resolution
(448 px by default, `image_size=...`). Inference runs under bf16 autocast; pass `use_bfloat16=False` for fp32.
Large query sets are decoded in chunks with `query_chunk_size=...`, which does not change the output.

The same models are available through Torch Hub; keyword arguments are passed to `ViTUpWrapper`:

```python
model = torch.hub.load("krispinwandel/vit-up", "vit_up_dinov3_splus", trust_repo=True)
features = model(images, query_coords)
```

See [notebooks/inference_example.ipynb](notebooks/inference_example.ipynb) for a PCA visualization of the features.

## Training

Training composes the Hydra configs in `configs/` with an experiment from `experiments/`:

| Experiment | Model | Initialization |
|---|---|---|
| `experiments/stage1/dinov3_splus` | `dinov3_splus_stage1` | – |
| `experiments/stage2/dinov3_splus` | `dinov3_splus` | `dinov3_splus_stage1` |
| `experiments/stage1/dinov3_splus_g28_56` | – | `dinov3_splus_stage1` |
| `experiments/stage2/dinov3_splus_g28_56` | `dinov3_splus_g28_56` | output of `stage1/dinov3_splus_g28_56` |
| `experiments/stage2/dinov3_base` | `dinov3_base` | `dinov3_base_stage1` |
| `experiments/stage1/dinov2_small` | – | – |
| `experiments/stage2/dinov2_small` | `dinov2_small` | output of `stage1/dinov2_small` |

```bash
python main.py --config-name experiments/stage1/dinov3_splus
python main.py --config-name experiments/stage2/dinov3_splus
```

ImageNet-1K is streamed from the Hugging Face Hub (`ILSVRC/imagenet-1k`, which requires accepting its license).
Runs write to `outputs/training/<name>/`. The model with the lowest validation loss and the final model are exported
to `checkpoints/best` and `checkpoints/last`, which work with `ViTUpWrapper`, `initialization=...` and the
evaluation configs. Lightning checkpoints for resuming are in `lightning/`.

Any setting can be overridden on the command line, for example:

```bash
python main.py --config-name experiments/stage2/dinov3_splus \
  initialization=outputs/training/stage1_dinov3_splus/checkpoints/best \
  logger=wandb schedule.batch_size=8 schedule.accumulate_grad_batches=2 \
  ckpt_path=outputs/training/stage2_dinov3_splus/lightning/last.ckpt   # resume
```

Add `--cfg job --resolve` to print the composed config without training. All recipes train on one GPU with bf16 mixed
precision; `val_check_interval` counts batches, so recipes with gradient accumulation scale it accordingly.

## Evaluation

The evaluation toolkits are configured with Hydra (`vit_up/eval_kits/config`). Select a method with
`model=<backbone>/<method>`:

| Backbone | Methods |
|---|---|
| `dinov3_splus` | `vit_up`, `vit_up_stage1`, `vit_up_g28_56`, `bilinear`, `anyup`, `jafar`, `naf`, `uplift`, `loftup_runtime` |
| `dinov3_base` | `vit_up`, `vit_up_stage1`, `bilinear`, `naf` |
| `dinov2_small` | `vit_up`, `bilinear`, `anyup`, `jafar`, `loftup`, `naf`, `uplift` |

A trained checkpoint is evaluated with `model=vit_up model.checkpoint=<directory> model.feature_dim=<C>`.
Results are written to `outputs/eval/`. Download the datasets into `data/` first:

```bash
python scripts/download_datasets.py   # --help lists the datasets; Cityscapes needs --city_un/--city_pwd
```

**Linear probing.** Train a segmentation head (VOC, Cityscapes, COCO, ADE20K) or a depth head (COCO), then evaluate it:

```bash
python -m vit_up.eval_kits.probing_toolkit.run_probing model=dinov3_splus/vit_up \
  schedule/mode=train schedule/dataset=voc dataset_evaluation=voc
python -m vit_up.eval_kits.probing_toolkit.run_probing model=dinov3_splus/vit_up task=depth \
  schedule/mode=train schedule/dataset=coco dataset_evaluation=coco
```

With `schedule/mode=eval`, heads trained on the native-resolution DINOv3-S+ backbone (`assets/pretrained_heads`) are
applied to the upsampled features instead (frozen-head evaluation).

**Correspondence.** SPair-71k PCK and NAVI 3D recall:

```bash
python -m vit_up.eval_kits.correspondence_2d_toolkit.run_correspondence_2d model=dinov3_splus/vit_up
python -m vit_up.eval_kits.geometric_correspondence_toolkit.evaluate_navi_correspondence model=dinov3_splus/vit_up
```

**Runtime.** Latency and peak memory from an image to dense features, including the backbone:

```bash
python -m vit_up.eval_kits.runtime_toolkit.run_runtime_bench model=dinov3_splus/vit_up model.query_chunk_size=null
```

**Output and backbone resolution.** `model.decode_size=112` decodes ViT-Up on a 112x112 grid and bilinearly resizes the
features to the output size. The bilinear baseline at higher backbone resolutions uses `model=dinov3_splus/bilinear
model.img_in_size=896` (56x56 tokens); add `model.out_size=448` for correspondence.

## Open-Vocabulary Segmentation (experimental)

The notebook [notebooks/talking_to_vit-up.ipynb](notebooks/talking_to_vit-up.ipynb) combines dense ViT-Up DINOv3-B
features with the Talk2DINO text projection model for prompt-based segmentation (`uv sync --extra demo`).

<p align="center">
  <img src="assets/readme/talking_to_vitup.png" alt="Open-vocabulary segmentation with Talk2DINO and ViT-Up" width="900">
</p>

This demo builds on [Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary
Segmentation](https://arxiv.org/abs/2411.19331).

## Video Encoding (experimental)

`scripts/encode_video.py` extracts dense ViT-Up features from every frame of a video, fits a global three-component PCA
and renders it as an RGB video. Frames are padded to squares for inference, while queries and outputs keep the video's
aspect ratio.

```bash
python scripts/encode_video.py input.mp4 --width 640 --output-dir outputs
python scripts/encode_video.py input.mp4 --width 640 --pca-nframes 10 --seg-video segmentation.mp4 --output-dir outputs
```

An optional segmentation video restricts queries and PCA to foreground pixels (grayscale threshold `--seg-threshold`,
default 127). For `input.mp4`, the output folder contains `input_features.npy` (`(T, H, W, D)` features),
`input_pca_features.npy`, `input_pca.npz`, `input_pca_rgb.mp4`, and metadata used to reuse cached features.
Read feature frames lazily with:

```python
from vit_up.video_features import make_read_ft_frame

read_ft_frame = make_read_ft_frame("outputs/input_features.npy")
frame = read_ft_frame(0)  # .features (H, W, D), .mask, .rgb and .pca_rgb when available
```

## Citation

```bibtex
@misc{wandel2026vitupfaithfulfeatureupsampling,
      title={ViT-Up: Faithful Feature Upsampling for Vision Transformers},
      author={Krispin Wandel and Jingchuan Wang and Hesheng Wang},
      year={2026},
      eprint={2606.14024},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.14024},
}
```
