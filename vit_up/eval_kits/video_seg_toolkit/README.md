## Video Segmentation Toolkit

DAVIS-2017 video object segmentation with the evaluation protocol of LILA ("Featurising Pixels
from Dynamic 3D Scenes with Linear In-Context Learners", Araslanov et al., CVPR 2026;
[code](https://github.com/google-research/google-research/tree/master/lila), Apache-2.0). Both
benchmarks transfer the first frame's object masks to the later frames of each of the 30
validation videos and report J&F (region IoU J and boundary F-measure F):

| Benchmark | LILA script | Method |
|---|---|---|
| `vos_probe` | `eval_vos.py` | a linear probe fit on the first frame (10 seeds; mean and std) |
| `vos_knn` | `eval_vosknn.py` | local k-NN label propagation from the first and the last 7 frames |
| `vos_knn_sweep` | – | `vos_knn` under one-at-a-time changes of its hand-set parameters |

### Data

```bash
python -m vit_up.eval_kits.video_seg_toolkit.prepare_data davis --data-root data   # DAVIS-2017, 0.8 GB
```

### Features

Each benchmark evaluates three feature modes in one pass (`benchmark.modes`); the features are
computed once per frame, and every mode gets its own probe or propagation history:

| Mode | Features | |
|---|---|---|
| `dense` | the model's dense features | reported for ViT-Up, LILA and the baselines |
| `backbone` | tokens of the original Hugging Face backbone, bilinearly upsampled | the "Bilinear" rows |
| `combined` | the sum of both (LILA's protocol) | |

The two branches use different backbones. ViT-Up computes its dense features from its own
backbone: the DINOv3 releases include the trained LoRA adapters (merged into the backbone), the
DINOv2-S releases have none. The `backbone` branch always uses the original, unadapted weights
(`encoder.backbone_model_name`, taken from the model by default). Dense features are
layer-normalized without affine parameters (`decoder_norm=layernorm`), as LILA normalizes its own.

`model=none` evaluates only the `backbone` mode. `model=dinov2_small/lila` runs the released LILA
checkpoint through the same code: it sparse-checks out LILA's code from google-research at a pinned
commit into the Torch Hub cache.

### Results

DAVIS-2017 val J&F (%) from the paper, using only the dense features (`dense` mode; Bilinear:
`backbone` mode). Linear probing is the mean over 10 seeds. ViT-Up is the Stage 2 model trained
jointly on two backbone grids; LILA uses its official DINOv2-S checkpoint.

| Method | DINOv2-S probing | DINOv2-S k-NN | DINOv3-S+ probing | DINOv3-S+ k-NN |
|---|---|---|---|---|
| Bilinear | 58.6 | 62.2 | 63.7 | 63.3 |
| LILA | 64.5 | 73.9 | – | – |
| NAF | 62.6 | 65.9 | 63.9 | 62.8 |
| UpLiFT | 65.1 | 69.9 | 66.0 | 67.4 |
| **ViT-Up** | **67.3** | **76.0** | **69.5** | **73.1** |

The ViT-Up, LILA and Bilinear rows:

```bash
BENCHMARK=vos_probe   # vos_probe, vos_knn
MODEL="model=dinov2_small/vit_up_g32_64 model.image_size=840"
# DINOv2-S LILA:      MODEL="model=dinov2_small/lila"
# DINOv2-S Bilinear:  MODEL="model=none encoder.backbone_model_name=facebook/dinov2-small"
# DINOv3-S+ ViT-Up:   MODEL="model=dinov3_splus/vit_up_g28_56 benchmark.resize_input_size=480"
# DINOv3-S+ Bilinear: MODEL="model=none encoder.backbone_model_name=facebook/dinov3-vits16plus-pretrain-lvd1689m benchmark.resize_input_size=480"
python -m vit_up.eval_kits.video_seg_toolkit.run_video_seg benchmark=$BENCHMARK $MODEL
```

Results go to `<run dir>/metrics.json` (one block per mode), by default under
`outputs/eval/video_seg/<model>/<date_time>/<benchmark>`. The k-NN sweep (`vos_knn_sweep`) compares
models with:

```bash
python -m vit_up.eval_kits.video_seg_toolkit.sweep_table vit_up=<run dir>/metrics.json lila=<run dir>/metrics.json
```

### Protocol details

- **Resolution.** Both benchmarks resample frames to 476x840 (shorter side 476, patch-aligned).
  The probe reads features at that resolution, k-NN on a 60x105 grid. ViT-Up is queried at the
  cell centers of that grid, with the frame centered on a black square as in training. Other
  models are bilinearly resized from their native output, as in LILA's evaluation.
- **Backbone resolution.** Each upsampler runs its own backbone at its configured resolution
  (ViT-Up: the release's training resolution, or `model.image_size`), which sets the grid above.
- **k-NN branch combination.** LILA's released `eval_vosknn.py` sums the two branches' soft masks
  and feeds the sum back as context, so in the `combined` mode the masks double every frame and
  overflow in fp16 after about 16 frames. This toolkit averages them instead
  (`normalize_branch_masks: true`); each frame's argmax is unchanged.
- **Comparison with the LILA paper.** The plain backbone reproduces LILA's linear-probing baseline
  (DINOv2-S: 58.6 here, 57.5 in the paper). Its k-NN baseline is lower (62.2 vs. 65.1) with LILA's
  released k-NN code; the paper states that k-NN "follows DINO's settings", which the release does
  not contain. LILA's own checkpoint reaches its published 73.9 k-NN score here. Compare methods
  with numbers from this toolkit only.
- The probes, metrics, k-NN propagation and transforms are otherwise ported unchanged, and were
  verified to be numerically identical to LILA's code on random inputs.
