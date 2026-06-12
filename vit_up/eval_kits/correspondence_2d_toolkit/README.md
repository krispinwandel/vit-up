# Correspondence 2D Evaluation

Hydra entrypoint for SPAIR-71k dense correspondence evaluation:

```bash
python nf_dino/eval_kits/correspondence_2d_toolkit/run_correspondence_2d.py \
  dataset_dir=/path/to/SPair-71k \
  model=dinov3/vit_up_probe \
  img_size=1024 \
  out_size=1024
```

The default config is:

```text
nf_dino/eval_kits/config/correspondence_2d/correspondence_2d.yaml
```

The selected upsampler is instantiated directly from `cfg.model`. Evaluation writes
`eval_config.json` and `pck_summary.json` into the Hydra run directory. The summary
contains `pck@0.1`, `pck@0.05`, and `pck@0.01` per category and averaged over all
categories.

`report/table_utils.py` remains available for building tables from saved per-category
metric JSON files.
