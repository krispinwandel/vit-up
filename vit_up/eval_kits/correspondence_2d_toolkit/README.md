# SPair-71k keypoint transfer

```bash
python -m vit_up.eval_kits.correspondence_2d_toolkit.run_correspondence_2d model=dinov3_splus/vit_up
```

Each image is zero-padded to a square, resized to `img_size`, and mapped to an
`out_size` x `out_size` feature grid. Every source keypoint takes the feature of
its grid cell and is transferred to the cell center with the highest cosine
similarity in the target image. A transfer is correct within `alpha` times the
longer side of the target object's bounding box (PCK@0.1/0.05/0.01).

`pck_summary.json` in the Hydra run directory holds PCK per category and its mean
over the 18 categories. Set `feature_cache_device=cpu` if a category's cached
feature maps do not fit in GPU memory.
