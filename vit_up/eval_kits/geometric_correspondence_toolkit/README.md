# NAVI geometric correspondence

```bash
python -m vit_up.eval_kits.geometric_correspondence_toolkit.evaluate_navi_correspondence model=dinov3_splus/vit_up
```

Features of both views are bicubically resized to the 0.25x geometry grid, matched
with the nearest/second-nearest cosine-distance ratio, and the best 1,000 matches
are scored by their 3D error under the ground-truth relative pose. Recall at 1, 2
and 5 cm is written to `navi_correspondence_summary.json`. The dataset is read from
`data/navi_v1` (`data_root`).

The dataset and matching code is adapted from
[Probing the 3D Awareness of Visual Foundation Models](https://github.com/mbanani/probe3d)
(El Banani et al., CVPR 2024; MIT license).
