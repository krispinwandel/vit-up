NAVI Geometric Correspondence
=============================

This copy is used in `nf_dino` only for NAVI geometric correspondence. The
runner instantiates models from `nf_dino/eval_kits/config/model`, so the same
wrappers used by the other eval kits can be selected with Hydra:

```bash
python nf_dino/eval_kits/geometric_correspondence_toolkit/evaluate_navi_correspondence.py \
  model=dinov3/backbone_probe \
  dataset.path=/path/to/navi_v1
```

Results are written under
`${mnt_dir}/output/geometric_correspondence/navi/<model>/...` and include
`navi_correspondence_summary.json`.

Upstream provenance
-------------------

This repository contains a re-implementation of the code for the paper [Probing the 3D Awareness of
Visual Foundation Models](https://arxiv.org/abs/2404.08636) (CVPR 2024) which presents an analysis of the 3D awareness of visual
foundation models.


[Mohamed El Banani](mbanani.github.io), [Amit Raj](https://amitraj93.github.io/), [Kevis-Kokitsi Maninis](https://www.kmaninis.com/), [Abhishek Kar](https://abhishekkar.info/), [Yuanzhen Li](https://people.csail.mit.edu/yzli/), [Michael Rubinstein](https://people.csail.mit.edu/mrub/), [Deqing Sun](https://deqings.github.io/), [Leonidas Guibas](https://geometry.stanford.edu/member/guibas/), [Justin Johnson](https://web.eecs.umich.edu/~justincj/),  [Varun Jampani](https://varunjampani.github.io/) 

If you find this code useful, please consider citing:  
```text
@inProceedings{elbanani2024probing,
  title={{Probing the 3D Awareness of Visual Foundation Models}},
  author={
        El Banani, Mohamed and Raj, Amit and Maninis, Kevis-Kokitsi and 
        Kar, Abhishek and Li, Yuanzhen and Rubinstein, Michael and Sun, Deqing and 
        Guibas, Leonidas and Johnson, Justin and Jampani, Varun
        },
  booktitle={CVPR},
  year={2024},
}
```

Acknowledgments
-----------------

We thank Prafull Sharma, Shivam Duggal, Karan Desai, Junhwa Hur, and Charles Herrmann for many helpful discussions.
We also thank Alyosha Efros, David Fouhey, Stella Yu, and Andrew Owens for their feedback. 
