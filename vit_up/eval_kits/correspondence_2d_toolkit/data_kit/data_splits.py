import glob
import os
import json
from tqdm import tqdm
from . import raw_data_utils


def load_img_file_splits(
    dataset_name: str, data_dir: str, category: str, cache_dir: str
):
    os.makedirs(cache_dir, exist_ok=True)
    # splits = ["trn", "val", "test"]
    splits = ["test"]  # for now just do test split
    img_file_splits = {k: [] for k in splits}
    img_file_pairs_split = {k: [] for k in splits}
    for split in splits:
        cache_path = os.path.join(cache_dir, f"{category}_img_files_{split}.json")
        pair_cache_path = os.path.join(
            cache_dir, f"{category}_img_file_pairs_{split}.json"
        )
        if os.path.exists(cache_path) and os.path.exists(pair_cache_path):
            print(
                f"Loading cached img file splits for category {category} and split {split}"
            )
            with open(cache_path, "r") as f:
                img_file_splits[split] = json.load(f)
            with open(pair_cache_path, "r") as f:
                img_file_pairs_split[split] = json.load(f)
        else:
            pairs = sorted(
                glob.glob(f"{data_dir}/PairAnnotation/{split}/*:{category}.json")
            )
            files = []
            file_pairs = []
            for pair in tqdm(
                pairs, desc=f"Processing category {category} and split {split}"
            ):
                src_trg_img_anno = raw_data_utils.load_normalized_src_tgt_img_anno(
                    pair, dataset_name
                )
                source_fp = f"{data_dir}/{src_trg_img_anno.src.rel_fp}"
                target_fp = f"{data_dir}/{src_trg_img_anno.trg.rel_fp}"
                files.append(source_fp)
                files.append(target_fp)
                file_pairs.append(source_fp)
                file_pairs.append(target_fp)
            img_file_splits[split] = list(set(files))
            img_file_pairs_split[split] = file_pairs
            with open(cache_path, "w") as f:
                json.dump(img_file_splits[split], f)
            with open(pair_cache_path, "w") as f:
                json.dump(img_file_pairs_split[split], f)
    return img_file_splits, img_file_pairs_split
