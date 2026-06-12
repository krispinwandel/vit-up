import importlib
import copy
from typing import Any, Dict, List, Optional, Type, cast

import torch.nn as nn


def _load_module_class(class_path: str) -> Type[nn.Module]:
    if class_path.startswith("nf_dino."):
        class_path = class_path.replace("nf_dino.", "vit_up.", 1)
    try:
        module_path, class_name = class_path.rsplit(".", 1)
    except ValueError as exc:
        raise ValueError(
            f"Invalid class path '{class_path}'. Expected format 'pkg.module.ClassName'."
        ) from exc

    module = importlib.import_module(module_path)
    class_obj = getattr(module, class_name, None)
    if class_obj is None:
        raise ValueError(
            f"Class '{class_name}' not found in module '{module_path}' for class path '{class_path}'."
        )
    if not isinstance(class_obj, type) or not issubclass(class_obj, nn.Module):
        raise TypeError(
            f"Class '{class_path}' must resolve to a torch.nn.Module subclass."
        )
    return cast(Type[nn.Module], class_obj)


class SmartModuleList(nn.ModuleList):
    """ModuleList that builds blocks from class spec and per-block init args.

    - Init args are shared by all blocks by default.
    - Per-block args must use a trailing '_' key, e.g. dims_.
    - For per-block args, the value must be a list of length n_blocks.
    - The trailing '_' is removed before passing kwargs to each block.
    """

    def __init__(
        self,
        n_blocks: int,
        block_class_path: str,
        block_init_args: Optional[Dict[str, Any]] = None,
        share_blocks: bool = False,
        skip_indices: Optional[List[int]] = None,
    ):
        if not isinstance(n_blocks, int) or n_blocks < 0:
            raise ValueError(f"n_blocks must be a non-negative int. Got {n_blocks}.")
        if not isinstance(block_class_path, str) or not block_class_path:
            raise ValueError("block_class_path must be a non-empty string.")
        if block_init_args is None:
            block_init_args = {}
        if not isinstance(block_init_args, dict):
            raise TypeError("block_init_args must be a dict.")

        self.n_blocks = (
            n_blocks - len(skip_indices) if skip_indices is not None else n_blocks
        )
        self.block_class_path = block_class_path
        self.block_init_args = dict(block_init_args)
        self.share_blocks = bool(share_blocks)
        self.skip_indices = set(skip_indices) if skip_indices is not None else set()

        self.access_index_to_index = {}
        for i in range(n_blocks):
            if i in self.skip_indices:
                continue
            access_index = len(self.access_index_to_index)
            self.access_index_to_index[access_index] = i

        super().__init__(
            modules=self._build_modules(
                n_blocks=self.n_blocks,
                block_class_path=self.block_class_path,
                block_init_args=self.block_init_args,
                share_blocks=self.share_blocks,
            )
        )

    def __getitem__(self, idx):
        index = self.access_index_to_index.get(idx)
        if index is None:
            return None  # Return None for skipped indices instead of raising IndexError
        if self.share_blocks and isinstance(idx, int):
            if self.n_blocks == 0:
                raise IndexError("index out of range")
            if idx < 0:
                idx += self.n_blocks
            if idx < 0 or idx >= self.n_blocks:
                raise IndexError("index out of range")
            return super().__getitem__(0)
        return super().__getitem__(index)

    @staticmethod
    def _build_modules(
        n_blocks: int,
        block_class_path: str,
        block_init_args: Dict[str, Any],
        share_blocks: bool = False,
    ) -> List[nn.Module]:
        block_class = _load_module_class(block_class_path)
        per_block_init_args: List[Dict[str, Any]] = [{} for _ in range(n_blocks)]

        for arg_name, arg_value in block_init_args.items():
            if arg_name.endswith("_"):
                clean_arg_name = arg_name[:-1]
                if not clean_arg_name:
                    raise ValueError(
                        "Per-block init arg key '_' is invalid. Provide a non-empty "
                        "name ending with '_', e.g. 'dims_'."
                    )
                if not isinstance(arg_value, list):
                    raise TypeError(
                        f"block_init_args['{arg_name}'] must be a list when using "
                        "the per-block '_' suffix."
                    )
                if len(arg_value) != n_blocks:
                    raise ValueError(
                        f"block_init_args['{arg_name}'] has length={len(arg_value)} "
                        f"but expected n_blocks={n_blocks}."
                    )
                for block_idx, block_arg_value in enumerate(arg_value):
                    per_block_init_args[block_idx][clean_arg_name] = copy.deepcopy(
                        block_arg_value
                    )
            else:
                for block_kwargs in per_block_init_args:
                    block_kwargs[arg_name] = copy.deepcopy(arg_value)

        if share_blocks and n_blocks > 0:
            return [block_class(**per_block_init_args[0])]

        return [block_class(**per_block_init_args[i]) for i in range(n_blocks)]
