import importlib
from typing import Any, Dict, List, Optional, Type, cast
import torch.nn as nn
import copy


def load_module_class(class_path: str) -> Type[nn.Module]:
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
            f"Class '{class_name}' not found in module '{module_path}' "
            f"for class path '{class_path}'."
        )
    if not isinstance(class_obj, type) or not issubclass(class_obj, nn.Module):
        raise TypeError(
            f"Class '{class_path}' must resolve to a torch.nn.Module subclass."
        )
    return cast(Type[nn.Module], class_obj)


def init_module(
    class_path: str,
    init_args: Dict[str, Any],
) -> nn.Module:
    module_class = load_module_class(class_path)
    module = module_class(**init_args)
    return module


def init_class_blocks(
    class_path: str,
    init_args: Dict[str, Any],
    n_blocks: int,
    overwrite_init_args: Dict[str, Any] = {},
) -> nn.ModuleList:
    module_class = load_module_class(class_path)
    class_init_args = dict(init_args)

    block_init_args: List[Dict[str, Any]] = [{} for _ in range(n_blocks)]
    for arg_name, arg_value in class_init_args.items():
        if isinstance(arg_value, list) and len(arg_value) == n_blocks:
            for block_idx, block_arg_value in enumerate(arg_value):
                block_init_args[block_idx][arg_name] = block_arg_value
        else:
            for block_kwargs in block_init_args:
                block_kwargs[arg_name] = arg_value

    for block_kwargs in block_init_args:
        for k, v in overwrite_init_args.items():
            block_kwargs[k] = v

    blocks = nn.ModuleList(
        [module_class(**block_init_args[i]) for i in range(n_blocks)]
    )
    return blocks


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if hasattr(value, "__dict__") and not isinstance(value, nn.Module):
        # jsonargparse may provide Namespace objects for nested class specs.
        return {k: to_builtin(v) for k, v in vars(value).items()}
    return value


def spec_to_dict(spec: Any, name: str) -> Dict[str, Any]:
    if isinstance(spec, dict):
        return cast(Dict[str, Any], to_builtin(spec))
    class_path = getattr(spec, "class_path", None)
    init_args = getattr(spec, "init_args", None)
    if class_path is not None:
        init_args_dict = to_builtin(init_args) if init_args is not None else {}
        if not isinstance(init_args_dict, dict):
            init_args_dict = {}
        return {
            "class_path": class_path,
            "init_args": init_args_dict,
        }
    raise TypeError(f"{name} must be a dict/class-spec or nn.Module, got {type(spec)}.")


def init_or_use_module(spec: Any, name: str) -> nn.Module:
    if isinstance(spec, nn.Module):
        return spec
    spec_dict = spec_to_dict(spec, name)
    return init_module(
        class_path=spec_dict["class_path"],
        init_args=spec_dict.get("init_args", {}),
    )


def init_or_use_blocks(spec: Any, n_blocks: int, name: str) -> nn.ModuleList:
    if isinstance(spec, nn.ModuleList):
        if len(spec) != n_blocks:
            raise ValueError(
                f"{name} ModuleList length must be {n_blocks}, got {len(spec)}."
            )
        return spec
    if isinstance(spec, nn.Module):
        return nn.ModuleList([copy.deepcopy(spec) for _ in range(n_blocks)])
    spec_dict = spec_to_dict(spec, name)
    return init_class_blocks(
        class_path=spec_dict["class_path"],
        init_args=spec_dict.get("init_args", {}),
        n_blocks=n_blocks,
    )


def normalize_class_init(raw: Dict[str, Any], name: str) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError(f"{name} must be a dict with class_path/init_args.")

    class_path = raw.get("class_path")
    if not isinstance(class_path, str) or not class_path:
        raise ValueError(f"{name}.class_path must be a non-empty string.")

    init_args = raw.get("init_args", {})
    if init_args is None:
        init_args = {}
    if not isinstance(init_args, dict):
        raise TypeError(f"{name}.init_args must be a dict when provided.")

    return {
        "class_path": class_path,
        "init_args": dict(init_args),
    }
