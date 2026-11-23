from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import get_class
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = sorted(p.relative_to(ROOT).with_suffix("").as_posix() for p in (ROOT / "experiments").rglob("*.yaml"))
EVAL_CONFIG = ROOT / "vit_up" / "eval_kits" / "config"
EVAL_MODELS = sorted(p.relative_to(EVAL_CONFIG / "model").with_suffix("").as_posix()
                     for p in (EVAL_CONFIG / "model").rglob("*.yaml") if p.name != "vit_up.yaml")


def targets(node):
    if isinstance(node, dict):
        if "_target_" in node:
            yield node["_target_"]
        for value in node.values():
            yield from targets(value)
    elif isinstance(node, list):
        for value in node:
            yield from targets(value)


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_experiment_composes(experiment):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = OmegaConf.to_container(compose(config_name=experiment), resolve=True, throw_on_missing=True)
    assert cfg["name"] == f"{Path(experiment).parent.name}_{Path(experiment).name}"
    for target in targets(cfg):
        get_class(target)


@pytest.mark.parametrize("config_dir, config_name", [("probing", "probing"), ("correspondence_2d", "correspondence_2d"),
                                                     ("geometric_correspondence", "navi_correspondence"),
                                                     ("runtime", "runtime_bench")])
@pytest.mark.parametrize("model", EVAL_MODELS)
def test_eval_config_composes(config_dir, config_name, model):
    with initialize_config_dir(config_dir=str(EVAL_CONFIG / config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=[f"model={model}"])
    cfg = OmegaConf.to_container(cfg.model, resolve=True, throw_on_missing=True)
    assert cfg["feature_dim"] in (384, 768)
    get_class(cfg["_target_"])
