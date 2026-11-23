"""Train ViT-Up, e.g. `python main.py --config-name experiments/stage1/dinov3_splus`."""
import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    from vit_up.training.run import train
    train(cfg)


if __name__ == "__main__":
    main()
