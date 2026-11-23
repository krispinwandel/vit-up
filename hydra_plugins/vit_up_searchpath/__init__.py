"""Make `experiments/...` selectable as config names next to the `configs/` directory."""
from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class ViTUpSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        for entry in list(search_path.get_path()):
            directory = Path(entry.path.removeprefix("file://"))
            if entry.provider == "main" and directory.name == "configs" and (directory.parent / "experiments").is_dir():
                search_path.append(provider="vit_up.experiments", path=str(directory.parent))
