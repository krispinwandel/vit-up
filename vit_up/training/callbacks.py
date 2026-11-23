from pathlib import Path

import lightning as L


class ExportCheckpoint(L.Callback):
    """Export the model with the lowest validation loss to `<directory>/best` and the final
    model to `<directory>/last`, in the checkpoint format used for inference and initialization."""

    def __init__(self, directory: str, monitor: str = "val/loss"):
        self.directory = Path(directory)
        self.monitor = monitor
        self.best = None

    def on_validation_end(self, trainer, module):
        value = trainer.callback_metrics.get(self.monitor)
        if trainer.sanity_checking or value is None or (self.best is not None and value >= self.best):
            return
        self.best = float(value)
        if trainer.is_global_zero:
            module.export(self.directory / "best")
            (self.directory / "best" / "step.txt").write_text(f"{trainer.global_step} {self.best}\n")

    def on_train_end(self, trainer, module):
        if trainer.is_global_zero:
            module.export(self.directory / "last")

    def state_dict(self):
        return {"best": self.best}

    def load_state_dict(self, state):
        self.best = state["best"]
