"""
src/utils/checkpoint_manager.py

Handles saving and resuming training state, so a Colab disconnect (or a
deliberate pause) never costs more than the current epoch's progress.
"""

import os
import json


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, run_name: str):
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.state_path = os.path.join(checkpoint_dir, f"{run_name}_state.json")

    def weights_path(self, epoch: int) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.run_name}_epoch{epoch}.weights.h5")

    def save(self, model, epoch: int, history: dict) -> None:
        model.save_weights(self.weights_path(epoch))
        with open(self.state_path, "w") as f:
            json.dump({"last_epoch": epoch, "history": history}, f, indent=2)
        print(f"[checkpoint] saved epoch {epoch} -> {self.weights_path(epoch)}")

    def try_resume(self, model) -> tuple[int, dict]:
        """
        If a previous checkpoint exists for this run, loads it and returns
        (next_epoch_to_run, history_so_far). Otherwise returns (0, {}).
        """
        if not os.path.exists(self.state_path):
            return 0, {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

        with open(self.state_path, "r") as f:
            state = json.load(f)

        last_epoch = state["last_epoch"]
        weights_file = self.weights_path(last_epoch)
        if os.path.exists(weights_file):
            model.load_weights(weights_file)
            print(f"[checkpoint] resumed from epoch {last_epoch}")
            return last_epoch + 1, state["history"]

        print("[checkpoint] state file found but weights missing — starting fresh")
        return 0, {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
