import csv
from pathlib import Path
from typing import Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class TrainingLossLogger(TrainerCallback):
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "training_loss.csv"
        self.csv_file = None
        self.csv_writer = None
        self._initialized = False

    def _initialize_csv(self):
        Called when logging occurs during training.
        Writes the loss to the CSV file.
        Called at the end of training.
        Closes the CSV file.