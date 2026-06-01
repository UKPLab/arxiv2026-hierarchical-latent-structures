"""
Callback for logging training loss to a CSV file in the run folder.
"""
import csv
from pathlib import Path
from typing import Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class TrainingLossLogger(TrainerCallback):
    """
    A callback that logs training loss at each logging step to a CSV file.

    The CSV file will be saved in the run folder with columns:
    - step: training step number
    - loss: training loss value
    - learning_rate: current learning rate
    - epoch: current epoch
    """

    def __init__(self, output_dir: str | Path):
        """
        Initialize the TrainingLossLogger.

        Args:
            output_dir: Directory where the loss log CSV will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "training_loss.csv"
        self.csv_file = None
        self.csv_writer = None
        self._initialized = False

    def _initialize_csv(self):
        """Initialize the CSV file with headers."""
        self.csv_file = open(self.log_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['step', 'loss', 'learning_rate', 'epoch'])
        self.csv_file.flush()
        self._initialized = True

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Called when logging occurs during training.
        Writes the loss to the CSV file.
        """
        if not self._initialized:
            self._initialize_csv()

        if logs is not None and 'loss' in logs:
            step = state.global_step
            loss = logs['loss']
            learning_rate = logs.get('learning_rate', 0.0)
            epoch = logs.get('epoch', 0.0)

            self.csv_writer.writerow([step, loss, learning_rate, epoch])
            self.csv_file.flush()  # Ensure data is written immediately

        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Called at the end of training.
        Closes the CSV file.
        """
        if self.csv_file is not None:
            self.csv_file.close()

        return control