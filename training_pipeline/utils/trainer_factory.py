from datasets import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
import json
from pathlib import Path
import uuid

from .evaluator_factory import EvaluatorFactory
from .training_loss_logger import TrainingLossLogger


def load_batch_sizes_from_json(model_config: dict) -> dict: