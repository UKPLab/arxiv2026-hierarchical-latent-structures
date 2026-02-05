import os
os.environ["WANDB_MODE"] = "offline"
import sys
import json
import torch
from joblib import Parallel, delayed
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
from tqdm import tqdm
import logging
from datetime import datetime
import traceback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = os.environ.get('OLMO_MODELS_DIR', str(PROJECT_ROOT / 'data' / 'olmo' / 'checkpoints'))
DATASETS_DIR = os.environ.get('OLMO_DATASETS_DIR', str(PROJECT_ROOT / 'data' / 'olmo' / 'paloma'))
RESULTS_DIR = os.environ.get('OLMO_RESULTS_DIR', str(PROJECT_ROOT / 'data' / 'olmo' / 'results'))


def _ensure_logging_setup():
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            checkpoint_dir: Directory to write results
            config: Evaluation configuration
            hydra_examples: Pre-loaded fixed examples for HydraEffect evaluator (optional)
    import gc

    _ensure_logging_setup()

    if log_memory and torch.cuda.is_available():
        allocated_before = torch.cuda.memory_allocated() / 1024**3
        reserved_before = torch.cuda.memory_reserved() / 1024**3
        logger.info(f'Memory before cleanup: {allocated_before:.2f} GiB allocated, {reserved_before:.2f} GiB reserved')

    for _ in range(3):
        gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

        try:
            torch.cuda.memory.empty_cache()
        except:
            pass

    gc.collect()

    if log_memory and torch.cuda.is_available():
        allocated_after = torch.cuda.memory_allocated() / 1024**3
        reserved_after = torch.cuda.memory_reserved() / 1024**3
        freed = allocated_before - allocated_after
        logger.info(f'Memory after cleanup: {allocated_after:.2f} GiB allocated, {reserved_after:.2f} GiB reserved (freed {freed:.2f} GiB)')

def log_gpu_memory(message: str):
    Pre-load and pre-process HydraEffect examples once to avoid redundant loading across checkpoints.

    Args:
        checkpoints: List of (checkpoint_path, checkpoint_name) tuples
        datasets: List of dataset names (unused, kept for compatibility)
        config: Evaluation configuration

    Returns:
        Dictionary with pre-loaded data:
        {
            'hydra_examples': [list of text examples for HydraEffect]
        }
    Evaluate all checkpoints assigned to a single GPU sequentially.

    Args:
        gpu_id: GPU device ID
        checkpoint_list: List of (checkpoint_path, checkpoint_name) tuples
        datasets: List of dataset names to evaluate
        config_dict: Configuration dictionary

    Returns:
        List of result dictionaries, one per checkpoint