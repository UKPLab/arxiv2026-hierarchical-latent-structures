import json
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, TrainerState


def find_checkpoints(checkpoints_dir: Path) -> List[int]:
    if not checkpoints_dir.exists():
        return []

    checkpoint_nums = []
    for checkpoint_dir in checkpoints_dir.iterdir():
        if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith("checkpoint-"):
            try:
                num = int(checkpoint_dir.name.split("-")[1])
                checkpoint_nums.append(num)
            except (IndexError, ValueError):
                continue

    return sorted(checkpoint_nums)


def parse_checkpoint_spec(spec: str, available_checkpoints: List[int]) -> List[int]:
    if spec.lower() == "all":
        return available_checkpoints

    requested = set()

    parts = spec.split(",")
    for part in parts:
        part = part.strip()
        if "-" in part:
            try:
                start, end = part.split("-")
                start, end = int(start.strip()), int(end.strip())
                requested.update(range(start, end + 1))
            except ValueError:
                raise ValueError(f"Invalid range specification: {part}")
        else:
            try:
                requested.add(int(part))
            except ValueError:
                raise ValueError(f"Invalid checkpoint number: {part}")

    result = [cp for cp in sorted(requested) if cp in available_checkpoints]

    if not result:
        raise ValueError(
            f"No valid checkpoints found for spec '{spec}'. "
            f"Available: {available_checkpoints}"
        )

    return result


def load_checkpoint_model(checkpoint_path: Path) -> AutoModelForCausalLM:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return model


def load_trainer_state(checkpoint_path: Path) -> TrainerState:
    trainer_state_file = checkpoint_path / "trainer_state.json"

    if not trainer_state_file.exists():
        raise FileNotFoundError(
            f"trainer_state.json not found in {checkpoint_path}"
        )

    with open(trainer_state_file, "r") as f:
        state_dict = json.load(f)

    trainer_state = TrainerState(
        epoch=state_dict.get("epoch", 0),
        global_step=state_dict.get("global_step", 0),
        max_steps=state_dict.get("max_steps", 0),
        num_train_epochs=state_dict.get("num_train_epochs", 0),
        total_flos=state_dict.get("total_flos", 0),
        log_history=state_dict.get("log_history", []),
    )

    return trainer_state


def get_checkpoint_path(run_path: Path, checkpoint_num: int) -> Path:
    return run_path / "checkpoints" / f"checkpoint-{checkpoint_num}"
