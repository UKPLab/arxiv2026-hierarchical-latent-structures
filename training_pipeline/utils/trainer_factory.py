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
# from .batch_size_optimizer import get_batch_sizes_for_config  # Removed - using JSON file instead


def load_batch_sizes_from_json(model_config: dict) -> dict:
    """Load batch sizes from optimal_batch_sizes.json based on model configuration."""
    # Create model key from config
    v = model_config.get("vocab_size", 1000)
    c = model_config.get("context_window", 256)
    d = model_config.get("model_dim", 128)
    l = model_config.get("num_layers", 2)
    h = model_config.get("num_heads", 2)
    
    # Format key as v{vocab}_c{context}_d{dim}_l{layers}_h{heads}
    model_key = f"v{v}_c{c}_d{d}_l{l}_h{h}"
    
    # Load batch sizes from JSON - try multiple possible locations
    possible_paths = [
        Path(__file__).parent.parent.parent / "optimal_batch_sizes.json",
        Path.cwd() / "optimal_batch_sizes.json",
        Path("/p/project1/westai0065/master-thesis/optimal_batch_sizes.json"),
    ]
    
    json_path = None
    for path in possible_paths:
        if path.exists():
            json_path = path
            break
    
    if json_path is None:
        raise FileNotFoundError(
            f"Batch sizes file not found. Tried: {[str(p) for p in possible_paths]}"
        )
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Access the summary section in the new consolidated format
    if "summary" in data:
        batch_sizes_data = data["summary"]
    else:
        # Fallback for old format
        batch_sizes_data = data
    
    if model_key not in batch_sizes_data:
        # Default to 128 for all batch sizes if model configuration not found
        return {
            "train": 512,
            "eval": 512,
            "hydra": 512,
            "inf": 512,
            "superposition": 512,
        }

    model_batch_sizes = batch_sizes_data[model_key]

    # Handle evaluators - they might be under 'evaluators' key or directly in the dict
    evaluators = model_batch_sizes.get("evaluators", {})

    # Return in the expected format with defaults if specific evaluator not found
    return {
        "train": model_batch_sizes["training_batch_size"],
        "eval": model_batch_sizes["training_batch_size"],  # Use same as train for eval
        "hydra": evaluators.get("HydraEffect", model_batch_sizes["training_batch_size"]),
        "inf": evaluators.get("ImplicitNeuralFunction", model_batch_sizes["training_batch_size"]),
        "superposition": evaluators.get("Superposition", model_batch_sizes["training_batch_size"]),
    }


class TrainerFactory:
    @staticmethod
    def create_trainer(
        model: GPT2LMHeadModel,
        train_dataset: Dataset,
        eval_dataset: Dataset | dict[str, Dataset],
        tokenizer: PreTrainedTokenizerBase,
        runner,
        evaluator_names: list[str] | None = None,
    ) -> Trainer:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # Load batch sizes from JSON file based on model configuration
        batch_sizes = None
        if hasattr(runner, "config") and hasattr(runner.config, "model_conf"):
            model_config_dict = runner.config.model_conf.model_dump()
            batch_sizes = load_batch_sizes_from_json(model_config_dict)

        # Use loaded batch sizes or raise error if not found
        if batch_sizes is None:
            raise ValueError("Unable to load batch sizes for model configuration")
            
        train_batch_size = batch_sizes["train"] - 16
        eval_batch_size = batch_sizes["eval"]
        
        # Always use the run directory for checkpoints
        if not hasattr(runner, 'config') or not hasattr(runner.config, 'run_path'):
            raise ValueError("Runner must have config.run_path for checkpoint saving")
        
        output_dir = str(runner.config.run_path / "checkpoints")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            save_strategy="steps",
            save_steps=0.01,  # Save every 1% of training steps = 100 checkpoints total
            save_total_limit=100,  # Keep exactly 100 checkpoints
            eval_strategy="no",
            #eval_steps=0,
            disable_tqdm=True,
            # auto_find_batch_size=True,
            fp16=True,
            dataloader_num_workers=6,
            dataloader_prefetch_factor=2,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
            torch_compile=True,
        )

        # Create training loss logger to save loss to CSV in run folder
        loss_logger = TrainingLossLogger(output_dir=runner.config.run_path)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[runner.aim_callback, loss_logger],
        )

        for name in evaluator_names or []:
            # Get batch size from experiment config if available
            batch_size = None
            if hasattr(runner, "experiment") and hasattr(
                runner.experiment, "evaluator_batch_sizes"
            ):
                batch_size = runner.experiment.evaluator_batch_sizes.get(name)

            # If no batch size in experiment config, use optimal batch size
            if batch_size is None and batch_sizes:
                if name == "HydraEffect":
                    batch_size = batch_sizes["hydra"]
                elif name == "ImplicitNeuralFunction":
                    batch_size = batch_sizes["inf"]
                elif name == "Superposition":
                    batch_size = batch_sizes["superposition"]

            callback: TrainerCallback = EvaluatorFactory.create(
                name=name,
                tokenizer=tokenizer,
                aim_run=runner.get_aim_run(),
                trainer=trainer,
                runner=runner,  # Pass runner to the factory
                batch_size=batch_size,
            )
            trainer.add_callback(callback)

        return trainer
