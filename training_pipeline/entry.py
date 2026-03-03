import json
import os
os.environ["WANDB_DISABLED"] = "true"
import random
import shutil
import subprocess
import uuid
from enum import Enum
from pathlib import Path
from typing import Dict, List
import multiprocessing as mp
from multiprocessing import Queue, Process
import torch.multiprocessing as torch_mp

import pendulum
import torch
from typer import Option, Typer

from training_pipeline.configs import ExperimentConfig, ModelConfig, RunnerConfig
from training_pipeline.runner import Runner
from training_pipeline.utils import (
    create_logger,
)

app = Typer(help="Run grid search experiments", pretty_exceptions_enable=False)

# Set spawn method at module level for both standard multiprocessing and torch
# This is crucial for torch.compile compatibility
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)
if torch_mp.get_start_method(allow_none=True) != "spawn":
    torch_mp.set_start_method("spawn", force=True)


def _train_single_process(
    cfg: RunnerConfig, config: ExperimentConfig, gpu_queue: Queue, result_queue: Queue
):
    """
    Worker process that trains a single model.
    Designed to work with non-daemon processes and torch.compile.
    """
    try:
        # Get a GPU from the queue FIRST, before any imports
        gpu_id = gpu_queue.get()

        try:
            # Set GPU visibility BEFORE importing torch or any CUDA-related modules
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            # Import torch here to ensure proper initialization in subprocess
            import torch
            
            # Verify correct device visibility
            if torch.cuda.is_available():
                num_visible = torch.cuda.device_count()
                if num_visible != 1:
                    print(f"WARNING: Process for GPU {gpu_id} sees {num_visible} GPUs instead of 1")
                # No need to call set_device since we only have one visible GPU

            # Create runner and train
            runner = Runner(cfg, config)
            runner.train_model(gpu_id)

            # Report success
            result_queue.put((cfg.run_id, "success", None))

        finally:
            # Always return GPU to the queue
            gpu_queue.put(gpu_id)

    except Exception as e:
        # Report failure with full traceback
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        result_queue.put((cfg.run_id, "failure", error_msg))


def _gpu_queue_initializer(num_gpus: int) -> Queue:
    """Create and initialize a queue with all available GPU IDs."""
    ctx = mp.get_context("spawn")
    gpu_queue = ctx.Queue()
    for gpu_id in range(num_gpus):
        gpu_queue.put(gpu_id)
    return gpu_queue


BATCH_SIZE = 8  # Process 8 experiments at a time


class ProcessingMode(Enum):
    SEQUENTIAL = "sequential"  # Initialize all, create all datasets, then train all
    BATCHED = "batched"  # Process in batches: create, train, delete per batch


class ExperimentStatus(Enum):
    NOT_INITIALIZED = "not_initialized"
    INITIALIZED = "initialized"
    DATASET_CREATED = "dataset_created"
    DATASETS_CREATED = "datasets_created"
    TRAINING_COMPLETE = "training_complete"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class GridSearchExperiment:
    def __init__(
        self,
        config: ExperimentConfig,
        experiment_name: str,
        output_dir: str | None = None,
        mode: ProcessingMode = ProcessingMode.BATCHED,
    ):
        self.config = config
        # Update the experiment name in config to match the actual experiment
        self.config.experiment_name = experiment_name
        self.name = experiment_name
        self.id = experiment_name
        self.logger = create_logger(self.__class__.__name__)
        self.mode = mode

        # Use external_experiment_path if provided, otherwise use output_dir
        if config.external_experiment_path:
            self.path = Path(config.external_experiment_path) / experiment_name
        else:
            base = output_dir or config.output_dir
            self.path = Path(base) / experiment_name

        # Set datasets path (used by both new and existing experiments)
        self.datasets_path = self.path / "datasets"

    @classmethod
    def from_config(
        cls,
        config_path: Path | str,
        experiment_name: str,
    ) -> "GridSearchExperiment":
        config = ExperimentConfig.from_yaml(Path(config_path))
        mode = ProcessingMode(config.mode.lower())
        return cls(config, experiment_name, mode=mode)

    @classmethod
    def from_existing(
        cls,
        experiment_name: str,
        output_dir: str = "experiments",
    ) -> "GridSearchExperiment":
        path = Path(output_dir) / experiment_name
        overview = path / "overview.json"

        # Check if it's a symlink pointing to an external path
        if path.is_symlink():
            actual_path = path.resolve()
            overview = actual_path / "overview.json"
            if not actual_path.exists() or not overview.exists():
                raise ValueError(
                    f"Experiment '{experiment_name}' not found at {actual_path}"
                )
        elif not path.exists() or not overview.exists():
            raise ValueError(f"Experiment '{experiment_name}' not found at {path}")

        with open(overview, "r") as f:
            data = json.load(f)

        config = ExperimentConfig(**data)
        mode = (
            ProcessingMode(config.mode.lower())
            if hasattr(config, "mode")
            else ProcessingMode.BATCHED
        )
        return cls(config, experiment_name, output_dir, mode=mode)

    @staticmethod
    def experiment_exists(
        experiment_name: str, output_dir: str = "experiments"
    ) -> bool:
        experiment_path = Path(output_dir) / experiment_name

        # Check if it's a symlink pointing to an external path
        if experiment_path.is_symlink():
            actual_path = experiment_path.resolve()
            return actual_path.exists() and (actual_path / "overview.json").exists()
        else:
            return (
                experiment_path.exists()
                and (experiment_path / "overview.json").exists()
            )

    def check_experiment_status(self) -> ExperimentStatus:
        overview_path = self.path / "overview.json"
        if not overview_path.exists():
            return ExperimentStatus.NOT_INITIALIZED

        self.sync_run_status()

        run_statuses = []
        for run_info in self.config.runs:
            try:
                run_statuses.append(ExperimentStatus(run_info.status))
            except ValueError:
                run_statuses.append(ExperimentStatus.UNKNOWN)

        if all(s == ExperimentStatus.TRAINING_COMPLETE for s in run_statuses):
            return ExperimentStatus.TRAINING_COMPLETE

        if all(s == ExperimentStatus.DATASET_CREATED for s in run_statuses):
            return ExperimentStatus.DATASETS_CREATED

        if all(s == ExperimentStatus.INITIALIZED for s in run_statuses):
            return ExperimentStatus.INITIALIZED

        return ExperimentStatus.PARTIAL

    def sample_hyperparameters(self, param_ranges: Dict) -> Dict:
        sampled_params = {}
        for param_name, param_range in param_ranges.items():
            sampled_params[param_name] = random.choice(param_range)
        return sampled_params

    def initialize_experiments(self) -> Path:
        """Initialize experiments based on language configurations"""
        self.path.mkdir(parents=True, exist_ok=True)
        
        # Create datasets directory at experiment level
        self.datasets_path = self.path / "datasets"
        self.datasets_path.mkdir(parents=True, exist_ok=True)

        # If using external_experiment_path, create a symlink in experiments directory
        if self.config.external_experiment_path:
            experiments_dir = Path(self.config.output_dir) / self.name
            if not experiments_dir.exists() and not experiments_dir.is_symlink():
                experiments_dir.parent.mkdir(parents=True, exist_ok=True)
                experiments_dir.symlink_to(self.path)
                self.logger.info(
                    f"Created symlink from {experiments_dir} to {self.path}"
                )

        param_ranges = self.config.param_ranges
        run_index = 0

        # Process each language configuration
        for lang_config in self.config.languages:
            # Extract language type and config
            if isinstance(lang_config, dict):
                # YAML format: - hierarchical: {...} or - bigram: {...}
                language_type = list(lang_config.keys())[0]
                language_params = lang_config[language_type]
            else:
                self.logger.warning(f"Skipping invalid language config: {lang_config}")
                continue

            # Create a unique dataset ID for this language configuration
            dataset_id = f"{language_type}_{uuid.uuid4().hex[:8]}"
            
            # Create experiments for this language
            for _ in range(self.config.num_experiments_per_grammar):
                run_id = str(uuid.uuid4())
                # No need to create run directories anymore
                run_path = self.path / run_id

                # Sample hyperparameters
                hyperparams = self.sample_hyperparameters(param_ranges)
                model_cfg = ModelConfig(**hyperparams)

                # Prepare language-specific configuration
                language_config_data = language_params.copy()
                language_config_data["vocab_size"] = hyperparams["vocab_size"]
                
                if language_type == "hierarchical":
                    # Calculate train/test splits if not provided
                    if "amount_of_docs" in language_config_data:
                        total_docs = language_config_data["amount_of_docs"]
                        train_ratio = 0.8  # 80/20 split
                        language_config_data["train_docs"] = int(total_docs * train_ratio)
                        language_config_data["test_docs"] = (
                            total_docs - language_config_data["train_docs"]
                        )

                # Create runner configuration with dataset_id
                runner_cfg = RunnerConfig(
                    run_id=run_id,
                    run_index=run_index,
                    language_type=language_type,
                    language_config=language_config_data,
                    model_conf=model_cfg,
                    run_path=run_path,
                    dataset_id=dataset_id,  # Reference to shared dataset
                    status=ExperimentStatus.INITIALIZED.value,
                    created_at=pendulum.now().isoformat(),
                )

                self.config.runs.append(runner_cfg)
                run_index += 1

        # Save the overview
        with open(self.path / "overview.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        total = len(self.config.runs)
        self.logger.info(f"Initialized {total} runs at {self.path}")
        self.logger.info(
            f"Languages: {[list(lang.keys())[0] for lang in self.config.languages]}"
        )

        return self.path

    def process_experiments(self) -> None:
        """Main entry point for processing experiments based on the selected mode."""
        if self.mode == ProcessingMode.SEQUENTIAL:
            self.process_sequential()
        else:  # ProcessingMode.BATCHED
            self.process_all_batches()

    def process_sequential(self) -> None:
        """Sequential mode: Initialize all, create all datasets, then train all."""
        mode_desc = "SEQUENTIAL mode (single run)" if self._is_single_run() else "SEQUENTIAL mode"
        self.logger.info(f"Processing in {mode_desc}")

        # Get all pending runs
        pending_runs = [
            r
            for r in self.config.runs
            if r.status
            in (
                ExperimentStatus.INITIALIZED.value,
                ExperimentStatus.DATASET_CREATED.value,
            )
        ]

        if not pending_runs:
            self.logger.info("No pending runs to process")
            return

        # Step 1: Generate all datasets first
        initialized_runs = [
            r
            for r in self.config.runs
            if r.status == ExperimentStatus.INITIALIZED.value
        ]

        if initialized_runs:
            self.logger.info(f"\n{'=' * 50}")
            self.logger.info(
                f"Step 1: Creating datasets for {len(initialized_runs)} runs"
            )
            self.logger.info(f"{'=' * 50}")

            # In sequential mode, provide all datasets to Rust at once
            self.logger.info(
                f"Generating all {len(initialized_runs)} datasets in a single call to Rust"
            )
            self._generate_datasets_rust(initialized_runs)
            self.sync_run_status()

        # Step 2: Train all models with datasets
        dataset_ready_runs = [
            r
            for r in self.config.runs
            if r.status == ExperimentStatus.DATASET_CREATED.value
        ]

        if dataset_ready_runs:
            self.logger.info(f"\n{'=' * 50}")
            self.logger.info(f"Step 2: Training {len(dataset_ready_runs)} models")
            self.logger.info(f"{'=' * 50}")

            # Train all models - _train_batch will handle GPU batching internally
            self._train_batch(dataset_ready_runs)

        # Step 3: Optional cleanup (only after all training is complete)
        if all(
            r.status == ExperimentStatus.TRAINING_COMPLETE.value
            for r in self.config.runs
        ):
            self.logger.info(f"\n{'=' * 50}")
            self.logger.info("Step 3: Cleaning up datasets")
            self.logger.info(f"{'=' * 50}")
            self._cleanup_all_datasets()

    def _cleanup_all_datasets(self) -> None:
        """Clean up all datasets after sequential training is complete."""
        # Force cleanup of any remaining DataLoader workers before proceeding
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Get unique dataset_ids that are fully processed
        processed_datasets = set()
        for run in self.config.runs:
            if run.status == ExperimentStatus.TRAINING_COMPLETE.value:
                processed_datasets.add(run.dataset_id)

        # Clean up each unique dataset
        datasets_path = self.path / "datasets"
        for dataset_id in processed_datasets:
            self._cleanup_dataset(dataset_id, datasets_path)

    def _cleanup_dataset(self, dataset_id: str, datasets_path: Path) -> None:
        return
        # """Clean up datasets for a specific dataset_id."""
        # dataset_dirs = [
        #     datasets_path / dataset_id / "train",
        #     datasets_path / dataset_id / "test",
        #     datasets_path / dataset_id / "non_rejection_dataset",
        #     datasets_path / dataset_id / "icl_dataset",
        #     datasets_path / dataset_id / "superposition_similar",
        #     datasets_path / dataset_id / "superposition_dissimilar",
        #     datasets_path / dataset_id / "inf_dataset",
        #     datasets_path / dataset_id / "tokenized_dataset",
        # ]

        # for dir_path in dataset_dirs:
        #     if dir_path.exists():
        #         self.logger.debug(f"Removing {dir_path}")
        #         shutil.rmtree(dir_path, ignore_errors=True)

        # # Remove the dataset directory itself if empty
        # dataset_dir = datasets_path / dataset_id
        # if dataset_dir.exists() and not any(dataset_dir.iterdir()):
        #     dataset_dir.rmdir()
            
        # self.logger.info(f"Cleaned up dataset {dataset_id}")

    def process_all_batches(self) -> None:
        """Batched mode: Process in batches - create datasets, train, and delete per batch."""
        mode_desc = "BATCHED mode (single run)" if self._is_single_run() else "BATCHED mode"
        self.logger.info(f"Processing in {mode_desc}")

        pending_runs = [
            r
            for r in self.config.runs
            if r.status
            in (
                ExperimentStatus.INITIALIZED.value,
                ExperimentStatus.DATASET_CREATED.value,
            )
        ]

        if not pending_runs:
            self.logger.info("No pending runs to process")
            return

        total_batches = (len(pending_runs) + BATCH_SIZE - 1) // BATCH_SIZE
        self.logger.info(
            f"Processing {len(pending_runs)} runs in {total_batches} batches"
        )

        for batch_idx in range(0, len(pending_runs), BATCH_SIZE):
            batch = pending_runs[batch_idx : batch_idx + BATCH_SIZE]
            batch_num = batch_idx // BATCH_SIZE + 1

            self.logger.info(f"\n{'=' * 50}")
            self.logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} runs)"
            )
            self.logger.info(f"{'=' * 50}")

            # Step 1: Generate datasets for this batch using Rust
            initialized_batch = [
                r for r in batch if r.status == ExperimentStatus.INITIALIZED.value
            ]
            if initialized_batch:
                self._generate_datasets_rust(initialized_batch)

            # Step 2: Train models for all runs with datasets
            dataset_ready_batch = [
                r for r in batch if r.status == ExperimentStatus.DATASET_CREATED.value
            ]
            if dataset_ready_batch:
                self._train_batch(dataset_ready_batch)

            # Step 3: Clean up datasets to save space
            self._cleanup_batch_datasets(batch)

            self.logger.info(f"Completed batch {batch_num}/{total_batches}")

    def _generate_datasets_rust(self, batch: List[RunnerConfig]) -> None:
        """Call Rust binary to generate datasets for specific runs."""
        
        # Group runs by dataset_id (language configuration)
        datasets_to_generate = {}
        for run in batch:
            if run.status == ExperimentStatus.INITIALIZED.value:
                dataset_id = run.dataset_id
                if dataset_id not in datasets_to_generate:
                    # Use the first run with this dataset_id as representative
                    datasets_to_generate[dataset_id] = run
        
        if not datasets_to_generate:
            self.logger.info("No datasets to generate")
            return
            
        self.logger.info(f"Generating {len(datasets_to_generate)} unique datasets...")

        # Process each dataset
        for dataset_id, run in datasets_to_generate.items():
            dataset_path = self.datasets_path / dataset_id
            
            # Check if dataset already exists
            if (dataset_path / "train.arrow").exists() and (dataset_path / "test.arrow").exists():
                self.logger.info(f"Dataset {dataset_id} already exists, skipping generation")
                # Mark all runs with this dataset_id as DATASET_CREATED
                for r in self.config.runs:
                    if r.dataset_id == dataset_id and r.status == ExperimentStatus.INITIALIZED.value:
                        r.status = ExperimentStatus.DATASET_CREATED.value
                        r.dataset_created_at = pendulum.now().isoformat()
                continue
            
            # Prepare configuration for Rust generator
            config = {
                "language_type": run.language_type,
                "language_config": run.language_config,
                "vocab_size": run.model_conf.vocab_size,
                "context_window": run.model_conf.context_window,
                "evaluators": self.config.evaluators or []
            }
            
            config_json = json.dumps(config)
            
            # Call Rust binary with the dataset path as argument
            self.logger.info(f"Generating dataset {dataset_id} at {dataset_path}")
            
            env = os.environ.copy()
            if self._is_single_run():
                # Use all available cores for single dataset generation
                num_cores = mp.cpu_count()
                env["RUST_NUM_WORKERS"] = str(num_cores)
                self.logger.info(f"Single run mode: Using all {num_cores} cores for dataset generation")
            
            result = subprocess.run(
                [self.config.generator_path, str(dataset_path)],
                input=config_json,
                capture_output=True,
                text=True,
                env=env if self._is_single_run() else None,
            )

            if result.returncode != 0:
                self.logger.error(f"Rust generator failed for dataset {dataset_id}: {result.stderr}")
                raise RuntimeError(f"Dataset generation failed: {result.stderr}")

            # Parse metrics from stdout
            metrics = None
            for line in result.stdout.split('\n'):
                if line.startswith('METRICS_JSON:'):
                    try:
                        metrics_str = line.replace('METRICS_JSON:', '').strip()
                        metrics = json.loads(metrics_str)
                        self.logger.info(f"Captured metrics for dataset {dataset_id}: {metrics}")
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse metrics JSON: {e}")

            # Store metrics with the dataset_id for later use
            if metrics:
                for r in self.config.runs:
                    if r.dataset_id == dataset_id:
                        r.language_metrics = metrics

            self.logger.info(f"Successfully generated dataset {dataset_id}")
            
            # Mark all runs with this dataset_id as DATASET_CREATED
            for r in self.config.runs:
                if r.dataset_id == dataset_id and r.status == ExperimentStatus.INITIALIZED.value:
                    r.status = ExperimentStatus.DATASET_CREATED.value
                    r.dataset_created_at = pendulum.now().isoformat()
        
        # Save the updated overview
        self._save_overview()

    def sync_run_status(self) -> None:
        """Sync run statuses from overview.json."""
        # Since we're not creating individual run directories anymore,
        # status is managed entirely through overview.json
        # This method is kept for backward compatibility
        pass

    def _is_single_run(self) -> bool:
        """Check if this is a single run configuration."""
        return len(self.config.runs) == 1

    def _train_single_run(self, run_cfg: RunnerConfig) -> None:
        """Train a single run using all available GPUs with accelerate."""
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPUs available for training")
            
        self.logger.info(f"Single run mode: Training on all {num_gpus} GPUs with accelerate")
        
        # Set all GPUs visible for single run
        gpu_list = ",".join(str(i) for i in range(num_gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        
        # Track training start time for throughput calculation
        import time
        start_time = time.time()
        
        # Create runner and train
        runner = Runner(run_cfg, self.config)
        runner.train_model(gpu_id=None)  # Use all visible GPUs via accelerate
        
        # Calculate and print throughput
        end_time = time.time()
        training_duration = end_time - start_time

        # Load dataset to get token count
        from datasets import Dataset

        # Determine dataset path (same logic as in runner)
        if run_cfg.dataset_id:
            experiment_path = run_cfg.run_path.parent
            dataset_base_path = experiment_path / "datasets" / run_cfg.dataset_id
        else:
            # Fallback to old structure for backward compatibility
            dataset_base_path = run_cfg.run_path

        # Load arrow file to get token count
        train_arrow_path = dataset_base_path / "train.arrow"
        if train_arrow_path.exists():
            train_dataset = Dataset.from_file(str(train_arrow_path))
            total_tokens = len(train_dataset)  # Number of tokens (one per row)
        else:
            # Fallback: try old structure
            try:
                from datasets import load_from_disk
                train_dataset = load_from_disk(run_cfg.run_path / "train")
                total_tokens = sum(len(ids) for ids in train_dataset["input_ids"])
            except:
                self.logger.warning("Could not load training dataset to calculate throughput")
                total_tokens = None

        if total_tokens:
            tokens_per_second = total_tokens / training_duration
        else:
            tokens_per_second = None
        self.logger.info(f"\n" + "="*50)
        self.logger.info(f"Single Run Training Complete")
        self.logger.info(f"Total training time: {training_duration:.2f} seconds")
        if total_tokens is not None:
            self.logger.info(f"Total tokens processed: {total_tokens:,}")
            self.logger.info(f"Throughput: {tokens_per_second:,.0f} tokens/second")
        self.logger.info("="*50 + f"\n")
        
        # Update status
        run_cfg.status = ExperimentStatus.TRAINING_COMPLETE.value
        run_cfg.training_completed_at = pendulum.now().isoformat()
        
        # Update the run in config and save overview
        for run in self.config.runs:
            if run.run_id == run_cfg.run_id:
                run.status = run_cfg.status
                run.training_completed_at = run_cfg.training_completed_at
                break
        self._save_overview()

    def _train_batch(self, batch: List[RunnerConfig]) -> None:
        """Train models using multiprocessing with non-daemon processes."""
        # Check if this is a single run - if so, use optimized single-run training
        if self._is_single_run() and len(batch) == 1:
            self._train_single_run(batch[0])
            return
            
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPUs available for training")

        self.logger.info(
            f"Training {len(batch)} models using multiprocessing on {num_gpus} GPUs..."
        )

        # Create multiprocessing context and queues
        ctx = mp.get_context("spawn")
        gpu_queue = _gpu_queue_initializer(num_gpus)
        result_queue = ctx.Queue()

        # Track active processes and pending configs
        active_processes = {}  # {run_id: process}
        pending_configs = list(batch)  # Configs waiting to be processed
        completed = 0
        failures = []

        # Start initial wave of processes (up to num_gpus)
        while len(active_processes) < num_gpus and pending_configs:
            cfg = pending_configs.pop(0)
            process = ctx.Process(
                target=_train_single_process,
                args=(cfg, self.config, gpu_queue, result_queue),
                daemon=False,  # Crucial: allows torch dataloaders to work
            )
            process.start()
            active_processes[cfg.run_id] = process
            self.logger.debug(f"Started training process for run {cfg.run_id}")

        # Process results and start new processes as GPUs become available
        while completed < len(batch):
            try:
                # Wait for a result with timeout
                run_id, status, error = result_queue.get(
                    timeout=300
                )  # 5 minute timeout
                completed += 1

                # Remove completed process
                if run_id in active_processes:
                    process = active_processes.pop(run_id)
                    process.join(timeout=10)  # Give it time to clean up
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)

                # Log result
                if status == "success":
                    self.logger.info(
                        f"Run {run_id} completed successfully ({completed}/{len(batch)})"
                    )
                else:
                    self.logger.error(f"Run {run_id} failed: {error}")
                    failures.append((run_id, error))

                # Start a new process if there are pending configs
                if pending_configs and len(active_processes) < num_gpus:
                    cfg = pending_configs.pop(0)
                    process = ctx.Process(
                        target=_train_single_process,
                        args=(cfg, self.config, gpu_queue, result_queue),
                        daemon=False,
                    )
                    process.start()
                    active_processes[cfg.run_id] = process
                    self.logger.debug(f"Started training process for run {cfg.run_id}")

            except Exception as e:
                self.logger.error(f"Error waiting for results: {e}")
                # Continue to try to collect other results

        # Clean up any remaining processes
        for run_id, process in active_processes.items():
            process.join(timeout=30)
            if process.is_alive():
                self.logger.warning(
                    f"Process for run {run_id} did not terminate cleanly, forcing..."
                )
                process.terminate()
                process.join(timeout=10)

        # Report any failures
        if failures:
            self.logger.error(f"{len(failures)} runs failed during training:")
            for run_id, error in failures:
                self.logger.error(f"  - {run_id}: {error}")

        # Sync statuses when all are done
        self.sync_run_status()
        self.logger.info(
            f"Completed training batch: {len(batch) - len(failures)} succeeded, {len(failures)} failed"
        )

    def _cleanup_batch_datasets(self, batch: List[RunnerConfig]) -> None:
        """Clean up datasets for runs in a batch that have completed training."""
        # Force cleanup of any remaining DataLoader workers before proceeding
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Get unique dataset_ids from completed runs
        completed_datasets = set()
        for run in batch:
            if run.status == ExperimentStatus.TRAINING_COMPLETE.value:
                # Check if all runs with this dataset_id are complete
                all_complete = all(
                    r.status == ExperimentStatus.TRAINING_COMPLETE.value
                    for r in self.config.runs
                    if r.dataset_id == run.dataset_id
                )
                if all_complete:
                    completed_datasets.add(run.dataset_id)

        # Clean up each completed dataset
        datasets_path = self.path / "datasets"
        for dataset_id in completed_datasets:
            self._cleanup_dataset(dataset_id, datasets_path)

    def _save_overview(self) -> None:
        """Save current config state to overview.json."""
        with open(self.path / "overview.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)


@app.command()
def main(
    experiment_name: str = Option(
        None,
        help="Name of the experiment (defaults to experiment_name from config)",
    ),
    config_path: str = Option(
        "experiment.yml",
        help="Path to config file",
    ),
) -> None:
    global RUST_GENERATOR_PATH

    logger = create_logger("GridSearch")

    # Load config to get the mode and experiment_name
    config = ExperimentConfig.from_yaml(Path(config_path))
    
    # Use experiment_name from config if not provided via CLI
    if experiment_name is None:
        experiment_name = config.experiment_name
        logger.info(f"Using experiment_name from config: '{experiment_name}'")

    # Validate mode from config
    try:
        ProcessingMode(config.mode.lower())
    except ValueError:
        logger.error(
            f"Invalid mode '{config.mode}' in config. Must be 'sequential' or 'batched'"
        )
        raise ValueError(
            f"Invalid mode '{config.mode}'. Must be 'sequential' or 'batched'"
        )

    if GridSearchExperiment.experiment_exists(experiment_name):
        logger.info(f"Loading existing experiment '{experiment_name}'...")
        gs = GridSearchExperiment.from_existing(experiment_name)
    else:
        logger.info(f"Creating new experiment '{experiment_name}'...")
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file '{config_path}' not found")
        gs = GridSearchExperiment.from_config(config_path, experiment_name)

    gs.logger = logger

    status = gs.check_experiment_status()
    logger.info(f"Experiment status: {status.value}")

    try:
        if status == ExperimentStatus.NOT_INITIALIZED:
            gs.initialize_experiments()
            gs.process_experiments()
        elif status == ExperimentStatus.INITIALIZED:
            gs.process_experiments()
        elif status == ExperimentStatus.TRAINING_COMPLETE:
            logger.info("All runs complete.")
        else:  # PARTIAL or other states
            logger.info("Resuming partial experiment...")
            gs.process_experiments()

    except Exception as exc:
        logger.exception("Experiment failed")
        raise


if __name__ == "__main__":
    app()
