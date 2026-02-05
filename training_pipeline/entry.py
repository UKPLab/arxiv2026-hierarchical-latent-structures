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

from training_pipeline.utils.aim_gpu_filter import patch_aim_gpu_tracking

patch_aim_gpu_tracking()

from training_pipeline.configs import ExperimentConfig, ModelConfig, RunnerConfig
from training_pipeline.runner import Runner
from training_pipeline.utils import (
    create_logger,
)

app = Typer(help="Run grid search experiments", pretty_exceptions_enable=False)

if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)
if torch_mp.get_start_method(allow_none=True) != "spawn":
    torch_mp.set_start_method("spawn", force=True)


def _train_single_process(
    cfg: RunnerConfig, config: ExperimentConfig, gpu_queue: Queue, result_queue: Queue
):
    try:
        gpu_id = gpu_queue.get()

        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            from training_pipeline.utils.aim_gpu_filter import patch_aim_gpu_tracking
            patch_aim_gpu_tracking()

            import torch
            
            if torch.cuda.is_available():
                num_visible = torch.cuda.device_count()
                if num_visible != 1:
                    print(f"WARNING: Process for GPU {gpu_id} sees {num_visible} GPUs instead of 1")

            runner = Runner(cfg, config)
            runner.train_model(gpu_id)

            result_queue.put((cfg.run_id, "success", None))

        finally:
            gpu_queue.put(gpu_id)

    except Exception as e:
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        result_queue.put((cfg.run_id, "failure", error_msg))


def _gpu_queue_initializer(num_gpus: int) -> Queue:
        self.path.mkdir(parents=True, exist_ok=True)
        
        self.datasets_path = self.path / "datasets"
        self.datasets_path.mkdir(parents=True, exist_ok=True)

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

        for lang_config in self.config.languages:
            if isinstance(lang_config, dict):
                language_type = list(lang_config.keys())[0]
                language_params = lang_config[language_type]
            else:
                self.logger.warning(f"Skipping invalid language config: {lang_config}")
                continue

            dataset_id = f"{language_type}_{uuid.uuid4().hex[:8]}"
            
            for _ in range(self.config.num_experiments_per_grammar):
                run_id = str(uuid.uuid4())
                run_path = self.path / run_id

                hyperparams = self.sample_hyperparameters(param_ranges)
                model_cfg = ModelConfig(**hyperparams)

                language_config_data = language_params.copy()
                language_config_data["vocab_size"] = hyperparams["vocab_size"]
                
                if language_type == "pcfg":
                    if "amount_of_docs" in language_config_data:
                        total_docs = language_config_data["amount_of_docs"]
                        train_ratio = 0.8
                        language_config_data["train_docs"] = int(total_docs * train_ratio)
                        language_config_data["test_docs"] = (
                            total_docs - language_config_data["train_docs"]
                        )

                runner_cfg = RunnerConfig(
                    run_id=run_id,
                    run_index=run_index,
                    language_type=language_type,
                    language_config=language_config_data,
                    model_conf=model_cfg,
                    run_path=run_path,
                    dataset_id=dataset_id,
                    status=ExperimentStatus.INITIALIZED.value,
                    created_at=pendulum.now().isoformat(),
                    token_limit=self.config.token_limit,
                )

                self.config.runs.append(runner_cfg)
                run_index += 1

        with open(self.path / "overview.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        total = len(self.config.runs)
        self.logger.info(f"Initialized {total} runs at {self.path}")
        self.logger.info(
            f"Languages: {[list(lang.keys())[0] for lang in self.config.languages]}"
        )

        return self.path

    def process_experiments(self) -> None:
        mode_desc = "SEQUENTIAL mode (single run)" if self._is_single_run() else "SEQUENTIAL mode"
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

            self.logger.info(
                f"Generating all {len(initialized_runs)} datasets in a single call to Rust"
            )
            self._generate_datasets_rust(initialized_runs)
            self.sync_run_status()

        dataset_ready_runs = [
            r
            for r in self.config.runs
            if r.status == ExperimentStatus.DATASET_CREATED.value
        ]

        if dataset_ready_runs:
            self.logger.info(f"\n{'=' * 50}")
            self.logger.info(f"Step 2: Training {len(dataset_ready_runs)} models")
            self.logger.info(f"{'=' * 50}")

            self._train_batch(dataset_ready_runs)

        if all(
            r.status == ExperimentStatus.TRAINING_COMPLETE.value
            for r in self.config.runs
        ):
            self.logger.info(f"\n{'=' * 50}")
            self.logger.info("Step 3: Cleaning up datasets")
            self.logger.info(f"{'=' * 50}")
            self._cleanup_all_datasets()


    def _cleanup_all_datasets(self) -> None:

            

    def process_all_batches(self) -> None:
        datasets_to_generate = {}
        for run in batch:
            if run.status == ExperimentStatus.INITIALIZED.value:
                dataset_id = run.dataset_id
                if dataset_id not in datasets_to_generate:
                    datasets_to_generate[dataset_id] = run
        
        if not datasets_to_generate:
            self.logger.info("No datasets to generate")
            return
            
        self.logger.info(f"Generating {len(datasets_to_generate)} unique datasets...")

        for dataset_id, run in datasets_to_generate.items():
            dataset_path = self.datasets_path / dataset_id
            
            if (dataset_path / "train.arrow").exists() and (dataset_path / "test.arrow").exists():
                self.logger.info(f"Dataset {dataset_id} already exists, skipping generation")
                for r in self.config.runs:
                    if r.dataset_id == dataset_id and r.status == ExperimentStatus.INITIALIZED.value:
                        r.status = ExperimentStatus.DATASET_CREATED.value
                        r.dataset_created_at = pendulum.now().isoformat()
                continue
            
            config = {
                "language_type": run.language_type,
                "language_config": run.language_config,
                "vocab_size": run.model_conf.vocab_size,
                "context_window": run.model_conf.context_window,
                "evaluators": self.config.evaluators or []
            }

            config_json = json.dumps(config)

            self.logger.info(f"Generating dataset {dataset_id} at {dataset_path}")
            
            env = os.environ.copy()
            if self._is_single_run():
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

            metrics = None
            for line in result.stdout.split('\n'):
                if line.startswith('METRICS_JSON:'):
                    try:
                        metrics_str = line.replace('METRICS_JSON:', '').strip()
                        metrics = json.loads(metrics_str)
                        self.logger.info(f"Captured metrics for dataset {dataset_id}: {metrics}")
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse metrics JSON: {e}")

            if metrics:
                for r in self.config.runs:
                    if r.dataset_id == dataset_id:
                        r.language_metrics = metrics

            self.logger.info(f"Successfully generated dataset {dataset_id}")
            
            for r in self.config.runs:
                if r.dataset_id == dataset_id and r.status == ExperimentStatus.INITIALIZED.value:
                    r.status = ExperimentStatus.DATASET_CREATED.value
                    r.dataset_created_at = pendulum.now().isoformat()
        
        self._save_overview()

    def sync_run_status(self) -> None:
        return len(self.config.runs) == 1

    def _train_single_run(self, run_cfg: RunnerConfig) -> None:
        if self._is_single_run() and len(batch) == 1:
            self._train_single_run(batch[0])
            return
            
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPUs available for training")

        self.logger.info(
            f"Training {len(batch)} models using multiprocessing on {num_gpus} GPUs..."
        )

        ctx = mp.get_context("spawn")
        gpu_queue = _gpu_queue_initializer(num_gpus)
        result_queue = ctx.Queue()

        active_processes = {}
        pending_configs = list(batch)
        completed = 0
        failures = []

        while len(active_processes) < num_gpus and pending_configs:
            cfg = pending_configs.pop(0)
            process = ctx.Process(
                target=_train_single_process,
                args=(cfg, self.config, gpu_queue, result_queue),
                daemon=False,
            )
            process.start()
            active_processes[cfg.run_id] = process
            self.logger.debug(f"Started training process for run {cfg.run_id}")

        while completed < len(batch):
            try:
                run_id, status, error = result_queue.get(
                    timeout=300
                )
                completed += 1

                if run_id in active_processes:
                    process = active_processes.pop(run_id)
                    process.join(timeout=10)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)

                if status == "success":
                    self.logger.info(
                        f"Run {run_id} completed successfully ({completed}/{len(batch)})"
                    )
                else:
                    self.logger.error(f"Run {run_id} failed: {error}")
                    failures.append((run_id, error))

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

        for run_id, process in active_processes.items():
            process.join(timeout=30)
            if process.is_alive():
                self.logger.warning(
                    f"Process for run {run_id} did not terminate cleanly, forcing..."
                )
                process.terminate()
                process.join(timeout=10)

        if failures:
            self.logger.error(f"{len(failures)} runs failed during training:")
            for run_id, error in failures:
                self.logger.error(f"  - {run_id}: {error}")

        self.sync_run_status()
        self.logger.info(
            f"Completed training batch: {len(batch) - len(failures)} succeeded, {len(failures)} failed"
        )

    def _cleanup_batch_datasets(self, batch: List[RunnerConfig]) -> None:
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

    config = ExperimentConfig.from_yaml(Path(config_path))
    
    if experiment_name is None:
        experiment_name = config.experiment_name
        logger.info(f"Using experiment_name from config: '{experiment_name}'")

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
        else:
            logger.info("Resuming partial experiment...")
            gs.process_experiments()

    except Exception as exc:
        logger.exception("Experiment failed")
        raise


if __name__ == "__main__":
    app()
