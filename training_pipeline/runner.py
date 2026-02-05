from __future__ import annotations

import json
import logging
import os
from typing import Optional

import pendulum

from training_pipeline.utils.aim_gpu_filter import patch_aim_gpu_tracking

patch_aim_gpu_tracking()

from aim.hugging_face import AimCallback
from datasets import load_from_disk
from training_pipeline.utils.arrow_loader import load_uint16_as_hf_input_ids
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from training_pipeline.configs import ExperimentConfig, RunnerConfig
from training_pipeline.utils import (
    ModelFactory,
    TrainerFactory,
    create_logger,
    format_duration,
    format_duration_from_timestamps,
)


class Runner:
    def __init__(
        self,
        config: "RunnerConfig",
        experiment: "ExperimentConfig" | None = None,
        logger=None,
    ) -> None:
        self.config = config
        self.experiment = experiment
        self.logger = logger or create_logger(self.__class__.__name__)
        self.aim_callback = AimCallback(repo=".", experiment=experiment.experiment_name)
        if self.aim_callback.experiment is None:
            self.logger.warning(
                "AimCallback experiment is None - metrics will not be tracked"
            )
        logging.getLogger("aim").setLevel(logging.ERROR)
        self.language_creation_time = None
        self.dataset_creation_duration = None
        self.dataset_max_depth = None
        self.superposition_datasets = None

    def get_aim_run(self):
        return self.aim_callback.experiment

    def train_model(self, gpu_id: Optional[int] = None) -> None:
        
        training_start = pendulum.now()
        
        if self.config.dataset_id:
            experiment_path = self.config.run_path.parent
            dataset_base_path = experiment_path / "datasets" / self.config.dataset_id
        else:
            dataset_base_path = self.config.run_path

        seq_length = self.config.model_conf.context_window
        
        self.logger.info(f"Loading datasets from {dataset_base_path}")
        self.logger.info("Loading train dataset...")
        train_tokenized = load_uint16_as_hf_input_ids(str(dataset_base_path / "train.arrow"), seq_length)
        self.logger.info("Train dataset loaded (streaming)")
        
        self.logger.info("Loading test dataset...")
        test_tokenzied = load_uint16_as_hf_input_ids(str(dataset_base_path / "test.arrow"), seq_length)
        self.logger.info("Test dataset loaded (streaming)")

        non_rej_path = dataset_base_path / "non_rejection.arrow"
        non_rej_dataset = None
        if non_rej_path.exists():
            self.logger.info("Loading non-rejection dataset...")
            non_rej_dataset = load_uint16_as_hf_input_ids(str(non_rej_path), seq_length)
            self.logger.info("Non-rejection dataset loaded (streaming)")

        icl_path = dataset_base_path / "icl.arrow"
        icl_dataset = None
        if icl_path.exists():
            self.logger.info("Loading ICL dataset...")
            icl_dataset = load_uint16_as_hf_input_ids(str(icl_path), seq_length)
            self.logger.info("ICL dataset loaded (streaming)")

   

        from transformers import AutoTokenizer

        tokenizer_dir = dataset_base_path / "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

        import random

        import numpy as np
        import torch
        from transformers import set_seed

        seed = random.randint(0, 2**32 - 1)
        set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        run = self.aim_callback.experiment
        if run is None:
            self.logger.warning(
                "Aim run is None, metrics will not be tracked for this experiment"
            )
        else:
            run["seed"] = seed
            run["runner_config"] = self.config.to_dict()

        if self.config.dataset_created_at and self.config.created_at:
            dataset_duration = format_duration_from_timestamps(
                self.config.created_at, self.config.dataset_created_at
            )
            if run is not None:
                run["dataset_creation_duration"] = dataset_duration
            self.logger.info(f"Dataset creation took: {dataset_duration}")

        if self.config.language_metrics and run is not None:
            for metric_name, metric_value in self.config.language_metrics.items():
                run[f"language_metrics/{metric_name}"] = metric_value
            self.logger.info(f"Tracked language metrics from config: {self.config.language_metrics}")

        metrics_paths = [
            dataset_base_path / "metrics.json",
            dataset_base_path / "language_metrics.json"
        ]

        for metrics_path in metrics_paths:
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    language_metrics = json.load(f)

                if run is not None:
                    for metric_name, metric_value in language_metrics.items():
                        run[f"language_metrics/{metric_name}"] = metric_value
                if "true_entropy_nats" in language_metrics:
                    self.logger.info(f"True entropy: {language_metrics['true_entropy_nats']:.4f} nats")
                if "true_entropy_bits" in language_metrics:
                    self.logger.info(f"True entropy: {language_metrics['true_entropy_bits']:.4f} bits")
                if "transition_matrix_rank" in language_metrics:
                    self.logger.info(f"Transition matrix rank: {language_metrics['transition_matrix_rank']}")
                if "ngram_transition_matrix_rank" in language_metrics:
                    self.logger.info(f"Ngram transition matrix rank: {language_metrics['ngram_transition_matrix_rank']}")
                if "pcfg_entropy_estimate" in language_metrics:
                    self.logger.info(f"PCFG entropy estimate: {language_metrics['pcfg_entropy_estimate']:.4f}")

                break  

        model = ModelFactory.create_model(len(tokenizer), self.config.model_conf)
        if run is not None:
            run["num_parameters"] = int(sum(p.numel() for p in model.parameters()))

        eval_datasets = {"test": test_tokenzied}
        if non_rej_dataset is not None:
            eval_datasets["NonRejectionDataset"] = non_rej_dataset
        if icl_dataset is not None:
            eval_datasets["InContextLearning"] = icl_dataset

        eval_names = None
        if self.experiment:
            eval_names = [
                n
                for n in self.experiment.evaluators
                if n not in ("NonRejectionDataset", "InContextLearning")
            ]

        self.logger.info("Starting args create...")


        trainer = TrainerFactory.create_trainer(
            model=model,
            train_dataset=train_tokenized,
            eval_dataset=eval_datasets,
            tokenizer=tokenizer,
            runner=self,
            evaluator_names=eval_names,
        )
        self.logger.info("End args create..")


        class EvaluatorDurationTracker(TrainerCallback):
            def __init__(self, aim_callback, logger, trainer):
                self.aim_callback = aim_callback
                self.logger = logger
                self.trainer = trainer

            def on_train_end(
                self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                **kwargs,
            ):
                run = self.aim_callback.experiment
                if run is None:
                    return

                for callback in self.trainer.callback_handler.callbacks:
                    if hasattr(callback, "total_duration") and hasattr(
                        callback, "__class__"
                    ):
                        evaluator_name = callback.__class__.__name__
                        if callback.total_duration > 0:
                            duration_formatted = format_duration(
                                callback.total_duration
                            )
                            run[f"{evaluator_name}_duration"] = duration_formatted
                            self.logger.info(
                                f"{evaluator_name} took: {duration_formatted}"
                            )

        duration_tracker = EvaluatorDurationTracker(
            self.aim_callback, self.logger, trainer
        )
        trainer.callback_handler.callbacks.insert(0, duration_tracker)
        self.logger.info("Starting training...")
        trainer.train()

        training_end = pendulum.now()
        training_duration = (training_end - training_start).total_seconds()
        training_duration_formatted = format_duration(training_duration)
        final_run = self.aim_callback.experiment
        if final_run is not None:
            final_run["training_duration"] = training_duration_formatted
        self.logger.info(f"Training took: {training_duration_formatted}")

        self.config.status = "training_complete"
        self.config.training_completed_at = training_end.isoformat()
        
        experiment_path = self.config.run_path.parent
        overview_path = experiment_path / "overview.json"
        print(overview_path)
        if overview_path.exists():
            with open(overview_path, "r") as f:
                overview = json.load(f)
            
            for run in overview.get("runs", []):
                if run["run_id"] == self.config.run_id:
                    run["status"] = self.config.status
                    run["training_completed_at"] = self.config.training_completed_at
                    break
            
            with open(overview_path, "w") as f:
                json.dump(overview, f, indent=2)

        test_start = pendulum.now()

        trainer.evaluate(eval_dataset=test_tokenzied, metric_key_prefix="test")
        if non_rej_dataset is not None:
            trainer.evaluate(
                eval_dataset=non_rej_dataset,
                metric_key_prefix="non_rejection",
            )
        if icl_dataset is not None:
            trainer.evaluate(
                eval_dataset=icl_dataset,
                metric_key_prefix="icl",
            )

        test_end = pendulum.now()
        test_duration = (test_end - test_start).total_seconds()
        test_duration_formatted = format_duration(test_duration)
        test_run = self.aim_callback.experiment
        if test_run is not None:
            test_run["test_duration"] = test_duration_formatted
        self.logger.info(f"Testing/evaluation took: {test_duration_formatted}")

    @staticmethod
    def format_duration(duration) -> str:
        return duration.in_words(locale="en")
