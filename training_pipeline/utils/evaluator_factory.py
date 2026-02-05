from __future__ import annotations

from typing import Optional
from transformers import TrainerCallback


class EvaluatorFactory:
    @staticmethod
    def create(
        name: str,
        tokenizer: PreTrainedTokenizerBase,
        aim_run,
        trainer: Trainer,
        runner=None,  
        batch_size: Optional[int] = None,  
    ) -> TrainerCallback:
        eval_dataset = trainer.eval_dataset
        if isinstance(eval_dataset, dict):
            eval_dataset = eval_dataset.get(
                name, eval_dataset.get("test") or list(eval_dataset.values())[0]
            )
        if trainer.args.eval_strategy == "steps":
            if hasattr(trainer, '_get_eval_steps'):
                repeat_steps = trainer._get_eval_steps(trainer.args.eval_steps)
            else:
                num_update_steps_per_epoch = len(trainer.get_train_dataloader())
                num_train_epochs = trainer.args.num_train_epochs
                max_steps = trainer.args.max_steps
                
                if max_steps > 0:
                    total_steps = max_steps
                else:
                    total_steps = int(num_update_steps_per_epoch * num_train_epochs)
                
                if trainer.args.eval_steps < 1:
                    repeat_steps = int(trainer.args.eval_steps * total_steps)
                else:
                    repeat_steps = int(trainer.args.eval_steps)
                    
                repeat_steps = max(1, repeat_steps)
        else:
            repeat_steps = 1

        if name == "HydraEffect":
            from training_pipeline.evaluators import HydraEffect

            return HydraEffect(
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                aim_run=aim_run,
                mode="train",
                repeat_steps=repeat_steps,
                max_samples=100,
                batch_size=batch_size or 1,
            )


        elif name == "ImplicitNeuralFunction":
            from training_pipeline.evaluators import ImplicitNeuralFunctionEvaluator

            inf_dataset = None
            if runner and hasattr(runner, "inf_dataset"):
                inf_dataset = runner.inf_dataset

            return ImplicitNeuralFunctionEvaluator(
                inf_dataset=inf_dataset,
                tokenizer=tokenizer,
                aim_run=aim_run,
                mode="train",
                repeat_steps=repeat_steps,
                few_shot_examples=5,  
                batch_size=batch_size or 8, 
            )

        raise ValueError(f"Unknown evaluator '{name}'")
