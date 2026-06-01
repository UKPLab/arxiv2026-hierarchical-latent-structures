import gc
import json
import os
import time
import random
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from transformers import AutoTokenizer
from typer import Typer, Option, Context
from training_pipeline.configs import RunnerConfig
from training_pipeline.utils.arrow_loader import load_uint16_as_hf_input_ids
from training_pipeline.utils import create_logger
from training_pipeline.utils.checkpoint_manager import (
    find_checkpoints,
    parse_checkpoint_spec,
    load_checkpoint_model,
    load_trainer_state,
    get_checkpoint_path,
)

from nnsight import CONFIG
CONFIG.APP.PYMOUNT = False
CONFIG.save()


app = Typer(help="Run Hydra Effect evaluation on saved checkpoints")

def load_datasets(
    run_config: RunnerConfig,
    seq_length: int,
    tokenizer=None,
):
    import traceback

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)

    run_config = RunnerConfig(**run_config_dict)
    logger = create_logger(f"HydraEffect-GPU{gpu_id}")
    start_time = time.time()
    error_msg = None

    try:
        logger.info(f"[GPU {gpu_id}] Evaluating checkpoint {checkpoint_num}")
        checkpoint_path = get_checkpoint_path(run_config.run_path, checkpoint_num)

        logger.info(f"[GPU {gpu_id}] Loading model from {checkpoint_path}...")
        model = load_checkpoint_model(checkpoint_path)

        trainer_state = load_trainer_state(checkpoint_path)
        logger.info(
            f"[GPU {gpu_id}] Loaded checkpoint at step {trainer_state.global_step}"
        )

        logger.info(f"[GPU {gpu_id}] Loading tokenizer from {tokenizer_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

        logger.info(f"[GPU {gpu_id}] Creating NNsight model...")
        import nnsight
        from nnsight import LanguageModel
        nnsight_model = LanguageModel(model, tokenizer=tokenizer)

        datasets = datasets_dict
        test_dataset = datasets.get("test")
        if test_dataset is None:
            raise ValueError("Test dataset not found in datasets_dict")

        results_dir = run_config.run_path / "post_eval_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(nnsight_model, '_graph_cache'):
            nnsight_model._graph_cache.clear()
        if hasattr(nnsight_model, '_trace_cache'):
            nnsight_model._trace_cache.clear()
        from nnsight.intervention.tracing.globals import Globals
       
        saves_count = len(Globals.saves)
        Globals.saves.clear()
        Globals.stack = 0
        logger.info(f'Cleared nnsight global state ({saves_count} stale object IDs removed)')

        logger.info(f"[GPU {gpu_id}] Running HydraEffect evaluation...")
        
        n_layers = len(nnsight_model.model.layers)
        
        lm_head = nnsight_model.lm_head if hasattr(nnsight_model, 'lm_head') else nnsight_model.model.lm_head

        from torch.utils.data import IterableDataset as TorchIterableDataset
        is_iterable_only = isinstance(test_dataset, TorchIterableDataset)

        samples = []
        if is_iterable_only:
            for i, sample in enumerate(test_dataset):
                if i >= max_samples:
                    break
                samples.append(sample)
        else:
            total = len(test_dataset)
            indices = list(range(total))
            if total > max_samples:
                indices = random.sample(indices, max_samples)
            for idx in indices:
                if hasattr(test_dataset, 'select'):
                    sample = test_dataset.select([idx])[0]
                else:
                    sample = test_dataset[idx]
                samples.append(sample)
        
        logger.info(f"[GPU {gpu_id}] Using {len(samples)} samples for evaluation")

        all_results = {}
        
        for k in [1, 2]:
            layer_diffs = {layer_idx: [] for layer_idx in range(k, n_layers)}
            
            for batch_start in range(0, len(samples), batch_size):
                batch_end = min(batch_start + batch_size, len(samples))
                batch_samples = samples[batch_start:batch_end]
                current_batch_size = len(batch_samples)
                
                batch_tokens = [torch.tensor(sample["input_ids"]).unsqueeze(0) for sample in batch_samples]
                
                for layer_idx in range(k, n_layers):
                    
                    base_gold_logits = []
                    with nnsight_model.trace() as tracer:
                        for i, tokens in enumerate(batch_tokens):
                            with tracer.invoke({"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}):
                                hidden_states = nnsight_model.model.layers[layer_idx].output[0]

                                logits = lm_head(hidden_states)

                                if logits.dim() == 3:
                                    last_token_logits = logits[0, -1, :]
                                else:
                                    last_token_logits = logits[-1, :]

                                base_gold_logits.append(nnsight.save(last_token_logits))

                    ablated_gold_logits = []
                    with nnsight_model.trace() as tracer:
                        for i, tokens in enumerate(batch_tokens):
                            with tracer.invoke({"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}):
                                nnsight_model.model.layers[layer_idx - k].output[:] = 0

                                hidden_states = nnsight_model.model.layers[layer_idx].output[0]

                                logits = lm_head(hidden_states)

                                if logits.dim() == 3:
                                    last_token_logits = logits[0, -1, :]
                                else:
                                    last_token_logits = logits[-1, :]

                                ablated_gold_logits.append(nnsight.save(last_token_logits))

                    for i in range(current_batch_size):
                        gold_token_id = batch_tokens[i][0, -1].item()

                        base_logit = base_gold_logits[i][gold_token_id].item()
                        ablated_logit = ablated_gold_logits[i][gold_token_id].item()

                        diff = base_logit - ablated_logit
                        layer_diffs[layer_idx].append(diff)

                    del base_gold_logits, ablated_gold_logits
                    
            k_results = {}
            for layer_idx, diffs in layer_diffs.items():
                if diffs:
                    k_results[layer_idx] = float(np.mean(diffs))
            all_results[f'k{k}'] = k_results
            
            del layer_diffs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        final_results = {
            'checkpoint': checkpoint_num,
            'global_step': trainer_state.global_step,
            'n_samples': len(samples),
            'n_layers': n_layers,
            'hydra_avg_drop_k1_eval': all_results.get('k1', {}),
            'hydra_avg_drop_k2_eval': all_results.get('k2', {}),
        }

        checkpoint_name = f"checkpoint-{checkpoint_num}"
        output_path = results_dir / f"{checkpoint_name}_HydraEffect.json"
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)

        logger.info(f"[GPU {gpu_id}] ✓ HydraEffect completed - results saved to {output_path.name}")

        del model
        del nnsight_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        success = True

    except Exception as e:
        logger.error(
            f"[GPU {gpu_id}] Failed to evaluate checkpoint {checkpoint_num}: {e}",
            exc_info=True,
        )
        print(
            f"\n========== ERROR [GPU {gpu_id}]: Checkpoint {checkpoint_num} ==========",
            flush=True,
        )
        traceback.print_exc()
        print(
            f"================================================================================\n",
            flush=True,
        )
        success = False
        error_msg = str(e)

    elapsed_time = time.time() - start_time
    return {
        "checkpoint_num": checkpoint_num,
        "gpu_id": gpu_id,
        "success": success,
        "elapsed_time": elapsed_time,
        "error": error_msg,
    }

@app.callback(invoke_without_command=True)
def main(
    ctx: Context,
    experiment: str = Option(..., help="Experiment name"),
    run_id: str = Option(..., help="Run ID (UUID)"),
    checkpoints: str = Option(
        "all", help="Checkpoint specification (e.g., 'all', '1,10,50', '1-10')"
    ),
    output_dir: str = Option(
        "experiments", help="Base directory for experiments"
    ),
    max_samples: int = Option(
        100, help="Maximum number of samples to evaluate (default: 20, use 100 to match original)"
    ),
    batch_size: int = Option(
        1, help="Batch size for evaluation (default: 1)"
    ),
) -> None: