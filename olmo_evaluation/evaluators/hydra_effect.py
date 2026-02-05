import torch
import numpy as np
from typing import List, Dict, Any, Optional
import warnings
import logging
import json
import os

warnings.filterwarnings('ignore', message='.*fast tokenizer.*')
warnings.filterwarnings('ignore', message='.*generation flags.*')
logging.getLogger('transformers').setLevel(logging.ERROR)


class HydraEffectEvaluator:
    def __init__(self, seq_len: int = 2048, num_samples: int = 10):
        self.seq_len = seq_len
        self.num_samples = num_samples

    def create_random_sequence(self, tokenizer) -> torch.Tensor:
        vocab_size = tokenizer.vocab_size

        random_token_ids = torch.randint(
            low=100,
            high=vocab_size - 100,
            size=(self.seq_len,)
        )

        return random_token_ids.unsqueeze(0)

    @torch.inference_mode()
    def evaluate(self, nnsight_model, tokenizer, texts: List[str] = None,
                 max_samples: int = None, batch_size: int = None,
                 output_path: Optional[str] = None) -> Dict[str, Any]:
        n_layers = len(nnsight_model.model.layers)
        lm_head = nnsight_model.lm_head if hasattr(nnsight_model, 'lm_head') else nnsight_model.model.lm_head

        all_results = {f'k{k}': {} for k in [1, 2]}

        for sample_idx in range(self.num_samples):
            input_ids = self.create_random_sequence(tokenizer)

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()

            for k in [1, 2]:
                for layer_idx in range(k, n_layers):
                    with nnsight_model.trace() as tracer:
                        with tracer.invoke({"input_ids": input_ids}):
                            hidden_states = nnsight_model.model.layers[layer_idx].output[0]
                            logits = lm_head(hidden_states)
                            clean_logits = logits.save()

                    if clean_logits.dim() == 3:
                        clean_last = clean_logits[0, -1, :]
                    else:
                        clean_last = clean_logits[-1]

                    with nnsight_model.trace() as tracer:
                        with tracer.invoke({"input_ids": input_ids}):
                            nnsight_model.model.layers[layer_idx - k].output[:] = 0
                            hidden_states = nnsight_model.model.layers[layer_idx].output[0]
                            logits = lm_head(hidden_states)
                            ablated_logits = logits.save()

                    if ablated_logits.dim() == 3:
                        ablated_last = ablated_logits[0, -1, :]
                    else:
                        ablated_last = ablated_logits[-1]

                    gold_token_id = input_ids[0, -1].item()
                    clean_logit = clean_last[gold_token_id].item()
                    ablated_logit = ablated_last[gold_token_id].item()
                    diff = clean_logit - ablated_logit

                    if layer_idx not in all_results[f'k{k}']:
                        all_results[f'k{k}'][layer_idx] = []
                    all_results[f'k{k}'][layer_idx].append(diff)

        final_results = {
            'n_samples': self.num_samples,
            'n_layers': n_layers,
            'k_impacts': {}
        }

        for k in [1, 2]:
            layer_avg = {}
            for layer_idx, diffs in all_results[f'k{k}'].items():
                layer_avg[layer_idx] = float(np.mean(diffs))
            final_results['k_impacts'][f'k{k}'] = layer_avg

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(final_results, f, indent=2)
            return {
                'status': 'success',
                'output_path': output_path,
                'n_layers': n_layers,
                'n_samples': self.num_samples
            }

        return final_results
