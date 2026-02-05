import torch
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import warnings
import logging
import json
import os

warnings.filterwarnings('ignore', message='.*fast tokenizer.*')
warnings.filterwarnings('ignore', message='.*generation flags.*')
logging.getLogger('transformers').setLevel(logging.ERROR)


class InductionHeadsEvaluator:
    def __init__(self, seq_len: int = 64, max_k: int = 10):
        self.seq_len = seq_len
        self.max_k = max_k

    def create_induction_sequence(self, tokenizer, seq_len: Optional[int] = None) -> torch.Tensor:
        seq_len = seq_len or self.seq_len
        vocab_size = tokenizer.vocab_size

        random_token_ids = torch.randint(
            low=100,
            high=vocab_size - 100,
            size=(seq_len,)
        )

        input_ids = torch.cat([random_token_ids, random_token_ids])

        return input_ids.unsqueeze(0)

    @torch.inference_mode()
    def get_k_order_scores(self, model, tokenizer, input_ids: torch.Tensor) -> np.ndarray:
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        n = self.seq_len
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads

        scores = torch.zeros((self.max_k, num_layers, num_heads)).to(device)

        original_attn_implementation = getattr(model.config, '_attn_implementation', None)
        model.set_attn_implementation('eager')

        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
            attentions = torch.stack(outputs.attentions)

        if original_attn_implementation is not None:
            model.set_attn_implementation(original_attn_implementation)

        for k in range(1, self.max_k + 1):
            start_idx = n + k
            if start_idx >= 2 * n:
                continue

            q_indices = torch.arange(start_idx, 2 * n)

            k_indices = torch.arange(k, n)

            batch_attn = attentions[:, 0, :, q_indices, k_indices]
            scores[k-1] = batch_attn.mean(dim=-1)

        max_scores_per_k = scores.amax(dim=(1, 2)).cpu().numpy()
        return max_scores_per_k

    @torch.inference_mode()
    def evaluate(self, model, tokenizer, texts: List[str],
                 max_samples: int = 20, batch_size: int = 8, output_path: Optional[str] = None) -> Dict[str, Any]:
        model.eval()

        all_k_scores = []

        for _ in tqdm(range(max_samples), desc="Evaluating k-order induction heads", leave=False):
            input_ids = self.create_induction_sequence(tokenizer)

            k_scores = self.get_k_order_scores(model, tokenizer, input_ids)
            all_k_scores.append(k_scores)

        scores_matrix = np.array(all_k_scores)

        results = {
            'n_samples': max_samples,
            'seq_len': self.seq_len,
            'max_k': self.max_k,
        }

        for k in range(1, self.max_k + 1):
            k_scores = scores_matrix[:, k-1]
            results[f'accuracy_k{k}'] = float(np.mean(k_scores))
            results[f'std_k{k}'] = float(np.std(k_scores))

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            return {'status': 'success', 'output_path': output_path, 'n_samples': max_samples}

        return results
