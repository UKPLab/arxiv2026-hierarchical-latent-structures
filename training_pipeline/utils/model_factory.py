import logging

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
)

from training_pipeline.configs import ModelConfig


class SuppressLossTypeWarning(logging.Filter):
    def filter(self, record):
        return (
            "loss_type=" not in record.getMessage()
            or "was set in the config but it is unrecognised" not in record.getMessage()
        )


logging.getLogger("transformers.modeling_utils").addFilter(SuppressLossTypeWarning())


class ModelFactory:
    @staticmethod
    def create_model(vocab_size, config: ModelConfig):
        """Create a model based on the architecture specified in config."""
        if config.architecture == "gpt2":
            return ModelFactory.create_gpt2_model(vocab_size, config)
        elif config.architecture == "llama":
            return ModelFactory.create_llama_model(vocab_size, config)
        else:
            raise ValueError(f"Unknown architecture: {config.architecture}")
    
    @staticmethod
    def create_gpt2_model(vocab_size, config: ModelConfig) -> GPT2LMHeadModel:
        gpt2_config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=config.context_window,
            n_ctx=config.context_window,
            n_embd=config.model_dim,
            n_layer=config.num_layers,
            n_head=config.num_heads,
        )
        return GPT2LMHeadModel(gpt2_config)
    
    @staticmethod
    def create_llama_model(vocab_size, config: ModelConfig) -> LlamaForCausalLM:
        # Calculate intermediate_size based on common Llama ratio (4x hidden_size)
        intermediate_size = config.model_dim * 4
        
        llama_config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=config.model_dim,
            intermediate_size=intermediate_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            max_position_embeddings=config.context_window,
        )
        return LlamaForCausalLM(llama_config)
