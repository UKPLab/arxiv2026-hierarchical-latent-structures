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