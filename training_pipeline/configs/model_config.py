from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    vocab_size: int = Field(..., description="Size of the vocabulary")
    context_window: int = Field(..., description="Context window length")
    
    architecture: str = Field(default="gpt2", description="Model architecture: 'gpt2' or 'llama'")

    model_dim: int = Field(..., description="Embedding dimension")
    num_layers: int = Field(..., description="Number of transformer layers")
    num_heads: int = Field(..., description="Number of attention heads per layer")

    @model_validator(mode="after")
    def check_dim_heads_compatibility(self) -> "ModelConfig":
        if self.num_heads and self.model_dim % self.num_heads != 0:
            raise ValueError(
                f"`model_dim` ({self.model_dim}) must be divisible by `num_heads` ({self.num_heads})"
            )
        return self

    @model_validator(mode="after")
    def check_architecture(self) -> "ModelConfig":
        valid_architectures = ["gpt2", "llama"]
        if self.architecture not in valid_architectures:
            raise ValueError(
                f"Invalid architecture '{self.architecture}'. Must be one of {valid_architectures}"
            )
        return self

    class Config:
        str_strip_whitespace = True
        validate_assignment = True
        arbitrary_types_allowed = True
