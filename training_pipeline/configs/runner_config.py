from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .model_config import ModelConfig


class RunnerConfig(BaseModel):
    run_id: str
    run_index: int = 0
    model_conf: ModelConfig
    run_path: Path
    dataset_id: Optional[str] = None
    status: str = "initialized"
    language_type: str = "ngram"
    language_config: Dict[str, Any] = {}
    language_metrics: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    dataset_created_at: Optional[str] = None
    training_completed_at: Optional[str] = None
    token_limit: Optional[int] = Field(default=None, description="Optional limit on total training tokens.")

    class Config:
        arbitrary_types_allowed = True

    @property
    def config_path(self) -> Path:
        return self.run_path.parent / "overview.json"

    @classmethod
    def from_dict(cls, data: Dict) -> "RunnerConfig":
        data = self.model_dump(exclude_none=True)
        data["run_path"] = str(self.run_path)
        return data

    def save_config(self):
        pass
