import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pendulum
import yaml
from pydantic import BaseModel, Field

from .runner_config import RunnerConfig


class ExperimentConfig(BaseModel):
    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    experiment_name: str
    num_experiments_per_grammar: int = Field(default=1)
    param_ranges: Dict[str, List[Any]] = Field(default_factory=dict)
    evaluators: Union[List[str], None] = Field(default_factory=list)    
    evaluator_batch_sizes: Dict[str, int] = Field(default_factory=dict)
    languages: List[Dict[str, Any]] = Field(default_factory=list)
    output_dir: str = Field(default="experiments")
    external_experiment_path: Optional[str] = Field(default=None)
    runs: List[RunnerConfig] = Field(default_factory=list)
    generator_path: str = Field(
        default_factory=lambda: str(Path(__file__).parent.parent.parent / "generator" / "target" / "release" / "generator")
    )
    created_at: str = Field(default_factory=lambda: pendulum.now().isoformat())
    mode: str = Field(default="batched")

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        with open(path, "r") as f:
            config = yaml.safe_load(f) or {}

        config.setdefault("experiment_id", str(uuid.uuid4()))
        config.setdefault("experiment_name", "default")
        config.setdefault("evaluators", [])
        config.setdefault("languages", [])

        return cls(**config)

    def get_language_config(self, language_type: str) -> Optional[Dict[str, Any]]:
        """Get language configuration by type"""
        for lang in self.languages:
            if isinstance(lang, dict):
                # Handle YAML format where language type is the key
                if language_type in lang:
                    return lang[language_type]
        return None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")