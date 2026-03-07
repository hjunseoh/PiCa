import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class PiCaConfig:
    """Configuration for PiCa (Parameter-Efficient Fine-tuning with Column Space Projection)."""
    target_modules: List[str] = field(default_factory=list)
    rank: int = 256
    base_model_name_or_path: Optional[str] = None

    def save(self, output_dir: str):
        """Save config to output_dir/pica_config.json."""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "pica_config.json"), "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, adapter_dir: str) -> "PiCaConfig":
        """Load config from adapter_dir/pica_config.json."""
        with open(os.path.join(adapter_dir, "pica_config.json"), "r") as f:
            data = json.load(f)
        return cls(**data)
