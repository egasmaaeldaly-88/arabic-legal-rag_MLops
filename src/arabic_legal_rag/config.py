from pathlib import Path
import yaml

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file with fallback paths."""
    path = Path(config_path)
    if not path.exists():
        alt_path = Path("../configs/config.yaml")
        if alt_path.exists():
            path = alt_path
            
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)