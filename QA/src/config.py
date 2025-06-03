# src/config.py
import yaml
from pathlib import Path

def load_config(config_path="configs/base.yaml", override_path=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if override_path:
        with open(override_path) as f:
            override = yaml.safe_load(f)
        # Merge override values
        config = merge_dicts(config, override)

    return config

def merge_dicts(base, override):
    for key, value in override.items():
        if isinstance(value, dict):
            base[key] = merge_dicts(base.get(key, {}), value)
        else:
            base[key] = value
    return base