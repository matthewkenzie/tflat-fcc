#!/usr/bin/env python3

from yaml import safe_load, dump


def load_config(config_path):
    """Load a YAML config file and return it as a dict."""
    with open(config_path) as f:
        return safe_load(f)

def save_config(config_path, config):
    with open(config_path, "w") as f:
        dump(config, f)

