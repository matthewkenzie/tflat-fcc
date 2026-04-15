#!/usr/bin/env python3

from yaml import safe_load


def load_config(config_path):
    """Load a YAML config file and return it as a dict."""
    with open(config_path) as f:
        return safe_load(f)
