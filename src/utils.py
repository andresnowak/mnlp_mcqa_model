import itertools
import logging
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union
import yaml

import pandas as pd
import torch
import transformers
from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizer

def load_config(config_path):
    """Loads the YAML config file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit(1)