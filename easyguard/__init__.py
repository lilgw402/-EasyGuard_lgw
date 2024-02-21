#!/usr/bin/env python
import os

from .version import __version__

# set the easyguard cache directory
# for local cache
EASYGUARD_CACHE = os.path.join(f"{os.environ['HOME']}/.cache", "easyguard")
EASYGUARD_MODEL_CACHE = os.path.join(EASYGUARD_CACHE, "models")
EASYGUARD_CONFIG_CACHE = os.path.join(EASYGUARD_CACHE, "config")
REMOTE_PATH_SEP = "/"
os.environ["EASYGUARD_CACHE"] = EASYGUARD_CACHE
os.environ["EASYGUARD_HOME"] = os.path.dirname(os.path.dirname(__file__))

from . import models
from .core import AutoImageProcessor, AutoModel, AutoProcessor, AutoTokenizer
from .modelzoo.config import MODELZOO_CONFIG

# This line will be programatically read/write by setup.py.
# Leave them at the bottom of this file and don't touch them.
