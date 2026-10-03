"""Apps module for QueryPilot."""

import os

from django.apps import AppConfig

os.environ.setdefault(
    "HF_HUB_DISABLE_PROGRESS_BARS",
    "1",
)

os.environ.setdefault(
    "TRANSFORMERS_VERBOSITY",
    "error",
)

os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)


class DocumentsConfig(AppConfig):
    name = "documents"
