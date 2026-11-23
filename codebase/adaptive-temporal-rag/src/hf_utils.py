"""Hugging Face load helpers — prefer local cache to avoid long offline retries."""

from __future__ import annotations

import logging
import os
import warnings


def prefer_local_hf_cache() -> None:
    """Reduce hub traffic and noisy load reports; still allow online if cache miss."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Shorter retry budget when DNS is broken (default is very chatty/slow)
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "2")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "5")
    # Hide DistilBERT "UNEXPECTED" vocab-head reports (harmless for encoder-only use)
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=".*UNEXPECTED.*",
    )


def from_pretrained_local_first(loader, name_or_path: str, **kwargs):
    """Try local_files_only first; fall back to normal download."""
    prefer_local_hf_cache()
    try:
        return loader(name_or_path, local_files_only=True, **kwargs)
    except Exception:
        return loader(name_or_path, local_files_only=False, **kwargs)
