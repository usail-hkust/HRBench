"""
Logging utilities for Hybrid Reasoning Benchmark.
"""
import logging
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Create a logger with file + console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(
            os.path.join(log_dir, f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def save_results_json(
    results: list,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Save experiment results to JSON with metadata."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(results),
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_results_json(path: str) -> dict:
    """Load experiment results from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
