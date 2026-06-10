"""
GaussianImageDistribution - File I/O Utilities

This module provides utility functions for common file operations.
"""

import json
from pathlib import Path
from typing import Any, Dict

import torch


def load_json(file_path: Path) -> Dict[str, Any]:
    """
    Load JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data as dictionary
    """
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Path, indent: int = 4) -> None:
    """
    Save data to JSON file.

    Args:
        data: Dictionary to save
        file_path: Output file path
        indent: JSON indentation (default: 4)
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
