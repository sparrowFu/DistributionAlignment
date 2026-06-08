"""
GaussianImageDistribution - File I/O Utilities

This module provides utility functions for common file operations.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch


def load_json(file_path: Path) -> Dict[str, Any]:
    """
    Load JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data as dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
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


def load_parquet(file_path: Path) -> pd.DataFrame:
    """
    Load parquet file using pandas.

    Args:
        file_path: Path to parquet file

    Returns:
        DataFrame containing the parquet data

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")

    return pd.read_parquet(file_path)


def save_checkpoint(
    state: Dict[str, Any],
    file_path: Path,
    save_optim: bool = True
) -> None:
    """
    Save model checkpoint.

    Args:
        state: State dictionary containing model state, optimizer state, etc.
        file_path: Output checkpoint file path
        save_optim: Whether to save optimizer state (default: True)
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not save_optim and "optimizer_state_dict" in state:
        state = state.copy()
        del state["optimizer_state_dict"]

    torch.save(state, file_path)


def load_checkpoint(file_path: Path, device: str = "cpu") -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        file_path: Path to checkpoint file
        device: Device to load checkpoint onto ('cpu' or 'cuda')

    Returns:
        Checkpoint state dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {file_path}")

    state = torch.load(file_path, map_location=device, weights_only=False)
    return state


def save_pickle(obj: Any, file_path: Path) -> None:
    """
    Save object to pickle file.

    Args:
        obj: Object to pickle
        file_path: Output file path
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_path: Path) -> Any:
    """
    Load object from pickle file.

    Args:
        file_path: Path to pickle file

    Returns:
        Unpickled object

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {file_path}")

    with open(file_path, "rb") as f:
        return pickle.load(f)


def ensure_dir_exists(dir_path: Path) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        dir_path: Directory path to ensure
    """
    dir_path.mkdir(parents=True, exist_ok=True)


def get_file_size(file_path: Path) -> str:
    """
    Get human-readable file size.

    Args:
        file_path: Path to file

    Returns:
        File size as human-readable string (e.g., "1.23 MB")
    """
    if not file_path.exists():
        return "N/A"

    size_bytes = file_path.stat().st_size

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.2f} PB"


if __name__ == "__main__":
    # Test I/O utilities
    from pathlib import Path

    # Test JSON
    test_data = {"test": "data", "numbers": [1, 2, 3]}
    test_json = Path("test_io.json")
    save_json(test_data, test_json)
    loaded = load_json(test_json)
    print(f"JSON test: {loaded}")

    # Test file size
    print(f"File size: {get_file_size(test_json)}")

    # Cleanup
    test_json.unlink()
