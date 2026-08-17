"""
Unified Logging System

This module provides a centralized logging system that outputs to both
console and log files with detailed formatting including timestamp,
level, module name, file name, line number, and message.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import traceback


# Store loggers to prevent duplicate handlers
_loggers = {}


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    also_console: bool = True,
    mode: str = "a"
) -> logging.Logger:
    """
    Set up a logger with console and/or file handlers.

    Args:
        name: Logger name (typically __name__ from calling module)
        log_file: Path to log file (optional)
        level: Logging level (default: INFO)
        also_console: Whether to also output to console (default: True)
        mode: File write mode - 'a' for append, 'w' for overwrite

    Returns:
        Configured logger instance
    """
    # Return existing logger if already created
    if name in _loggers:
        return _loggers[name]

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent duplicate logs from parent loggers

    # Clear any existing handlers
    logger.handlers.clear()

    # Detailed format: timestamp, level, module, file, line, message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Add console handler if requested
    if also_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if log file is specified
    if log_file is not None:
        # Ensure parent directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode=mode, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Store logger to prevent duplicates
    _loggers[name] = logger

    return logger


def get_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    also_console: bool = True
) -> logging.Logger:
    """
    Get an existing logger or create a new one.

    This is a convenience function that wraps setup_logger but uses
    append mode by default to avoid overwriting existing logs.

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level (default: INFO)
        also_console: Whether to also output to console (default: True)

    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]

    return setup_logger(name, log_file, level, also_console, mode="a")


def log_exception(logger: logging.Logger, exc: Exception, context: str = "") -> None:
    """
    Log an exception with full traceback.

    Args:
        logger: Logger instance
        exc: Exception to log
        context: Optional context message
    """
    if context:
        logger.error(f"{context}: {type(exc).__name__}: {exc}")
    else:
        logger.error(f"{type(exc).__name__}: {exc}")

    logger.error("Traceback:")
    logger.error(traceback.format_exc())


if __name__ == "__main__":
    from pathlib import Path
    test_log = Path("test.log")

    logger = setup_logger(
        name="test_logger",
        log_file=test_log,
        level=logging.DEBUG
    )

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    try:
        1 / 0
    except Exception as e:
        log_exception(logger, e, "Test exception")

    print(f"Test logs written to: {test_log}")
    print("Check both console output and the file.")
