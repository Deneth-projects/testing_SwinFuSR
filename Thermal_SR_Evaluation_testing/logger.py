"""
logger.py
=========
Central logging utility for the Thermal-Guided Super-Resolution evaluation
project. Every module obtains its logger through `get_logger()` so that log
output is consistent across the whole project and no module prints directly
to the console on its own.
"""

import logging
from pathlib import Path

_LOGGER_NAME = "thermal_sr_eval"


def setup_logging(logs_folder: Path) -> logging.Logger:
    """Configure the project-wide logger and attach file + console handlers.

    A single logger writes to both the console and a timestamped
    'evaluation.log' file inside the given logs folder, preserving a
    complete, reproducible audit trail of the evaluation run.

    Parameters
    ----------
    logs_folder : Path
        Directory in which 'evaluation.log' will be created.

    Returns
    -------
    logging.Logger
        Configured logger instance. Call `get_logger()` from any other
        module to retrieve this same instance after setup.
    """
    logs_folder.mkdir(parents=True, exist_ok=True)
    log_path = logs_folder / "evaluation.log"

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers on re-run in same process

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_logger() -> logging.Logger:
    """Retrieve the shared project logger.

    Any module can call this after `setup_logging()` has been called once
    from `evaluation.py`, without needing to pass the logger instance
    around explicitly.

    Returns
    -------
    logging.Logger
        The shared 'thermal_sr_eval' logger. If `setup_logging()` has not
        been called yet, this returns a logger with no handlers attached
        (messages will be silently discarded until handlers exist).
    """
    return logging.getLogger(_LOGGER_NAME)
