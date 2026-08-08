import logging
import os
from datetime import datetime

import config


def setup_logger(name="thermal_eval"):
    logs_folder = os.path.join(config.OUTPUT_FOLDER, "Logs")
    os.makedirs(logs_folder, exist_ok=True)
    log_path = os.path.join(logs_folder, f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    log.propagate = False

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(console_handler)

    return log, log_path
