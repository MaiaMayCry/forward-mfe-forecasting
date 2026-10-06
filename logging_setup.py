import logging
import sys
from pathlib import Path

def setup_logging(log_file: str = "run.log", level: int = logging.INFO) -> None:
    """Configure root logger with file + console handlers."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-6s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # writes logs to the file with the formatter
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)

    # send logs to console as well
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # get parent logger, set level and clears existing handlers
    # before adding both file and console
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Reduce noise from sklearn/yfinance
    #logging.getLogger("sklearn").setLevel(logging.WARNING)
    #logging.getLogger("yfinance").setLevel(logging.WARNING)