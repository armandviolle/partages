import logging
import sys


def setup_logger(level: int = logging.DEBUG):
    """
    Initializes a logger that sends INFO and DEBUG to stdout,
    and WARNING and higher levels to stderr.

    Parameters
    ----------
    name : str
        Name of the logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.propagate = False  # Avoid duplicated logs if root logger is configured

    # Clear existing handlers if any
    root_logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Handler for stdout (DEBUG and INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)
    stdout_handler.setFormatter(formatter)

    # Handler for stderr (WARNING and above)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)

    # Add handlers to logger
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)
