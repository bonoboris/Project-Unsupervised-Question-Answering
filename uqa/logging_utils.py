"""Logging related helpers."""

import logging

FORMATTER = logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s")
CONSOLE_FORMATER = logging.Formatter("%(message)s")


def init_root_logger(
    quiet=False, no_log=False, log_file="uqa.log", verbosity=logging.INFO, file_verbosity=logging.DEBUG
):
    """Initialize the root logger to log to the console and/or to a file.

    Parameters
    ----------
    quiet: bool, default=False
        If ``True`` does not log to the console.
    no_log: bool, default=False
        If ``True`` does not log to a file
    log_file: str, default="uqa.log"
        Log file path
    verbosity: int, default=logging.INFO
        Verbosity for console handler
    file_verbosity: int, default=logging.DEBUG
        Verbosity for file handler
    """
    handlers = []
    if not no_log:
        file_handler = logging.FileHandler(log_file, "a", "utf8")
        file_handler.setLevel(file_verbosity)
        file_handler.setFormatter(FORMATTER)
        handlers.append(file_handler)

    if not quiet:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(verbosity)
        stream_handler.setFormatter(CONSOLE_FORMATER)
        handlers.append(stream_handler)

    logging.basicConfig(level=logging.DEBUG, handlers=handlers)

    logging.debug("Logger initialized")
