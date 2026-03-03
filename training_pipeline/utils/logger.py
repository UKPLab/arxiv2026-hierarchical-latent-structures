import logging

from rich.logging import RichHandler


def create_logger(name: str) -> logging.Logger:
    """Create a Rich-configured logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    rich_handler = RichHandler(rich_tracebacks=True, show_time=False)
    formatter = logging.Formatter(
        "%(asctime)s • %(name)s • [%(levelname)s] • %(message)s",
        "%d.%m.%Y %H:%M:%S",
    )
    rich_handler.setFormatter(formatter)
    logger.addHandler(rich_handler)

    return logger
