"""Logging setup for ModelGate."""

import logging

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_HANDLER_MARKER = "_modelgate_console_handler"


def configure_logging(
    level: int | str = logging.INFO,
    logger_name: str = "modelgate",
) -> logging.Logger:
    """Configure and return a package logger without duplicating its handler."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    handler = next(
        (
            existing_handler
            for existing_handler in logger.handlers
            if getattr(existing_handler, _HANDLER_MARKER, False)
        ),
        None,
    )

    if handler is None:
        handler = logging.StreamHandler()
        setattr(handler, _HANDLER_MARKER, True)
        logger.addHandler(handler)

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    return logger
