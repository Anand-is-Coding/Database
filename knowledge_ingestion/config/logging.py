"""Centralized logging configuration.

Every module in this project should obtain its logger via `get_logger(__name__)`
rather than instantiating its own handlers, so log formatting and verbosity
remain consistent across the pipeline.
"""

import logging

from rich.logging import RichHandler

from config.settings import settings

_CONFIGURED = False


def _configure_root_logger() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_path=False,
                # markup=False is deliberate: log messages routinely embed
                # dynamic, unsanitized content (PDF filenames, chapter/
                # section headings, student questions). With markup=True,
                # Rich parses that content as markup and silently DROPS
                # anything that looks like a tag - e.g. a filename literally
                # containing "[...]" loses that substring from the log
                # entirely, not just its formatting. Disabling markup makes
                # dynamic content render verbatim; only literal Rich markup
                # written by this codebase itself would need markup=True,
                # and none of the logging call sites use it.
                markup=False,
            )
        ],
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger backed by the shared Rich handler."""
    _configure_root_logger()
    return logging.getLogger(name)
