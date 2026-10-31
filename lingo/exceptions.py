"""Exception hierarchy for Lingo.

All errors derive from :class:`LingoError`. Each class also subclasses the built-in
exception raised by earlier releases (``ValueError``), so existing ``except`` clauses
keep working.
"""

from __future__ import annotations


class LingoError(Exception):
    """Base class for all Lingo errors."""


class UnsupportedTaskError(LingoError, ValueError):
    """The requested task is not one of ``Pipeline.SUPPORTED_TASKS``."""


class InvalidInputError(LingoError, ValueError):
    """Inputs are missing or malformed for the selected task."""


__all__ = ["InvalidInputError", "LingoError", "UnsupportedTaskError"]
