"""Installable OpenPI and GR00T vendor-side configuration templates."""

from importlib.resources import files


def template_directory():
    """Return the installed template resource tree for copy/integration workflows."""

    return files(__package__)


__all__ = ["template_directory"]
