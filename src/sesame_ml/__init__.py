"""Sesame Robot simulation and learning stack."""

from importlib.metadata import PackageNotFoundError, version

from sesame_ml.constants import JOINT_NAMES, STAND_ANGLES_RAD

__all__ = ["JOINT_NAMES", "STAND_ANGLES_RAD"]
try:
    __version__ = version("sesame-ml")
except PackageNotFoundError:  # Source tree imported without installation metadata.
    __version__ = "0+unknown"
