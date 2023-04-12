"""Identification of the current operating system."""

import platform
from enum import Enum

COV_SCREEN_NAME = "coverage"
DOC_SCREEN_NAME = "sphinx-docs"


class OperatingSystem(Enum):
    """Initializes the operating system."""
    WINDOWS = 'Windows'
    LINUX = 'Linux'
    MAC = 'Darwin'


def get_current_system():
    """Returns the current operating system."""
    system = platform.system()  # pylint: disable=W0621

    if system == 'Linux':
        return OperatingSystem.LINUX
    if system == 'Windows':
        return OperatingSystem.WINDOWS
    if system == 'Darwin':
        return OperatingSystem.MAC

    raise ValueError(f'Invalid operating system: {system}')


system = get_current_system()

if system in [OperatingSystem.LINUX, OperatingSystem.MAC]:
    COV_DOC_BUILD_DIR = "_build/htmlcov"
    DOCS_BUILD_DIR = "docs"
elif system == OperatingSystem.WINDOWS:
    COV_DOC_BUILD_DIR = r"_build\htmlcov"
    DOCS_BUILD_DIR = r"_build\docs"
else:
    raise ValueError(f'System {system} is not supported')
