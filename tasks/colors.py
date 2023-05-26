from enum import Enum

ENDC = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"


class Color(Enum):
    """Color class."""
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    ERROR = "\033[91m"


def colorize(message, color=Color.OKGREEN, underline=False, bold=False):
    """Colorize a message."""
    msg = color.value
    if underline:
        msg += UNDERLINE
    if bold:
        msg += BOLD
    msg += message
    msg += ENDC
    return msg
