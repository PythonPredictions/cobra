from .version import __version__
from cobra.utils import log_tutorial
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

log_tutorial()
