# stdlib
import sys

# wfaudit relative
from . import logger  # noqa: F401
from .process_pcaps import process_pcaps  # noqa: F401

logger.add(sink=sys.stderr, level="DEBUG")
