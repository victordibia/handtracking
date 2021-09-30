import logging
from handtrack.version import VERSION as __version__
from handtrack.handmodel import HandModel

logging.getLogger("tensorflow").setLevel(logging.ERROR)

__all__ = ["HandModel"]