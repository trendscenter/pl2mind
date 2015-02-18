"""
Module for general logger.
"""

import logging


logging.basicConfig(format="[%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)