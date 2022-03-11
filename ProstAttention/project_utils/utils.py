import logging
from itertools import chain


def flatten(list_of_list):
    """ This function flattens a list of list.

    Args:
        list_of_list: a list of list

    Returns:
        list, flattened
    """
    return list(chain(*list_of_list))


def init_logging():
    """ Initialize the logging handlers.
    """
    logger = logging.getLogger()
    logger.handlers = []  # Reset handlers
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
