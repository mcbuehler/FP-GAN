import json
import logging
import sys
from collections import OrderedDict

from models.base_gazenet import BaseGazeNet


def write_parameter_summaries(model: BaseGazeNet, path_out: str):
    """
    Write a json summary for the parameters in model.
    Args:
        model: a GazeNet model
        path_out: where to write the summary
    """
    data = OrderedDict([
        ('name', model.name),
        ('batch_size', model.batch_size),
        ('image_size', model.image_size),
        ('norm', model.norm),
        ('learning_rate', model.learning_rate),
        ('Optimiser', model.Optimiser.__name__),
        ('beta1', model.beta1),
        ('beta2', model.beta2)
    ])
    with open(path_out, 'a') as file:
        file.write(json.dumps(data, indent=4))


def get_logger():
    """
    Creates and returns a logger
    Returns: Logger
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
