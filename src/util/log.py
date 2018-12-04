import json
from collections import OrderedDict

import sys

from models.base_gazenet import BaseGazeNet
import logging


def write_parameter_summaries(model: BaseGazeNet, path_out: str):
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
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger