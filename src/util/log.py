import json
from collections import OrderedDict

from models.base_gazenet import BaseGazeNet


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