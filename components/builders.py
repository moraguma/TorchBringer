import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


LAYER_STRING_TO_CLASS = {
    "linear": nn.Linear,
    "relu": nn.ReLU
}
OPTIMIZER_STRING_TO_CLASS = {
    "adamw": optim.AdamW
}


def build_sequential(config) -> nn.Sequential:
    """
    Receives a config array such that each of its elements is a dictionary that specifies a layer of the NN. While
    each layer has the attribute "type", other required attributes depend on the type of layer
    """
    layers = []

    for layer_spec in config:
        layers.append(LAYER_STRING_TO_CLASS[layer_spec["type"]](**build_kwargs(layer_spec)))
    
    return nn.Sequential(*layers)


def build_optimizer(config) -> optim.Optimizer:
    """
    Receives a config dictionary that specifies the type and parameters of optimizer. The specific parameters
    required depend on the optimizer type
    """
    return OPTIMIZER_STRING_TO_CLASS[config["type"]](**build_kwargs(config))


def build_kwargs(config):
    """
    Should look for any necessary substitutions to specific objects or formats before building kwargs
    """
    excludes = "type"
    kwargs = {}
    for k in config:
        if not k in excludes:
            kwargs[k] = config[k]
    return kwargs