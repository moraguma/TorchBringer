import torch.nn as nn
import torch.nn.functional as F

def build_from(config):
    """
    Receives a config array such that each of its elements is a dictionary that specifies a layer of the NN. While
    each layer has the attribute "type", other required attributes depend on the type of layer
    """
    layers = []

    for layer_spec in config:
        match layer_spec["type"]:
            case "linear":
                layers.append(nn.Linear(layer_spec["in"], layer_spec["out"]))
            case "relu":
                layers.append(nn.ReLU())
    
    return nn.Sequential(*layers)