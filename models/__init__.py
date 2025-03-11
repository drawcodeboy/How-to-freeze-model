from .encoder import Encoder
from .mlp import MLP
from .convnet import ConvNet

def load_model(name:str = 'Encoder'):
    if name == 'Encoder':
        return Encoder()
    elif name == 'MLP':
        return MLP()
    elif name == 'ConvNet':
        return ConvNet()