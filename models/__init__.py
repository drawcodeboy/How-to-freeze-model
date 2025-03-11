from .encoder import Encoder
from .mlp import MLP

def load_model(name:str = 'Encoder'):
    if name == 'Encoder':
        return Encoder()
    elif name == 'MLP':
        return MLP()