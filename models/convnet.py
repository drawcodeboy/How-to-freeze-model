from torch import nn

from .encoder import Encoder
from .mlp import MLP

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = Encoder()
        self.mlp = MLP()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp(x)
        
        return x