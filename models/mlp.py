import torch
from torch import nn
from einops import rearrange

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.li = nn.Linear(2, 1, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = rearrange(x, 'b c w h -> b (w h) c')
        x = torch.mean(x, dim=1)
        
        return self.relu(self.li(x))