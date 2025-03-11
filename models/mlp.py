from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.li = nn.Linear(4, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.li(x))