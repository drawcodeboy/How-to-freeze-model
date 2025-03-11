from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        
        self.bn1 = nn.BatchNorm2d()
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bn1(self.conv1(x)))