import torch
from torch import nn

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.li = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.li(x)

class TwoBlocks(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = Block()
        self.block2 = Block()
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
    
        return x

class BlockNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.two_blocks_1 = TwoBlocks()
        self.two_blocks_2 = TwoBlocks()
    
    def forward(self, x):
        x = self.two_blocks_1(x)
        x = self.two_blocks_2(x)
        
        return x

if __name__ == '__main__':
    model = BlockNet()
    
    for param in model.two_blocks_1.parameters():
        param.requires_grad = False
    
    # Check freeze
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")
        
    '''
    [Output]
    two_blocks_1.block1.li.weight: False
    two_blocks_1.block1.li.bias: False
    two_blocks_1.block2.li.weight: False
    two_blocks_1.block2.li.bias: False
    two_blocks_2.block1.li.weight: True
    two_blocks_2.block1.li.bias: True
    two_blocks_2.block2.li.weight: True
    two_blocks_2.block2.li.bias: True
    '''