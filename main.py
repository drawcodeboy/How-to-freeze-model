import torch
from torch import nn
from torch.optim import Adam
from models import load_model

def main():
    # Assumption
    # Let's assume a scenario where the ConvNet consists of an encoder and an MLP. 
    # The encoder is pre-trained, its weights are loaded and frozen, and only the MLP is trained.
    
    # (1) Declare ConvNet
    model = load_model(name='ConvNet')
    
    # (2) Load Pre-trained Encoder
    ckpt = torch.load('saved/pre-trained_encoder.pth', weights_only=True)
    model.encoder.load_state_dict(ckpt)
    
    # (3) Freeze
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    # (Additional) Check freeze
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")
        
    # (4) Test
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters())
    
    print('============[Before Back-Propagation]============')
    for name, param in model.named_parameters():
        print(f"Name: {name}\n{param}")
    
    optimizer.zero_grad()
    
    x, target = torch.randn(2, 1, 10, 10), torch.tensor([[1000], [1000]], dtype=torch.float32) # (B, C, H, W), (B, D)
    
    output = model(x) # (B, D)
    
    loss = loss_fn(output, target)
    
    loss.backward()
    
    optimizer.step()
    
    print('============[After Back-Propagation]============')
    for name, param in model.named_parameters():
        print(f"Name: {name}\n{param}")
        
    '''
    [Output]
    encoder.conv1.weight: False
    mlp.li.weight: True
    ============[Before Back-Propagation]============
    Name: encoder.conv1.weight
    Parameter containing:
    tensor([[[[ 0.1819, -0.2629, -0.2961],
            [-0.0302, -0.3108,  0.2362],
            [ 0.0499, -0.0626, -0.1266]]],


            [[[-0.1539, -0.1248, -0.3083],
            [-0.1207, -0.1153, -0.2869],
            [-0.1683, -0.0942, -0.2953]]]])
    Name: mlp.li.weight
    Parameter containing:
    tensor([[-0.1500,  0.4760]], requires_grad=True)
    ============[After Back-Propagation]============
    Name: encoder.conv1.weight
    Parameter containing:
    tensor([[[[ 0.1819, -0.2629, -0.2961],
            [-0.0302, -0.3108,  0.2362],
            [ 0.0499, -0.0626, -0.1266]]],


            [[[-0.1539, -0.1248, -0.3083],
            [-0.1207, -0.1153, -0.2869],
            [-0.1683, -0.0942, -0.2953]]]])
    Name: mlp.li.weight
    Parameter containing:
    tensor([[-0.1490,  0.4770]], requires_grad=True)
    '''
        

if __name__ == '__main__':
    main()