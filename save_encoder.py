import torch

from models import load_model

def main():
    encoder = load_model(name='Encoder')
    
    torch.save(encoder.state_dict(), './saved/pre-trained_encoder.pth')

if __name__ == '__main__':
    main()