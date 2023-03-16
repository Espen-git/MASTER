# network
import torch
import torch.nn as nn

torch.manual_seed(0)
    
class FFNeuralNetwork(nn.Module):
    def __init__(self):
        super(FFNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16*16, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,16*16),
        )        
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits