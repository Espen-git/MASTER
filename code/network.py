# network
import torch
import torch.nn as nn

torch.manual_seed(0)
    
class FFNeuralNetwork(nn.Module):
    def __init__(self, config):
        super(FFNeuralNetwork, self).__init__()

        is_complex = config['is_complex']
        use_upper_triangular = config['use_upper_triangular']

        input_scale = 1 if is_complex else 2
        numb_input_corr = 16*16 if not use_upper_triangular else 136
        numb_input_features = numb_input_corr*input_scale

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(numb_input_features, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,16*input_scale),
        )        
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits