import torch
import torch.nn as nn
    
class FFNeuralNetwork(nn.Module):
    def __init__(self, config):
        super(FFNeuralNetwork, self).__init__()

        is_complex = config['is_complex']
        use_upper_triangular = config['use_upper_triangular']

        input_scale = 1 if is_complex else 2
        numb_input_corr = 16*16 if not use_upper_triangular else 136
        numb_input_features = numb_input_corr*input_scale
        numb_output_features = 16*input_scale

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(numb_input_features, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,numb_output_features),
        )        
    
    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
class FFNeuralNetwork2(nn.Module):
    def __init__(self, config):
        super(FFNeuralNetwork2, self).__init__()

        is_complex = config['is_complex']
        use_upper_triangular = config['use_upper_triangular']

        input_scale = 1 if is_complex else 2
        numb_input_corr = 16*16 if not use_upper_triangular else 136
        numb_input_features = numb_input_corr*input_scale
        numb_output_features = 16*input_scale

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(numb_input_features, 128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,numb_output_features),
        )        
    
    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
class FFNeuralNetwork3(nn.Module):
    def __init__(self, config):
        super(FFNeuralNetwork3, self).__init__()

        is_complex = config['is_complex']
        use_upper_triangular = config['use_upper_triangular']

        input_scale = 1 if is_complex else 2
        numb_input_corr = 16*16 if not use_upper_triangular else 136
        numb_input_features = numb_input_corr*input_scale
        numb_output_features = 16*input_scale

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(numb_input_features, 128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,numb_output_features),
        )        
    
    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits