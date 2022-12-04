import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layer_dim_list, output_dim, logit_output=True):
        super().__init__()
        self.logit_output = logit_output
        self.hidden_layer_list = nn.ModuleList()
        curr_input_dim = input_dim
        for layer_dim in hidden_layer_dim_list:
            curr_output_dim = layer_dim 
            self.hidden_layer_list.append(nn.Linear(curr_input_dim, curr_output_dim))
            curr_input_dim = curr_output_dim
        self.last_layer = nn.Linear(curr_input_dim, output_dim)
        
    def forward(self, x):
        curr_input = x
        for layer in self.hidden_layer_list:
            curr_output = F.relu(layer(curr_input)) 
            curr_input = curr_output
        if not self.logit_output:
            return torch.sigmoid(self.last_layer(curr_output))
        return self.last_layer(curr_input)