import torch
from torch import nn

"""
This code is the implementation of the dynamics model for the F1TENTH Gym environment.
The model takes input of the states and actions of the car in the past T time steps,
    and outputs the next state of the car.
"""

class MLPdynamics(torch.nn.Module):
    def __init__(self, 
                 state_dim=5, 
                 act_dim=2,
                 history_len=10,
                 hidden_dim=256, 
                 ):
        super(MLPdynamics, self).__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim

        input_dim = (state_dim + act_dim)* history_len
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, state, action):
        """
        Input:
            x: [Batch_size, history_len, state_dim + act_dim]
        Output:
            next_state: [Batch_size, state_dim] 
        """
        x = torch.cat([state, action], dim=-1)
        x = x.flatten(start_dim=1)
        return self.net(x)

    def parameter_num(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class Transformerdynamics(torch.nn.Module):
    def __init__(self, 
                 state_dim=5, 
                 act_dim=2,
                 history_len=10,
                 hidden_dim=64, 
                 num_layers=4
                 ):
        super(Transformerdynamics, self).__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim

        input_dim = (state_dim + act_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, 
                                                   nhead=4,
                                                   dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Sequential(
            nn.flatten(),
            nn.Linear(hidden_dim * history_len, state_dim)
        )

    def forward(self, x):
        """
        Input:
            x: [Batch_size, history_len, state_dim + act_dim]
        Output:
            next_state: [Batch_size, state_dim] 
        """
        embedded = self.transformer(x)
        return self.output_layer(embedded)

    def parameter_num(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)