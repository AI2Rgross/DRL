import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1   = nn.Linear(state_size,512)
        self.fc2   = nn.Linear(512,64)
        
        self.fc_A  = nn.Linear(64,action_size)
        self.fc_V  = nn.Linear(64,1)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
 
        A =  F.relu(self.fc_A(x))
        V =  F.relu(self.fc_V(x))
        x = V+A-A.mean(1).expand(A.size)
 
        return x 
 
