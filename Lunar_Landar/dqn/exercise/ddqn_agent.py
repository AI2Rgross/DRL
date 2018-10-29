import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3                # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ddqn_Agent():
    """Interacts with and learns from the environment."""

    
    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.wQ =0
        self.wQ1=0
        self.wQ2=0
        # Q-Network
        self.qnetwork_Qa  = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_Qb  = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer_Qa = optim.Adam(self.qnetwork_Qa.parameters(), lr=LR)
        self.optimizer_Qb = optim.Adam(self.qnetwork_Qb.parameters(), lr=LR)

        # Replay memory
        self.memory = ddqn_ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def print_result():
        somme=self.wQ1+self.wQ2+0.0000001
        print("qQ1=",self.wQ1/somme," qQ2=",self.wQ2/somme)
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

                
    def act_a(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    
        self.qnetwork_Qa.eval()
        with torch.no_grad():
            action_values = self.qnetwork_Qa(state)
        self.qnetwork_Qa.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
 

    def act_b(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    
        self.qnetwork_Qb.eval()
        with torch.no_grad():
            action_values = self.qnetwork_Qb(state)
        self.qnetwork_Qb.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        

    def act(self,state,eps=0.):
        self.wQ=np.random.choice([0, 1])
        if(self.wQ):
            return self.act_a(state, eps)
        else:
            return self.act_b(state, eps)


    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        #
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        if self.wQ:
            # wQ take either 0 or 1 based on uniform random function.
            yj=self.qnetwork_Qb.forward(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets=rewards+gamma*yj*(1.0-dones)
 
            # Get expected Q values from local model
            Q_expected = self.qnetwork_Qa.forward(states).gather(1, actions)
            # Compute loss: Mean Square Error by element
            loss = F.mse_loss(Q_expected, Q_targets)
  
            # Minimize the loss
            self.optimizer_Qa.zero_grad()
            loss.backward()
            self.optimizer_Qa.step()
            # ------------------- update target network ------------------- #
            self.soft_update(self.qnetwork_Qa, self.qnetwork_Qb, TAU)
        else:
            
            yj=self.qnetwork_Qa.forward(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets=rewards+gamma*yj*(1.0-dones)
 
            # Get expected Q values from local model
            Q_expected = self.qnetwork_Qb.forward(states).gather(1, actions)
            # Compute loss: Mean Square Error by element
            loss = F.mse_loss(Q_expected, Q_targets)
  
            # Minimize the loss
            self.optimizer_Qb.zero_grad()
            loss.backward()
            self.optimizer_Qb.step()
            # ------------------- update target network ------------------- #
            self.soft_update(self.qnetwork_Qb, self.qnetwork_Qa, TAU)      
        
        

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        teta_target = ro*teta_local + (1 - ro)*teta_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ddqn_ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)