import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 4        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
UNIFORM_PROB = 1.
UNIFORM_RATE = 0.99999
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class dqn_replay_Agent():
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

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = dqn_replay_ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
       
    
#    def step(self, state, action, reward, next_state, done,uniform=True):
        # Save experience in replay memory
     #   if(len(self.memory)<BUFFER_SIZE):
     #       self.memory.add( state, action, reward, next_state, done)
        # Modification in case buffer is too small
     #   else:
     #       self.memory.rem()
     #         self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
     #   self.t_step = (self.t_step + 1) % UPDATE_EVERY

     #   if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
     #       if len(self.memory) >= BATCH_SIZE:
     #           experiences,ind,weights = self.memory.sample(uniform=True)
     #           self.learn(experiences, GAMMA,ind,weights)

    def step(self, state, action, reward, next_state, done,uniform=True):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences,ind,weight = self.memory.sample(uniform=True)
                self.learn(experiences, GAMMA,uniform,ind,weight)              
 

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
 


    def wmse_loss(self,Q_expected,Q_targets,weights):
        mean=0
        for i in range(len(weights)):
 #           mean+=weights[i]*((Q_expected[i]-Q_targets)**2)
 #       return mean/len(weights)
            Q_expected[i] = Q_expected[i] * weights[i]**0.5
            Q_targets[i]  = Q_targets[i]  * weights[i]**0.5
        return Q_expected,Q_targets
    
    
    def learn(self, experiences, gamma,uniform,ind, weights):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
             
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
  
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        #update priority
        if(not uniform):
            losses=(Q_expected-Q_targets)
            self.memory.update(losses,ind)
            Q_expected, Q_targets= self.wmse_loss(Q_expected,Q_targets,weights)

        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        return losses

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


class dqn_replay_ReplayBuffer:
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
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done","error"])
        self.seed = random.seed(seed)
        self.max_error=0.00001
        # uniform prob is the oppposite (1-Uni prob) ... just easier to do like that 
        self.uniform_rate = UNIFORM_RATE
        self.uniform_prob = UNIFORM_PROB  
        self.uniforme=0
        self.total= torch.zeros(1,dtype=torch.float)
        self.alpha=0.5
        self.beta=0.5
        
    def sample(self,uniform=True):
        """Randomly sample a batch of experiences from memory."""
        #Old version:      
        #experiences = random.sample(self.memory, k=self.batch_size)
 
        # Prioritize distribution
        ind,weights=self.distribution(uniform=True)
        experiences=[]
        
        for i in ind:           
            experiences.append(self.memory[i])
 
        states  = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones),ind,weights

   
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done,self.max_error)
        self.memory.appendleft(e)
        self.total= self.total+self.max_error

        
    def rem(self):
 #       ind = np.random.choice(np.arange(len(self.memory)-BATCH_SIZE), 1)
 #       ind2= len(self.memory)-1
 #       ind=ind[0]+BATCH_SIZE
        
 #       self.memory[ind]=self.memory[ind]._replace(state=self.memory[ind2].state)
 #       self.memory[ind]=self.memory[ind]._replace(action=self.memory[ind2].action)
 #       self.memory[ind]=self.memory[ind]._replace(reward=self.memory[ind2].reward)
 #       self.memory[ind]=self.memory[ind]._replace(next_state=self.memory[ind2].next_state)
 #       self.memory[ind]=self.memory[ind]._replace(done=self.memory[ind2].done)
 #       self.memory[ind]=self.memory[ind]._replace(error=self.memory[ind2].error)     
        self.memory.pop()

    def update(self,losses,ind):
        if(ind != []):     
            j=0 
            for i in ind:
                self.total=self.total-self.memory[i].error
                e=float((abs(losses[j])+1e-4)**self.alpha)   
                self.memory[i]=self.memory[i]._replace(error=e)
                self.total=self.total+e
                if(self.max_error < e):
                    self.max_error = e
                
                
   
    def distribution(self,uniform=True):
        ind=[]
        weights=[]
        N=len(self.memory)
        self.uniforme=uniform
        if(uniform):
            distrib=np.empty(N);
            distrib.fill(1./N)
            ind = np.random.choice(a=np.arange(N), size=self.batch_size,replace=False,p=distrib)
            weights=np.ones(self.batch_size)
        else:
            distrib=[float(i.error/self.total)  for i in self.memory] 
            total=0
            for i in distrib:
                total+=i
            distrib[0]=distrib[0]+(1.-total)
            ind = np.random.choice(a=np.arange(N), size=self.batch_size,replace=False,p=distrib)
            weights=[ (distrib[i]*N)**(-self.beta) for i in ind] 
            max_w=max(weights)
            weights=weights/max_w
        return ind,weights

