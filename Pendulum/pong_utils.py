from parallelEnv import parallelEnv 
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib import animation
from IPython.display import display
import random as rand



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# collect trajectories for a parallelized parallelEnv object
def collect_trajectories(envs, policy, tmax=200, nrand=5):
    
    # number of parallel instances
    n=len(envs.ps)

    #initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    prob_list=[]
    action_list=[]

    envs.reset()
    
    #start all parallel agents
    envs.step([1]*n)
    
    # perform nrand random steps
    for _ in range(nrand):
        next_states, rewards, dones, _ = envs.step(np.random.uniform([-2, 2],n))
    
    for t in range(tmax):

        # probs will only be used as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        distributions, values = policy(next_states).squeeze().cpu().detach().numpy()
        actions = [distribution.sample() for distribution in distributions]
        log_probs = [distribution.log_prob(action) for distribution, action in zip(distributions,actions)]
  
        
        # advance the game (0=no action)
        # we take one action and skip game forward
        next_states, rewards, dones, _ = envs.step(actions)
   

        # store the result
        state_list.append(next_states)
        reward_list.append(rewards)
        prob_list.append(log_probs)
        action_list.append(actions)
        
        # stop if any of the trajectories is done
        # we want all the lists to be retangular
        if is_done.any():
            break


    # return pi_theta, states, actions, rewards, probability
    return prob_list, state_list,action_list, reward_list


    
# clipped surrogate function
# similar as -policy_loss for REINFORCE, but for PPO
def clipped_surrogate(policy, old_log_probs, states, actions, rewards,
                      discount=0.995,
                      epsilon=0.1, beta=0.01):

    discount = discount**np.arange(len(rewards))
    rewards = np.asarray(rewards)*discount[:,np.newaxis]
    
    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    
    # normalize the reward: (x-mean)/std 
    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
    
    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)
    # convert states to policy (or probability)
    # evaluate the new prob: this is what we have:(s,a,r,s') for old prob.     

    distributions, values = policy(states).squeeze().cpu().detach().numpy()
    actions = [distribution.sample() for distribution in distributions]
    new_log_probs = [distribution.log_prob(action) for distribution, action in zip(distributions,actions)]
  
  
    # ratio for clipping
    ratio = ne(w_log_probs - old_log_probs).exp()

    # clipped function
    clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    clipped_surrogate = torch.min(ratio*rewards, clip*rewards)

    # include a regularization term
    # this steers new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs.exp()*torch.log(old_probs.exp()+1.e-10)+ \
        (1.0-new_probs.exp())*torch.log(1.0-old_probs.exp()+1.e-10))

    
    # this returns an average of all the entries of the tensor
    # effective computing L_sur^clip / T
    # averaged over time-step and number of trajectories
    # this is desirable because we have normalized our rewards
    return torch.mean(clipped_surrogate + beta*entropy)

import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, state_size, action_size, fc1_units, std=0.0):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2v = nn.Linear(fc1_units, 1)
        self.fc2a = nn.Linear(fc1_units, action_size)
        self.reset_parameters()
        self.log_std = nn.Parameter(torch.ones(1, action_size) * std)   
   

    def reset_parameters(self):
        self.fc1.weight.data.normal_(mean=0., std=0.1)
        self.fc2a.weight.data.normal_(mean=0., std=0.1)
        self.fc2v.weight.data.normal_(mean=0., std=0.1)
       
    def forward(self, state):
 
       value = F.relu(self.fc1(state))
       value = self.fc2v(value)

       mu = F.relu(self.fc1(state))
       mu = self.fc2a(mu)


       std   = self.log_std.exp().expand_as(mu)
       distribution  = Normal(mu, std)
       return distribution, value
    
