# Continous Control

## Introduction
In this project, I trained multiple agents (20 in total) to touch a moving target. Each agent, control a double-jointed arm. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of each agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

To solve the environment the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 


## Download the main files
Download the repo and unzip it:
https://github.com/AI2Rgross/DRL/p2_continous_control

Download the unity environment:
**_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
 

Place the zip file in the p2_continous_control folder and unzip the file.


## Requirement
As I am working under ubuntu 16.04, I will give you a few tips to make it run smoothly. But, I recommand you to follow the instruction on the Udacity DRL github: https://github.com/udacity/deep-reinforcement-learning/blob/master/p2_continous_control/README.md. Especially, if you work under a different OS.

You can also have a look at:
https://github.com/udacity/deep-reinforcement-learning/blob/master/README.md
 

## Tips to prepare the environement
- Python 3
Check if you have a one of the following python versionn: 3.5.x or 3.6.x
If not, it s time to install it using the "sudo apt-get install" command.

- pip 3
You will also need to install pip for python 3. 
sudo apt-get install python3-pip

- Pytorch:
check this link https://pytorch.org and get the command to install pytorch

- Ml agents for unity
First dowmload Ml agent ripo on https://github.com/Unity-Technologies/ml-agents.git

- Install ml agent:
pip3 install .

if there is any trouble have a look at:
https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md

- Install unity
pip3 install unity

- ipython
In order to run the notebook with python 3 also install ipyton.
sudo pip3 install ipython[all]


## Requirements for the Continous control project
Go inside the Continous control repo, and open a terminal in order to install the requirements:
pip3 install -r requirements.txt
Then run the next command to open a notebook:
ipython3 notebook

click on Continous_control.ipynb to open it.
Try to run the first cell where the libraries are imported to check if UnityEnvironment is well installed.

Then, in the next cell correct the path of the Reacher.x86_64 and try running it. Be careful to use the multi-agent environment.

env = UnityEnvironment(file_name="the/path/here/Reacher.x86_64")

Now everything is ready to either train your own solution or run the pre-computed solution.


## My files
Main files of the repository:

    - The main part of the code: point for starting the environment, train the agent or test a solution.
        Continous_control.ipynb

    - the Agent class with DDPG, and othes basic functions to interact with the environment.
        agent.py

    - The Pytorch neural networks used to approximate the actor-critic functions used by each agent.
        model.py

    - the weights of the pytorch model for DDPG of the solved environment for the actor and critic.
        checkpoint_actor.pth  (dqn)
        checkpoint_critic.pth (ddqn)
        
    - Installation notes and tips, brief description of the project
        README.md

    - Udacity original readme for instalation of the environment.
        Udacity_README.md

    - My notes about DDPG
        Report.pdf

The code I developped to solve the Reacher environement is based on the p2_continous_control and ddpg_bipedal exercices from the UDACITY DRL github available here: https://github.com/udacity/deep-reinforcement-learning.
