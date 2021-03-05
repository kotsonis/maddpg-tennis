# maddpg-tennis
A Multi Agent Deep Deterministic Actor-Critic reinforcement learning solution in python for the Unity ML (Udacity) [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment

## Introduction

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Getting Started
To set up your python environment and run the code in this repository, follow the instructions below.
### setup Conda Python environment

Create (and activate) a new environment with Python 3.6.

- __Linux__ or __Mac__: 
```shell
	conda create --name ddpg-rl python=3.6
	source activate ddpg-rl
```
- __Windows__: 
```bash
	conda create --name ddpg-rl python=3.6 
	activate ddpg-rl
```
### Download repository
 Clone the repository and install dependencies

```shell
	git clone https://github.com/kotsonis/ddpg-reacher.git
	cd ddpg-reacher
	pip install -r requirements.txt
```

### Install Reacher environment
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
   - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
   - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Place the file in the `ddpg-reacher` folder, and unzip (or decompress) the file.

3. edit [hyperparams.py](hyperparams.py) to and set the `banana_location` entry to point to the right location. Example :
```python 
std_learn_params = {
        # Unity Environment parameters
        "banana_location": "./Banana_Windows_x86_64/Banana.exe",
```
## Instructions
### Training
To train an agent, [train.py](train.py) reads the hyperparameters from [hyperparams.py](hyperparams.py) and accepts command line options to modify parameters and/or set saving options.You can get the CLI options by running
```bash
python train.py -h
```
### Playing with a trained model
you can see the agent playing with the trained model as follows:
```bash
python play.py
```
You can also specify the number of episodes you want the agent to play, as well as the non-default trained model as follows:
```bash
python play.py --episodes 20 --model v2_model.pt
```

## Implementation and results
You can read about the implementation details and the results obtained in [Report.md](Report.md)