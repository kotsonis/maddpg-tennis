# maddpg-tennis
A Multi Agent Deep Deterministic Actor-Critic reinforcement learning solution in python for the Unity ML (Udacity) [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment

## Introduction

![Trained Agent](https://github.com/kotsonis/maddpg-tennis/blob/main/images/trained_agent_gif.gif)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

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
	git clone https://github.com/kotsonis/maddpg-tennis.git
	cd maddpg-tennis
	pip install -r requirements.txt
```

### Install Tennis environment
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Place the file in the `maddpg-reacher` folder, and unzip (or decompress) the file.
3. edit [tennis_env.cfg](tennis_env.cfg) to set the `env` entry to point to the right location. Example :
```python 
--env=./Tennis_Windows_x86_64/Tennis.exe
```

### Commandline arguments
maddpg-reacher uses the [Abseil](https://abseil.io/docs/python/quickstart.html) library for logging and argument parsing
You can get the CLI options by running
```bash
python tennis.py -h
```

### Training
To train the agents, [tennis.py](tennis.py) reads the hyperparameters from [training.cfg](training.cfg) and accepts command line options to modify parameters and/or set saving options.You can train the agents with standard parameters as follows
```bash
python tennis.py --flagfile=training.cfg
```
### Playing with a trained model
you can see the agents playing with a pre-trained model as follows:
```bash
python tennis.py --flagfile=play.cfg
```
You can also specify the number of episodes you want the agent to play, as well as the non-default trained model as follows:
```bash
python tennis.py --play --episodes 20 --load v2_model.pt
```

## Implementation and results
You can read about the implementation details and the results obtained in [Report.md](Report.md)
