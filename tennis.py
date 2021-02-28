import os
from absl import app
from unityagents import UnityEnvironment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque
from multiagent import MADDPG
# from agents.ppo import PPOAgent
from absl import logging
from absl import flags
config = flags.FLAGS

flags.DEFINE_string(name='env', default='./Tennis_Windows_x86_64/Tennis.exe',
                    help='Unity Environment to load')
flags.DEFINE_boolean(name='render', default=False, help="execute Unity Enviroment with display")
flags.DEFINE_string(name='load',default=None,
                    help='model file to load with path')
flags.DEFINE_bool(name='play', default=None,
                  help='play environment with model')
flags.DEFINE_bool(name='train', default=None, 
                  help='train the agent')
flags.DEFINE_integer(name='episodes', default=20,
                     help='number of episodes to run')
flags.DEFINE_float(name='gamma',default=0.995,
                   help='discount factor for future rewards (0,1]')
flags.DEFINE_float(name='tau',default=0.01,
                   help='network update factor (0,1]')

flags.DEFINE_integer(name='trajectories',default=2048,
                     help='number of trajectories to sample per iteration')
flags.DEFINE_integer(name='training_iterations',default=2048,
                     help='number of iterations to train on')
flags.DEFINE_integer(name='memory_batch_size',default=512,
                     help='batch size of memory samples per epoch')
flags.DEFINE_bool(name='tb', default=True,
                  help='enable tensorboard logging')
flags.DEFINE_string(name='device', default='cpu',
                    help="Device to use for torch")

def main(argv):
    del argv
    if config.log_dir != '':
        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)
        logging.get_absl_handler().use_absl_log_file()
        logging.set_verbosity('info')
    env = UnityEnvironment(file_name=config.env, worker_id = 1, no_graphics=config.render)
    
    model = MADDPG(env=env,
                gamma=config.gamma,
                tau=config.tau,
                memory_size = 1e6,
                batch_size = config.memory_batch_size)
    if config.load is not None:
        model.load_model(load_model = config.load)
    if config.play is not None:
        model.play(episodes= config.episodes)
    if config.train is not None:
        model.train(
            iterations=config.training_iterations,
            )
    
    env.close()

if __name__ == '__main__':
    app.run(main)