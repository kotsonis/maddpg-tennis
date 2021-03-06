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

flags.DEFINE_float(name='eps_start',default=1.0,
    help='starting exploration rate (0,1]')
flags.DEFINE_float(name='eps_minimum',default=0.001,
    help='minimum exploration rate')
flags.DEFINE_float(name='eps_decay',default=0.99,
    help='eps decay rate. eps=eps*eps_decay')
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
flags.DEFINE_integer(name='n_steps',default=5,
                     help='number of steps for N step returns calculation')

flags.DEFINE_bool(name='tb', default=True,
                  help='enable tensorboard logging')
flags.DEFINE_string(name='device', default='cpu',
                    help="Device to use for torch")

flags.DEFINE_float(name='PER_alpha',default = 0.5,
                   help='α factor (prioritization) for Prioritized Replay Buffer')
flags.DEFINE_float(name='PER_beta_min',default = 0.5,
                   help='starting β factor (randomness) for Prioritized Replay Buffer')
flags.DEFINE_float(name='PER_beta_max',default=1.0,
                   help='ending β factor (randomness) for Prioritized Replay Buffer')
flags.DEFINE_float(name='PER_minimum_priority',default=1e-5,
                   help='minimum priority to set when updating priorities')

flags.DEFINE_spaceseplist(name='actor_dnn_dims', default= [128,128,64],
                          help='layer dimensions of actor NN')
flags.DEFINE_spaceseplist(name='critic_dnn_dims', default= [128,128,64],
                          help='layer dimensions of critic NN')
flags.DEFINE_float(name='actor_lr',default=1e-4,
                   help='lr for actor optimizer')
flags.DEFINE_float(name='critic_lr',default=1e-4,
                   help='lr for actor optimizer')
def main(argv):
    del argv
    if config.log_dir != '':
        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)
        logging.get_absl_handler().use_absl_log_file()
        logging.set_verbosity('info')
    env = UnityEnvironment(file_name=config.env, worker_id = 1, no_graphics=config.render)
    # convert dnn dims command line parameters to ints
    actor_dnn_dims = [int(i) for i in config.actor_dnn_dims]
    critic_dnn_dims = [int(i) for i in config.critic_dnn_dims]
    model = MADDPG(env=env,
                eps_start = config.eps_start,
                eps_minimum = config.eps_minimum,
                eps_decay = config.eps_decay,
                gamma=config.gamma,
                tau=config.tau,
                memory_size = 1e6,
                batch_size = config.memory_batch_size,
                learn_every=4,
                n_steps=config.n_steps,
                PER_alpha = config.PER_alpha,
                PER_beta_min = config.PER_beta_min,
                PER_beta_max = config.PER_beta_max,
                PER_minimum_priority = config.PER_minimum_priority,
                actor_hidden_dims= actor_dnn_dims,
                actor_lr = config.actor_lr,
                critic_hidden_dims= critic_dnn_dims,
                critic_lr = config.critic_lr
                )
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