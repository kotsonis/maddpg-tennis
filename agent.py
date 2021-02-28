import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from network import Actor, CentralActionValueFn
from utils import OUNoise

class DDPG():
    def __init__(self,
                 state_size, 
                 action_size, 
                 total_agents=1, 
                 actor_activation_fc=F.relu, 
                 lr_actor = 0.01,
                 critic_activation_fc=F.relu,
                 lr_critic = 0.01,
                 actor_hidden_dims = (64,64),
                 critic_hidden_dims = (64,64),
                 gamma = 0.99,
                 tau = 0.1):
        super(DDPG,self).__init__()
        self.ds = state_size
        self.da = action_size
        self.tot_agents = total_agents
        self.tau = tau
        self.noise = OUNoise(action_dimension=self.da)
        
        #self.network_update_factor = tau
        # create actor / critic networks

        self.critic = CentralActionValueFn(state_size=self.ds, 
                                           action_size=self.da, 
                                           num_agents=self.tot_agents, 
                                           hidden_dims=critic_hidden_dims, 
                                           activation_fc=critic_activation_fc)
        self.target_critic = CentralActionValueFn(state_size=self.ds, 
                                                  action_size=self.da, 
                                                  num_agents=self.tot_agents, 
                                                  hidden_dims=critic_hidden_dims, 
                                                  activation_fc=critic_activation_fc)
                                                
        self.actor = Actor(state_size=self.ds, 
                           action_size=self.da, 
                           hidden_dims=actor_hidden_dims, 
                           activation_fc = actor_activation_fc)
        self.target_actor = Actor(state_size=self.ds, 
                                  action_size=self.da, 
                                  hidden_dims=actor_hidden_dims, 
                                  activation_fc = actor_activation_fc)
        
        # make target & current networks identical
        self.soft_update(tau=1.0)
        # set up optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1.e-5)
        
    def act(self, obs, noise=0.0):
        action = self.actor(obs) + noise*self.noise.noise()
        #print('actor: got obs {} and will take action {}'.format(obs,action))
        action = action.clip(-1.0,1.0)
        return action

    def target_act(self, obs, noise=0.0):
        action = self.target_actor(obs) + noise*self.noise.noise()
        action = action.clip(-1.0,1.0)
        return action
    
    
    def soft_update(self,tau=0.01):
        """Soft update model parameters.
        """
        # update actor
        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        # update critic
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
