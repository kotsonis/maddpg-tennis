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
                 action_bounds,
                 total_agents=1, 
                 actor_activation_fc=F.relu, 
                 lr_actor = 0.01,
                 critic_activation_fc=F.relu,
                 lr_critic = 0.01,
                 actor_hidden_dims = (64,64),
                 critic_hidden_dims = (64,64),
                 actor_class = Actor,
                 critic_class = CentralActionValueFn,
                 critic_loss = torch.nn.MSELoss(),
                 gamma = 0.99,
                 tau = 0.1
                 ):
        super(DDPG,self).__init__()
        self.ds = state_size
        self.da = action_size
        self.tot_agents = total_agents
        self.tau = tau
        self.critic_loss = critic_loss

        self.noise = OUNoise(action_dimension=self.da)
        
        #self.network_update_factor = tau
        # create actor / critic networks
                                                
        self.actor = actor_class(state_size=self.ds, 
                           action_size=self.da, 
                           hidden_dims=actor_hidden_dims, 
                           activation_fc = actor_activation_fc,
                           action_bounds=action_bounds)
        self.target_actor = actor_class(state_size=self.ds, 
                                  action_size=self.da, 
                                  hidden_dims=actor_hidden_dims, 
                                  activation_fc = actor_activation_fc,
                                  action_bounds=action_bounds)
        self.critic = critic_class(state_size=self.ds, 
                                           action_size=self.da, 
                                           num_agents=total_agents, 
                                           hidden_dims=critic_hidden_dims, 
                                           activation_fc=critic_activation_fc)
        self.target_critic = critic_class(state_size=self.ds, 
                                                  action_size=self.da, 
                                                  num_agents=total_agents, 
                                                  hidden_dims=critic_hidden_dims, 
                                                  activation_fc=critic_activation_fc)
        # set up optimizers
        # 
        self.critic_optimizer = Adam(
                                     self.critic.parameters(), 
                                     lr=lr_critic, 
                                     weight_decay=1.e-5)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        # make target & current networks identical
        self.hard_update()
        
        
    def act(self, obs, noise=0.0):
        action = (1-noise)*self.actor(obs) + noise*self.noise.noise()
        #print('actor: got obs {} and will take action {}'.format(obs,action))
        action = action.clip(-1.0,1.0)
        return action

    def target_act(self, obs, noise=0.0):
        action = (1-noise)*self.target_actor(obs) + noise*self.noise.noise()
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

    def hard_update(self):
        """copies local model parameters into target.

        """
        # update actor
        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(local_param.data)
        # update critic
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(local_param.data)

