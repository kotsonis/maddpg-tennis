import numpy as np 
from numpy.random import default_rng
import torch
from collections import namedtuple, deque

class Buffer():
    def __init__(self, size, gamma=0.95, n_steps=5):
        super(Buffer,self).__init__()
        self.buffer = size*[]
        self.next_idx = 0
        self.max_size = size
        self.gamma = gamma
        self.n_steps = n_steps
        self.rng = default_rng()
        self.experience = namedtuple(
                                     "Experience", 
                                     field_names=[
                                         "obs",
                                         "state", 
                                         "action", 
                                         "all_actions",
                                         "reward", 
                                         "next_obs",
                                         "next_state",
                                         "done"
                                         ])
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.n_step_buffer = deque(maxlen = n_steps)

    def __len__(self):
        return len(self.buffer)

    def add(self, data):
        if self.next_idx >= len(self.buffer):
            self.buffer.append(data)
        else:
            self.buffer[self.next_idx] = data
        self.next_idx = int((self.next_idx + 1) % self.max_size)

    def push(self, obs, actions, rewards, next_obs, done):
        """push an experience (obs,actions,rewards,next_obs) to the memory buffer"""
        #combine all agent observations
        state = [item for values in obs for item in values]
        next_state = [item for values in next_obs for item in values]
        all_actions = [item for values in actions for item in values]

        data = self.experience(obs, state, actions, all_actions,rewards, next_obs, next_state, done)
        self.n_step_buffer.append(data)
        if np.any(done) or len(self.n_step_buffer)>=self.n_steps:
            obs, state, actions, all_actions,rewards, next_obs_f, next_state_f, done_f = self.n_step_buffer.pop()
            returns = np.array(rewards)
            while len(self.n_step_buffer) > 0:
                obs, state, actions, all_actions,rewards, next_obs, next_state, done = self.n_step_buffer.pop()
                returns = np.array(rewards) + returns*self.gamma
            data = self.experience(obs, state, actions, all_actions,returns, next_obs_f, next_state_f, done_f)
            self.add(data)

    def sample(self, batch_size):
        """return a random subset of size batch_size of the memory"""
        assert batch_size <= len(self.buffer), "not enough samples"
        idxs = list(self.rng.choice(len(self.buffer), batch_size))
        return self.encode_sample(idxs)
    
    def encode_sample(self,idxes):
        obs, states, actions, all_actions,rewards, next_obs, next_states,dones = [], [], [], [],[], [], [],[]
        for idx in idxes:
            obs.append(self.buffer[idx].obs)
            states.append(self.buffer[idx].state)
            actions.append(self.buffer[idx].action)
            all_actions.append(self.buffer[idx].all_actions)
            rewards.append(self.buffer[idx].reward)
            next_obs.append(self.buffer[idx].next_obs)
            next_states.append(self.buffer[idx].next_state)
            dones.append(self.buffer[idx].done)
        # cast to tensors
        obs, states, actions, all_actions,rewards, next_obs, next_states,dones = map(
                lambda x: torch.tensor(x).float().to(self.device),
                (obs, states, actions, all_actions,rewards, next_obs, next_states,dones)
            )
        # change from batch x actors x ... to actors x batch x ...
        obs, actions, rewards, next_obs,dones = map(
                lambda x: x.permute(1,0,-1),
                (obs, actions, rewards.unsqueeze(-1), next_obs, dones.unsqueeze(-1))
            )
        return (obs, states, actions, all_actions,rewards, next_obs, next_states,dones)
