import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import OUNoise, hidden_init

class Actor(nn.Module):
    def __init__(self, 
                 state_size, 
                 action_size, 
                 hidden_dims=(64,64), 
                 activation_fc = torch.tanh
                 ):
        super(Actor,self).__init__()
        self.ds = state_size
        self.da = action_size
        self.activation_fc = activation_fc
        self.means = torch.tensor([0.0, 0.5], dtype=torch.float32)
        self.spans = torch.tensor([1.0, 0.5], dtype=torch.float32)
        self.input_layer = nn.Linear(self.ds, hidden_dims[0])
        #self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            #hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1],self.da)
        #self.output_layer.weight.data.uniform_(*hidden_init(self.output_layer))

        self.Noise = OUNoise(self.da)
        # move to GPU if available
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)
    
    def _format(self, state):
        """cast state to torch tensor and unroll """
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.view(-1,self.ds)
        return x

    def forward(self,states):
        states = self._format(states)
        x = self.activation_fc(self.input_layer(states))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = torch.tanh(self.output_layer(x))
        x = x*self.spans + self.means
        return x
    
    def np_act(self,
               states, 
               noise=0.0
               ):
        action = self.forward(states)
        np_action = action.detach().cpu().numpy()
        return np_action 

class CentralActionValueFn(nn.Module):
    def __init__(self, 
                 state_size, 
                 action_size, 
                 num_agents, 
                 hidden_dims=(64,64), 
                 activation_fc=torch.tanh
                 ):
        super(CentralActionValueFn, self).__init__()
        self.ds = state_size*num_agents
        self.da = action_size*num_agents
        self.num_agents = num_agents
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(self.ds+self.da, hidden_dims[0])
        #self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            #hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1],1)
        #self.output_layer.weight.data.uniform_(*hidden_init(self.output_layer))
        # move to GPU if available
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)
    
    def forward(self, 
                states, 
                actions
                ):
        x = torch.cat((states,actions), dim=1)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        return x
