import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import hidden_init

class Actor(nn.Module):
    def __init__(self, 
                 state_size, 
                 action_size, 
                 hidden_dims=(64,64), 
                 activation_fc = torch.tanh,
                 action_bounds = ([-0.999, 0],[0.999,0.999])
                 ):
        super(Actor,self).__init__()
        self.ds = state_size
        self.da = action_size
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(self.ds, hidden_dims[0])
        #self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            #hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1],self.da)
        #self.output_layer.weight.data.uniform_(*hidden_init(self.output_layer))

        # move to GPU if available
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.env_min, self.env_max = action_bounds
        self.env_min = torch.tensor(self.env_min,
                                    device=self.device, 
                                    dtype=torch.float32)

        self.env_max = torch.tensor(self.env_max,
                                    device=self.device, 
                                    dtype=torch.float32)
        
        self.nn_min = torch.tanh(torch.Tensor([float('-inf')])).to(self.device)
        self.nn_max = torch.tanh(torch.Tensor([float('inf')])).to(self.device)
        self.rescale_fn = lambda x: (x - self.nn_min) * (self.env_max - self.env_min) / \
                                    (self.nn_max - self.nn_min) + self.env_min
        self.to(self.device)
    
    def _format(self, state):
        """cast state to torch tensor and unroll """
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = x.view(-1,self.ds)
        return x

    def forward(self,states):
        states = self._format(states)
        x = self.activation_fc(self.input_layer(states))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.rescale_fn(torch.tanh(self.output_layer(x)))
        
        return x

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
        self.state_input_layer = nn.Linear(self.ds, hidden_dims[0])
        #self.action_input_layer = nn.Linear(self.da, hidden_dims[0]).double()
        self.input_layer = nn.Linear(hidden_dims[0]+self.da, hidden_dims[0])
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
        state_layer = self.activation_fc(self.state_input_layer(states))
        #action_layer = self.activation_fc(self.action_input_layer(actions))
        x = torch.cat((state_layer,actions), dim=1)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        return x

class G_Actor(nn.Module):
    def __init__(self, 
                 state_size, 
                 action_size, 
                 hidden_dims=(64,64), 
                 activation_fc = F.leaky_relu,
                 action_bounds = ([-0.999, 0],[0.999,0.999]),
                 ):
        super(G_Actor,self).__init__()
        self.ds = state_size
        self.da = action_size
        self.activation_fc = activation_fc
        
        self.input_layer = nn.Linear(self.ds, hidden_dims[0])
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
            self.hidden_layers.append(hidden_layer)
        self.output_layer_mean = nn.Linear(hidden_dims[-1],self.da)
        self.output_layer_log_std = nn.Linear(hidden_dims[-1],self.da)
        self.output_layer_mean.weight.data.uniform_(*hidden_init(self.output_layer_mean))


        # move to GPU if available
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.env_min, self.env_max = action_bounds
        self.env_min = torch.tensor(self.env_min,
                                    device=self.device, 
                                    dtype=torch.float32)

        self.env_max = torch.tensor(self.env_max,
                                    device=self.device, 
                                    dtype=torch.float32)
        
        self.nn_min = torch.tanh(torch.Tensor([float('-inf')])).to(self.device)
        self.nn_max = torch.tanh(torch.Tensor([float('inf')])).to(self.device)
        self.rescale_fn = lambda x: (x - self.nn_min) * (self.env_max - self.env_min) / \
                                    (self.nn_max - self.nn_min) + self.env_min
        self.to(self.device)
    
    def _format(self, state):
        """cast state to torch tensor and unroll """
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.view(-1,self.ds)
        return x

    def forward(self,state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        
        # calculate non-linear higher dimension representation of state
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        # calculate policy mean 
        x_mean = self.output_layer_mean(x)
        # calculate the log of the standard deviation and clamp within reasonable values
        x_log_std = self.output_layer_log_std(x) #.clamp(-20.0,0.0)
        return x_mean, x_log_std
        
    def full(self, state, epsilon = 1e-6):
        """returns sampled action, log_prob of action, and greedy action for state"""
        # get mean and std log for states
        mean, log_std = self.forward(state)
        # get a Normal distribution with those values
        policy = torch.distributions.Normal(mean, log_std.exp())
        # sample an action input
        pre_tanh_action = policy.rsample()
        # convert to -1 ... 1 value
        tanh_action = torch.tanh(pre_tanh_action)
        # rescale action to action bounds
        action = self.rescale_fn(tanh_action)
        # get log probability and rescale to action bounds
        log_prob = policy.log_prob(pre_tanh_action)
        #- torch.log((1-tanh_action.pow(2)).clamp(0,1) + epsilon)
        # multiply the probs of each action dimension (sum the log_probs)
        log_prob = log_prob.sum(-1).unsqueeze(-1)
        return action, log_prob, self.rescale_fn(torch.tanh(mean))
    
    def np_action(self, state, eps=0.5):
        """returns an action and log probs in numpy format for environment step"""
        
        if np.random.random() < eps:
            mean, log_std = self.forward(state)
            policy = torch.distributions.Normal(mean, log_std.exp())
            action = self.rescale_fn(torch.tanh(mean))
            log_prob = policy.log_prob(mean)
            log_prob = log_prob.sum(-1).unsqueeze(-1)
            
        else:
            action, log_prob, mean = self.full(state)
        action_np = action.squeeze().detach().cpu().numpy()
        log_prob_np = log_prob.squeeze().detach().cpu().numpy()
        

        return action_np, log_prob_np

    def action_means(self, state):
        action, log_prob, mean = self.full(state)
        return mean

    def get_probs(self, state, action):
        """returns log probs and entropy for choosing provided action at given state"""
        mean, log_std = self.forward(state)
        # get a Normal distribution with those values
        policy = torch.distributions.Normal(mean, log_std.exp())
        # convert action back to pre-tanh value
        pre_tanh_action = torch.atanh(action)

        log_prob = policy.log_prob(pre_tanh_action)
        #- torch.log((1-action.pow(2)).clamp(0,1) + 1e-6)

        log_prob = log_prob.sum(-1).unsqueeze(-1)
        entropy = policy.entropy().unsqueeze(-1)
        return log_prob, entropy


