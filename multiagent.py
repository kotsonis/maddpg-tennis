import datetime
import os
from absl import logging
from absl import flags
config = flags.FLAGS
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from agent import DDPG
from replay import Buffer

class MADDPG():
    def __init__(self,
                env,
                gamma=0.99,
                tau=0.1,
                memory_size = 1e6,
                batch_size = 1024,
                learn_every = 100,
                ** kwargs):
        super(MADDPG,self).__init__()
        self.discount = gamma
        self.network_update_factor = tau
        self.env = env
        self.memory = Buffer(size=int(memory_size))
        self.batch_size = batch_size
        self.learn_every = learn_every

        self.brain_name, self.num_agents, self.da, self.ds = self._get_env_params(env)
        self.agents = [DDPG(state_size = self.ds, 
                            action_size = self.da, 
                            total_agents=self.num_agents, 
                            actor_activation_fc=torch.tanh, 
                            lr_actor=0.01,
                            critic_activation_fc=F.relu,
                            lr_critic=0.01,
                            actor_hidden_dims = (128,128),
                            critic_hidden_dims = (128,128),
                            gamma = self.discount,
                            tau = self.network_update_factor) for _ in range(self.num_agents)]
        self.log_dir = kwargs.setdefault('log_dir', config.log_dir)
        self.model_save_dir = os.path.join(self.log_dir, 'model')
        self.model_save_period = 50
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        
        self.tb_logging = kwargs.get('tb',config.tb)
        if (self.tb_logging):
            self._tb_init()
        self.iteration = 0
        # tb tracking fields
        self.tb_loss_agent = self.num_agents*[0.0]
        self.tb_loss_critic = self.num_agents*[0.0]
        self.tb_loss_value = 0.0
        self.tb_loss_policy = 0.0
        self.tb_avg_return = 0.0
        self.tb_traj_mean_value = 0.0
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
    
    def soft_update_models(self,tau=0.01):
        for agent in self.agents:
            agent.soft_update(tau=tau)
        
    def save_model(self, path=None):
        """Saves the model."""
        data = {}
        data['iteration'] = self.iteration
        for i,agent in enumerate(self.agents):
            data['agent_'+str(i)+'_actor_state_dict'] = agent.actor.state_dict()
            data['agent_'+str(i)+'_critic_state_dict'] = agent.critic.state_dict()
            data['agent_'+str(i)+'_actor_optim_state_dict'] =agent.actor_optimizer.state_dict()
            data['agent_'+str(i)+'_critic_optim_state_dict'] =agent.critic_optimizer.state_dict()

        torch.save(data, path)
    
    def load_model(self, **kwargs):
        """loads a model from a given path."""
        load_path = kwargs.setdefault(
                                    'load',
                                    config.load)
        checkpoint = torch.load(load_path,map_location=torch.device(self.device))
        
        self.saved_iteration = checkpoint['iteration']
        self.iteration += self.saved_iteration
        for i,agent in enumerate(self.agents):
            agent.actor.load_state_dict(checkpoint['agent_'+str(i)+'_actor_state_dict'])
            agent.target_actor.load_state_dict(checkpoint['agent_'+str(i)+'_actor_state_dict'])
            agent.critic.load_state_dict(checkpoint['agent_'+str(i)+'_critic_state_dict'])
            agent.target_critic.load_state_dict(checkpoint['agent_'+str(i)+'_critic_state_dict'])
            agent.actor_optimizer.load_state_dict(checkpoint['agent_'+str(i)+'_actor_optim_state_dict'])
            agent.critic_optimizer.load_state_dict(checkpoint['agent_'+str(i)+'_critic_optim_state_dict'] )
            agent.actor.train()
            agent.critic.train()
        
        logging.info('Loaded model: {}'.format(load_path))
        logging.info('iteration: {}'.format(self.iteration))

    def _next_iter(self):
        """increases training iterator and performs logging/model saving"""
        self.iteration += 1
        
        if (self.iteration+1) % self.model_save_period ==0:
            self.save_model(os.path.join(self.model_save_dir, 'model_{:4d}.pt'.format(self.iteration)))
        if self.tb_logging and ((self.iteration+1) % self.tensorboard_update_period ==0):
            # save latest model
            self.save_model(os.path.join(self.model_save_dir, 'model_latest.pt'))
            # write to tb log
            self._tb_write()
        
        return self.iteration
    
    def _tb_init(self):
        """initialize tensorboard logging"""

        ts = datetime.datetime.now().replace(microsecond=0).strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(os.path.join(self.log_dir, 'tb',ts))
        self.tensorboard_update_period = 1
        return

    def _tb_write(self):
        """Writes training data to tensorboard summary writer. Should be overloaded by sub-classes."""
        it = self.iteration
        # tb tracking fields
        self.writer.add_scalar('agent1/actor_loss', self.tb_loss_agent[0], it)
        self.writer.add_scalar('agent2/actor_loss', self.tb_loss_agent[1], it)
        self.writer.add_scalar('agent1/critic_loss', self.tb_loss_critic[0], it)
        self.writer.add_scalar('agent2/critic_loss', self.tb_loss_critic[1], it)
        self.writer.add_scalar('mean/return', self.tb_avg_return, it)

        self.writer.flush()
    
    def _get_env_params(self, env):
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        # number of agents 
        num_agents = len(env_info.agents)
        # get the action size
        action_size = brain.vector_action_space_size
        # get the state space 
        states = env_info.vector_observations
        state_size = states.shape[1]
        return brain_name, num_agents, action_size, state_size

    def act_np(self, obs, noise=0.0):
        actions = []
        #obs = torch.tensor(obs).float()
        for agent, observation in zip(self.agents, obs):
            actions.append(agent.target_act(observation, noise).detach().squeeze().cpu().data.numpy())
        actions = np.array(actions)
        return actions
    def act(self,obs,noise=0.0):
        actions = []
        #obs = torch.tensor(obs).float()
        for agent, observation in zip(self.agents, obs):
            actions.append(agent.act(observation, noise))
        actions = torch.cat(actions, dim=1)
        return actions
    def target_act(self,obs,noise=0.0):
        target_actions = []
        #obs = torch.tensor(obs).float()
        for agent, observation in zip(self.agents, obs):
            target_actions.append(agent.target_act(observation, noise))
        target_actions = torch.cat(target_actions, dim=1)
        return target_actions

    def train(self, iterations):
        step = 0
        while self.iteration < iterations:
            self.agents[0].noise.reset()
            self.agents[1].noise.reset()
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            states = env_info.vector_observations   
            while True:
                actions = self.act_np(states, noise=0.1)
                #print(actions)
                env_info = self.env.step(actions)[self.brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards 
                dones = env_info.local_done
                self.memory.push(states, actions, rewards, next_states, dones)
                step += 1
                if (len(self.memory)> self.batch_size):
                    if (step +1) % self.learn_every == 0:
                        self.learn_step()
                        self._next_iter()
                #scores += env_info.rewards
                if np.any(dones):
                    break
                states = next_states
    def learn_step(self):
        obs, states, actions, all_actions, rewards, next_obs, next_states,dones = self.memory.sample(self.batch_size)
        
        for i, agent in enumerate(self.agents):

            agent.critic_optimizer.zero_grad()
            future_actions = self.target_act(next_obs)
            #print('future actions shape {}'.format(future_actions.shape))
            q_next = agent.target_critic(next_states, future_actions)
            #print('shape q_next :{}, rewards[i]:{}, dones[i]:{}'.format(
            #    q_next.shape, rewards[i].shape, dones[i].shape
            #))
            y = rewards[i].view(-1, 1)
            y +=self.discount * q_next*(1 - dones[i].view(-1, 1))
            q = agent.critic(states, all_actions)
            huber_loss = torch.nn.SmoothL1Loss()
            critic_loss = huber_loss(q, y.detach())
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
            agent.critic_optimizer.step()

            agent.actor_optimizer.zero_grad()
            new_actions = self.act(obs)
            #print('new actions shape {}'.format(new_actions.shape))
            #print('obs shape {}'.format(obs.shape))
            actor_loss = -agent.critic(states, new_actions).mean()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
            agent.actor_optimizer.step()
            with torch.no_grad():
                self.tb_loss_agent[i] = actor_loss
                self.tb_loss_critic[i] = critic_loss
        self.soft_update_models(self.network_update_factor)

    def play(self, episodes):
        for it in range(episodes):
            print('\rIteration {}'.format(it))
            env_info = self.env.reset(train_mode=False)[self.brain_name]
            states = env_info.vector_observations   
            while True:
                #actions = [[1.0,1.0],[-1.0,0.6]]
                actions = self.act_np(states, noise=0.0)
                #print(actions)
                env_info = self.env.step(actions)[self.brain_name]
                next_states = states = env_info.vector_observations
                rewards = env_info.rewards 
                dones = env_info.local_done
                if np.any(dones):
                    break
                states = next_states
