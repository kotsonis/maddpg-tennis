import datetime
import os
from absl import logging
from absl import flags
config = flags.FLAGS
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from agent import DDPG

from replay import PriorityReplay as Buffer
from network import CentralActionValueFn
from collections import deque

class MADDPG():
    def __init__(self,
                env,
                eps_start = 0.9,
                eps_minimum = 0.05,
                eps_decay = 0.99,
                gamma=0.99,
                tau=0.1,
                memory_size = 1e6,
                batch_size = 1024,
                learn_every = 20,
                n_steps = 10,
                actor_hidden_dims = (128,128,64),
                actor_lr = 1e-4,
                critic_hidden_dims = (128,128,64),
                critic_lr = 1e-4,
                ** kwargs):
        super(MADDPG,self).__init__()
        self.eps_start = eps_start
        self.eps_minimum = eps_minimum
        self.eps_decay = eps_decay
        self.eps = self.eps_start
        self._min_epsilon = torch.finfo(torch.float).eps
        self.discount = gamma
        self.network_update_factor = tau
        self.env = env
        
        self.batch_size = batch_size
        self.learn_every = learn_every
        self.brain_name, self.num_agents, self.da, self.ds = self._get_env_params(env)
        self.memory = [
                     Buffer(size=int(memory_size), gamma=gamma, n_steps=n_steps, **kwargs)
                     for _ in range(self.num_agents)]

        low_bound = np.ones((self.da))*-0.99999
        upper_bound = np.ones((self.da))*0.99999
        self.action_bounds = (low_bound,upper_bound)

        self.agents = [DDPG(state_size = self.ds, 
                            action_size = self.da, 
                            total_agents=self.num_agents, 
                            actor_activation_fc=F.leaky_relu, 
                            lr_actor=1e-4,                      
                            critic_activation_fc=F.leaky_relu,
                            lr_critic=1e-4,                     
                            actor_hidden_dims = actor_hidden_dims, 
                            actor_lr = actor_lr,  
                            critic_hidden_dims = critic_hidden_dims,  
                            critic_lr = critic_lr,
                            gamma = self.discount,
                            tau = self.network_update_factor,
                            action_bounds = self.action_bounds) for _ in range(self.num_agents)]

        self.log_dir = kwargs.setdefault('log_dir', config.log_dir)
        self.model_save_dir = os.path.join(self.log_dir, 'model')
        self.model_save_period = 500
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        
        self.iteration = 0
        
        self.tb_logging = kwargs.get('tb',config.tb)
        if (self.tb_logging):
            self.tb_init = self._tb_init
            self.tb_write = self._tb_write
        else:
            self.tb_init = lambda *args: None
            self.tb_write = lambda *args: None
        # tb tracking fields
        self.tb_loss_agent = self.num_agents*[0.0]
        self.tb_loss_critic = self.num_agents*[0.0]
        self.tb_loss_value = 0.0
        self.tb_loss_policy = 0.0
        self.tb_avg_return = 0.0
        self.tb_traj_mean_value = 0.0
        
        self.tb_init()

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
    
    def soft_update_models(self,tau=0.01):
        for agent in self.agents:
            agent.soft_update(tau=tau)
        # update critic
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def critic_weight_sync(self, tau=0.01):
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update_models(self):
        for agent in self.agents:
            agent.hard_update()
        # update critic
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(local_param.data)

    def _next_eps(self):
        """updates exploration factor"""
        self.eps = max(self.eps_minimum, self.eps*self.eps_decay)

    def save_model(self, path=None):
        """Saves the model."""
        data = {}
        data['iteration'] = self.iteration
        data['agent_0_critic_state_dict'] = self.agents[0].critic.state_dict()
        data['agent_0_critic_optim_state_dict'] =self.agents[0].critic_optimizer.state_dict()
        data['agent_0_actor_state_dict'] = self.agents[0].actor.state_dict()
        data['agent_0_target_actor_state_dict'] = self.agents[0].target_actor.state_dict()
        data['agent_0_actor_optim_state_dict'] = self.agents[0].actor_optimizer.state_dict()
        data['agent_1_actor_state_dict'] = self.agents[1].actor.state_dict()
        data['agent_1_target_actor_state_dict'] = self.agents[1].target_actor.state_dict()
        data['agent_1_actor_optim_state_dict'] = self.agents[1].actor_optimizer.state_dict()
        data['agent_1_critic_state_dict'] = self.agents[1].critic.state_dict()
        data['agent_1_critic_optim_state_dict'] =self.agents[1].critic_optimizer.state_dict()

        torch.save(data, path)
    
    def load_model(self, **kwargs):
        """loads a model from a given path."""
        load_path = kwargs.setdefault(
                                    'load',
                                    config.load)
        checkpoint = torch.load(load_path,map_location=torch.device(self.device))
        
        self.saved_iteration = checkpoint['iteration']
        
        self.iteration += self.saved_iteration
        self.agents[0].actor.load_state_dict(checkpoint['agent_0_actor_state_dict'])
        self.agents[0].target_actor.load_state_dict(checkpoint['agent_0_target_actor_state_dict'])
        self.agents[0].actor_optimizer.load_state_dict(checkpoint['agent_0_actor_optim_state_dict'])
        self.agents[0].critic.load_state_dict(checkpoint['agent_0_critic_state_dict'])
        self.agents[0].target_critic.load_state_dict(checkpoint['agent_0_critic_state_dict'])
        self.agents[0].critic_optimizer.load_state_dict(checkpoint['agent_0_critic_optim_state_dict'] )
        self.agents[0].actor.train()
        self.agents[0].critic.train()
        self.agents[1].actor.load_state_dict(checkpoint['agent_1_actor_state_dict'])
        self.agents[1].target_actor.load_state_dict(checkpoint['agent_1_target_actor_state_dict'])
        self.agents[1].actor_optimizer.load_state_dict(checkpoint['agent_1_actor_optim_state_dict'])
        self.agents[1].critic.load_state_dict(checkpoint['agent_1_critic_state_dict'])
        self.agents[1].target_critic.load_state_dict(checkpoint['agent_1_critic_state_dict'])
        self.agents[1].critic_optimizer.load_state_dict(checkpoint['agent_1_critic_optim_state_dict'] )
        self.agents[1].actor.train()
        self.agents[1].critic.train()
        #self.critic.train()
        
        logging.info('Loaded model: {}'.format(load_path))
        logging.info('iteration: {}'.format(self.iteration))

    def _next_iter(self):
        """increases training iterator and performs logging/model saving"""
        self.iteration += 1
        self._next_eps()
        if (self.iteration+1) % self.model_save_period ==0:
            self.save_model(os.path.join(self.model_save_dir, 'model_{:4d}.pt'.format(self.iteration)))
        if (self.iteration+1) % self.tensorboard_update_period ==0:
            # save latest model
            self.save_model(os.path.join(self.model_save_dir, 'model_latest.pt'))
            # write to tb log
            self.tb_write()
        
        return self.iteration
    
    def _tb_init(self):
        """initialize tensorboard logging"""

        ts = datetime.datetime.now().replace(microsecond=0).strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(os.path.join(self.log_dir, 'tb',ts))
        self.tensorboard_update_period = 1
        self.tb_avg_100_score = 0.0
        return

    def _tb_write(self):
        """Writes training data to tensorboard summary writer. Should be overloaded by sub-classes."""
        it = self.iteration
        # tb tracking fields
        self.writer.add_scalar('agent1/actor_loss', self.tb_loss_agent[0], it)
        self.writer.add_scalar('agent2/actor_loss', self.tb_loss_agent[1], it)
        self.writer.add_scalar('agent1/critic_loss', self.tb_loss_critic[0], it)
        self.writer.add_scalar('agent2/critic_loss', self.tb_loss_critic[1], it)
        self.writer.add_scalar('last_episode_score', self.tb_avg_return, it)
        self.writer.add_scalar('score_running_mean', self.tb_avg_100_score, it)
        self.writer.add_scalar('eps', self.eps, it)
        self.writer.add_scalar('episodes', self.episodes, it)

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

    def act_np(self, obs, noise=0.0, eps=0.0):
        actions = []
        #obs = torch.tensor(obs).float()
        
        for agent, observation in zip(self.agents, obs):
            actions.append(agent.target_act(observation, noise=eps).detach().squeeze().cpu().data.numpy())
        actions = np.array(actions)
        return actions.clip(-1.0,1.0)

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
        self.episodes = 0
        self.rewards_scale = 5.0
        episode_scores = deque(maxlen = 100)
        scores = np.zeros(self.num_agents)
        env_reset = lambda : self.env.reset(train_mode=True)[self.brain_name]
        env_step = lambda act: self.env.step(act)[self.brain_name]

        for replay in self.memory:
            replay.initialize(
                                iterations=5000, 
                                env_reset_fn=env_reset, 
                                env_step_fn=env_step,
                                action_size=self.da, 
                                num_agents=self.num_agents,
                                rewards_scale = self.rewards_scale)
        warmup_exploration_steps = 2000
        learn_every = self.memory[0].n_steps +1

        while self.iteration < iterations:
            self.agents[0].noise.reset()
            self.agents[1].noise.reset()
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            states = env_info.vector_observations   
            while True:
                step += 1
                actions = self.act_np(states, eps=self.eps)
                #print(actions)
                env_info = self.env.step(actions)[self.brain_name]
                next_states = env_info.vector_observations
                rewards = self.rewards_scale*np.array(env_info.rewards)
                scores += env_info.rewards
                dones = env_info.local_done
                for replay in self.memory:
                    replay.push(states, actions, rewards, next_states, dones)
                states = next_states
                #scores += env_info.rewards
                if (len(self.memory[0])> self.batch_size) and (step+1) % learn_every == 0:
                    self.learn_step()
                    self._next_iter()
                if np.any(dones):
                    self.episodes += 1
                    self.tb_avg_return = np.max(scores)
                    episode_scores.append(np.max(scores))
                    self.tb_avg_100_score = np.mean(episode_scores)
                    scores = scores*0.0
                    break
                
            
            
            
    
    def learn_step(self):
        
        for i,agent in enumerate(self.agents):
            replay = self.memory[i]
            for training_epoch in range(20):  #winner: 5
                replay.sample(self.batch_size)
                s = replay.obs
                a = replay.actions
                r = replay.rewards
                s1 = replay.next_obs
                d = replay.dones
                g = replay.gammas
                w = replay.weights
                mem_idx = replay.idxes
                all_s = s.permute(1,0,2).reshape(self.batch_size, -1)
                all_s1 = s1.permute(1,0,2).reshape(self.batch_size, -1)
                all_a = a.permute(1,0,2).reshape(self.batch_size, -1)
            
                a1 = torch.stack(
                    [ag.target_act(s1[ag_num],noise=0)
                    for ag_num,ag in enumerate(self.agents)]
                    )
                a1 = a1.permute(1,0,2).reshape(self.batch_size, -1)
                s1_a1_val = agent.target_critic(all_s1, a1)
                q_pred = r[i] + g[i]*(1-d[i])*s1_a1_val
                q = agent.critic(all_s,all_a)
                q_loss = w*F.mse_loss(q, q_pred.detach(),reduction='none')
                q_loss = q_loss.mean()
                
                agent.critic_optimizer.zero_grad()
                q_loss.backward()
                agent.critic_optimizer.step()

                pred_a = torch.stack(
                    [ag.act(s[ag_num],noise=0)
                    for ag_num,ag in enumerate(self.agents)]
                    ).permute(1,0,2).reshape(self.batch_size, -1)
                actor_loss = -1*agent.critic(all_s, pred_a)
                actor_loss = actor_loss.mean()

                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                agent.actor_optimizer.step()

                agent.soft_update(tau=self.network_update_factor)

            with torch.no_grad():
                td_error = q_pred - q
                updated_priorities = td_error.abs()
                new_p = updated_priorities.squeeze()
                self.tb_loss_critic[i] = q_loss
                self.tb_loss_agent[i] = actor_loss
                replay.update_priorities(mem_idx, new_p.cpu().data.numpy().tolist())
        



    def learn_step_with_sorting(self):
        self.memory.sample(self.batch_size)
        obs = self.memory.obs
        actions = self.memory.actions
        rewards = self.memory.rewards
        next_obs = self.memory.next_obs
        dones = self.memory.dones
        gammas = self.memory.gammas
        weights = self.memory.weights
        indices = self.memory.idxes
        updated_priorities = torch.zeros_like(weights)
        for i, agent in enumerate(self.agents):
            agent_idx_list = list(range(self.num_agents))
            agent_idx_list.remove(i)
            agent_idx_list = [i] + agent_idx_list
            idx = torch.tensor(agent_idx_list)
            next_states_sorted = next_obs[idx].permute(1,0,2)\
                                              .reshape(self.batch_size,-1)
            states_sorted = obs[idx].permute(1,0,2)\
                                    .reshape(self.batch_size,-1)
            actions_sorted = actions[idx].permute(1,0,2)\
                                         .reshape(self.batch_size,-1)
            # optimize critic
            future_actions = torch.stack([
                ag.target_act(next_obs[ag_num],noise=0).detach() 
                for ag_num,ag in enumerate(self.agents)])
            future_actions_sorted = future_actions[idx].permute(1,0,2) \
                                                       .reshape(self.batch_size, -1)
               
            q_next = self.target_critic(next_states_sorted, future_actions_sorted)
            goes_on = (1-dones[i])
            future_q = q_next * goes_on
            discounted_future_q = gammas[i]*future_q
            y = rewards[i]
            y = y + discounted_future_q
            q = self.critic(states_sorted, actions_sorted)
            huber_loss = torch.nn.SmoothL1Loss(reduction='none')
            critic_loss = huber_loss(q, y.detach())
            with torch.no_grad():
                td_error = y - q
                updated_priorities += td_error.abs()
                self.tb_loss_critic[i] = critic_loss.mean()
            
            critic_loss = torch.mean(critic_loss * weights)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # optimize actor
            
            new_actions = torch.stack([
                ag.act(obs[ag_num],noise=0) if ag_num==i 
                else ag.act(obs[ag_num],noise=0).detach()
                for ag_num,ag in enumerate(self.agents)])
            combined_new_actions = new_actions[idx].permute(1,0,2)\
                                                   .reshape(self.batch_size, -1)
            actor_loss = - self.critic(states_sorted, combined_new_actions).mean(dim=1).mean()
            
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            with torch.no_grad():
                self.tb_loss_agent[i] = actor_loss
        with torch.no_grad(): 
            new_p = updated_priorities.squeeze()
            # -------------------- update PER priorities ----------------------- #
            self.memory.update_priorities(indices, new_p.cpu().data.numpy().tolist())

        self.soft_update_models(self.network_update_factor)

    def play(self, episodes):
        scores = np.zeros(self.num_agents)
        episode_scores = deque(maxlen = 100)
        
        for it in range(episodes):
            env_info = self.env.reset(train_mode=False)[self.brain_name]
            states = env_info.vector_observations   
            while True:
                #actions = [[1.0,1.0],[-1.0,0.6]]
                actions = self.act_np(states, noise=0.0)
                #print(actions)
                env_info = self.env.step(actions)[self.brain_name]
                states = env_info.vector_observations
                rewards = env_info.rewards 
                scores += env_info.rewards
                dones = env_info.local_done
                if np.any(dones):
                    episode_scores.append(np.max(scores))
                    print('Episode: {}, max_score: {}, avg_score over episodes: {}'\
                           .format(it, np.max(scores), np.mean(episode_scores)))
                    scores *= 0.0
                    break
                
