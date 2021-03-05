import numpy as np 
from numpy.random import default_rng
import torch
from collections import namedtuple, deque
from utils import SumSegmentTree, MinSegmentTree
class Buffer():
    def __init__(self, size, gamma=0.95, n_steps=5):
        super(Buffer,self).__init__()
        self._sampling_results = dict()
        self.buffer = []
        self.next_idx = 0
        self._maxsize = size
        self.gamma = gamma
        self.n_steps = n_steps
        self.rng = default_rng()
        self.experience = namedtuple(
                                     "Experience", 
                                     field_names=[
                                         "obs",
                                         "action", 
                                         "reward", 
                                         "next_obs",
                                         "done",
                                         "gammas"
                                         ])
        
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.n_step_buffer = deque(maxlen = n_steps)

    @property
    def obs(self):
        return self._sampling_results['obs']
    @property
    def next_obs(self):
        return self._sampling_results['next_obs']
    @property
    def actions(self):
        return self._sampling_results['actions']
    @property
    def rewards(self):
        return self._sampling_results['rewards']
    @property
    def dones(self):
        return self._sampling_results['dones']
    @property
    def gammas(self):
        return self._sampling_results['gammas']

    def __len__(self):
        return len(self.buffer)

    def add(self, data):
        if self.next_idx >= len(self.buffer):
            self.buffer.append(data)
        else:
            self.buffer[self.next_idx] = data
        self.next_idx = int((self.next_idx + 1) % self._maxsize)

    def push(self, obs, actions, rewards, next_obs, dones):
        """push an experience (obs,actions,rewards,next_obs) to the memory buffer"""
        #combine all agent observations
        obs_t = torch.tensor(obs, dtype=torch.float)
        next_obs_t = torch.tensor(next_obs,dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(-1)
        dones = torch.tensor(dones, dtype=torch.uint8).unsqueeze(-1)
        actions = torch.tensor(actions,dtype=torch.float)
        # option: count opponents reward into agents reward
        # rewards = 2*rewards - rewards.sum()
        
        data = self.experience(obs_t, actions,rewards, next_obs_t, dones, 1.0)
        self.n_step_buffer.append(data)
        if (len(self.n_step_buffer) == self.n_steps):
            cur_obs, cur_action, returns, gammas = self._calc_back_rewards()
            data = self.experience(cur_obs, cur_action, returns, next_obs_t, dones,gammas)
            self.add(data)
            
        if torch.any(dones):
            while (len(self.n_step_buffer) > 0):
                cur_obs, cur_action, returns, gammas = self._calc_back_rewards()
                data = self.experience(cur_obs, cur_action, returns, next_obs_t, dones,gammas)
                self.add(data)

    def _calc_back_rewards(self):
        gamma = self.gamma
        obs_t, actions_t, reward_t, _, _,_ = self.n_step_buffer.popleft()
        cum_reward = reward_t
        for data in self.n_step_buffer:
            next_rewards = data.reward
            cum_reward = cum_reward+ gamma*next_rewards
            # if (torch.any(data.done)):cum_reward[data.reward.argmax()] += 1.0
            gamma = gamma*self.gamma

        gammas = torch.ones_like(cum_reward)*gamma

        return obs_t, actions_t, cum_reward, gammas

        """if np.any(done) or len(self.n_step_buffer)>=self.n_steps:
            obs, state, actions, all_actions,rewards, next_obs_f, next_state_f, done_f = self.n_step_buffer.pop()
            returns = np.array(rewards)
            while len(self.n_step_buffer) > 0:
                obs, state, actions, all_actions,rewards, next_obs, next_state, done = self.n_step_buffer.pop()
                returns = np.array(rewards) + returns*self.gamma
            data = self.experience(obs, state, actions, all_actions,returns, next_obs_f, next_state_f, done_f)
            self.add(data)
        """

    def sample(self, batch_size):
        """return a random subset of size batch_size of the memory"""
        assert batch_size <= len(self.buffer), "not enough samples"
        idxs = list(self.rng.choice(len(self.buffer), batch_size))
        return self.encode_sample(idxs)
    
    def encode_sample(self,idxes):
        obs, actions, rewards, next_obs, dones,gammas = [], [], [], [],[], []
        for idx in idxes:
            obs.append(self.buffer[idx].obs)
            actions.append(self.buffer[idx].action)
            rewards.append(self.buffer[idx].reward)
            next_obs.append(self.buffer[idx].next_obs)
            dones.append(self.buffer[idx].done)
            gammas.append(self.buffer[idx].gammas)
        # cast to tensors
        obs, actions,rewards, next_obs, dones,gammas = map(
                lambda x: torch.stack(x).to(self.device),
                (obs, actions, rewards, next_obs, dones,gammas)
            )
        # change from batch x actors x ... to actors x batch x ...
        obs, actions, next_obs = map(
                lambda x: x.permute(1,0,2),
                (obs,actions, next_obs)
            )
        rewards, dones,gammas  = map(
                lambda x: x.permute(1,0,2),
                (rewards, dones,gammas)
            )
        self._sampling_results['obs'] = obs
        self._sampling_results['actions'] = actions
        self._sampling_results['rewards'] = rewards
        self._sampling_results['next_obs'] = next_obs
        self._sampling_results['dones'] = dones
        self._sampling_results['gammas'] = gammas

    def initialize(self, iterations, env_reset_fn, env_step_fn, action_size, num_agents, rewards_scale):
        it = 0
        while it < iterations:
            env_info = env_reset_fn()
            states = env_info.vector_observations   
            while True:
                actions = np.random.randn(num_agents, action_size)
                env_info = env_step_fn(actions)
                next_states = env_info.vector_observations
                rewards = rewards_scale*np.array(env_info.rewards)
                dones = env_info.local_done
                self.push(states, actions, rewards, next_states, dones)
                it += 1
                states = next_states
                if np.any(dones):
                    break

class PriorityReplay(Buffer):
    """Prioritized replay buffer.
    
    Agnostic with regards to underlying experiences being stored/sampled. """
    def __init__(self, 
                 PER_alpha, 
                 PER_beta_min, 
                 PER_beta_max, 
                 PER_minimum_priority, 
                 **kwargs):
        """Create Prioritized replay buffer.
        
        parameters:
         PER_alpha: 
            prioritization factor       (CLI `--PER_alpha x.xx`)
         PER_beta_min: 
            initial beta factor         (CLI `--PER_beta_min x.xx`)
         PER_beta_max: 
            final beta factor           (CLI `--PER_beta_max x.xx`)
         PER_minimum_priority: 
            minimum priority for updated indexes
                                        (CLI `--PER_minimum_priority x.xx) """
        # read configuration parameters from arguments or defaults
        self._alpha = PER_alpha
        assert self._alpha >= 0, "negative alpha not allowed"
        self.beta_min = PER_beta_min
        self.beta_max = PER_beta_max
        self.min_priority = PER_minimum_priority
        # intialize parent
        super(PriorityReplay, self).__init__(**kwargs)
        # find minimum power of 2 size for sumtree and mintree and create them
        st_capacity = 1
        while st_capacity < self._maxsize:
            st_capacity *= 2
        self._st_sum = SumSegmentTree(st_capacity)
        self._st_min = MinSegmentTree(st_capacity)
        # initialize internal parameters
        self._beta_decay = 1
        self._beta = self.beta_min
        self._max_priority = 1.0
        self.vf = np.vectorize(self._st_sum.find_prefixsum_idx)
    
    def compute_beta_decay(self,training_iterations=1):
        """calculates the beta decay factor according to total training iterations.

        beta decay = (`PER_beta_max` - `PER_beta_min`)/`training_iterations`
        
        Beta is an annealling factor for randomness. In early training we want to focus on prioritized experiences
        while at the end of training, we should be sampling uniformly."""
        self._beta_decay = (self.beta_max-self.beta_min)/training_iterations

    def decay_beta(self):
        """increases beta used in weights calculation.
        
        beta = max(`PER_beta_max`, beta + (`PER_beta_max` - `PER_beta_min`)/(total training steps to do)) """
        self._beta += self._beta_decay
        self._beta = max(self._beta, self.beta_max)


    def add(self, data):
        """adds experience tuple into underlying buffer and updates sumtree/mintree entry with initial priority."""
        idx = self.next_idx                # obtain next available index to store at from the replay buffer parent class
        super().add(data)        # add to the replay buffer
        self._st_sum[idx] = self._max_priority ** self._alpha   # put it in the sum tree with max priority
        self._st_min[idx] = self._max_priority ** self._alpha   # put it in the min tree with max priority

    def _sample_proportional(self, batch_size):
        """returns list of indexes from uniformly sampling within `batch_size` segments.
        
        args:
            `batch_size`: number of samples to return (available as commandline parameter) """
        results = []
        p_total = self._st_sum.sum(0, len(self.buffer) - 1)       # get total sum of priorites in the whole replay buffer
        bin_interval = p_total / batch_size                      # split the total sum of priorities into batch_size segments
        points_in_bin = np.random.random(size=batch_size)*bin_interval
        offset_per_bin = np.arange(0,batch_size)*bin_interval
        mass = offset_per_bin + points_in_bin
        results = self.vf(mass).tolist()
        return results

    def sample(self, batch_size):
        """ sample a batch of experiences from memory and also returns importance weights and idxes of sampled experiences"""
        idxes = self._sample_proportional(batch_size=batch_size)
        weights = []
        # find maximum weight factor, ie. smallest P(i) since we are dividing by this
        p_sum = self._st_sum.sum()
        p_min = self._st_min.min() / p_sum
        max_weight = (p_min * len(self.buffer)) ** (-self._beta)
        
        for idx in idxes:
            p_sample = self._st_sum[idx] / p_sum
            weight_sample = (p_sample * len(self.buffer)) ** (-self._beta) 
            weights.append(weight_sample / max_weight)
        #expand weights dimension from (batch_size,) to (batch_size,1)
        weights_t = torch.tensor(weights, requires_grad=False,device=self.device).unsqueeze(1)
        
        self.encode_sample(idxes)
        self._sampling_results['weights'] = weights_t
        self._sampling_results['idxes'] = idxes
        return
    @property
    def weights(self):
        return self._sampling_results['weights']
    @property
    def idxes(self):
        return self._sampling_results['idxes']

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to transitions at the sampled idxes denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        
        for idx, priority in zip(idxes, priorities):
            assert priority >= 0, "priority must be greater than zero"
            priority = max(priority, self.min_priority)
            assert 0 <= idx < len(self.buffer)
            self._st_sum[idx] = priority ** self._alpha     # update value and parent values in sum-tree
            self._st_min[idx] = priority ** self._alpha     # update value and parent values in min-tree

            self._max_priority = max(self._max_priority, priority)