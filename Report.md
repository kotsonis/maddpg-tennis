# Tennis Unity environment solution Report
# Udacity Reinforcement Learning Nanodegree Project 3: Collaboration and Competition

This project implements a multi-agent DDPG to learn two agents how to collaborate and play Tennis.

## Environment particularities
<p align="center">
  <img  src="./images/trained_agent_gif.gif">
</p>

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation which consists of the last 3 observations concatenated.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

## Reinforcement Learning and Policy Gradient methods background

Reinforcement learning (RL) is one of the three basic machine learning paradigms, together with supervised learning and unsupervised learning. Whereas both supervised and unsupervised learning are concerned with finding an accurate system to predict an output given inputs, RL focuses on Markov Decision Processes (MDPs) and is concerned with the transition from states to states and the reward/cost associated with these transitions.  
This environment/agent interaction is depicted in below figure:
<p align="center">
  <img  src="images/agent_environment.png">
</p>


Basically, the agent and environment interact in a sequence of discrete time steps t. At each time step t, the agent receives some representation of the environment's state, S<sub>t</sub>&#8712;S, and on that basis selects an action, A<sub>t</sub>&#8712;A(S). At the next timestep t+1, in part as a consequence of its action, the agent receives a scalar reward, R<sub>t+1</sub>&#8712;&#8477;, as well as an new state S<sub>t+1</sub>. The MDP and agent together create a trajectory from an initial time step t over n transitions of states,actions,rewards, and next states as follows:

<img alt="formula" src="https://render.githubusercontent.com/render/math?math=\displaystyle S_t,A_t,R_t,S_{t%2b 1},A_{t%2b1},\cdots,S_{t%2bn},A_{t%2bn},R_{t%2bn}"/>


We represent the sum of rewards accumulated over a trajectory as <img alt="formula" src="https://render.githubusercontent.com/render/math?math=\displaystyle G_t = R_t %2b R_{t%2b1} %2b \cdots %2b R_{t%2bn}"/>. Clearly the limit of G<sub>t</sub> as the trajectory steps n increase is unbounded, so to make sure that we can have a bounded maximum total reward, we discount the rewards from the next transaction by a factor &gamma;&#8712;(0,1] with the case of γ=1 being useful only when a task is episodic, ie with a fixed number of transitions.
In the case that the sets S,R,and A are finite, then the MDP is finite and the following hold:
* Random variables <img alt="formula" src="https://render.githubusercontent.com/render/math?math=\displaystyle R_{t%2B1},S_{t%2b1}"/> have well defined discrete probability distributions **depending only on the preceding state <img alt="formula" src="https://render.githubusercontent.com/render/math?math=\displaystyle S_t"/> & action <img alt="formula" src="https://render.githubusercontent.com/render/math?math=\displaystyle A_t"/>**
* Given a random state s' ∈ S and reward r ∈ R the probability of s' and r occuring at time t given a preceding state s and action a is given by the four argument MDP <strong>dynamics function</strong> <img alt="formula" src="https://render.githubusercontent.com/render/math?math=\displaystyle p:S \times R \times S \times A \to [0,1]"/>
  <img alt="formula" src="https://render.githubusercontent.com/render/math?math=p(s',r|s,a) \doteq \Pr \{ {S_t} = {s'},{R_t} = {r'}|{S_{t - 1}} = s,{A_{t - 1}} = a\} \quad \forall (s',s) \in S,r \in R,a \in A(s)%0A"/>
* Namely, given a state s and an action a, a probability can be assigned to reaching a state s' and receiving reward r, however the sum of these probabilities over all the possible next states and rewards is 1. That is:
  * <img alt="formula" src="https://render.githubusercontent.com/render/math?math=\displaystyle\sum_{s'\in S} {\sum_{r \in R} {p(s',r\space| s,a)}} = 1 \quad \forall s\in S,a \in A(s)%0A"/>

From the <em>dynamics</em> function <em>**p**</em> we can derive other useful functions:
* <em>state transition probabilities</em> : <img alt="formula" src="https://render.githubusercontent.com/render/math?math=\begin{align}%0a\displaystyle p(s'|s,a) = \sum_{r \in R} {p(s',r|s,a)} \quad \quad p:S\times S\times A \to [0,1]\end{align}%0A"/>
* <em>expected rewards</em>: <img alt="formula" src="https://render.githubusercontent.com/render/math?math=\begin{align}%0a\displaystyle r(s,a) = \mathbb{E}[R_t|S_{t-1}=s,A_{t-1}=a] = \sum_{r \in R}\sum_{s'\in S} {p(s',r|s,a)} \quad \quad r:S\times A \to \mathbb{R}\end{align}%0A"/>

We define a policy, ![formula](https://render.githubusercontent.com/render/math?math=\pi(s)), as a function ![formula](https://render.githubusercontent.com/render/math?math=\pi:S\to%20A) that produces an action a given a state s. Thus the expected rewards at a state s can be expressed as ![formula](https://render.githubusercontent.com/render/math?math=r(s,a)=r(s,\pi(s))=\mathbb{E}[R_t|S_{t-1}=s,A_{t-1}=\pi(s)]). This allows us to define the action-value function ![formula](https://render.githubusercontent.com/render/math?math=q_\pi(s,a)), which defines the value of taking action a in state s under the policy π, and continuing the trajectory by taking following the policy π as follows:

[comment]: <> (%2B is plus sign)
[comment]: <> (%0A is line feed, %26 is &)

<img src="https://render.githubusercontent.com/render/math?math=\begin{align*}%0a\displaystyle q_\pi(s,a) %26= \mathbb{E}[G_t | S_t = s, A_t = a] \\%0A%26= \mathbb{E}_\pi\left[\displaystyle\sum_{k=0}^{\infty}{\gamma^kR_{t%2bk%2b1}|S_t=s,A_t=a }\right] \\%0A\end{align*}%0A">


A fundamental property of the action value function is that it satisfies recursive relationships. That is, for any policy ![formula](https://render.githubusercontent.com/render/math?math=\pi) and any state ![formula](https://render.githubusercontent.com/render/math?math=s), the following consistency holds between the action value of ![formula](https://render.githubusercontent.com/render/math?math=(s,a)) and the action value of (![formula](https://render.githubusercontent.com/render/math?math=s_{%2b1},a_{%2b1})):

<img src="https://render.githubusercontent.com/render/math?math=\begin{align*}%0a\displaystyle q_\pi(s,a) %26= \mathbb{E}[R_{t%2b 1} %2b \gamma G_{t%2b 1}|S_t=s,A_t=a] \\%0A%26= \displaystyle\sum_{s'}\sum_rp(s',r|s,a)r%2b \sum_{s'}\sum_rp(s',r|s,a)\gamma(\mathbb{E}[G_{t%2b 1}|S_{t%2b1}=s',A_{t%2b1}=a=\pi(s)]) \\%0A%26=\displaystyle\sum_{s'}\sum_rp(s',r|s,a)[r%2bq_\pi(s',a')]\end{align*}%0A">

The sum of probabilities over all possible next states s' and rewards r reflects the stochastisticy of the system. If we focus on a specific time-step t, then we can reformulate this with bootstrapping as follows:

![Image](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%20q_%5Cpi%28s%2Ca%29%20%26%3D%20r_t&plus;%5Cgamma%20q_%5Cpi%28s%27%2Ca%27%7Ca%27%3D%5Cpi%28s%29%29%5Cimplies%5C%5C%20%5Cdelta_t%20%26%3D%20%5Cdisplaystyle%20%5Cunderbrace%7B%5Coverbrace%7Br_t&plus;%5Cgamma%20q_%5Cpi%28s%27%2Ca%27%29%7D%5E%5Ctext%7Bestimated%20q%7D-q_%5Cpi%28s%2Ca%29%7D_%7B%5Ctext%7BTD%20Error%7D%7D%20%5Cend%7Baligned%7D)

When ![formula](https://render.githubusercontent.com/render/math?math=q_\pi) is approximated as a non-linear function with a neural network with parameters θ, then we denote the action-value function as ![formula](https://render.githubusercontent.com/render/math?math=Q^\pi%28%20s,a|\theta%20%29).
### Q - learning
In Q-learning, we consider that π is an optimal policy, that selects the next action based on the maximum q value of the future state. We learn the parameters of ![formula](https://render.githubusercontent.com/render/math?math=Q^\pi) through gradient descent, trying to fit the above function for a Bellmann error of 0. With a learning rate ![formula](https://render.githubusercontent.com/render/math?math=\alpha) the formula and gradient is thus:
<img src="https://render.githubusercontent.com/render/math?math=\begin{align*}%0a\Delta q_t %26= \text{TD Error}= (r_t%2b\gamma \max_a Q(s_{t%2b1},a , \theta) - Q(s_t,a_t , \theta) q_\pi(s,a) \\%0A \Delta \theta %26= \alpha\Delta q_t \nabla_\theta Q(s,a,\theta) \end{align*}%0A">

### Deep Q Learning Networks
Mnih et al., in their [DQN](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) paper, used deep neural networks as a function approximator for the above Q-Learning.
Specifically, two Q-networks are trained by minimising a sequence of loss functions ![formula](https://render.githubusercontent.com/render/math?math=L_i(\theta_i)) that changes at each iteration i. One Q-network has parameters ![formula](https://render.githubusercontent.com/render/math?math=\theta) and is actively learning, while the second, target, Q-network has parameters ![formula](https://render.githubusercontent.com/render/math?math=\overline{\theta}) and it's parameters are gradually updated with the online's parameters. The loss function is thus:

<img alt="formula" src="https://render.githubusercontent.com/render/math?math=\begin{align*}%0a\displaystyle L_i(\theta_i)%26=\mathbb{E}_{s,a\sim \rho(\cdot)}[(y_i-Q(s,a|\theta_i))^2], \\%0Ay_i %26= \mathbb{E}_{s\prime\sim Env}[r%2b\gamma\displaystyle \max_{a\prime}Q(s',a'|\overline{\theta_i})|s,a]\quad =\text{estimated}\:q\:\text{at iteration}\:i,\\%0A\rho(a,s)%26=probability\: distribution\: over\: sequences\: s\: and\: actions\: a\end{align*}%0A">

### Deep Deterministic Policy Gradient

in the [DDPG](https://arxiv.org/abs/1509.02971) paper, Lillicrap et. al define an actor-critic method to train an agent on a continuous action space.

Finding the optimal policy with Q-learning is not trivial on continuous action spaces, since we would need to optimize the selected action at every timestep, requiring computation that will slow down our algorithm.
Instead, in [policy gradient methods](http://proceedings.mlr.press/v32/silver14.pdf?CFID=6293331&CFTOKEN=eaaee2b6cc8c9889-7610350E-DCAB-7633-E69F572DC210F301), the actor uses a learned policy function approximator <img src="https://render.githubusercontent.com/render/math?math=\mu(s|\theta^\mu):S\gets A "> to select the best action. Learning what action is best given a state is done by maximizing the Q value for the state. Assuming we are sampling actions from a behavior <img src="https://render.githubusercontent.com/render/math?math=\beta=\pi(s,a)"> the objective of the actor becomes:

<img alt="formula" src="https://render.githubusercontent.com/render/math?math=\begin{align*}%0a\displaystyle J_\beta (\mu_\theta) %26= \int_S\rho^\beta (s)Q^\mu(s,\mu_\theta (s))ds \\%0A%26=\mathbb{E}_{s\sim \rho^\beta}[Q^\mu(s,\mu_\theta (s))]\\%0A Loss%26=-\mathbb{E}_{s\sim \rho^\beta}[Q^\mu(s,\mu_\theta (s))]\end{align*}%0A">
And the gradient wrt θ becomes:

<img alt="formula" src="https://render.githubusercontent.com/render/math?math=\begin{align*}%0a\displaystyle \nabla_\theta J_\beta (\mu_\theta) %26\approx \int_S\rho^\beta (s)\nabla_\theta \mu_\theta(a|s)Q^\mu(s,a)ds \\%0A%26=\mathbb{E}_{s\sim \rho^\beta}[\nabla_\theta \mu_\theta (s) \nabla_aQ^\mu(s,a)|_{a=\mu_\theta(s)}]\\%0A\end{align*}%0A">

In the actor-critic setting, critic learns the state value and the state action pair value. This is used to improve the actor (acting as a critic), as depicted in below figure:
<p align="center">
  <img  src="./images/actor_critic_methods.png">
</p>


Using a experience replay buffer and having two separate actor-critic networks (one being the target), the DDPG algorithm is as follows:
![DDPG Algorithm](images/DDPG_algorithm.png)

With the above refresher and definitions, we can move to the presentation of the algorithm implemented.

## Multi Agent Deep Deterministic Policy Gradient Method (MADDPG)
The algorithm implemented for solving the tennis environment is based on the [MADDPG](https://arxiv.org/pdf/1706.02275.pdf) paper by Lowe et al.
MADDPG addresses the situation where agents are either competing or need to collaborate in order to maximize their rewards. For example, in the tennis enviroment, the agents will have to learn how to collaborate, since learning to bounce the ball over the net once will not achieve a high score. Instead, the agent needs to make sure that he will send the ball over the net at a place where the opponent can hit it back.

MADDPG proposes the following methodology for tackling this:
* During execution, an agent selects an action based solely on the observation it receives. This is important as the agent will not be able to use the observations or actions of other agents to decide on it's action
* The agent will learn it's policy using the DDPG method previously. Ie, it will learn to take actions that end in a state of maximal value
* The agent's critic Q, will be producing a value not for the agent's observation/action pair, but will have access to the observations and actions of all agents in the system. Thus the critic will be producing a value for a state (all agent observations concatenated) / actions (all agent actions concatenated) pair
* The agent's critic is used only in training

The above is depicted in below excerpt from the MADDPG paper:
<p align=center><img src="./images/MADDPG_training_vs_execution.png"></p>

The algorithm for MADDPG is given below:
<p><img src="./images/MADDPG_algorithm.png"></p>

### Implementation details
In this project, we made the following modifications to the above algorithm:
1. **Noise Process**: each agent has it's own Ornstein-Uhlenbeck noise generator, which is used for action exploration. The noise contribution to an action is calculated as follows (with the `noise` factor decaying during training):
   ```python
   action = (1-noise)*self.actor(obs) + noise*self.noise.noise()
   ```
2. **Replay buffer**: Instead of sampling stochastically from a replay buffer, we implemented a [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf) solution. To overcome the fact that each agent may have a different TD Error for a given experience, each agent has a separate buffer, from which he samples with priority during his training step
3. **n step returns**: As the rewards to an agent are not immediate, to speed up training, an n-step return calculation was implemented. This required that we also make sure that the agents do not take a learning step after each iteration, but only every n-step iterations (to allow for a feedback of the previously selected action to be available in the buffer)
4. **multiple training steps**: Inspired by REINFORCE, we decided that each time the agents perform a learning iterations, that they sample their buffer and learn multiple times. This allows us to have a smaller batch size, while still covering a considerable amount of experiences in every learning iteration


### Neural network architecture

#### Actor Network
The actor is a function approximator from the observation received by the agent (24 floating point values) to an action (2 floating point values).
We use a deep neural net to approximate this, with the following characteristics:

Below is the summary for the actor network. On the hidden layers we use the `leaky_relu` activation function. On the outputs we use `tanh` to produce actions that are in the desired range -1.0 to 1.0.

| Layer | Type | Input | Output | Activation Fn | Parameters
------------ | ------------- | ------------- | ------------- | ------------- | -------------
input_layer | Fully Connected | obs (24x1) | 128 | `leaky_relu` | 3200 (24x128 + 128 bias)
hidden_layer[0] | Fully Connected | 128 | 64 | `leaky_relu` | 8256 (128x64 + 64 bias)
hidden_layer[1] | Fully Connected | 64 | 64 | `leaky_relu` | 4160 (64x64 + 64 bias)
output_layer | Fully Connected | 64 | 2 | `tanh` | 130 (64x2 + 2 bias)
|||||| **15746 total**

The outputs are then scaled and shifted accordingly, since the horizontal movement ranges from [-1.0, 1.0] while the vertical action ranges from [0.0, 1.0]

#### Critic network
The actor is a function approximator from the current state of the game and all agents actions to a state action value (1 floating point value).
We use a deep neural net to approximate this, with the following characteristics:

Since the observation size is 24 floating point values, the **state**=(obs size) * num_agents = 24*2 = **48**
After running the **state** through a first layer of abstraction (`state_input_layer`), we concatenate all the agent **actions** to produce the input to our `input_layer`

Below is the summary for the critic network. On the hidden layers we use the `leaky_relu` activation function. At the output layer we have no activation funtion.

| Layer | Type | Input | Output | Activation Fn | Parameters
------------ | ------------- | ------------- | ------------- | ------------- | -------------
state_input_layer | Fully Connected | state (48) | 128 | `leaky_relu` | 6272 (48x128 + 128 bias)
input_layer | Fully Connected | 128 + actions (4) | 128 | `leaky_relu` | 17024 (132x128 + 128 bias)
hidden_layer[0] | Fully Connected | 128 | 64 | `leaky_relu` | 8256 (128x64 + 64 bias)
hidden_layer[1] | Fully Connected | 64 | 64 | `leaky_relu` | 4160 (64x64 + 64 bias)
output_layer | Fully Connected | 64 | 1 | **none** | 65 (64x1 + 1 bias)
|||||| **29505 total**

## Training parameters
The following parameters were used for training:
|Parameter | command line flag | Value
|---------- | ----------------- | -----
|**Training**
|iterations |`--training_iterations`|8000
|batch size | `--memory_batch_size`|512
|Network update rate |`--tau`|0.05
|**Noise**
|Starting Noise factor | `--eps_start` |1.0
|Minimum Noise factor | `--eps_minimum` |0.1
|Noise factor decay rate |`--eps_decay` |0.999
|**Prioritized Experience Replay**
|buffer size ||1'000'000
|n steps |`--n_steps`|15
|reward discount rate |`--gamma`|0.995
|α factor (prioritization) for Prioritized Replay Buffer|`--PER_alpha`|0.5
|starting β factor (randomness) for Prioritized Replay Buffer|`--PER_beta_min`|0.5
|ending β factor (randomness) for Prioritized Replay Buffer|`--PER_beta_max`|1.0
|minimum priority to set when updating priorities|`--PER_minimum_priority`|1e-5
|**Actor**
|dimension of layers |`--actor_dnn_dims`|[128,64,64]
|activation fn||`leaky_relu`
|output activation fn||`tanh`
|Optimizer||`Adam`
|Learning rate|`--actor_lr`|0.0001
|**Critic**
|dimension of layers |`--actor_dnn_dims`|[128,64,64]
|activation fn||`leaky_relu`
|output activation fn||None
|Optimizer||`Adam`
|Learning rate|`--critic_lr`|0.0001

## Results - Plot of scores
With the above parameters, the agents were able to solve solve the game (average max score over 100 episodes > 0.5) in 2294 episodes (7781 iterations).


Below is the 100 episode average score per episode, as well as the last episode score per episode.
<p align=center>
<img width=400 src="./images/Agents_performance.png"></p>

During training, the losses for each agent were as follows:
<p align=center>
<img width=400 src="./images/agents_training_losses.png"></p>

