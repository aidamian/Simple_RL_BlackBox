import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

"""

Stohastic policy initialized for env with state size 4 and action size 2
Episode 100     Average Score: 38.08
Episode 200     Average Score: 46.19
Episode 300     Average Score: 42.05
Episode 400     Average Score: 48.72
Episode 500     Average Score: 47.77
Episode 600     Average Score: 51.19
Episode 700     Average Score: 52.86
Episode 800     Average Score: 62.09
Episode 900     Average Score: 57.16
Episode 1000    Average Score: 56.75
Deterministic policy initialized for env with state size 4 and action size 2
Episode 100     Average Score: 124.48
Environment solved in 156 episodes!     Average Score: 195.71


   Score  Stohastic    Eps  SimAnn  AdaNoi
0  200.0      False    104    True    True
1  200.0      False    492   False   False
2  200.0       True    608   False   False
3  200.0      False    880    True    True
4  200.0       True   3705    True    True
5   15.0       True  10000    True    True
6   14.0       True  10000   False   False
7  181.0      False  10000   False   False

"""


class Agent():
  """
   softmax regression policy
  """ 
  def __init__(self, s_size=4, a_size=2, stohastic_policy=False):
    # weights for simple linear policy: state_space x action_space
    self.w = 1e-4 * np.random.rand(s_size, a_size)  
    self.stohastic_policy = stohastic_policy
    if self.stohastic_policy:
      self.name = "Stohastic"
    else:
      self.name = "Deterministic"
    print("{} policy initialized for env with state size {} and action size {}".format(
        self.name, s_size, a_size))
      
  def forward(self, state):
    x = np.dot(state, self.w)
    x -= x.max()
    return np.exp(x)/sum(np.exp(x))
  
  def update(self, delta_w):
    assert self.w.shape == delta_w.shape
    self.w += delta_w
  
  def act(self, state):
    probs = self.forward(state)
    if self.stohastic_policy:      
      action = np.random.choice(2, p=probs) # option 1: stochastic policy
    else:
      action = np.argmax(probs)              # option 2: deterministic policy
    return action
      
def discounted_rewards(rewards, gamma):
  """
  Calculates sum(R0 * gamma**0 + R1 * gamma**1 + R2 * gamma**2 + ...)
  """
  discounts = [gamma ** step for step in range(len(rewards))]
  return np.dot(discounts, rewards)

def run_episode(env, agent, max_t=1000):
    rewards = []
    state = env.reset()
    for t in range(max_t):
      action = agent.act(state)
      state, reward, done, _ = env.step(action)
      rewards.append(reward)
      if done:
        break 
    return rewards
  

def hill_climbing(env, agent, n_episodes=10000, max_t=1000, 
                  gamma=1.0, print_every=100, noise_scale=1e-2,
                  simulated_annealing=True, adaptive_noise=True):
  """Implementation of hill climbing with adaptive noise scaling.
      
  Params
  ======
      env: the environment object
      agent: the policy object (must have `w` prop and `act` method)
      n_episodes (int): maximum number of training episodes
      max_t (int): maximum number of timesteps per episode
      gamma (float): discount rate
      print_every (int): how often to print average score (over last 100 episodes)
      noise_scale (float): standard deviation of additive noise
      simulated_annealing: True to decrease noise scaling at good rewards
      adaptive_noise: True to increase noise scaling when bad rewards
  """
  scores_deque = deque(maxlen=100)
  scores = []
  best_R = -np.Inf
  best_w = agent.w
  max_noise_scale = 2
  print("Starting training with:")
  print("  simulated_annealing: {}".format(simulated_annealing))
  print("  adaptive_noise:      {}".format(adaptive_noise))
  for i_episode in range(1, n_episodes+1):
    rewards = run_episode(env=env, agent=agent, max_t=max_t)
    scores_deque.append(sum(rewards))
    scores.append(sum(rewards))

    #discounts = [gamma**i for i in range(len(rewards)+1)]
    #R = sum([a*b for a,b in zip(discounts, rewards)])
    R = discounted_rewards(rewards, gamma)

    if R >= best_R: # found better weights
      best_R = R
      best_w = agent.w
      # now we decrease the noise radius (simulated annealing)
      if simulated_annealing:
        noise_scale = max(1e-3, noise_scale / 2)
      agent.w += noise_scale * np.random.rand(*agent.w.shape) 
    else: # did not find better weights
      if adaptive_noise:
        noise_scale = min(max_noise_scale, noise_scale * 2)
      agent.w = best_w + noise_scale * np.random.rand(*agent.w.shape)

    if i_episode % print_every == 0:
        print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    if np.mean(scores_deque)>=195.0:
        print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_deque)))
        agent.w = best_w
        break
      
  return scores, i_episode
      
      
      
e = gym.make('CartPole-v0')
print('observation space:', e.observation_space)
print('action space:', e.action_space)
res = []
for policy_stohastic in [True, False]:
  for sa in [True, False]:
    for an in [True, False]:
      p = Agent(s_size=4, a_size=2, stohastic_policy=policy_stohastic)
      sc, nr_ep = hill_climbing(env=e, agent=p, 
                                simulated_annealing=sa,
                                adaptive_noise=an)
      res.append((sc, nr_ep, p.name, sa, an, policy_stohastic))

res = sorted(res, key=lambda x:x[1])
  
scores, nr_eps, name, _sa, _an, pol = res[0]
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.title(name+' policy, sa={}  an={}'.format(_sa, _an))
plt.show()
import pandas as pd
df = pd.DataFrame({
      "Score": [x[0][-1] for x in res],
      "Stohastic" :  [x[-1] for x in res],
      "Eps" : [x[1] for x in res],
      "SimAnn" : [x[3] for x in res],
      "AdaNoi" : [x[3] for x in res]      
    })
print(df.sort_values("Eps"))
