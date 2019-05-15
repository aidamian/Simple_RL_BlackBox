# Simple Black Box methods for searching for Reinforcement Learning policies

### Hill Ascent

Grid search results on `CartPole-v0`

```
   Score  Stohastic    Eps  SimAnn  AdaNoi
0  200.0      False    104    True    True
1  200.0      False    492   False   False
2  200.0       True    608   False   False
3  200.0      False    880    True    True
4  200.0       True   3705    True    True
5   15.0       True  10000    True    True
6   14.0       True  10000   False   False
7  181.0      False  10000   False   False
```
Scores for model 0 (best)

![Hill climb](https://github.com/andreidi/Simple_RL_BlackBox/blob/master/deterministic_hil_climbing.png)


### Cross Entropy (random) Method

Solved `MountainCarContinuous-v0` in 147 episodes

```
Episode 139     Average Score: 89.25
Episode 140     Average Score: 89.48
Episode 141     Average Score: 89.57
Episode 142     Average Score: 89.61
Episode 143     Average Score: 89.60
Episode 144     Average Score: 89.64
Episode 145     Average Score: 89.66
Episode 146     Average Score: 89.66
Episode 147     Average Score: 90.83

```