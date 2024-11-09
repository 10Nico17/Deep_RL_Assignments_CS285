Source of Exercises: Berkeley CS 285: Deep Reinforcement Learning, Decision Making, and Control

### Homework 2:


##### Environment:
<img src="hw2/images/env1.png" alt="Environment 1" width="600px">

##### Reward-to-go:
<img src="hw2/images/reward_to_go.png" alt="Reward to go" width="600px">

##### Baseline-average-reward:
<img src="hw2/images/baseline.png" alt="Policy Gradient" width="600px">

##### Continuous action space:
<img src="hw2/images/continous_action_space.png" alt="Continuous Action Space" width="600px">

#### Implementation:

#### Policy
<img src="hw2/images/policy.png" alt="Policy" width="600px">

#### create action distribution
<img src="hw2/images/action_distr.png" alt="Policy" width="600px">


-n 1:   Specifies the number of iterations for training. Each iteration consists of a series of actions 
        taken by the agent in the environment, followed by a training step to improve the policy 
        until a terminal state is reached or the maximum number of steps is exceeded.

-b 1:   Batch size, which indicates the number of collected state-action pairs used per iteration. 
        A batch size of 1 means that an update is made after each individual state-action pair. 
        Typically, larger values are used for more stable learning.

#### Task1, 100 iterations/episodes: 
#### no critic use q-values, no reward-to-go
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name experiment1

<img src="hw2/images/experiment1.png" alt="Continuous Action Space" width="600px">

#### no critic use q-values, reward-to-go
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name experiment2

<img src="hw2/images/experiment2.png" width="600px">


#### no critic use q-values, no reward-to-go, Normalize advantage
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name experiment3

<img src="hw2/images/experiment3.png" width="600px">

#### no critic use q-values, reward-to-go, Normalize advantage
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name experiment4

<img src="hw2/images/experiment4.png" width="600px">



#### no critic use q-values, no reward-to-go, batch_size 4000
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 --exp_name experiment5

<img src="hw2/images/experiment5.png" width="600px">


#### no critic use q-values, reward-to-go, batch_size 4000
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 --exp_name experiment6

<img src="hw2/images/experiment6.png" width="600px">


#### no critic use q-values, no reward-to-go, Normalize advantage, batch_size 4000
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -na --exp_name experiment7

<img src="hw2/images/experiment7.png" width="600px">


#### no critic use q-values, reward-to-go, Normalize advantage, batch_size 4000
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg -na --exp_name experiment8

<img src="hw2/images/experiment8.png" width="600px">


