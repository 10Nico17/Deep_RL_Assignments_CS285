#### Source of Exercises: Berkeley CS 285: Deep Reinforcement Learning, Decision Making, and Control



## Homework 1:

### Behavioral Cloning

<pre style="font-size: 16px; font-weight: bold;">
    python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Ant.pkl \
    --env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
    --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
    --video_log_freq 1
</pre>

<img src="hw1/images/BC_ant.gif" width="800px">

### DAgger

<pre style="font-size: 16px; font-weight: bold;">
     python cs285/scripts/run_hw1.py \
     -expert_policy_file cs285/policies/experts/Ant.pkl \
     -env_name Ant-v4 --exp_name dagger_ant --n_iter 10 \
     -do_dagger --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
     -video_log_freq 1
</pre>


<img src="hw1/images/dagger_return.png" width="800px">

<img src="hw1/images/dagger.gif" width="800px">

############################################################################################################################################

## Homework 2:

##### Environment1:
<img src="hw2/images/env1.png" alt="Environment 1" width="800px">

##### Reward-to-go:
<img src="hw2/images/reward_to_go.png" alt="Reward to go" width="800px">

##### Baseline-average-reward:
<img src="hw2/images/baseline.png" alt="Policy Gradient" width="800px">

##### Continuous action space:
<img src="hw2/images/continous_action_space.png" alt="Continuous Action Space" width="800px">

#### Implementation:

#### 1. Policy Gradients
<img src="hw2/images/policy.png" alt="Policy" width="800px">

#### sample action from prob. distr. 
<img src="hw2/images/action_distr.png" alt="Policy" width="800px">


-n 1:   Specifies the number of iterations for training. Each iteration consists of a series of actions 
        taken by the agent in the environment, followed by a training step to improve the policy 
        until a terminal state is reached or the maximum number of steps is exceeded.

-b 1:   Batch size, which indicates the number of collected state-action pairs used per iteration. 
        A batch size of 1 means that an update is made after each individual state-action pair. 
        Typically, larger values are used for more stable learning.


## Task1, 100 iterations/episodes: 
#### no critic, no reward-to-go
###### 1. python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole

#### no critic, reward-to-go
###### 2. python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg

#### no critic, no reward-to-go, Normalize advantage
###### 3. python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_na

#### no critic, reward-to-go, Normalize advantage
###### 4. python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na

<img src="hw2/images/overview_cartpole_sb.png" width="800px">

#### no critic, no reward-to-go, batch_size 4000
###### 1. python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 --exp_name cartpole_lb

#### no critic, reward-to-go, batch_size 4000
###### 2. python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg

#### no critic, no reward-to-go, Normalize advantage, batch_size 4000
###### 3. python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -na --exp_name cartpole_lb_na

#### no critic, reward-to-go, Normalize advantage, batch_size 4000
###### 4.python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na

<img src="hw2/images/overview_cartpole_lb.png" width="800px">

<img src="hw2/images/cartPole1.gif" width="800px">


### Task2, Using a Neural Network Baseline

##### Continous Environment2:

<img src="hw2/images/etask2.png" width="800px">

#### no critic use q-values, reward-to-go, batch_size 5000
###### 1. python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --exp_name cheetah

#### with critic NN (Baseline), reward-to-go, batch_size 5000
<img src="hw2/images/actor_critic.png" width="800px">

###### 2. python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline

<img src="hw2/images/overview_HalfCheetah.png" width="800px">

<img src="hw2/images/overview_HalfCheetah_baseline_loss.png" width="800px">


### Task3, Generalized Advantage Estimation

##### Environment3:
<img src="hw2/images/ludar.png" width="800px">

<img src="hw2/images/networks_LunarLander.png" width="800px">

###### python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda <λ> --exp_name lunar_lander_lambda<λ>

<img src="hw2/images/gae_lamba_compare.png" width="800px">

### Task4: Different seeds and hyperparameters

<pre style="font-size: 16px; font-weight: bold;">
for seed in $(seq 1 5); do
    python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 100 --exp_name pendulum_tune_lr0.005_bs2000_s$seed -rtg --use_baseline -na --batch_size 2000 --seed $seed --learning_rate 0.005
done
</pre>

<img src="hw2/images/env_pendelum.png" width="800px">

<img src="hw2/images/pendelum_seeds.png" width="800px">

<pre style="font-size: 16px; font-weight: bold;">
for lr in 0.001 0.005 0.01; do          # Verschiedene Lernraten testen
    for bs in 1000 2000 5000; do        # Verschiedene Batch-Größen testen
        python cs285/scripts/run_hw2.py \
            --env_name InvertedPendulum-v4 \
            -n 100 \
            --exp_name pendulum_tune_lr${lr}_bs${bs} \
            -rtg \
            --use_baseline \
            -na \
            --batch_size $bs \
            --learning_rate $lr
    done
done
</pre>


<img src="hw2/images/pedulum_diff_hyp.png" width="800px">


### Task5

##### Environment4:
<img src="hw2/images/humanoid.png" width="800px">

<img src="hw2/images/networks_humanoid.png" width="800px">

<img src="hw2/images/Humanoid.png" width="800px">

<img src="hw2/images/humanoid.gif" width="800px">


https://www.gymlibrary.dev/environments/mujoco/humanoid/


<pre style="font-size: 16px; font-weight: bold;">
    SSH video rendering: xvfb-run -s "-screen 0 1400x900x24" python cs285/scripts/run_hw2.py --env_name Humanoid-v4 --ep_len 1000 --discount 0.99 -n 1 -l 3 -s 256 -b 50000 -lr 0.001 --baseline_gradient_steps 50 -na --use_reward_to_go --use_baseline --gae_lambda 0.97 --exp_name humanoid --video_log_freq 5
</pre>


<pre style="font-size: 16px; font-weight: bold;">
    tensorboard --logdir=.
</pre>

############################################################################################################################################


## Homework 3:

#### Basic Q-Learning, basic DQN algorithm

<pre style="font-size: 16px; font-weight: bold;">
    python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 1
    python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 2
</pre>

<img src="hw3/images/DQN.png" width="800px">

<img src="hw3/images/critic_target.png" width="800px">

<img src="hw3/images/lunar1.png" width="800px">

#### Double Q-Learning

<img src="hw3/images/double1.png" width="800px">

<img src="hw3/images/code_doubleq.png" width="800px">


<pre style="font-size: 16px; font-weight: bold;">
    python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 1
    python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 2
</pre>


<img src="hw3/images/compare_doubleq_dqn.png" width="800px">

<pre style="font-size: 16px; font-weight: bold;">
    python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/mspacman.yaml    
</pre>

<img src="hw3/images/pacman.png" width="800px">



#### Continuous Actions with Soft-Actor-Critic:

<img src="hw3/images/pendulum.png" width="800px">

<img src="hw3/images/networks.png" width="800px">


### Actor: Class MLP_Policy
Take an observation and output a distribution over actions. Options depending on config file.

<img src="hw3/images/MLPPolicyActor.png" width="800px">


### Critics: Class StateActionCritic

Critic: Q(s,a)≈r+γQ(s′,a′)

Mehrere Critic-Netzwerke können verwendet werden, z. B. für Double Q-Learning oder Redundant Q-Learning
Ein Critic ist ein vollständig verbundenes neuronales Netzwerk (MLP), das die Eingabe (s,a)(s,a) verarbeitet und einen einzigen QQ-Wert ausgibt.

Mechanismen zum Einsatz mehrerer Critic-Netzwerke
- Unabhängige QQ-Netzwerke
- Minimale QQ-Schätzung (Clipped Double Q):
- Mittelwert der QQ-Schätzungen
- REDQ (Redundant Q-Learning)

Jedes Critic-Netzwerk wird separat trainiert und liefert eine Schätzung für Q(s,a)Q(s,a).

Update Critic:

<img src="hw3/images/updatecritic.png" width="800px">


### Target_critics: Class StateActionCritic

Die Target Critic-Netzwerke werden für das Bootstrapping verwendet. Sie liefern stabile Zielwerte für die Bellman-Gleichung.

y=r+γ(1−d)Qtarget​(s′,a′)

Updates:

Soft Update: Die Target-Netzwerke werden langsam an die Critic-Netzwerke angepasst (Polyak averaging):
    θ′←τθ+(1−τ)θ′

Hard Update: Die Target-Netzwerke werden periodisch vollständig auf die Werte der Critic-Netzwerke gesetzt:
    θ′←θ


<img src="hw3/images/updatetarget.png" width="800px">


### Entropy 
In continuous spaces, we have several options for generating exploration noise. One of the most common is providing an entropy bonus to encourage the actor to have high entropy (i.e. to
be “more random”), scaled by a “temperature” coefficient β. 


Testing:

<pre style="font-size: 16px; font-weight: bold;">
    python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/sanity_pendulum.yaml
</pre>


Testing:

<img src="hw3/images/bootstrap1.png" width="800px">

<img src="hw3/images/bootstrap2.png" width="800px">

<img src="hw3/images/bootstrap3.png" width="800px">


Testing entropy: 

<img src="hw3/images/control_entropy.png" width="800px">

Compare entropy if actor trained with actor loss only consists of the entropy bonus (lila) or without (yellow, pink), no maximizing return !


<img src="hw3/images/entropy_plot.png" width="800px">

For this experiment: num_critic_networks=1



python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/sanity_invertedpendulum_reinforce.yaml


python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/sanity_invertedpendulum_reparametrize.yaml


############################################################################################################################################

## Homework4:

#### ENV: conda activate cs285hw3

### Model-based Reinforcement Learning:
- Learn Dynamic Model
- Action Selection via random-shooting optimization and CEM
- On-Policy data collection
- Ensembles

#### Task1:
Collect a large dataset by executing random actions. Train a neural network dynamics model on this fixed
dataset.

<img src="hw4/images/DynamicModel.png" width="800px">

MBRL with ensemble models:

<img src="hw4/images/NNDynamics.png" width="800px">

<pre style="font-size: 16px; font-weight: bold; width: 800px;">
    python cs285/scripts/run_hw4.py -cfg experiments/mpc/halfcheetah_0_iter.yaml
</pre>

<img src="hw4/images/itr_0_loss_curve.png" width="800px">

#### Task2:
Action selection using your learned dynamics model and a given reward function.

Model Predictice Control:
We first collect data with our policy (which starts as random), we
train our model on that collected data, we evaluating the resulting MPC policy (which now uses the trained
model), and repeat.

<img src="hw4/images/action_selection.png" width="800px">


<pre style="font-size: 16px; font-weight: bold; width: 800px;">
python cs285/scripts/run_hw4.py -cfg experiments/mpc/obstacles_1_iter.yaml
</pre>

Average eval return: -25.955617904663086

#### Task3:

MBRL algorithm with on-policy data collection and iterative model training.

<pre style="font-size: 16px; font-weight: bold; width: 800px;">
python cs285/scripts/run_hw4.py -cfg experiments/mpc/obstacles_multi_iter.yaml
python cs285/scripts/run_hw4.py -cfg experiments/mpc/reacher_multi_iter.yaml
python cs285/scripts/run_hw4.py -cfg experiments/mpc/halfcheetah_multi_iter.yaml
</pre>

obstacles_multi_iter: 
    Average eval return:-26.595630645751953

reacher_multi_iter:
    Average eval return: -268.4715969465834

halfcheetah_multi_iter: 
    Average eval return: 421.34788151243447


<img src="hw4/images/evalReturnTask3.png" width="800px">

<img src="hw4/images/dynamamicLoss.png" width="800px">


#### Task4:
Compare the performance of your MBRL algorithm as a function of three hyperparameters: the
number of models in your ensemble, the number of random action sequences considered during each action
selection, and the MPC planning horizon

<pre style="font-size: 16px; font-weight: bold; width: 800px;">
python cs285/scripts/run_hw4.py -cfg experiments/mpc/reacher_ablation.yaml
</pre>

fixed: num_layers: 2, hidden_size: 250, num_iters: 15, initial_batch_size: 5000, batch_size: 800, num_agent_train_steps_per_iter: 1000, num_eval_trajectories: 10, mpc_strategy: random

1. mpc_num_action_sequences: 1000, mpc_horizon: 10, ensemble_size: 3
    Average eval return: -259.5900037229883

2. mpc_num_action_sequences: 1250, mpc_horizon: 10, ensemble_size: 3
    Average eval return: -269.16030872614294

3. mpc_num_action_sequences: 1000, mpc_horizon: 15, ensemble_size: 3
    Average eval return: -279.0613245690755

4. mpc_num_action_sequences: 1000, mpc_horizon: 10, ensemble_size: 6
    Average eval return: -275.9346172796258

5. mpc_num_action_sequences: 500, mpc_horizon: 10, ensemble_size: 3

6. mpc_num_action_sequences: 1000, mpc_horizon: 5, ensemble_size: 3

7. mpc_num_action_sequences: 1000, mpc_horizon: 10, ensemble_size: 1


-> no changes

<img src="hw4/images/task4Compare.png" width="800px">



#### Task5:
You will compare the performance of your MBRL algorithm with action selecting performed by random-
shooting (what you have done up to this point) and CEM

<img src="hw4/images/cem.png" width="800px">


<pre style="font-size: 16px; font-weight: bold; width: 800px;">
python cs285/scripts/run_hw4.py -cfg experiments/mpc/halfcheetah_cem.yaml
</pre>


#### Task6:  Model-Based Policy Optimization (MBPO)

<div style="max-width: 800px;">

In this homework you will also be implementing a variant of MBPO. Another way of leveraging the learned
model is through generating additional samples to train the policy and value functions. Since RL often
requires many environment interaction samples, which can be costly, we can use our learned model to generate
additional samples to improve sample complexity. In MBPO, we build on your SAC implementation from HW3
and use the learned model you implemented in the earlier questions for generating additional samples to train
our SAC agent

Learn Policy:
<img src="hw4/images/SACagent.png" width="800px">

<pre style="font-size: 16px; font-weight: bold; width: 800px;">
python cs285/scripts/run_hw4.py -cfg experiments/mpc/halfcheetah_mbpo.yaml --sac_config_file experiments/sac/halfcheetah_clipq.yaml
</pre>


The different settings (Model-Free SAC, Dyna, MBPO) differ in how and from where the training data originates.



1. (grey) Model-free SAC baseline: no additional rollouts from the learned model.
Model-Free SAC Baseline: SAC agent is trained directly on real interactions with the environment.The replay buffer of the SAC agent only contains real observations, actions and rewards. There is no additional use of learnt models

2. (blue) Dyna (technically “dyna-style” - the original Dyna algorithm is a little different): add single-step rollouts
from the model to the replay buffer and incorporate additional gradient steps per real world step. Here, single-step rollouts are generated with the learnt dynamic model and added to the replay buffer of the SAC agent.
This additional data improves the sampling efficiency as the SAC agent sees not only real but also simulated data.
For each real interaction in the environment, additional gradient steps are performed to further optimise the policy.
The SAC agent optimises its policy based on the mix of data.

<img src="hw4/images/Dyna2.png" width="800px">


3. (pink) MBPO: add in 10-step rollouts from the model to the replay buffer and incorporate additional gradient
steps per real world step: Here, longer, multi-step rollouts (e.g. 10 steps) are generated with the dynamic model and added to the replay buffer. These simulated rollouts provide even more data, which is used to train the SAC agent.
Again, additional gradient steps are performed per real environment interaction. The SAC agent processes both the real and simulated data to learn a better policy. The multi-step rollouts are more challenging as errors in the dynamics model prediction can accumulate over several steps.

<img src="hw4/images/MPO.png" width="800px">

<img src="hw4/images/mbpo1.png" width="800px">

<img src="hw4/images/mbpo2.png" width="800px">


Overview modules:
<img src="hw4/images/offpolicy.png" width="800px">



Results:

<img src="hw4/images/task6.png" width="800px">

############################################################################################################################################
</div>
