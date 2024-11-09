from typing import Optional, Sequence
import numpy as np
import torch

from cs285.networks.policies import MLPPolicyPG
from cs285.networks.critics import ValueCritic
from cs285.infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        print('self.actor (policy): ', self.actor)

        # create the critic (baseline) network, if needed
        if use_baseline:
            print('create critic (baseline)')
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            print('no critic (baseline) used')
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """
        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # TODO: flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.
        obs = np.concatenate(obs, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        terminals = np.concatenate(terminals, axis=0)
        q_values = np.concatenate(q_values, axis=0)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )
        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # TODO: update the PG actor/policy network once using the advantages
        #info: dict = None
        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        info = self.actor.update(obs, actions, advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            print("Update the critic")
            critic_info = {}
            for _ in range(self.baseline_gradient_steps):
                # Führe das Update durch und sammle die Rückgabewerte für jedes Schritt
                step_info = self.critic.update(obs, q_values)

                # Aktualisiere die gesammelten Informationen
                for key, value in step_info.items():
                    if key not in critic_info:
                        critic_info[key] = []
                    critic_info[key].append(value)

            # Füge die durchschnittlichen Werte über alle Schritte zum `info`-Dictionary hinzu
            for key, values in critic_info.items():
                info[f"Critic {key}"] = np.mean(values)

        return info


    """
    Trajectory-Based Policy Gradient (Gesamtsumme aller Belohnungen für die gesamte Trajektorie)
    Reward-to-Go Policy Gradient (Nur zukünftige Belohnungen ab dem aktuellen Zeitschritt werden berücksichtigt)      
    """

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""
        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # TODO: use the helper function self._discounted_return to calculate the Q-values
            print('No use_reward_to_go')
            q_values = [self._discounted_return(reward_traj) for reward_traj in rewards]
            #for i, q_value in enumerate(q_values):
            #    print(f'q_values[{i}] shape:', q_value.shape)
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # TODO: use the helper function self._discounted_reward_to_go to calculate the Q-values
            print('Use_reward_to_go')
            q_values = [self._discounted_reward_to_go(reward_traj) for reward_traj in rewards]
            #for i, q_value in enumerate(q_values):
            #    print(f'q_values[{i}] shape:', q_value.shape)

        return q_values


    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.
        Operates on flat 1D NumPy arrays.
        """

        if self.critic is None:
            # TODO: if no baseline, then what are the advantages?
            print('no baseline use q-values')
            advantages = q_values.copy() 
            #print('################## advantages1: ', advantages.shape)
        else:
            
            # Falls ein Kritiker vorhanden ist, berechne die Wertschätzungen für obs und ziehe diese von den Q-Werten ab
            print('Use neural network as critic')
            obs_tensor = ptu.from_numpy(obs)  # Conversion to torch.Tensor
            values = self.critic(obs_tensor).squeeze() 
           
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                print('No gae_lambda used, only critic NN')
                # TODO: if using a baseline, but not GAE, what are the advantages?
                #advantages = q_values - values
                advantages = q_values - values.detach().cpu().numpy()
                #print('################## advantages1: ', advantages.shape)

            else:
                print('GAE_lambda used')                
                batch_size = obs.shape[0]
                # Add a dummy T+1 value to values for easier recursive calculation
                values = np.append(values.detach().cpu().numpy(), [0])  # Detach, transfer to CPU, then convert to NumPy
                advantages = np.zeros(batch_size + 1)  # Include dummy advantage value at the end for computation

                # Loop through the batch in reverse to compute GAE advantages
                for i in reversed(range(batch_size)):
                    # Calculate delta for time step i
                    delta = rewards[i] + self.gamma * values[i + 1] * (1 - terminals[i]) - values[i]

                    # Recursive GAE formula with λ to compute advantage estimates
                    advantages[i] = delta + self.gamma * self.gae_lambda * (1 - terminals[i]) * advantages[i + 1]

                # Remove the dummy advantage value at the end
                advantages = advantages[:-1]

        # TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            print("Normalize advantage")
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages
    



    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        T = len(rewards)
        discounts = np.array([self.gamma ** t for t in range(T)])
        total_return = np.sum(rewards * discounts)  
        return [total_return] * T


    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        T = len(rewards)
        discounted_rewards = np.zeros(T)
        for t in range(T):
            discounts = np.array([self.gamma ** (t_prime - t) for t_prime in range(t, T)])
            discounted_rewards[t] = np.sum(rewards[t:] * discounts)
        return discounted_rewards
