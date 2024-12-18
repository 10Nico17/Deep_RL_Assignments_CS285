from typing import Callable, Optional, Tuple
import numpy as np
import torch.nn as nn
import torch
import gym
from cs285.infrastructure import pytorch_util as ptu


class ModelBasedAgent(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        make_dynamics_model: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[nn.ParameterList], torch.optim.Optimizer],
        ensemble_size: int,
        mpc_horizon: int,
        mpc_strategy: str,
        mpc_num_action_sequences: int,
        cem_num_iters: Optional[int] = None,
        cem_num_elites: Optional[int] = None,
        cem_alpha: Optional[float] = None,
    ):
        super().__init__()
        self.env = env
        self.mpc_horizon = mpc_horizon
        self.mpc_strategy = mpc_strategy
        self.mpc_num_action_sequences = mpc_num_action_sequences
        self.cem_num_iters = cem_num_iters
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha
        self.printed_random_action = False  # Flag-Variable für "Random Action"
        self.printed_cem_action = False  # Flag-Variable für "CEM Action"


        assert mpc_strategy in (
            "random",
            "cem",
        ), f"'{mpc_strategy}' is not a valid MPC strategy"

        # ensure the environment is state-based
        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1

        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        self.ensemble_size = ensemble_size
        self.dynamics_models = nn.ModuleList(
            [
                make_dynamics_model(
                    self.ob_dim,
                    self.ac_dim,
                )
                for _ in range(ensemble_size)
            ]
        )
        self.optimizer = make_optimizer(self.dynamics_models.parameters())
        self.loss_fn = nn.MSELoss()

        # keep track of statistics for both the model input (obs & act) and
        # output (obs delta)
        self.register_buffer(
            "obs_acs_mean", torch.zeros(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_acs_std", torch.ones(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_delta_mean", torch.zeros(self.ob_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_delta_std", torch.ones(self.ob_dim, device=ptu.device)
        )




    def update(self, i: int, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray):
        """
        Update self.dynamics_models[i] using the given batch of data.

        Args:
            i: index of the dynamics model to update
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
            next_obs: (batch_size, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)
        # TODO(student): update self.dynamics_models[i] using the given batch of data
        # HINT: make sure to normalize the NN input (observations and actions)
        # *and* train it with normalized outputs (observation deltas) 
        # HINT 2: make sure to train it with observation *deltas*, not next_obs
        # directly
        # HINT 3: make sure to avoid any risk of dividing by zero when
        # normalizing vectors by adding a small number to the denominator!

        # Schritt 2: Beobachtung und Aktion zusammenführen
        obs_acs = torch.cat([obs, acs], dim=1)    
        # Schritt 3: Eingabe normalisieren
        obs_acs_norm = (obs_acs - self.obs_acs_mean) / (self.obs_acs_std + 1e-8)
        # Schritt 4: Beobachtungsdeltas berechnen
        delta = next_obs - obs
        # Schritt 5: Deltas normalisieren
        delta_norm = (delta - self.obs_delta_mean) / (self.obs_delta_std + 1e-8)
        # Schritt 6: Vorhersage des Modells für die Deltas
        pred_delta_norm = self.dynamics_models[i](obs_acs_norm)
        # Schritt 7: Verlust berechnen
        loss = self.loss_fn(pred_delta_norm, delta_norm)
        # Schritt 8: Optimierung durchführen
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Schritt 9: Verlustwert zurückgeben
        return ptu.to_numpy(loss)


    @torch.no_grad()
    def update_statistics(self, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray):
        """
        Update the statistics used to normalize the inputs and outputs of the dynamics models.

        Args:
            obs: (n, ob_dim)
            acs: (n, ac_dim)
            next_obs: (n, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)
        
        
        # Schritt 2: Beobachtungen und Aktionen kombinieren
        obs_acs = torch.cat([obs, acs], dim=1)
        # Schritt 3: Berechne Beobachtungs-Deltas
        delta = next_obs - obs
        # Schritt 4: Berechne Mittelwerte
        self.obs_acs_mean = obs_acs.mean(dim=0)
        self.obs_delta_mean = delta.mean(dim=0)
        # Schritt 5: Berechne Standardabweichungen
        self.obs_acs_std = obs_acs.std(dim=0) + 1e-8
        self.obs_delta_std = delta.std(dim=0) + 1e-8


    @torch.no_grad()
    def get_dynamics_predictions(
        self, i: int, obs: np.ndarray, acs: np.ndarray
    ) -> np.ndarray:
        """
        Takes a batch of each current observation and action and outputs the
        predicted next observations from self.dynamics_models[i].

        Args:
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
        Returns: (batch_size, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        # TODO(student): get the model's predicted `next_obs`
        # HINT: make sure to *unnormalize* the NN outputs (observation deltas)
        # Same hints as `update` above, avoid nasty divide-by-zero errors when
        # normalizing inputs!
        # Beobachtungen und Aktionen kombinieren und normalisieren

        obs_acs = torch.cat([obs, acs], dim=1)
        obs_acs_norm = (obs_acs - self.obs_acs_mean) / (self.obs_acs_std + 1e-8)
        # Vorhersage des Modells für die normalisierten Deltas
        pred_delta_norm = self.dynamics_models[i](obs_acs_norm)
        # Denormalisierte Deltas berechnen
        pred_delta = pred_delta_norm * self.obs_delta_std + self.obs_delta_mean
        # Vorhersage des nächsten Zustands
        pred_next_obs = obs + pred_delta
        return ptu.to_numpy(pred_next_obs)
    


    def evaluate_action_sequences(self, obs: np.ndarray, action_sequences: np.ndarray):
        """
        Evaluate a batch of action sequences using the ensemble of dynamics models.

        Args:
            obs: starting observation, shape (ob_dim,)
            action_sequences: shape (mpc_num_action_sequences, horizon, ac_dim)
        Returns:
            sum_of_rewards: shape (mpc_num_action_sequences,)
        """
        # We are going to predict (ensemble_size * mpc_num_action_sequences)
        # distinct rollouts, and then average over the ensemble dimension to get
        # the reward for each action sequence.

        # We start by initializing an array to keep track of the reward for each
        # of these rollouts.
        sum_of_rewards = np.zeros(
            (self.ensemble_size, self.mpc_num_action_sequences), dtype=np.float32
        )
        # We need to repeat our starting obs for each of the rollouts.
        obs = np.tile(obs, (self.ensemble_size, self.mpc_num_action_sequences, 1))

        
        
        # Für jeden Zeitschritt der Horizon
        for t in range(self.mpc_horizon):
            acs = action_sequences[:, t, :]  # Aktionen für diesen Schritt
            acs = np.tile(acs, (self.ensemble_size, 1, 1))  # Dupliziere für Ensemble

            # Vorhersage des nächsten Zustands
            next_obs = np.stack([
                self.get_dynamics_predictions(i, obs[i], acs[i])
                for i in range(self.ensemble_size)
            ])

            # Berechne Belohnungen
            rewards, _ = self.env.get_reward(
                next_obs.reshape(-1, self.ob_dim),
                acs.reshape(-1, self.ac_dim)
            )
            rewards = rewards.reshape(self.ensemble_size, self.mpc_num_action_sequences)

            sum_of_rewards += rewards
            obs = next_obs  # Aktuelle Beobachtung aktualisieren    
        
         # now we average over the ensemble dimension
        return sum_of_rewards.mean(axis=0)

    def get_action(self, obs: np.ndarray):
        """
        Choose the best action using model-predictive control.

        Args:
            obs: (ob_dim,)
        """
        # always start with uniformly random actions
        action_sequences = np.random.uniform(
            self.env.action_space.low,
            self.env.action_space.high,
            size=(self.mpc_num_action_sequences, self.mpc_horizon, self.ac_dim),
        )

        if self.mpc_strategy == "random":
            if not self.printed_random_action:  
                print("Random Action")
                self.printed_random_action = True  

            # evaluate each action sequence and return the best one
            rewards = self.evaluate_action_sequences(obs, action_sequences)
            assert rewards.shape == (self.mpc_num_action_sequences,)
            best_index = np.argmax(rewards)
            return action_sequences[best_index][0]
        
        
        elif self.mpc_strategy == "cem":
            if not self.printed_cem_action:  
                print("CEM Action")
                self.printed_cem_action = True  

            elite_mean, elite_std = None, None
            for i in range(self.cem_num_iters):
                pass
                # TODO(student): implement the CEM algorithm
                # HINT: you need a special case for i == 0 to initialize
                # the elite mean and std
        else:
            raise ValueError(f"Invalid MPC strategy '{self.mpc_strategy}'")
