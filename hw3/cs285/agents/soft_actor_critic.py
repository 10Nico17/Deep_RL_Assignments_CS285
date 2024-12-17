from typing import Callable, Optional, Sequence, Tuple
import copy

import torch
from torch import nn
import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class SoftActorCritic(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,


        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_actor_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],


        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        make_critic_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: Optional[int] = None,
        soft_target_update_rate: Optional[float] = None,
        # Actor-critic configuration
        actor_gradient_type: str = "reinforce",  # One of "reinforce" or "reparametrize"
        num_actor_samples: int = 1,
        num_critic_updates: int = 1,
        # Settings for multiple critics
        num_critic_networks: int = 1,
        target_critic_backup_type: str = "mean",  # One of "doubleq", "min", "redq", or "mean"
        # Soft actor-critic
        use_entropy_bonus: bool = False,
        temperature: float = 0.0,
        backup_entropy: bool = True,
    ):
        super().__init__()

        assert target_critic_backup_type in [
            "doubleq",
            "min",
            "mean",
            "redq",
        ], f"{target_critic_backup_type} is not a valid target critic backup type"

        assert actor_gradient_type in [
            "reinforce",
            "reparametrize",
        ], f"{actor_gradient_type} is not a valid type of actor gradient update"

        assert (
            target_update_period is not None or soft_target_update_rate is not None
        ), "Must specify either target_update_period or soft_target_update_rate"



        self.actor = make_actor(observation_shape, action_dim)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.actor_lr_scheduler = make_actor_schedule(self.actor_optimizer)



        self.critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )

        self.critic_optimizer = make_critic_optimizer(self.critics.parameters())
        self.critic_lr_scheduler = make_critic_schedule(self.critic_optimizer)
        self.target_critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )
        self.update_target_critic()

        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.discount = discount
        self.target_update_period = target_update_period
        self.target_critic_backup_type = target_critic_backup_type
        self.num_critic_networks = num_critic_networks
        self.use_entropy_bonus = use_entropy_bonus
        self.temperature = temperature
        self.actor_gradient_type = actor_gradient_type
        self.num_actor_samples = num_actor_samples
        self.num_critic_updates = num_critic_updates
        self.soft_target_update_rate = soft_target_update_rate
        self.backup_entropy = backup_entropy

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

        self.first_update = True                     # Attribut für den ersten Aufruf
        self.first_update_critic = True              # Attribut für den ersten Aufruf
        self.first_target_critic_backup_type = True  # Attribut für den ersten Aufruf
        self.first_entropy = True                    # Attribut für den ersten Aufruf
        self.first_actor_gradient_type = True

        self.train_actor_entropy_bonus_only = False    # only train entropy

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the action for a given observation.
        """
        with torch.no_grad():
            observation = ptu.from_numpy(observation)[None]
            """
            Der Actor gibt eine Wahrscheinlichkeitsverteilung zurück, 
            die das Modell für die Aktionen in diesem Zustand gelernt hat. 

            action_distribution: Der Name der Variablen.
            torch.distributions.Distribution: Der erwartete Datentyp dieser Variablen.
            self.actor(observation): Der Wert, der der Variablen zugewiesen wird.
            """

            action_distribution: torch.distributions.Distribution = self.actor(observation)
            """
            Eine Aktion wird aus der Wahrscheinlichkeitsverteilung gesampelt, 
            die der Actor berechnet hat. Dies sorgt dafür, dass der Agent weiterhin explorativ bleibt.
            """    

            action: torch.Tensor = action_distribution.sample()
            assert action.shape == (1, self.action_dim), action.shape
            return ptu.to_numpy(action).squeeze(0)




    def critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) Q-values for the given state-action pair.
        """
        return torch.stack([critic(obs, action) for critic in self.critics], dim=0)

    def target_critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) target Q-values for the given state-action pair.
        """
        return torch.stack(
            [critic(obs, action) for critic in self.target_critics], dim=0
        )

    def q_backup_strategy(self, next_qs: torch.Tensor) -> torch.Tensor:
        """
        Handle Q-values from multiple different target critic networks to produce target values.

        For example:
         - for "vanilla", we can just leave the Q-values as-is (we only have one critic).
         - for double-Q, swap the critics' predictions (so each uses the other as the target).
         - for clip-Q, clip to the minimum of the two critics' predictions.

        Parameters:
            next_qs (torch.Tensor): Q-values of shape (num_critics, batch_size). 
                Leading dimension corresponds to target values FROM the different critics.
        Returns:
            torch.Tensor: Target values of shape (num_critics, batch_size). 
                Leading dimension corresponds to target values FOR the different critics.
        """
        if self.first_target_critic_backup_type:
            print("target_critic_backup_type: ", self.target_critic_backup_type)
            self.first_target_critic_backup_type = False 



        assert (
            next_qs.ndim == 2
        ), f"next_qs should have shape (num_critics, batch_size) but got {next_qs.shape}"
        num_critic_networks, batch_size = next_qs.shape
        assert num_critic_networks == self.num_critic_networks
        # print("next_qs: ", next_qs.shape)
        # TODO(student): Implement the different backup strategies.
        
        
        if self.target_critic_backup_type == "doubleq":
            next_qs = next_qs.flip(0)
        elif self.target_critic_backup_type == "min":
            next_qs, _ = torch.min(next_qs, dim=0)
        elif self.target_critic_backup_type == "mean":
            next_qs = torch.mean(next_qs, dim=0)
        else:
            pass


        # If our backup strategy removed a dimension, add it back in explicitly
        # (assume the target for each critic will be the same)
        # print("next_qs: ", type(next_qs))
        # print(next_qs)
        if next_qs.shape == (batch_size,):
            next_qs = next_qs[None].expand((self.num_critic_networks, batch_size)).contiguous()
        # print("num_critic_networks: ", self.num_critic_networks)
        # print("batch_size: ", batch_size)
        assert next_qs.shape == (
            self.num_critic_networks,
            batch_size,
        ), next_qs.shape
        return next_qs



    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ):
        """
        Update the critic networks by computing target values and minimizing Bellman error.
        """

        if self.first_update_critic:
            print("First call to update critic!")
            self.first_update_critic = False 


        (batch_size,) = reward.shape

        # Compute target values
        # Important: we don't need gradients for target values!
        with torch.no_grad():
            # TODO(student)
            
            # Sample from the actor for next state according to policy
            next_action_distribution: torch.distributions.Distribution = self.actor(obs=next_obs)
            next_action = next_action_distribution.sample()
            # Compute the next Q-values for the sampled actions
            next_qs = self.target_critic(obs=next_obs, action=next_action)


            ## if we use multiple target and critic networks
            # Handle Q-values from multiple different target critic networks (if necessary)
            # (For double-Q, clip-Q, etc.)
            # print("here: next_qs: ", next_qs.shape)

            # Check if values changes if we only use one network values must stay the same
            #next_qs_before = next_qs.clone()            
            next_qs = self.q_backup_strategy(next_qs)            
            # Logge die Differenz
            #print(f"Differenz in next_qs: {torch.sum(next_qs - next_qs_before).item()}")

            assert next_qs.shape == (
                self.num_critic_networks,
                batch_size,
            ), next_qs.shape



            ## if we use extra entropy
            if self.use_entropy_bonus and self.backup_entropy:
                # TODO(student): Add entropy bonus to the target values for SAC
                next_action_entropy = self.entropy(next_action_distribution)
                
                next_action_entropy = next_action_entropy[None].expand((self.num_critic_networks, batch_size)).contiguous()
                assert next_action_entropy.shape == next_qs.shape, next_action_entropy.shape
                next_qs -= self.temperature * next_action_entropy


            # Compute the target Q-value
            target_values: torch.Tensor = reward[None] + self.discount * (1 - 1.0 * (done[None])) * next_qs
            assert target_values.shape == (
                self.num_critic_networks,
                batch_size
            ), target_values.shape



        # TODO(student): Update the critic
        # Predict Q-values
        q_values = self.critic(obs=obs, action=action)
        assert q_values.shape == (self.num_critic_networks, batch_size), q_values.shape

        # Compute loss
        loss: torch.Tensor = self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
        }

    def entropy(self, action_distribution: torch.distributions.Distribution):
        """
        Compute the (approximate) entropy of the action distribution for each batch element.
        """

        if self.first_entropy:
            print("First call entropy!")
            self.first_entropy = False 

        # TODO(student): Compute the entropy of the action distribution.
        # Note: Think about whether to use .rsample() or .sample() here...
        sampled_action = action_distribution.rsample()
        entropy = -action_distribution.log_prob(sampled_action)
        return entropy



    """
    def actor_loss_reinforce(self, obs: torch.Tensor):
        batch_size = obs.shape[0]

        # TODO(student): Generate an action distribution
        action_distribution: torch.distributions.Distribution = self.actor(obs)
        # print(action_distribution.batch_shape)
        with torch.no_grad():
            # TODO(student): draw num_actor_samples samples from the action distribution for each batch element
            action = action_distribution.sample(sample_shape=(self.num_actor_samples,))
            assert action.shape == (
                self.num_actor_samples,
                batch_size,
                self.action_dim,
            ), action.shape

            # TODO(student): Compute Q-values for the current state-action pair
            q_values = self.critic(obs[None].expand(self.num_actor_samples, -1, -1), action)
            # print("obs: ", obs.shape)
            # print("action: ", action.shape)
            # print("q_values: ", q_values.shape)
            assert q_values.shape == (
                self.num_critic_networks,
                self.num_actor_samples,
                batch_size,
            ), q_values.shape

            # Our best guess of the Q-values is the mean of the ensemble
            q_values = torch.mean(q_values, dim=0)
            advantage = q_values

        # Do REINFORCE: calculate log-probs and use the Q-values
        # TODO(student)
        log_probs = action_distribution.log_prob(action)
        torch.nan_to_num_(log_probs, nan=0.0, posinf=0.0, neginf=0.0)
        # print("advantage: ", advantage.shape)
        # print("log_probs: ", log_probs.shape)
        loss = -torch.mean(log_probs * advantage)
        # print("entropy: ", self.entropy(action_distribution).shape)

        return loss, torch.mean(self.entropy(action_distribution))
    """
    

    
    def actor_loss_reinforce(self, obs: torch.Tensor):
        batch_size = obs.shape[0]

        # Generate an action distribution from the actor
        action_distribution: torch.distributions.Distribution = self.actor(obs)

        with torch.no_grad():
            # Draw num_actor_samples samples from the action distribution for each batch element
            action = action_distribution.sample((self.num_actor_samples,))  # Shape: (num_actor_samples, batch_size, action_dim)
            assert action.shape == (
                self.num_actor_samples,
                batch_size,
                self.action_dim,
            ), f"Action shape mismatch: {action.shape}"

            # Compute Q-values for the current state-action pair
            q_values = self.critic(
                obs.repeat(self.num_actor_samples, 1, 1).reshape(-1, *obs.shape[1:]),
                action.view(-1, self.action_dim),
            ).view(self.num_critic_networks, self.num_actor_samples, batch_size)
            
            
            
            assert q_values.shape == (
                self.num_critic_networks,
                self.num_actor_samples,
                batch_size,
            ), f"Q-values shape mismatch: {q_values.shape}"

            # Our best guess of the Q-values is the mean of the ensemble
            q_values = torch.mean(q_values, axis=0)  # Shape: (num_actor_samples, batch_size)
            advantage = q_values.mean(dim=0)  # Shape: (batch_size,)

        # Calculate log-probs and ensure shapes are correct
        log_probs = action_distribution.log_prob(action)  # Shape: (num_actor_samples, batch_size)
        if log_probs.ndim == 2:  # If action_dim is automatically summed
            assert log_probs.shape == (
                self.num_actor_samples,
                batch_size,
            ), f"log_probs shape mismatch: {log_probs.shape}"
        else:  # Manually sum over action_dim
            log_probs = log_probs.sum(-1)  # Shape: (num_actor_samples, batch_size)
            assert log_probs.shape == (
                self.num_actor_samples,
                batch_size,
            ), f"log_probs shape mismatch after summing: {log_probs.shape}"

        # Compute REINFORCE loss as negative weighted log probabilities
        loss = -(log_probs.mean(dim=0) * advantage).mean()

        return loss, torch.mean(self.entropy(action_distribution))










    def actor_loss_reparametrize(self, obs: torch.Tensor):
        batch_size = obs.shape[0]

        # Sample from the actor
        action_distribution: torch.distributions.Distribution = self.actor(obs)

        # TODO(student): Sample actions
        # Note: Think about whether to use .rsample() or .sample() here...
        action = action_distribution.rsample(sample_shape=(self.num_actor_samples,))

        # TODO(student): Compute Q-values for the sampled state-action pair
        q_values = self.critic(obs[None].repeat((self.num_actor_samples, 1, 1)), action)

        # TODO(student): Compute the actor loss
        loss = -torch.mean(q_values)

        return loss, torch.mean(self.entropy(action_distribution))





    def update_actor(self, obs: torch.Tensor):
        """
        Update the actor by one gradient step using either REPARAMETRIZE or REINFORCE.
        """
        # print("actor_gradient_type: ", self.actor_gradient_type)
        if self.first_actor_gradient_type:
            print("actor_gradient_type: ", self.actor_gradient_type)
            print("use_entropy_bonus: ", self.use_entropy_bonus)
            print("self.temperature: ", self.temperature)
            print("self.train_actor_entropy_bonus_only_to_test_entropy:: ", self.train_actor_entropy_bonus_only)
            self.first_actor_gradient_type = False 


        if self.train_actor_entropy_bonus_only:
            # Berechne die Entropie (ohne andere Optimierungsziele wie Q-Werte oder Rewards)
            action_distribution: torch.distributions.Distribution = self.actor(obs)
            entropy = self.entropy(action_distribution)
            if self.use_entropy_bonus:
                loss = -self.temperature * entropy.mean()

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            return {"actor_loss": loss.item(), "entropy": entropy.mean().item()}

        else:    
            if self.actor_gradient_type == "reparametrize":
                loss, entropy = self.actor_loss_reparametrize(obs)
            elif self.actor_gradient_type == "reinforce":
                loss, entropy = self.actor_loss_reinforce(obs)
            
            if self.use_entropy_bonus:
                loss -= self.temperature * entropy    

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            return {"actor_loss": loss.item(), "entropy": entropy.item()}       

    def update_target_critic(self):
        self.soft_update_target_critic(1.0)

    def soft_update_target_critic(self, tau):
        for target_critic, critic in zip(self.target_critics, self.critics):
            for target_param, param in zip(
                target_critic.parameters(), critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
        ):
        """
        Update the actor and critic networks.
        """

        if self.first_update:
            print("First call to update!")
            self.first_update = False 
    

        critic_infos = []
        # TODO(student): Update the critic for num_critic_upates steps, and add the output stats to critic_infos
        critic_infos = [self.update_critic(obs=observations, action=actions, reward=rewards, next_obs=next_observations, done=dones) for _ in range(self.num_critic_updates)]
        
        
        
        # TODO(student): Update the actor
        actor_info = self.update_actor(obs=observations)

        
        
        
        # TODO(student): Perform either hard or soft target updates.
        # Relevant variables:
        #  - step
        #  - self.target_update_period (None when using soft updates)
        #  - self.soft_target_update_rate (None when using hard updates)
        if self.target_update_period is not None and step % self.target_update_period == 0:
            self.update_target_critic()
        elif self.soft_target_update_rate is not None:
            self.soft_update_target_critic(self.soft_target_update_rate)
        # Average the critic info over all of the steps
        critic_info = {
            k: np.mean([info[k] for info in critic_infos]) for k in critic_infos[0]
        }

        # Deal with LR scheduling
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

        return {
            **actor_info,
            **critic_info,
            "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
            "critic_lr": self.critic_lr_scheduler.get_last_lr()[0],
        }