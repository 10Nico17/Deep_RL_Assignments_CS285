from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np
import random

import cs285.infrastructure.pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int], num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()


    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
        observation = ptu.from_numpy(np.asarray(observation))[None]

        if random.random() < epsilon:  # Exploration
            action = torch.randint(0, self.num_actions, (1,))
        else:  # Exploitation
            critic_values = self.critic(observation)
            action = torch.argmax(critic_values, dim=1)

        return ptu.to_numpy(action).squeeze(0).item()




    """
    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict, dict]:

        # TODO(student): paste in your code from HW3, and make sure the return values exist
        raise NotImplementedError
        with torch.no_grad():
            next_qa_values = ...

            if self.use_double_q:
                next_action = ...
            else:
                next_action = ...

            next_q_values = ...
            assert next_q_values.shape == (batch_size,), next_q_values.shape

            target_values = ...
            assert target_values.shape == (batch_size,), target_values.shape

        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": q_values,
            },
        )
    """



    """
    Compute the loss for the DQN critic.
    Returns:
     - loss: torch.Tensor, the MSE loss for the critic
     - metrics: dict, a dictionary of metrics to log
     - variables: dict, a dictionary of variables that can be used in subsequent calculations
    """

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict, dict]:
        """
        Berechnet den Verlust für den DQN-Critic.
        """
        (batch_size,) = reward.shape

        with torch.no_grad():
            # Berechne Q-Werte für den nächsten Zustand
            next_qa_values = self.target_critic(next_obs)

            if self.use_double_q:
                # Double DQN: Auswahl der Aktion mit dem Online-Critic
                next_action = torch.argmax(self.critic(next_obs), dim=1).unsqueeze(dim=1)
            else:
                # Standard DQN: Auswahl der besten Aktion direkt aus dem Target-Critic
                next_action = torch.argmax(next_qa_values, dim=1).unsqueeze(dim=1)

            # Wähle die Q-Werte für die ausgewählten Aktionen
            next_q_values = next_qa_values.gather(dim=1, index=next_action).squeeze(1)

            # Berechne Zielwerte
            target_values = reward + (1 - done.float()) * self.discount * next_q_values

        # Berechne die Q-Werte für die aktuellen Aktionen
        qa_values = self.critic(obs)        
        q_values = qa_values.gather(dim=1, index=action.unsqueeze(dim=1)).squeeze(1)

        # Berechne den Verlust
        loss = self.critic_loss(q_values, target_values)

        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": next_q_values,
            },
        )

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        loss, metrics, _ = self.compute_critic_loss(obs, action, reward, next_obs, done)

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        metrics["grad_norm"] = grad_norm.item()
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return metrics

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())



    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # Update the critic and get training statistics
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)

        # Periodically update the target critic
        if step % self.target_update_period == 0:
            self.update_target_critic()

        return critic_stats