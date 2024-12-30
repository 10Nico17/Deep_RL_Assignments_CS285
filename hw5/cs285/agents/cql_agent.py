from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import cs285.infrastructure.pytorch_util as ptu
from cs285.agents.dqn_agent import DQNAgent


class CQLAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        cql_alpha: float,
        cql_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )
        self.cql_alpha = cql_alpha
        self.cql_temperature = cql_temperature

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: bool,
    ) -> Tuple[torch.Tensor, dict, dict]:
        loss, metrics, variables = super().compute_critic_loss(
            obs,
            action,
            reward,
            next_obs,
            done,
        )


        # Berechne den Standard-DQN-Verlust
        loss, metrics, variables = super().compute_critic_loss(
            obs, action, reward, next_obs, done
        )

        # Extrahiere Q-Werte aus den Variablen
        qa_values = variables['qa_values']  # Q-Werte für die aktuellen Aktionen
        q_values = variables['q_values']    # Q-Werte für alle möglichen Aktionen

        #print("q_values shape:", q_values.shape)
        #print("qa_values shape:", qa_values.shape)

        # Konservativer Q-Learning-Term hinzufügen
        # LogSumExp für alle Q-Werte (OOD-Aktionen) berechnen
        random_actions = torch.randint(
            0, self.num_actions, (obs.shape[0], self.num_actions), device=obs.device
        )
        random_q_values = self.critic(obs).gather(1, random_actions)

        #logsumexp_q = torch.logsumexp(q_values / self.cql_temperature, dim=1).mean()
        logsumexp_q = torch.logsumexp(qa_values / self.cql_temperature, dim=1).mean()



        dataset_q = qa_values.mean()

        # Konservativer Verlustterm hinzufügen
        cql_loss = self.cql_alpha * (logsumexp_q - dataset_q)

        # Gesamtverlust berechnen
        loss = loss + cql_loss

        # Metriken für das Logging
        metrics.update({
            "critic_loss/cql_loss": cql_loss.item(),
            "critic_loss/logsumexp_q": logsumexp_q.item(),
            "critic_loss/dataset_q": dataset_q.item(),
        })

        return loss, metrics, variables

