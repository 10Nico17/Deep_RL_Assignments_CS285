from typing import Callable, Optional, Sequence, Tuple, List
import torch
from torch import nn
from cs285.agents.dqn_agent import DQNAgent


class AWACAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        temperature: float,
        **kwargs,
    ):
        super().__init__(observation_shape=observation_shape, num_actions=num_actions, **kwargs)

        self.actor = make_actor(observation_shape, num_actions)
        print("actor: ", self.actor)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.temperature = temperature




    def compute_critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        
        
        with torch.no_grad():
            # Berechne die Verteilung des Actors für die nächsten Beobachtungen
            next_action_distribution = self.actor(next_observations)

            # Extrahiere die Wahrscheinlichkeiten aus der Categorical Distribution
            next_action_probs = next_action_distribution.probs

            # Q-Werte für den nächsten Zustand berechnen
            next_qa_values = self.critic(next_observations)
            next_qs = (next_action_probs * next_qa_values).sum(dim=1)

            # TD-Target berechnen
            target_values = rewards + (1 - dones.float()) * self.discount * next_qs




        # Q-Werte für die aktuellen Aktionen berechnen
        qa_values = self.critic(observations).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Verlust berechnen
        q_values = qa_values
        assert q_values.shape == target_values.shape

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
                "q_values": q_values,
            },
        )
 

    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):

        # Q-Werte für die gegebenen Aktionen
        qa_values = self.critic(observations).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Erwartete Q-Werte als Wertfunktion V(s)
        action_distribution = self.actor(observations)
        action_probs = action_distribution.probs  # Wahrscheinlichkeiten aus der Verteilung extrahieren
        q_values = self.critic(observations)
        values = (action_probs * q_values).sum(dim=1)

        # Advantage berechnen
        advantages = qa_values - values

        return advantages


    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Aktualisiert den Actor basierend auf Advantage-Gewichtung.
        """
        # Advantage berechnen
        advantages = self.compute_advantage(observations, actions)

        # Berechne Log-Wahrscheinlichkeiten der Aktionen
        action_distribution = self.actor(observations)
        log_probs = action_distribution.log_prob(actions)

        # Advantage-basierte Gewichtung
        weights = torch.exp(advantages / self.temperature).detach()
        loss = -(weights * log_probs).mean()

        # Optimierung
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss.item()


    def update(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, dones: torch.Tensor, step: int):
        metrics = super().update(observations, actions, rewards, next_observations, dones, step)

        # Update the actor.
        actor_loss = self.update_actor(observations, actions)
        metrics["actor_loss"] = actor_loss

        return metrics
