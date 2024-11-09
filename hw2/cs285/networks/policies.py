import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:            
            """
            Wenn der Aktionsraum diskret ist, erzeugt das Netzwerk Wahrscheinlichkeiten (Logits) für jede mögliche Aktion:
            self.logits_net: Ein neuronales Netzwerk, das Logits (unskalierte Wahrscheinlichkeiten) für jede mögliche Aktion ausgibt. 
            Die Anzahl der Ausgaben (output_size) entspricht der Anzahl der möglichen Aktionen (ac_dim).
            parameters = self.logits_net.parameters(): Alle Parameter dieses Netzwerks werden dem Optimierer zur Verfügung gestellt, 
            damit sie während des Trainings angepasst werden können.
            """
            print('discrete')
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
       
        else:
            """
            Wenn der Aktionsraum kontinuierlich ist, erzeugt das Netzwerk eine Normalverteilung für jede Aktionsdimension:
            self.mean_net: Ein neuronales Netzwerk, das für jede Aktionsdimension den Mittelwert (mean) ausgibt.
            self.logstd: Ein trainierbarer Parameter (nn.Parameter), der die logarithmierte Standardabweichung (logstd) 
            für die Aktionsverteilung repräsentiert. Dieser Wert wird exponentiiert, um die Standardabweichung (std) zu berechnen.    
            """
            print('continous')
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete


    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        
        # TODO: implement get_action
        if isinstance(obs, tuple):
            obs = obs[0]

        obs_tensor = ptu.from_numpy(obs).unsqueeze(0)  
        action_dist = self.forward(obs_tensor)
    
        if self.discrete:
            action = action_dist.sample().item()  
        else:
            action = action_dist.sample().cpu().numpy().flatten()  

        return action

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        #print("forward")

        if self.discrete:
            # Berechne die Logits für jede mögliche Aktion
            logits = self.logits_net(obs)
            # Erstelle eine diskrete Wahrscheinlichkeitsverteilung basierend auf den Logits
            action_dist = distributions.Categorical(logits=logits)
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd)
            action_dist = distributions.Normal(mean, std)
        return action_dist

        
        
        '''
        else:
            # Berechne die Mittelwerte für jede Aktionsdimension
            mean = self.mean_net(obs)
            # Erstelle die Standardabweichung (aus logstd) für die Normalverteilung
            std = torch.exp(self.logstd)
            # Erstelle eine Normalverteilung basierend auf Mittelwert und Standardabweichung
            action_dist = distributions.Normal(mean, std)   
        return action_dist
        '''
    



    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError



class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        
        #print("update policies")
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        optimizer = self.optimizer
        optimizer.zero_grad()


        # Der Verlust L(θ) für die Policy-Gradient-Methode:
        # 
        # L(θ) = - (1 / N) * Σ_t [log π_θ(a_t | s_t) * A_t]
        #
        # wobei:
        # - θ die Parameter der Policy darstellt (die Gewichte des neuronalen Netzwerks),
        # - N die Anzahl der gesammelten Datenpunkte (oder Trajektorien) ist,
        # - log π_θ(a_t | s_t) die Log-Wahrscheinlichkeit der gewählten Aktion a_t im Zustand s_t ist,
        # - A_t der Vorteil (Advantage) ist, der für den Zustand s_t und die Aktion a_t berechnet wurde.
        #
        # Ziel ist es, die Erwartung der gewichteten Log-Wahrscheinlichkeiten zu maximieren,
        # was gleichbedeutend ist mit der Minimierung des negativen Erwartungswerts.
        #
        # Der Gradient von L(θ) bezüglich der Policy-Parameter θ:
        # 
        # ∇_θ L(θ) = - (1 / N) * Σ_t [∇_θ log π_θ(a_t | s_t) * A_t]
        #
        # In PyTorch wird dies automatisch durch `loss.backward()` berechnet.

        log_pi = self.forward(obs).log_prob(actions)

        '''
        Um eine einzige Log-Wahrscheinlichkeit pro Aktion zu erhalten (z. B. [batch_size] anstelle von [batch_size, action_dim]), 
        müssen die Log-Wahrscheinlichkeiten der einzelnen Aktionsdimensionen aufsummiert werden. 
        Das entspricht der gemeinsamen Wahrscheinlichkeit der multidimensionalen Aktion
        '''

        if not self.discrete:
            log_pi = log_pi.sum(axis=-1)

        #print("log_pi shape:", log_pi.shape)
        #print("advantages shape:", advantages.shape)


        loss = torch.neg(torch.mean(torch.mul(log_pi, advantages)))

        loss.backward()
        optimizer.step()

        return {
            "Actor Loss": ptu.to_numpy(loss),

        }
