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
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
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
        action = self.forward(obs).sample(); 
        return ptu.to_numpy(action)

    # dimension should be (batch size, observation size)
    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        # print(obs.shape)
        assert obs.ndim <= 2, "SHOULD NOT HAVE MORE THAN 2 DIMENSIONS"

        # dim = (# of samples)
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            return distributions.Categorical(logits=self.logits_net(obs))
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            return distributions.Normal(self.mean_net(obs), torch.exp(self.logstd))
        return None
    
    # for A3C
    def syncGradients(self, gnet): 
        for lp, gp in zip(self.parameters(), gnet.parameters()): 
            gp.grad = lp.grad

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        pass


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        nostep: bool = False
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: implement the policy gradient actor update.
        dist = self.forward(obs)

        # log(probability of every action multiplied by each other) -> sum of logs
        logprobs = dist.log_prob(actions)
        if(logprobs.ndim == 2): 
            logprobs = logprobs.sum(dim=1)

        assert logprobs.shape == advantages.shape, "Log Probs and Advantages Don't Match Same Shape"

        # divide by the # of total timesteps
        loss = -torch.mean(logprobs * advantages)

        self.optimizer.zero_grad(); 
        loss.backward(); 
        
        # in the case of A3C, don't step yet
        if not nostep: 
            self.optimizer.step(); 

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
