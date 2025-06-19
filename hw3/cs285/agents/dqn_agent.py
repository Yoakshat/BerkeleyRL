from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import cs285.infrastructure.pytorch_util as ptu
import random


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
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
        with torch.no_grad():
            observation = ptu.from_numpy(np.asarray(observation))[None]

            # TODO(student): get the action from the critic using an epsilon-greedy strategy
            # with probability epsilon, explore
            # otherwise pick best action
            if(random.random() >= epsilon): 
                # pick best action
                action = torch.argmax(self.critic(observation), dim=1)
                action = ptu.to_numpy(action).squeeze(0).item()
            else: 
                action = random.randint(0, self.num_actions-1)

        return action

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values from target critic
        with torch.no_grad():
            # TODO(student): compute target values
            # e.g move right move left move up move down

            # (batch_size, actions)
            next_qa_values = self.target_critic(next_obs)
            assert(next_qa_values.shape[1] == self.num_actions)

            if self.use_double_q:
                # select action from our online network
                next_action = torch.argmax(self.critic(next_obs), dim=1)
            else:
              # assume we greedily pick best action here
                next_action = torch.argmax(next_qa_values, dim=1)

            
            # (batch_size,)
            assert next_action.ndim == 1 and next_action.shape[0] == batch_size
            
            # like an index operation (next_qa_values[:, next_action])
            next_q_values = torch.gather(next_qa_values, 1, next_action.unsqueeze(1)).squeeze(1)

            # no gradient will be propagated through this
            # multiply by 1 - done
            # the target value for a terminal state is just reward
            target_values = reward + self.discount * next_q_values * (1 - done.float())

            assert target_values.ndim == 1

        # TODO(student): train the critic with the target values
        # THESE are from the critic
        # propagate gradients through these
        qa_values = self.critic(obs)
        # get the qvalues that we predict
        q_values = torch.gather(qa_values, 1, action.unsqueeze(1)).squeeze(1) # Compute from the data actions; see torch.gather

        # check if gather works right
        assert (qa_values[0][action[0]].item() == q_values[0].item()) and \
                (qa_values[1][action[1]].item() == q_values[1].item()), "Using gather wrongly"
        
        assert q_values.shape == target_values.shape

        assert target_values.requires_grad == False 
        assert q_values.requires_grad == True

        loss = self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    # done: if the agent lost their life or the game crashed 
    # terminal: if anything ended it, including truncation
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
        # TODO(student): update the critic, and the target if needed
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)

        # for every step, we're sampling
        # every N steps update
        if(step % self.target_update_period == 0):
          self.update_target_critic()

        return critic_stats
    
