from typing import Callable, List, Optional, Tuple

import gym
import rlf.policies.utils as putils
import rlf.rl.utils as rutils
import torch.nn as nn
from rlf.policies.base_net_policy import BaseNetPolicy
from rlf.rl.model import def_mlp_weight_init, weight_init


class ActorCritic(BaseNetPolicy):
    """
    Defines an actor and critic type policy
    """

    def __init__(
        self,
        get_critic_fn: Optional[
            Callable[[List[int], List[int], gym.Space, int], nn.Module]
        ] = None,
        get_critic_head_fn: Optional[Callable[[int], nn.Module]] = None,
        use_goal: bool = False,
        fuse_states: List[str] = [],
        get_base_net_fn: Optional[Callable[[Tuple[int], bool], nn.Module]] = None,
    ):
        """
        :param get_critic_fn: Callback that takes as input the input
            observation shape, the base encoder output shape, the action space, and
            number of hidden dimensions.
        """
        super().__init__(use_goal, fuse_states, get_base_net_fn)

        if get_critic_fn is None:
            get_critic_fn = putils.get_def_critic
        if get_critic_head_fn is None:
            get_critic_head_fn = putils.get_def_critic_head

        self.get_critic_fn = get_critic_fn
        self.get_critic_head_fn = get_critic_head_fn

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)

        obs_shape = rutils.get_obs_shape(obs_space, args.policy_ob_key)

        self.critic = self.get_critic_fn(
            obs_shape,
            self._get_base_out_shape(),
            action_space,
            self.args.policy_hidden_dim,
        )
        self.critic_head = self.get_critic_head_fn(self.critic.output_shape[0])

    def _get_value_from_features(self, base_features, hxs, masks):
        """
        - base_features: post fusion base features
        """
        critic_features, hxs = self.critic(base_features, hxs, masks)
        return self.critic_head(critic_features)

    def get_value(self, inputs, add_inputs, hxs, masks):
        base_features, hxs = self.base_net(inputs, hxs, masks)
        base_features = self._fuse_base_out(base_features, add_inputs)

        return self._get_value_from_features(base_features, hxs, masks)

    def get_critic_params(self):
        return (
            list(self.base_net.parameters())
            + list(self.critic.parameters())
            + list(self.critic_head.parameters())
        )

    def get_actor_params(self):
        return list(self.base_net.parameters()) + list(self.actor_net.parameters())
