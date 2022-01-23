from collections import defaultdict
from typing import Callable, List, Optional, Tuple

import gym
import rlf.policies.utils as putils
import rlf.rl.utils as rutils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rlf.policies.actor_critic.base_actor_critic import ActorCritic
from rlf.policies.base_policy import ActionData


class DistActorCritic(ActorCritic):
    """
    Defines an actor/critic where the actor outputs an action distribution
    """

    def __init__(
        self,
        get_actor_fn: Optional[Callable[[list, list, int], nn.Module]] = None,
        get_dist_fn: Optional[Callable[[list, gym.Space], nn.Module]] = None,
        get_critic_fn: Optional[
            Callable[[list, list, gym.Space, int], nn.Module]
        ] = None,
        get_critic_head_fn: Optional[Callable[[int], nn.Module]] = None,
        fuse_states: List[str] = [],
        use_goal: bool = False,
        get_base_net_fn: Optional[Callable[[Tuple[int], bool], nn.Module]] = None,
    ):
        """
        For all the network callbacks, if None is provided, a default is used.
        :param get_actor_fn: A callback to get the actor network which takes
            the output of the base_net as input. The arguments to the callback are
            the observation space shape, the output shape of the base network, and
            the network hidden dim.
        :param get_dist_fn: A callback to get the distribution network. The
            arguments to the callback are the output shape of the actor network the
            Gym action space.
        """
        super().__init__(
            get_critic_fn, get_critic_head_fn, use_goal, fuse_states, get_base_net_fn
        )
        if get_actor_fn is None:
            get_actor_fn = putils.get_def_actor
        self.get_actor_fn = get_actor_fn

        if get_dist_fn is None:
            get_dist_fn = putils.get_def_dist
        self.get_dist_fn = get_dist_fn

    def get_actor_params(self):
        return (
            list(self.base_net.parameters())
            + list(self.actor.parameters())
            + list(self.dist.parameters())
        )

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        self.actor = self.get_actor_fn(
            rutils.get_obs_shape(obs_space, args.policy_ob_key),
            self._get_base_out_shape(),
            self.args.policy_hidden_dim,
        )
        self.dist = self.get_dist_fn(self.actor.output_shape, self.action_space)

    def get_action(self, state, add_state, hxs, masks, step_info):
        dist, value, hxs = self.forward(state, add_state, hxs, masks)
        if self.args.deterministic_policy:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy()

        return ActionData(
            value, action, action_log_probs, hxs, {"dist_entropy": dist_entropy}
        )

    def forward(self, state, add_state, hxs, masks):
        base_features, hxs = self._apply_base_net(state, add_state, hxs, masks)
        base_features = self._fuse_base_out(base_features, add_state)

        value = self._get_value_from_features(base_features, hxs, masks)

        actor_features, _ = self.actor(base_features, hxs, masks)
        dist = self.dist(actor_features)

        return dist, value, hxs

    def evaluate_actions(self, state, add_state, hxs, masks, action):
        dist, value, hxs = self.forward(state, add_state, hxs, masks)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy()
        return {
            "value": value,
            "log_prob": action_log_probs,
            "ent": dist_entropy,
        }
