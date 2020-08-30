from rlf.policies.base_net_policy import BaseNetPolicy
import torch.nn.functional as F
import random
import math
import torch.nn as nn
import torch
import rlf.policies.utils as putils
import rlf.rl.utils as rutils
from rlf.policies.base_policy import create_simple_action_data


class SVGDPolicy(BaseNetPolicy):
    def __init__(self,
            get_actor_fn=None,
            get_actor_head_fn=None,
            get_critic_fn=None,
            get_critic_head_fn=None,
            use_goal=False,
            get_base_net_fn=None):

        super().__init__(use_goal, get_base_net_fn)

        if get_critic_fn is None:
            get_critic_fn = putils.get_def_ac_critic
        if get_critic_head_fn is None:
            get_critic_head_fn = putils.get_def_critic_head
        if get_actor_fn is None:
            get_actor_fn = putils.get_def_actor
        if get_actor_head_fn is None:
            get_actor_head_fn = putils.get_def_actor_head

        self.get_critic_fn = get_critic_fn
        self.get_critic_head_fn = get_critic_head_fn
        self.get_actor_fn = get_actor_fn
        self.get_actor_head_fn = get_actor_head_fn

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        if rutils.is_discrete(action_space):
            raise ValueError(("Currently only support continuous actions. "
                    "However it only requires a small coding change to fix."))

        ac_dim = action_space.shape[0]
        # Create networks.
        self.actor_net = self.get_actor_fn(obs_space.shape, self.base_net.output_shape[0] + ac_dim)
        self.actor_head = self.get_actor_head_fn(self.actor_net.output_shape[0], ac_dim)
        self.critic = self.get_critic_fn(obs_shape, self.base_net.output_shape, action_space)
        self.critic_head = self.get_critic_head_fn(self.critic.output_shape[0])

    def forward(self, state, add_state, hxs, masks):
        base_features, _ = self._apply_base_net(state, add_state, hxs, masks)
        import ipdb; ipdb.set_trace()
        xi = torch.randn([*state.shape, self.action.space.shape[0]])
        base_features_with_xi = torch.cat([base_features, xi])
        actor_features, _ = self.actor_net(base_features_with_xi, hxs, masks)
        return self.actor_head(actor_features)

    def get_value(self, state, action, add_state, hxs, masks):
        base_features, hxs = self._apply_base_net(state, add_state, hxs, masks)
        critic_features, hxs = self.critic(base_features, action, hxs, masks)
        return self.critic_head(critic_features)

    def get_action(self, state, add_state, hxs, masks, step_info):
        action = self.forward(state, add_state, hxs, masks)
        return create_simple_action_data(action, hxs)

    def get_critic_params(self):
        return list(self.base_net.parameters()) + \
                list(self.critic.parameters()) + \
                list(self.critic_head.parameters())

    def get_actor_params(self):
        return list(self.base_net.parameters()) + \
                list(self.actor_net.parameters()) + \
                list(self.actor_head.parameters())
