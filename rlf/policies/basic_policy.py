from rlf.policies.base_net_policy import BaseNetPolicy
import torch.nn.functional as F
import random
import math
import gym
import torch.nn as nn
import torch
from rlf.policies.base_policy import create_simple_action_data
import rlf.policies.utils as putils
import rlf.rl.utils as rutils


class BasicPolicy(BaseNetPolicy):
    def __init__(self,
            get_base_net_fn=None):
        super().__init__(get_base_net_fn)
        self.state_norm_fn = lambda x: x
        self.action_denorm_fn = lambda x: x

    def set_state_norm_fn(self, state_norm_fn):
        self.state_norm_fn = state_norm_fn

    def set_action_denorm_fn(self, action_denorm_fn):
        self.action_denorm_fn = action_denorm_fn

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        ac_dim = rutils.get_ac_dim(action_space)
        self.head = nn.Linear(self.base_net.output_shape[0], ac_dim)

    def forward(self, state, hxs, mask):
        base_features, _ = self.base_net(state, hxs, mask)
        return self.head(base_features), None, None

    def get_action(self, state, add_state, hxs, mask, step_info):
        ret_action, _, _ = self.forward(state, hxs, mask)
        ret_action = rutils.get_ac_compact(self.action_space, ret_action)
        return create_simple_action_data(ret_action)
