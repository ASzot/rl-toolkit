import torch.nn as nn
import numpy as np
import torch
import rlf.policies.utils as putils
import rlf.rl.utils as rutils
from rlf.policies.base_policy import BasePolicy


class BaseNetPolicy(nn.Module, BasePolicy):
    def __init__(self,
            get_base_net_fn=None):
        """
        - get_base_fn: (tuple(int) -> rlf.rl.model.BaseNet)
            returned module should take as input size of observation space and
            return the size `hidden_size`.
          default: none, use the default
        """
        super().__init__()
        if get_base_net_fn is None:
            get_base_net_fn = putils.def_get_hidden_net

        self.get_base_net_fn = get_base_net_fn

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        self.base_net = self.get_base_net_fn(rutils.get_obs_shape(obs_space))

    def watch(self, logger):
        super().watch(logger)
        logger.watch_model(self)

    def save_to_checkpoint(self, checkpointer):
        checkpointer.save_key('policy', self.state_dict())
