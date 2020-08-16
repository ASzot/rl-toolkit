import torch.nn as nn
import numpy as np
import torch
import rlf.policies.utils as putils
import rlf.rl.utils as rutils
from rlf.policies.base_policy import BasePolicy


class BaseNetPolicy(nn.Module, BasePolicy):
    """
    The starting point for all neural network policies. Includes an easy way to
    goal condition policies. Defines a base neural network transformation that
    outputs a hidden representation.
    """

    def __init__(self,
            use_goal=False,
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
        self.use_goal = use_goal

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        if self.use_goal:
            use_obs_shape = rutils.get_obs_shape(obs_space)
            if len(use_obs_shape) != 1:
                raise ValueError(('Goal conditioning only ',
                    'works with flat state representation'))
            use_obs_shape = (use_obs_shape[0] + obs_space['desired_goal'].shape[0],)
        else:
            use_obs_shape = rutils.get_obs_shape(obs_space)
        self.base_net = self.get_base_net_fn(use_obs_shape)

    def _apply_base_net(self, state, add_state, hxs, masks):
        if self.use_goal:
            # Combine the goal and the state
            combined_state = torch.cat([state, add_state['desired_goal']], dim=-1)
            return self.base_net(combined_state, hxs, masks)
        else:
            return self.base_net(state, hxs, masks)

    def watch(self, logger):
        super().watch(logger)
        #logger.watch_model(self)
        print('Using policy network:')
        print(self)

    def save_to_checkpoint(self, checkpointer):
        checkpointer.save_key('policy', self.state_dict())
