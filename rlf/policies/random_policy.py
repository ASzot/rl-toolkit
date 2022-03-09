import numpy as np
import rlf.rl.utils as rutils
import torch
from gym import spaces
from rlf.policies.base_policy import BasePolicy, create_simple_action_data


class RandomPolicy(BasePolicy):
    def get_action(self, state, add_state, hxs, masks, step_info):
        n_procs = rutils.get_def_obs(state).shape[0]
        action = torch.tensor(
            np.array([self.action_space.sample() for _ in range(n_procs)])
        ).to(self.args.device)
        if isinstance(self.action_space, spaces.Discrete):
            action = action.unsqueeze(-1)

        return create_simple_action_data(action, hxs)
