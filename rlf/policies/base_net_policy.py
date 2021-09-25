import inspect
from functools import partial
from typing import Callable, List, Optional, Tuple

import numpy as np
import rlf.policies.utils as putils
import rlf.rl.utils as rutils
import torch
import torch.nn as nn
from rlf.args import str2bool
from rlf.policies.base_policy import BasePolicy


class BaseNetPolicy(nn.Module, BasePolicy):
    """
    The starting point for all neural network policies. Includes an easy way to
    goal condition policies. Defines a base neural network transformation that
    outputs a hidden representation.
    """

    def __init__(
        self,
        use_goal: bool = False,
        fuse_states: List[str] = [],
        get_base_net_fn: Optional[Callable[[Tuple[int], bool], nn.Module]] = None,
    ):
        """
        For all the network callbacks, if None is provided, a default is used.
        :param use_goal: If true, the goal observation in the goal observation
            dictionary will be concatenated to the observation.
        :fuse_states: The keys of the 1D state observations to fuse with the encoded state.
        :param get_base_net_fn: Callback to get the network that encodes
            observations. Arguments are the input shape, and whether to use a
            recurrent encoder.
        """
        super().__init__()
        if get_base_net_fn is None:
            get_base_net_fn = putils.get_img_encoder

        self.get_base_net_fn = get_base_net_fn
        self.fuse_states = fuse_states
        self.use_goal = use_goal

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        if "recurrent" in inspect.getfullargspec(self.get_base_net_fn).args:
            self.get_base_net_fn = partial(
                self.get_base_net_fn, recurrent=self.args.recurrent_policy
            )
        if self.use_goal:
            use_obs_shape = rutils.get_obs_shape(obs_space, args.policy_ob_key)
            if len(use_obs_shape) != 1:
                raise ValueError(
                    ("Goal conditioning only ", "works with flat state representation")
                )
            use_obs_shape = (use_obs_shape[0] + obs_space["desired_goal"].shape[0],)
        else:
            use_obs_shape = rutils.get_obs_shape(obs_space, args.policy_ob_key)

        self.base_net = self.get_base_net_fn(use_obs_shape)
        base_out_dim = self.base_net.output_shape[0]
        for k in self.fuse_states:
            if len(obs_space.spaces[k].shape) != 1:
                raise ValueError("Can only fuse 1D states")
            base_out_dim += obs_space.spaces[k].shape[0]
        self.base_out_shape = (base_out_dim,)

    def _get_base_out_shape(self) -> Tuple[int]:
        return self.base_out_shape

    def _fuse_base_out(self, base_features, add_input):
        if len(self.fuse_states) == 0:
            return base_features
        fuse_states = torch.cat([add_input[k] for k in self.fuse_states], dim=-1)
        fused = torch.cat([base_features, fuse_states], dim=-1)
        return fused

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument(
            "--policy-hidden-dim",
            default=64,
            type=int,
            help="""
                Default hidden size used for all networks related to the policy. This includes value functions.
                """,
        )
        parser.add_argument("--load-policy", type=str2bool, default=True)
        parser.add_argument(
            "--recurrent-policy",
            action="store_true",
            default=False,
            help="use a recurrent policy",
        )

    def get_storage_hidden_states(self):
        hxs = super().get_storage_hidden_states()
        if self.args.recurrent_policy:
            hxs["rnn_hxs"] = self.base_net.gru.hidden_size
        return hxs

    def _apply_base_net(self, state, add_state, hxs, masks):
        if self.use_goal:
            # Combine the goal and the state
            combined_state = torch.cat([state, add_state["desired_goal"]], dim=-1)
            return self.base_net(combined_state, hxs, masks)
        else:
            return self.base_net(state, hxs, masks)

    def watch(self, logger):
        super().watch(logger)
        # logger.watch_model(self)
        print("Using policy network:")
        print(self)

    def load(self, checkpointer):
        super().load(checkpointer)
        if self.args.load_policy:
            self.load_state_dict(checkpointer.get_key("policy"))

    def save(self, checkpointer):
        super().save(checkpointer)
        checkpointer.save_key("policy", self.state_dict())
