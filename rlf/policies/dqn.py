from rlf.policies.base_net_policy import BaseNetPolicy
import torch.nn.functional as F
import random
import math
import torch.nn as nn
import torch
import rlf.policies.utils as putils
from rlf.policies.base_policy import create_simple_action_data


class DQN(BaseNetPolicy):
    """
    Defines approximation of Q(s,a) using deep neural networks. The value for
    each action are output as heads of the network.
    """

    def __init__(self,
            get_base_net_fn=None,
            get_actor_head_fn=None):
        super().__init__(get_base_net_fn)
        if get_actor_head_fn is None:
            get_actor_head_fn = putils.get_def_actor_head
        self.get_actor_head_fn = get_actor_head_fn

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        assert self.action_space.__class__.__name__ == 'Discrete'
        self.head = self.get_actor_head_fn(self.base_net.output_shape[0],
                action_space.n)

    def forward(self, state, rnn_hxs, masks):
        base_features, _ = self.base_net(state, rnn_hxs, masks)
        return self.head(base_features)

    def get_action(self, state, add_state, rnn_hxs, masks, step_info):
        if step_info.is_eval:
            eps_threshold = 0
        else:
            num_steps = step_info.cur_num_steps
            eps_threshold = self.args.eps_end + \
                (self.args.eps_start - self.args.eps_end) * \
                math.exp(-1.0 * num_steps / self.args.eps_decay)

        sample = random.random()
        if sample > eps_threshold:
            q_vals = self.forward(state, rnn_hxs, masks)
            ret_action = q_vals.max(1)[1].unsqueeze(-1)
        else:
            # Take a random action.
            ret_action = torch.LongTensor([[random.randrange(self.action_space.n)]
                for i in range(state.shape[0])])
            if self.args.cuda:
                ret_action = ret_action.cuda()

        return create_simple_action_data(ret_action)

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--eps-start', type=float, default=0.9)
        parser.add_argument('--eps-end', type=float, default=0.05)
        parser.add_argument('--eps-decay', type=float, default=200)
