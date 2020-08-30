from rlf.policies.base_policy import BasePolicy, create_np_action_data
import numpy as np
import math
import random
import torch

class QTable(BasePolicy):
    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        self.Q = np.zeros((obs_space.n, action_space.n))
        if self.args.eps_end == None:
            self.args.eps_end = self.args.eps_start

    def get_action(self, state, add_state, hxs, masks, step_info):
        q_s = self.Q[state[0].long().item()]

        if step_info.is_eval:
            eps_threshold = 0
        else:
            num_steps = step_info.cur_num_steps
            eps_threshold = self.args.eps_end + \
                (self.args.eps_start - self.args.eps_end) * \
                math.exp(-1.0 * num_steps / self.args.eps_decay)

        sample = random.random()
        if sample > eps_threshold:
            ret_action = np.argmax(q_s)
        else:
            ret_action = random.randrange(self.action_space.n)

        return create_np_action_data(ret_action, {
            'alg_add_eps': eps_threshold
            })

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--eps-start', type=float, default=0.9)
        parser.add_argument('--eps-end', type=float, default=None,
            help="""
            If "None" then no decay is applied.
            """)
        parser.add_argument('--eps-decay', type=float, default=200)
