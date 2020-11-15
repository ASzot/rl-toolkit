"""
Code is heavily based off of https://github.com/denisyarats/pytorch_sac.
The license is at `rlf/algos/off_policy/denis_yarats_LICENSE.md`
"""
from rlf.policies.base_policy import create_simple_action_data
from rlf.policies.actor_critic.base_actor_critic import ActorCritic
import torch
import rlf.policies.utils as putils
from rlf.policies.base_net_policy import BaseNetPolicy
import rlf.rl.utils as rutils

class DistActorQ(BaseNetPolicy):
    def __init__(self,
                get_actor_fn=None,
                get_dist_fn=None,
                get_critic_fn=None,
                use_goal=False,
                fuse_states=[],
                get_base_net_fn=None):

            super().__init__(use_goal, fuse_states, get_base_net_fn)

            if get_critic_fn is None:
                get_critic_fn = putils.get_def_ac_critic
            self.get_critic_fn = get_critic_fn

            if get_actor_fn is None:
                get_actor_fn = putils.get_def_actor
            self.get_actor_fn = get_actor_fn

            if get_dist_fn is None:
                get_dist_fn = putils.get_def_dist
            self.get_dist_fn = get_dist_fn

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)

        obs_shape = rutils.get_obs_shape(obs_space, args.policy_ob_key)
        self.critic = self.get_critic_fn(obs_shape, self._get_base_out_shape(),
                action_space)
        self.actor = self.get_actor_fn(
            rutils.get_obs_shape(obs_space, args.policy_ob_key),
            self._get_base_out_shape())

        log_std_bounds = [float(x) for x in self.args.log_std_bounds.split(',')]
        self.dist = self.get_dist_fn(
            self.actor.output_shape, self.action_space,
            log_std_bounds)

        self.ac_low_bound = torch.tensor(self.action_space.low).to(args.device)
        self.ac_high_bound = torch.tensor(self.action_space.high).to(args.device)

    def forward(self, state, add_state, hxs, masks):
        base_features, hxs = self._apply_base_net(state, add_state, hxs, masks)
        base_features = self._fuse_base_out(base_features, add_state)

        actor_features, _ = self.actor(base_features, hxs, masks)
        dist = self.dist(actor_features)

        return dist

    def get_action(self, state, add_state, hxs, masks, step_info):
        n_procs = state.shape[0]
        cur_step = step_info.cur_num_steps

        if not step_info.is_eval and cur_step < self.args.n_rnd_steps:
            action = torch.tensor([self.action_space.sample()
                for _ in range(n_procs)]).to(self.args.device)
            return create_simple_action_data(action, hxs)

        dist = self.forward(state, add_state, hxs, masks)
        if step_info.is_eval:
            action = dist.mean
        else:
            action = dist.sample()

        action = torch.min(torch.max(action, self.ac_low_bound), self.ac_high_bound)

        return create_simple_action_data(action, hxs)


    def get_value(self, state, action, add_state, hxs, masks):
        base_features, hxs = self._apply_base_net(state, add_state, hxs, masks)
        return self.critic(base_features, action)

    def get_actor_params(self):
        return list(self.base_net.parameters()) + \
                list(self.dist.parameters()) + list(self.actor.parameters())

    def get_critic_params(self):
        return list(self.base_net.parameters()) + \
                list(self.critic.parameters())

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--n-rnd-steps', type=int, default=10000)
        parser.add_argument("--log-std-bounds", type=str, default='-5,2')
