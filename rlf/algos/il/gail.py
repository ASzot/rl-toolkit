from rlf.algos.il.base_irl import BaseIRLAlgo
import torch
import torch.nn as nn
import torch.nn.functional as F
import rlf.rl.utils as rutils
import rlf.algos.utils as autils
from collections import defaultdict
from rlf.baselines.common.running_mean_std import RunningMeanStd
from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.on_policy.ppo import PPO
from rlf.args import str2bool
import torch.optim as optim
import numpy as np
from rlf.rl.model import ConcatLayer
from rlf.rl.model import InjectNet
from functools import partial


def get_default_discrim(ac_dim, in_shape):
    """
    - ac_dim: int will be 0 if no action are used.
    Returns: (nn.Module) Should take state AND actions as input if ac_dim
    != 0. If ac_dim = 0 (discriminator does not use actions) then ONLY take
    state as input.
    """
    if len(in_shape) != 1:
        raise ValueError("""
                Default discriminator only supports 1D state spaces. Create your
                own discriminator if you want to do this.
                """)
    hidden_dim = 64
    layers = [
            #nn.Linear(in_shape[0] + ac_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
            ]

    return nn.Sequential(*layers)

class GAIL(NestedAlgo):
    def __init__(self, agent_updater=PPO(), get_discrim=None):
        super().__init__([GailDiscrim(get_discrim), agent_updater], 1)

class GailDiscrim(BaseIRLAlgo):
    def __init__(self, get_discrim=None):
        super().__init__()
        if get_discrim is None:
            get_discrim = get_default_discrim
        self.get_discrim = get_discrim

    def _create_discrim(self):
        ob_shape = rutils.get_obs_shape(self.policy.obs_space)
        ac_dim = rutils.get_ac_dim(self.action_space)
        discrim_head = InjectNet(
            self.policy.get_base_net_fn(ob_shape).net,
            partial(self.get_discrim, in_shape=ob_shape),
            ob_shape[0], 64, ac_dim,
            self.args.action_input)

        return discrim_head.to(self.args.device)

    def init(self, policy, args):
        super().init(policy, args)
        self.action_space = self.policy.action_space

        self.discrim_net = self._create_discrim()

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

        self.opt = optim.Adam(
            self.discrim_net.parameters(), lr=self.args.disc_lr)

    def _get_sampler(self, storage):
        agent_experience = storage.get_generator(None,
                                                 mini_batch_size=self.expert_train_loader.batch_size)
        return self.expert_train_loader, agent_experience

    def _trans_batches(self, expert_batch, agent_batch):
        return expert_batch, agent_batch

    def _norm_expert_state(self, state, obsfilt):
        state = state.cpu().numpy()
        if obsfilt is not None:
            state = obsfilt(state, update=False)
        state = torch.tensor(state).to(self.args.device)
        return state

    def _compute_discrim_loss(self, agent_batch, expert_batch, obsfilt):
        expert_actions = expert_batch['actions'].to(self.args.device)
        expert_actions = self._adjust_action(expert_actions)
        expert_states = self._norm_expert_state(expert_batch['state'],
                obsfilt)

        agent_states = agent_batch['state']
        agent_actions = agent_batch['action']

        agent_actions = rutils.get_ac_repr(
            self.action_space, agent_actions)
        expert_actions = rutils.get_ac_repr(
            self.action_space, expert_actions)

        expert_d = self._compute_disc_val(expert_states, expert_actions)
        agent_d = self._compute_disc_val(agent_states, agent_actions)

        grad_pen = self.args.disc_grad_pen * autils.wass_grad_pen(expert_states,
                expert_actions, agent_states, agent_actions,
                self.args.action_input, self._compute_disc_val)

        return expert_d, agent_d, grad_pen

    def _compute_disc_val(self, state, action):
        return self.discrim_net(state, action)

    def _update_reward_func(self, storage):
        self.discrim_net.train()

        d = self.args.device
        log_vals = defaultdict(lambda: 0)
        obsfilt = self.get_env_ob_filt()

        n = 0
        expert_sampler, agent_sampler = self._get_sampler(storage)
        for epoch_i in range(self.args.n_gail_epochs):
            for expert_batch, agent_batch in zip(expert_sampler, agent_sampler):
                expert_batch, agent_batch = self._trans_batches(
                    expert_batch, agent_batch)
                n += 1
                expert_d, agent_d, grad_pen = self._compute_discrim_loss(agent_batch, expert_batch,
                        obsfilt)
                expert_loss = F.binary_cross_entropy_with_logits(expert_d,
                                                                 torch.ones(expert_d.shape).to(d))
                agent_loss = F.binary_cross_entropy_with_logits(agent_d,
                                                                torch.zeros(agent_d.shape).to(d))
                discrim_loss = expert_loss + agent_loss

                if self.args.disc_grad_pen != 0.0:
                    log_vals['grad_pen'] += grad_pen.item()
                    total_loss = discrim_loss + grad_pen
                else:
                    total_loss = discrim_loss

                self.opt.zero_grad()
                total_loss.backward()
                self.opt.step()

                log_vals['discrim_loss'] += discrim_loss.item()
                log_vals['expert_loss'] += expert_loss.item()
                log_vals['agent_loss'] += agent_loss.item()

        for k in log_vals:
            log_vals[k] /= n

        return log_vals

    def _compute_discrim_reward(self, storage, step, add_info):
        state = rutils.get_def_obs(storage.get_obs(step))
        action = storage.actions[step]
        action = rutils.get_ac_repr(self.action_space, action)
        d_val = self._compute_disc_val(state, action)
        s = torch.sigmoid(d_val)
        eps = 1e-20
        if self.args.reward_type == 'airl':
            reward = (s + eps).log() - (1 - s + eps).log()
        elif self.args.reward_type == 'gail':
            reward = (s + eps).log()
        elif self.args.reward_type == 'raw':
            reward = d_val
        else:
            raise ValueError(f"Unrecognized reward type {self.args.reward_type}")
        return reward

    def _get_reward(self, step, storage, add_info):
        masks = storage.masks[step]
        with torch.no_grad():
            self.discrim_net.eval()
            reward = self._compute_discrim_reward(storage, step, add_info)

            if self.args.gail_reward_norm:
                if self.returns is None:
                    self.returns = reward.clone()

                self.returns = self.returns * masks * self.args.gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

                return reward / np.sqrt(self.ret_rms.var[0] + 1e-8), {}
            else:
                return reward, {}

    def get_add_args(self, parser):
        super().get_add_args(parser)
        #########################################
        # Overrides

        #########################################
        # New args
        parser.add_argument('--action-input', type=str2bool, default=False)
        parser.add_argument('--gail-reward-norm', type=str2bool, default=False)
        parser.add_argument('--disc-lr', type=float, default=0.0001)
        parser.add_argument('--disc-grad-pen', type=float, default=0.0)
        parser.add_argument('--n-gail-epochs', type=int, default=1)
        parser.add_argument('--reward-type', type=str, default='airl', help="""
                One of [airl, raw, gail]. Changes the reward computation. Does
                not change training.
                """)

    def load_resume(self, checkpointer):
        super().load_resume(checkpointer)
        self.opt.load_state_dict(checkpointer.get_key('gail_disc_opt'))
        self.discrim_net.load_state_dict(checkpointer.get_key('gail_disc'))

    def save(self, checkpointer):
        super().save(checkpointer)
        checkpointer.save_key('gail_disc_opt', self.opt.state_dict())
        checkpointer.save_key('gail_disc', self.discrim_net.state_dict())
