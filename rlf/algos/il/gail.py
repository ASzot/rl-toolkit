from rlf.algos.il.base_irl import BaseIRLAlgo
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlf.rl import utils
from rlf.rl.model import InjectNet
from collections import defaultdict
from rlf.baselines.common.running_mean_std import RunningMeanStd
from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.on_policy.ppo import PPO
from rlf.args import str2bool
import torch.optim as optim
from torch import autograd
import numpy as np


def get_default_discrim(hidden_dim=64):
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, 1)), hidden_dim


class GAIL(NestedAlgo):
    def __init__(self, agent_updater=PPO(), get_discrim=None):
        super().__init__([GailDiscrim(get_discrim), agent_updater], 1)


class GailDiscrim(BaseIRLAlgo):
    def __init__(self, get_discrim=None):
        super().__init__()
        if get_discrim is None:
            get_discrim = get_default_discrim
        self.get_base_discrim = get_discrim

    def _create_discrim(self):
        state_enc = self.policy.get_base_net_fn(
            utils.get_obs_shape(self.policy.obs_space))
        discrim_head, in_dim = self.get_base_discrim()

        return InjectNet(state_enc.net, discrim_head,
                                     state_enc.output_shape[0], in_dim,
                                     utils.get_ac_dim(self.action_space),
                                     self.args.action_input).to(self.args.device)

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

        agent_actions = utils.get_ac_repr(
            self.action_space, agent_actions)
        expert_actions = utils.get_ac_repr(
            self.action_space, expert_actions)

        expert_d = self.discrim_net(expert_states, expert_actions)
        agent_d = self.discrim_net(agent_states, agent_actions)

        grad_pen = self.compute_grad_pen(expert_states, expert_actions,
                                         agent_states, agent_actions, self.args.gail_grad_pen)

        return expert_d, agent_d, grad_pen

    def get_env_settings(self, args):
        settings = super().get_env_settings(args)
        def mod_render_frames(frame, obs, next_obs, info, action, next_mask, **kwargs):
            with torch.no_grad():
                use_obs = utils.get_def_obs(obs).unsqueeze(0)
                if next_mask.item() == 0:
                    use_next_obs = torch.FloatTensor(info['final_obs']).unsqueeze(0)
                else:
                    use_next_obs = utils.get_def_obs(next_obs).unsqueeze(0)
                d_val = self._compute_disc_val(use_obs, use_next_obs, action)
                frame = utils.render_text(frame, "Discrim %.3f" % d_val.item(), 0)

                s = torch.sigmoid(d_val)
                eps = 1e-20
                reward = (s + eps).log() - (1 - s + eps).log()
                frame = utils.render_text(frame, "Reward %.3f" % reward.item(), 1)
            return frame
        settings.mod_render_frames_fn = mod_render_frames
        return settings

    def _compute_disc_val(self, state, next_state, action):
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
                # expert_loss = F.mse_loss(torch.sigmoid(expert_d), torch.ones(expert_d.shape).to(d))
                # agent_loss = F.mse_loss(torch.sigmoid(agent_d), torch.zeros(agent_d.shape).to(d))
                discrim_loss = expert_loss + agent_loss

                if self.args.gail_grad_pen != 0.0:
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
        state = utils.get_def_obs(storage.get_obs(step))
        action = storage.actions[step]
        action = utils.get_ac_repr(self.action_space, action)
        d_val = self.discrim_net(state, action)
        s = torch.sigmoid(d_val)
        eps = 1e-20
        reward = (s + eps).log() - (1 - s + eps).log()
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

    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10):

        num_dims = len(expert_state.shape) - 1
        alpha = torch.rand(expert_state.size(0), 1)
        alpha_state = alpha.view(-1, *[1 for _ in range(num_dims)]
                                 ).expand_as(expert_state).to(expert_state.device)
        mixup_data_state = alpha_state * expert_state + \
            (1 - alpha_state) * policy_state
        mixup_data_state.requires_grad = True

        alpha_action = alpha.expand_as(expert_action).to(expert_action.device)
        mixup_data_action = alpha_action * expert_action + \
            (1 - alpha_action) * policy_action
        mixup_data_action.requires_grad = True

        disc = self.discrim_net(mixup_data_state, mixup_data_action)
        ones = torch.ones(disc.size()).to(disc.device)

        inputs = [mixup_data_state]
        if self.args.action_input:
            inputs.append(mixup_data_action)

        grad = autograd.grad(
            outputs=disc,
            inputs=inputs,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def get_add_args(self, parser):
        super().get_add_args(parser)
        #########################################
        # Overrides

        #########################################
        # New args
        parser.add_argument('--action-input', type=str2bool, default=False)
        parser.add_argument('--gail-reward-norm', type=str2bool, default=False)
        parser.add_argument('--disc-lr', type=float, default=0.0001)
        parser.add_argument('--gail-grad-pen', type=float, default=0.0)
        parser.add_argument('--n-gail-epochs', type=int, default=1)

    def load_resume(self, checkpointer):
        super().load_resume(checkpointer)
        self.opt.load_state_dict(checkpointer.get_key('gail_disc_opt'))
        self.discrim_net.load_state_dict(checkpointer.get_key('gail_disc'))

    def save(self, checkpointer):
        super().save(checkpointer)
        checkpointer.save_key('gail_disc_opt', self.opt.state_dict())
        checkpointer.save_key('gail_disc', self.discrim_net.state_dict())
