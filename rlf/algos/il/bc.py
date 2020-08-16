from rlf.algos.il.base_il import BaseILAlgo
import torch.nn.functional as F
import torch
from rlf.storage.base_storage import BaseStorage
import gym
import numpy as np
import rlf.rl.utils as rutils
import rlf.algos.utils as autils
from tqdm import tqdm
import copy


class BehavioralCloning(BaseILAlgo):
    def __init__(self, set_arg_defs=True):
        super().__init__()
        self.set_arg_defs = set_arg_defs

    def init(self, policy, args):
        super().init(policy, args)
        self.num_epochs = 0
        self.norm_mean = self.expert_stats['state'][0]
        self.norm_var = torch.pow(self.expert_stats['state'][1], 2)
        self.num_bc_updates = 0

    def get_env_settings(self, args):
        settings = super().get_env_settings(args)
        if args.bc_state_norm:
            print('Setting environment state normalization')
            settings.state_fn = self._norm_state
        return settings

    def _norm_state(self, x):
        obs_x = torch.clamp((rutils.get_def_obs(x) - self.norm_mean) / torch.pow(self.norm_var + 1e-8, 0.5), -10.0, 10.0)
        if isinstance(x, dict):
            x['observation'] = obs_x
        return x

    def get_num_updates(self):
        if self.exp_generator is None:
            return len(self.expert_train_loader) * self.args.bc_num_epochs
        else:
            return self.args.exp_gen_num_trans * self.args.bc_num_epochs

    def get_completed_update_steps(self, num_updates):
        return num_updates * self.args.traj_batch_size

    def _reset_data_fetcher(self):
        super()._reset_data_fetcher()
        self.num_epochs += 1

    def full_train(self, update_iter=0):
        action_loss = []
        prev_num = 0

        # First BC
        with tqdm(total=self.args.bc_num_epochs) as pbar:
            while self.num_epochs < self.args.bc_num_epochs:
                super().pre_update(self.num_bc_updates)
                log_vals = self._bc_step()
                self.num_bc_updates += 1
                action_loss.append(log_vals['pr_action_loss'])

                pbar.update(self.num_epochs - prev_num)
                prev_num = self.num_epochs

        rutils.plot_line(action_loss, f"action_loss_{update_iter}.png",
                         self.args, not self.args.no_wb,
                         self.get_completed_update_steps(self.update_i))
        self.num_epochs = 0

    def pre_update(self, cur_update):
        # Override the learning rate decay
        pass

    def _bc_step(self):
        expert_batch = self._get_next_data()

        if expert_batch is None:
            self._reset_data_fetcher()
            expert_batch = self._get_next_data()

        states = expert_batch['state'].to(self.args.device)
        if self.args.bc_state_norm:
            states = self._norm_state(states)

        if self.args.bc_noise is not None:
            add_noise = torch.randn(states.shape) * self.args.bc_noise
            states += add_noise.to(self.args.device)
            states = states.detach()

        true_actions = expert_batch['actions'].to(self.args.device)
        true_actions = self._adjust_action(true_actions)

        pred_actions, _, _ = self.policy(states, None, None)
        loss = autils.compute_ac_loss(pred_actions, true_actions,
                self.policy.action_space)

        self._standard_step(loss)

        return {
                'pr_action_loss': loss.item()
                }

    def update(self, storage):
        return self._bc_step()

    def get_storage_buffer(self, policy, envs, args):
        return BaseStorage()

    def get_add_args(self, parser):
        if not self.set_arg_defs:
            # This is set when BC is used at the same time as another optimizer
            # that also has a learning rate.
            self.set_arg_prefix('bc')

        super().get_add_args(parser)
        #########################################
        # Overrides
        if self.set_arg_defs:
            parser.add_argument('--num-processes', type=int, default=1)
            parser.add_argument('--num-steps', type=int, default=0)
            ADJUSTED_INTERVAL = 200
            parser.add_argument('--log-interval', type=int,
                                default=ADJUSTED_INTERVAL)
            parser.add_argument('--save-interval', type=int,
                                default=100*ADJUSTED_INTERVAL)
            parser.add_argument('--eval-interval', type=int,
                                default=100*ADJUSTED_INTERVAL)
        parser.add_argument('--no-wb', default=False, action='store_true')

        #########################################
        # New args
        parser.add_argument('--bc-num-epochs', type=int, default=1)
        parser.add_argument('--bc-state-norm', action='store_true')
        parser.add_argument('--bc-noise', type=float, default=None)

