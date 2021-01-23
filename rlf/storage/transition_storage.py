from rlf.storage.base_storage import BaseStorage
import random
import torch
import rlf.rl.utils as rutils
from collections import defaultdict
import numpy as np


class TransitionStorage(BaseStorage):
    def __init__(self, obs_space, action_space, capacity, args, hidden_states={}):
        super().__init__()
        self.args = args
        self.capacity = capacity
        self.position = 0
        self.last_seen = None
        self.set_device = None
        self.hidden_state_dims = hidden_states
        self.full = False
        self.d = self.args.device

        obs_shape = obs_space.shape
        if rutils.is_discrete(action_space):
            self.actions = torch.empty((capacity, 1),
                    dtype=torch.long)
        else:
            self.actions = torch.empty((capacity, action_space.shape[0]),
                    dtype=torch.float32)

        self.rewards = torch.empty((capacity, 1), dtype=torch.float32)
        self.masks = torch.empty((capacity, 1), dtype=torch.float32)
        # Hidden states
        self.hidden_states = {}
        for k, dim in hidden_states.items():
            self.hidden_states[k] = torch.empty((capacity, dim), dtype=torch.float32)
        # Obs
        self.ob_keys = rutils.get_ob_shapes(obs_space)
        self.obs = {}
        for k, space in self.ob_keys.items():
            ob = torch.empty((capacity, *space), dtype=torch.float32)
            if k is None:
                self.obs = ob
            else:
                self.obs[k] = ob
        # Next obs
        self.next_obs = {}
        for k in self.ob_keys:
            if k is None:
                self.next_obs = torch.empty(self.obs.shape, dtype=torch.float32)
            else:
                self.next_obs[k] = torch.empty(self.obs[k].shape, dtype=torch.float32)

    def _push_transition(self, trans):
        for k in self.ob_keys:
            if k is None:
                self.obs[self.position].copy_(trans['state'])
                self.next_obs[self.position].copy_(trans['next_state'])
            else:
                self.obs[k][self.position].copy_(trans['state'][k])
                self.next_obs[k][self.position].copy_(trans['next_state'][k])

        for k, dim in self.hidden_states.items():
            self.hidden_states[k][self.idx].copy_(trans['hxs'][k])

        self.actions[self.position].copy_(trans['action'])
        self.rewards[self.position].copy_(trans['reward'])
        self.masks[self.position].copy_(trans['mask'])

        self.position = (self.position + 1) % self.capacity
        self.full = self.full or self.position == 0

    def insert(self, obs, next_obs, reward, done, infos, ac_info):
        super().insert(obs, next_obs, reward, done, infos, ac_info)

        masks, bad_masks = self.compute_masks(done, infos)

        batch_size = rutils.get_def_obs(obs).shape[0]
        for i in range(batch_size):
            if bad_masks[i] == 0 and self.args.use_proper_time_limits:
                # we are not actually done.
                masks[i] = 1.0
            self._push_transition({
                    'action': ac_info.action[i],
                    'state': rutils.obs_select(obs, i),
                    'reward': reward[i],
                    'next_state': rutils.obs_select(next_obs, i),
                    'mask': masks[i],
                    'hxs': rutils.deep_dict_select(ac_info.hxs, i),
                    })

        self.last_seen = {
                'obs': next_obs,
                'masks': masks,
                'hxs': ac_info.hxs,
                }

    def sample(self, batch_size):
        raise ValueError('Depricated')
        return random.sample(self.memory, batch_size)

    def sample_tensors(self, sample_size):
        #idxs = torch.randint(0, len(self), size=(sample_size,))
        idxs = np.random.randint(0, len(self), size=sample_size)
        obs = None
        next_obs = None
        other_obs = {}
        other_next_obs = {}
        for k in self.ob_keys:
            if k is None:
                obs = self.obs[idxs].to(self.d)
                next_obs = self.next_obs[idxs].to(self.d)
            else:
                if k == self.args.policy_ob_key:
                    obs = self.obs[k][idxs].to(self.d)
                    next_obs = self.next_obs[k][idxs].to(self.d)
                else:
                    other_obs[k] = self.obs[k][idxs].to(self.d)
                    other_next_obs[k] = self.next_obs[k][idxs].to(self.d)
        assert obs is not None, f"Found not find {self.args.policy_ob_key}"

        hxs = {}
        for k, dim in self.hidden_states.items():
            hxs[k] = torch.as_tensor(self.hidden_states[k][idxs],
                    device=self.d).float()

        actions = self.actions[idxs].to(self.d)
        rewards = self.rewards[idxs].to(self.d)
        masks = self.masks[idxs].to(self.d)

        cur_add = {
            'hxs': None,
            'masks': None,
            'add_state': other_obs,
        }
        next_add = {
            'hxs': hxs,
            'masks': masks,
            'add_state': other_next_obs,
        }

        return obs, next_obs, actions, rewards, cur_add, next_add

    def __len__(self):
        return self.capacity if self.full else self.position

    def init_storage(self, obs):
        super().init_storage(obs)
        batch_size = rutils.get_def_obs(obs).shape[0]
        hxs = {}
        for k, dim in self.hidden_state_dims.items():
            hxs[k] = torch.zeros(batch_size, dim)
        self.last_seen = {
                'obs': obs,
                'masks': torch.zeros(batch_size, 1),
                'hxs': hxs,
                }

    def get_obs(self, step):
        ret_obs = self.last_seen['obs']
        return ret_obs

    def get_hidden_state(self, step):
        return self.last_seen['hxs']

    def get_masks(self, step):
        return self.last_seen['masks']

    def to(self, device):
        self.set_device = device
