"""
Code is heavily based off of https://github.com/denisyarats/pytorch_sac.
The license is at `rlf/algos/off_policy/denis_yarats_LICENSE.md`
"""
import numpy as np
import rlf.rl.utils as rutils
import torch
from rlf.storage.base_storage import BaseStorage


class ReplayBuffer(BaseStorage):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, capacity, device, args):
        super().__init__()
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.add_data = {}

        self.idx = 0
        self.last_save = 0
        self.full = False
        self._modify_reward_fn = None

    def add_info_key(self, key_name, data_size):
        super().add_info_key(key_name, data_size)
        self.add_data[key_name] = np.empty(
            (self.capacity, *data_size), dtype=np.float32
        )

    def to(self, device):
        return self

    def init_storage(self, obs):
        batch_size = rutils.get_def_obs(obs).shape[0]
        hxs = {}
        self.last_seen = {
            "obs": obs,
            "masks": torch.zeros(batch_size, 1).to(self.device),
            "hxs": hxs,
        }

    def get_obs(self, step):
        ret_obs = self.last_seen["obs"]
        return ret_obs

    def get_hidden_state(self, step):
        return self.last_seen["hxs"]

    def get_masks(self, step):
        return self.last_seen["masks"]

    def __len__(self):
        return self.capacity if self.full else self.idx

    def insert(self, obs, next_obs, reward, done, infos, ac_info):
        action = ac_info.take_action
        masks, bad_masks = self.compute_masks(done, infos)
        self.last_seen = {
            "obs": next_obs,
            "masks": masks,
            "hxs": ac_info.hxs,
        }
        np.copyto(self.obses[self.idx], obs[0].cpu().numpy())
        np.copyto(self.actions[self.idx], action[0].cpu().numpy())
        np.copyto(self.rewards[self.idx], reward[0].cpu().numpy())
        np.copyto(self.next_obses[self.idx], next_obs[0].cpu().numpy())
        np.copyto(self.not_dones[self.idx], masks[0].cpu().numpy())
        np.copyto(self.not_dones_no_max[self.idx], bad_masks[0].cpu().numpy())

        for i, inf in enumerate(infos):
            for k in self.get_extract_info_keys():
                if k in inf:
                    if isinstance(inf[k], torch.Tensor):
                        assign_val = inf[k].cpu().numpy()
                    else:
                        assign_val = inf[k]
                    np.copyto(self.add_data[k][self.idx], assign_val.cpu().numpy())

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def set_modify_reward_fn(self, modify_reward_fn):
        self._modify_reward_fn = modify_reward_fn

    def sample_tensors(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(
            self.not_dones_no_max[idxs], device=self.device
        )
        add_data = {
            k: torch.as_tensor(self.add_data[k][idxs], device=self.device)
            for k in self.add_data
        }
        if self._modify_reward_fn is not None:
            rewards = self._modify_reward_fn(
                obses, actions, next_obses, not_dones, add_data
            )
        return {
            "state": obses,
            "next_state": next_obses,
            "action": actions,
            "reward": rewards,
            "mask": not_dones,
        }
