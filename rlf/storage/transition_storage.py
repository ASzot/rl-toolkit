"""
Code is heavily based off of https://github.com/denisyarats/pytorch_sac.
The license is at `rlf/algos/off_policy/denis_yarats_LICENSE.md`
"""
import pickle
import random
from collections import defaultdict
from typing import Optional

import gym.spaces as spaces
import numpy as np
import rlf.rl.utils as rutils
import torch
from rlf.storage.base_storage import BaseStorage


class TransitionStorage(BaseStorage):
    def __init__(self, obs_space, action_shape, capacity, storage_device, args):
        super().__init__()

        self.capacity = capacity
        self.device = args.device
        self.args = args
        self.obs_space = obs_space
        self.action_shape = action_shape

        if isinstance(obs_space, spaces.Dict):
            self.ob_keys = {k: space.shape for k, space in obs_space.spaces.items()}
        else:
            self.ob_keys = {self.args.policy_ob_key: obs_space.shape}

        self.obses = {}
        self.next_obses = {}
        for k, obs_shape in self.ob_keys.items():
            obs_dtype = torch.float32 if len(obs_shape) == 1 else torch.uint8

            self.obses[k] = torch.zeros(
                capacity, *obs_shape, dtype=obs_dtype, device=storage_device
            )
            self.next_obses[k] = torch.zeros(
                capacity, *obs_shape, dtype=obs_dtype, device=storage_device
            )

        self.actions = torch.zeros(capacity, *action_shape, device=storage_device)
        self.rewards = torch.zeros(capacity, 1, device=storage_device)
        self.masks = torch.zeros(capacity, 1, device=storage_device)
        self.masks_no_max = torch.zeros(capacity, 1, device=storage_device)

        self.idx = 0
        self.last_save = 0
        self.full = False

        self._modify_reward_fn = None

    def get_generator(
        self,
        from_recent: bool,
        num_samples: Optional[int],
        mini_batch_size: int,
        n_mini_batches: int = -1,
        **kwargs,
    ):
        """To do the same thing as the on policy rollout storage, this does not
        return the next state.
        :param from_recent: If True, this will grab the most recent `num_samples`.
        :param num_samples:
        :param n_mini_batches: The maximum number of batches of size mini_batch_size to return.
            If -1, then the number of mini batches is not restricted and will be computed based off the buffer size.
        """
        if num_samples is None:
            num_samples = len(self)
        if num_samples > len(self):
            return None
        if from_recent:
            all_indices = []
            max_side = self.idx
            all_indices = list(range(max(self.idx - num_samples, 0), self.idx))

            overflow_amount = num_samples - self.idx
            if self.full and overflow_amount > 0:
                all_indices.extend(
                    [list(range(self.capacity - overflow_amount, self.capacity))]
                )
        else:
            all_indices = list(range(0, self.capacity if self.full else self.idx))

        if n_mini_batches > 0:
            num_batches = min(num_samples // mini_batch_size, n_mini_batches)
        else:
            num_batches = num_samples // mini_batch_size
        for _ in range(num_batches):
            idxs = np.random.choice(all_indices, mini_batch_size)

            obses, other_obses = self._dict_sel(self.obses, idxs)
            next_obses, other_next_obses = self._dict_sel(self.next_obses, idxs)
            actions = self.actions[idxs]
            rewards = self.rewards[idxs]
            masks = self.masks[idxs]

            if len(obses) == 1:
                obses = next(iter(obses.values()))
                next_obses = next(iter(next_obses.values()))

            yield {
                "state": obses,
                "other_state": other_obses,
                "next_state": next_obses,
                "other_next_state": other_next_obses,
                "reward": rewards,
                "action": actions,
                "mask": masks,
            }

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample_tensors(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses, other_obses = self._dict_sel(self.obses, idxs)
        next_obses, other_next_obses = self._dict_sel(self.obses, idxs)

        actions = self.actions[idxs].to(self.device)
        rewards = self.rewards[idxs].to(self.device)
        masks = self.masks[idxs].to(self.device)
        masks_no_max = self.masks_no_max[idxs].to(self.device)

        if self._modify_reward_fn is not None:
            rewards = self._modify_reward_fn(obses, actions, next_obses, masks)

        if len(obses) == 1:
            obses = next(iter(obses.values()))
            next_obses = next(iter(next_obses.values()))

        return (
            obses,
            next_obses,
            actions,
            rewards,
            {"other_state": other_obses},
            {"mask": masks, "other_state": other_next_obses},
        )

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

    def _dict_sel(self, obs, idx: torch.Tensor):
        obs_batch = None
        other_obs_batch = {}
        for k, ob_shape in self.ob_keys.items():
            use_obs = obs[k][idx].to(self.device)
            if k == self.args.policy_ob_key:
                obs_batch = use_obs
            else:
                other_obs_batch[k] = use_obs
        return obs_batch, other_obs_batch

    def insert(self, obs, next_obs, reward, done, infos, ac_info):
        masks, bad_masks = self.compute_masks(done, infos)
        self.last_seen = {
            "obs": next_obs,
            "masks": masks,
            "hxs": ac_info.hxs,
        }

        use_next_obs = {}
        use_obs = {}

        if not isinstance(obs, dict):
            obs = {self.args.policy_ob_key: obs}
            next_obs = {self.args.policy_ob_key: next_obs}

        storage_device = self.actions.device

        for k in self.ob_keys:
            if isinstance(obs[k], torch.Tensor):
                use_obs[k] = obs[k].to(storage_device)
                use_next_obs[k] = next_obs[k].to(storage_device)
            elif isinstance(obs[k], np.ndarray):
                use_obs[k] = torch.tensor(obs[k]).to(storage_device)
                use_next_obs[k] = torch.tensor(next_obs[k]).to(storage_device)
            else:
                raise ValueError(f"Unrecognized observation data format {type(obs[k])}")

        action = ac_info.take_action

        def copy_from_to(buffer_start, batch_start, how_many):
            buffer_slice = slice(buffer_start, buffer_start + how_many)
            batch_slice = slice(batch_start, batch_start + how_many)

            for k, ob_shape in self.ob_keys.items():
                self.obses[k][buffer_slice] = use_obs[k][batch_slice].clone().detach()
                self.next_obses[k][buffer_slice] = (
                    use_next_obs[k][batch_slice].clone().detach()
                )

            self.actions[buffer_slice] = action[batch_slice].clone().detach()
            self.rewards[buffer_slice] = reward[batch_slice].clone().detach()
            self.masks[buffer_slice] = masks[batch_slice].clone().detach()
            self.masks_no_max[buffer_slice] = bad_masks[batch_slice].clone().detach()

        batch_start = 0
        obs_len = rutils.get_def_obs(use_obs).shape[0]
        buffer_end = self.idx + obs_len
        if buffer_end > self.capacity:
            copy_from_to(self.idx, batch_start, self.capacity - self.idx)
            batch_start = self.capacity - self.idx
            self.idx = 0
            self.full = True

        how_many = obs_len - batch_start
        copy_from_to(self.idx, batch_start, how_many)
        self.idx = (self.idx + how_many) % self.capacity
        self.full = self.full or self.idx == 0

    def set_modify_reward_fn(self, modify_reward_fn):
        self._modify_reward_fn = modify_reward_fn
