"""
Code is heavily based off of https://github.com/denisyarats/pytorch_sac.
The license is at `rlf/algos/off_policy/denis_yarats_LICENSE.md`
"""
import random
from collections import defaultdict

import numpy as np
import rlf.rl.utils as rutils
import torch
from rlf.storage.base_storage import BaseStorage


class TransitionStorage(BaseStorage):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, capacity, device):
        super().__init__()

        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.masks = np.empty((capacity, 1), dtype=np.float32)
        self.masks_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

        self._modify_reward_fn = None

    def get_generator(self, from_recent, num_samples, mini_batch_size, **kwargs):
        """To do the same thing as the on policy rollout storage, this does not
        return the next state.
        """
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
        num_batches = num_samples // mini_batch_size
        for _ in range(num_batches):
            idxs = np.random.choice(all_indices, mini_batch_size)

            obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
            actions = torch.as_tensor(self.actions[idxs], device=self.device)
            rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
            masks = torch.as_tensor(self.masks[idxs], device=self.device)

            yield {
                "state": obses,
                "reward": rewards,
                "action": actions,
                "mask": masks,
            }

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample_tensors(self, batch_size):
        (
            state,
            action,
            reward,
            n_state,
            not_done,
            not_done_no_max,
        ) = self.sample(batch_size)
        return state, n_state, action, reward, {}, {"mask": not_done}

    def init_storage(self, obs):
        batch_size = obs.shape[0]
        hxs = {}
        self.last_seen = {
            "obs": obs,
            "masks": torch.zeros(batch_size, 1),
            "hxs": hxs,
        }

    def get_obs(self, step):
        ret_obs = self.last_seen["obs"]
        return ret_obs

    def get_hidden_state(self, step):
        return self.last_seen["hxs"]

    def get_masks(self, step):
        return self.last_seen["masks"]

    def insert(self, obs, next_obs, reward, done, infos, ac_info):
        masks, bad_masks = self.compute_masks(done, infos)
        self.last_seen = {
            "obs": next_obs,
            "masks": masks,
            "hxs": ac_info.hxs,
        }
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(next_obs, torch.Tensor):
            next_obs = next_obs.cpu().numpy()
        action = ac_info.take_action

        def copy_from_to(buffer_start, batch_start, how_many):
            buffer_slice = slice(buffer_start, buffer_start + how_many)
            batch_slice = slice(batch_start, batch_start + how_many)
            np.copyto(self.obses[buffer_slice], obs[batch_slice])
            np.copyto(self.actions[buffer_slice], action[batch_slice])
            np.copyto(self.rewards[buffer_slice], reward[batch_slice])
            np.copyto(self.next_obses[buffer_slice], next_obs[batch_slice])
            np.copyto(self.masks[buffer_slice], done[batch_slice])
            np.copyto(self.masks_no_max[buffer_slice], bad_masks[batch_slice])

        _batch_start = 0
        buffer_end = self.idx + len(obs)
        if buffer_end > self.capacity:
            copy_from_to(self.idx, _batch_start, self.capacity - self.idx)
            _batch_start = self.capacity - self.idx
            self.idx = 0
            self.full = True

        _how_many = len(obs) - _batch_start
        copy_from_to(self.idx, _batch_start, _how_many)
        self.idx = (self.idx + _how_many) % self.capacity
        self.full = self.full or self.idx == 0

    def set_modify_reward_fn(self, modify_reward_fn):
        self._modify_reward_fn = modify_reward_fn

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        masks = torch.as_tensor(self.masks[idxs], device=self.device)
        masks_no_max = torch.as_tensor(self.masks_no_max[idxs], device=self.device)

        if self._modify_reward_fn is not None:
            rewards = self._modify_reward_fn(obses, actions, next_obses, masks)

        return obses, actions, rewards, next_obses, masks, masks_no_max
