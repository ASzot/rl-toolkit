from rlf.storage.base_storage import BaseStorage
import random
import torch
import rlf.rl.utils as rutils
from collections import defaultdict


class TransitionStorage(BaseStorage):
    def __init__(self, capacity, args, hidden_states={}):
        super().__init__()
        self.args = args
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.last_seen = None
        self.set_device = None
        self.hidden_state_dims = hidden_states

    def _push_transition(self, transition):
        if len(self.memory) < self.capacity:
            # Add a new element to the list and then populate it.
            self.memory.append(None)

        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def insert(self, obs, next_obs, reward, done, infos, ac_info):
        super().insert(obs, next_obs, reward, done, infos, ac_info)
        masks, bad_masks = self.compute_masks(done, infos)

        batch_size = rutils.get_def_obs(obs).shape[0]
        for i in range(batch_size):
            self._push_transition({
                    'action': ac_info.action[i],
                    'state': rutils.obs_select(obs, i),
                    'mask': self.last_seen['masks'][i],
                    'hxs': rutils.deep_dict_select(self.last_seen['hxs'], i),
                    'reward': reward[i],
                    'next_state': rutils.obs_select(next_obs, i),
                    'next_mask': masks[i],
                    'next_hxs': rutils.deep_dict_select(ac_info.hxs, i),
                    })

        self.last_seen = {
                'obs': next_obs,
                'masks': masks,
                'hxs': ac_info.hxs,
                }

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample_tensors(self, sample_size):
        transitions = self.sample(sample_size)

        states = []
        add_states = defaultdict(list)
        actions = []
        masks = []
        hxs = defaultdict(list)
        rewards = []

        next_states = []
        next_add_states = defaultdict(list)
        next_masks = []
        next_hxs = defaultdict(list)

        for x in transitions:
            states.append(rutils.get_def_obs(x['state']))
            for k, v in rutils.get_other_obs(x['state']).items():
                add_states[k].append(v)
            actions.append(x['action'])
            masks.append(x['mask'])
            for k, v in x['hxs'].items():
                hxs[k].append(x['hxs'][k])
            rewards.append(x['reward'])

            next_states.append(rutils.get_def_obs(x['next_state']))
            for k, v in rutils.get_other_obs(x['next_state']).items():
                next_add_states[k].append(v)
            next_masks.append(x['next_mask'])
            for k, v in x['next_hxs'].items():
                next_hxs[k].append(x['next_hxs'][k])

        states = torch.stack(states)
        for k,v in add_states.items():
            add_states[k] = torch.stack(v)
        actions = torch.stack(actions)
        masks = torch.stack(masks)
        for k,v in hxs.items():
            hxs[k] = torch.stack(hxs[k])
        rewards = torch.stack(rewards)

        next_states = torch.stack(next_states)
        for k,v in next_add_states.items():
            next_add_states[k] = torch.stack(v)
        next_masks = torch.stack(next_masks)
        for k,v in next_hxs.items():
            next_hxs[k] = torch.stack(next_hxs[k])

        if self.set_device is not None:
            masks = masks.to(self.set_device)
            rewards = rewards.to(self.set_device)

        cur_add = {
            'hxs': hxs,
            'masks': masks,
            'add_state': add_states,
        }
        next_add = {
            'hxs': next_hxs,
            'masks': next_masks,
            'add_state': next_add_states,
        }
        return states, next_states, actions, rewards, cur_add, next_add

    def __len__(self):
        return len(self.memory)

    def init_storage(self, obs):
        super().init_storage(obs)
        batch_size = rutils.get_def_obs(obs).shape[0]
        hxs = {}
        for k, dim in self.hidden_state_dims.items():
            hxs[k] = torch.zeros(batch_size, dim)
        self.last_seen = {
                'obs': obs,
                # Start with saying we are done since this is the start of the
                # first episode
                'masks': torch.zeros(batch_size, 1),
                'hxs': hxs,
                }

    def get_obs(self, step):
        ret_obs = self.last_seen['obs']
        if self.set_device is not None:
            for k, v in ret_obs.items():
                ret_obs[k] = v.to(self.set_device)
        return ret_obs

    def get_hidden_state(self, step):
        return self.last_seen['hxs']

    def get_masks(self, step):
        return self.last_seen['masks']

    def to(self, device):
        self.set_device = device
