from rlf.storage.base_storage import BaseStorage
import random
import torch


class TransitionStorage(BaseStorage):
    def __init__(self, capacity, create_transition_fn):
        super().__init__()
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.create_transition_fn = create_transition_fn
        self.last_seen = None
        self.set_device = None

    def insert(self, obs, next_obs, reward, done, infos, ac_info):
        masks, bad_masks = self.compute_masks(done, infos)

        if len(self.memory) < self.capacity:
            # Add a new element to the list and then populate it.
            self.memory.append(None)
        self.memory[self.position] = self.create_transition_fn(obs, next_obs,
                reward, done, masks, bad_masks, ac_info, self.last_seen)

        self.last_seen = {
                'obs': next_obs,
                'masks': masks,
                'rnn_hxs': ac_info.rnn_hxs,
                }
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def init_storage(self, obs):
        batch_size = obs.shape[0]
        self.last_seen = {
                'obs': obs,
                # Start with saying we are done since this is the start of the
                # first episode
                'masks': torch.tensor([[0.0] for _ in range(batch_size)]),
                'rnn_hxs': torch.tensor([0 for _ in range(batch_size)])
                }

    def get_obs(self, step):
        ret_obs = self.last_seen['obs']
        if self.set_device is not None:
            ret_obs = ret_obs.to(self.set_device)
        return ret_obs

    def get_recurrent_hidden_state(self, step):
        return self.last_seen['rnn_hxs']

    def get_masks(self, step):
        return self.last_seen['masks']

    def to(self, device):
        self.set_device = device
