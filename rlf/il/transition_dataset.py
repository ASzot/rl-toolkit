from rlf.il.il_dataset import ImitationLearningDataset
import torch
import rlf.rl.utils as rutils
import numpy as np


class TransitionDataset(ImitationLearningDataset):
    def __init__(self, load_path, transform_dem_dataset_fn):
        self.trajs = torch.load(load_path)
        self.trajs = transform_dem_dataset_fn(self.trajs)

        # Convert all to floats
        self.trajs['obs'] = self.trajs['obs'].float()
        self.trajs['done'] = self.trajs['done'].float()
        self.trajs['actions'] = self.trajs['actions'].float()
        self.trajs['next_obs'] = self.trajs['next_obs'].float()

        self.state_mean = torch.mean(self.trajs['obs'], dim=0)
        self.state_std = torch.std(self.trajs['obs'], dim=0)
        self._compute_action_stats()

    def get_num_trajs(self):
        return int((self.trajs['done'] == 1).sum())

    def _compute_action_stats(self):
        self.action_mean = torch.mean(self.trajs['actions'], dim=0)
        self.action_std = torch.std(self.trajs['actions'], dim=0)

    def clip_actions(self, low_val, high_val):
        self.trajs['actions'] = rutils.multi_dim_clip(self.trajs['actions'],
                low_val, high_val)
        self._compute_action_stats()

    def get_expert_stats(self, device):
        return {
                'state': (self.state_mean.to(device), self.state_std.to(device)),
                'action': (self.action_mean.to(device), self.action_std.to(device))
                }

    def __len__(self):
        return len(self.trajs['obs'])

    def __getitem__(self, i):
        return {
                'state': self.trajs['obs'][i],
                'next_state': self.trajs['next_obs'][i],
                'done': self.trajs['done'][i],
                'actions': self.trajs['actions'][i]
                }


    def compute_split(self, traj_frac):
        use_count = int(len(self) * traj_frac)
        all_idxs = np.arange(0, len(self))
        np.random.shuffle(all_idxs)
        idxs = all_idxs[:use_count]
        return torch.utils.data.Subset(self, idxs)
