import torch.utils.data
from abc import ABC, abstractmethod

class ImitationLearningDataset(torch.utils.data.Dataset, ABC):
    """
    The data should be a dictionary saved with `torch.save`, consisting of
        {
        'done': torch.tensor
        'obs': torch.tensor
        'next_obs': torch.tensor
        'actions': torch.tensor
        }
        All lists should be exactly the same length.
    """
    def viz(self, args):
        pass

    @abstractmethod
    def get_expert_stats(self, device):
        pass

    @abstractmethod
    def get_num_trajs(self):
        pass

    @abstractmethod
    def compute_split(self, traj_frac):
        pass

    def clip_actions(self, low_val, high_val):
        pass

    def to(self, device):
        return self
