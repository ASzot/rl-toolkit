import torch.utils.data
from abc import ABC, abstractmethod

class ImitationLearningDataset(torch.utils.data.Dataset, ABC):
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
