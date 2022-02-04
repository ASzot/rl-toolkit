from collections import defaultdict

import numpy as np
import rlf.rl.utils as rutils
import torch
from rlf.il.il_dataset import ImitationLearningDataset, convert_to_tensors
from rlf.il.transition_dataset import TransitionDataset


class TrajDataset(TransitionDataset):
    """
    See `rlf/il/il_dataset.py` for notes about the demonstration dataset
    format.
    """

    def __init__(
        self,
        load_path,
        transform_dem_dataset_fn,
        override_data=None,
    ):
        super().__init__(load_path, transform_dem_dataset_fn, override_data)
        self.trajs = self.group_into_trajs()

    def get_add_data_loader_kwargs(self):
        """
        When batching multiple trajectories, flattening the horizon dimension together.
        """

        def coallate(batch):
            return {
                k: torch.cat([batch[i][k] for i in range(len(batch))], dim=0)
                for k in batch[0].keys()
            }

        return {"collate_fn": coallate}

    def viz(self, args):
        import seaborn as sns

        traj_lens = [len(t) for t in self.trajs]
        if len(traj_lens) == 0 or np.var(traj_lens) == 0:
            return
        p = sns.distplot(traj_lens)
        p.set_title(f"{len(traj_lens)} trajs")
        p.set_xlabel("Trajectory Lengths")
        save_path = rutils.plt_save(rutils.get_save_dir(args), "traj_len_dist.png")
        print(f"Saved expert data visualization to {save_path}")

    def compute_split(self, traj_frac, rnd_seed):
        use_count = int(len(self.trajs) * traj_frac)
        idxs = np.arange(len(self.trajs))

        rng = np.random.default_rng(rnd_seed)
        rng.shuffle(idxs)
        idxs = idxs[:use_count]

        return torch.utils.data.Subset(self, idxs)

    def to(self, device):
        for i, traj in enumerate(self.trajs):
            self.trajs[i] = [t.to(device) for t in traj]
        return self

    def __getitem__(self, i):
        return {
            "state": torch.stack([t.obs for t in self.trajs[i]]),
            "next_state": torch.stack([t.next_obs for t in self.trajs[i]]),
            "done": torch.stack([t.done for t in self.trajs[i]]),
            "actions": torch.stack([t.action for t in self.trajs[i]]),
        }

    def __len__(self):
        return len(self.trajs)

    def get_num_trajs(self):
        return len(self.trajs)
