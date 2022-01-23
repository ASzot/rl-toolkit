from collections import namedtuple
from typing import List, NamedTuple

import numpy as np
import rlf.rl.utils as rutils
import torch
from rlf.il.il_dataset import ImitationLearningDataset, convert_to_tensors


class DatasetTransition(NamedTuple):
    index: int
    obs: torch.Tensor
    done: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor

    def to(self, d):
        return DatasetTransition(
            self.index,
            self.obs.to(d),
            self.done.to(d),
            self.action.to(d),
            self.next_obs.to(d),
        )


class DatasetTrajectory:
    def __init__(self):
        self._dataset_transitions = []

    def append(self, t: DatasetTransition):
        self._dataset_transitions.append(t)

    def __getitem__(self, key):
        return self._dataset_transitions[key]

    def __iter__(self):
        return iter(self._dataset_transitions)

    def __next__(self):
        return next(self._dataset_transitions)

    def __len__(self):
        return len(self._dataset_transitions)

    def obs_to_tensor(self) -> torch.Tensor:
        obs = torch.stack([t.obs for t in self], dim=0)
        last_t = self[-1].next_obs
        return torch.cat([obs, last_t.view(1, -1)], dim=0)


class TransitionDataset(ImitationLearningDataset):
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
        super().__init__(load_path, transform_dem_dataset_fn)
        if override_data is not None:
            trajs = override_data
        elif load_path.endswith(".npz"):
            trajs = self._load_npz(load_path)
        else:
            trajs = self._load_pt(load_path)

        trajs = self._transform_dem_dataset_fn(trajs)
        trajs = convert_to_tensors(trajs)

        # Convert all to floats
        self.trajs = {}
        self.trajs["obs"] = trajs["obs"].float()
        self.trajs["done"] = trajs["done"].float()
        self.trajs["actions"] = trajs["actions"].float()
        self.trajs["next_obs"] = trajs["next_obs"].float()

        self.state_mean = torch.mean(self.trajs["obs"], dim=0)
        self.state_std = torch.std(self.trajs["obs"], dim=0)
        self._compute_action_stats()

    def viz(self, args):
        import seaborn as sns

        traj_lens = []
        cur_len = 0
        for done in self.trajs["done"]:
            if done:
                traj_lens.append(cur_len)
                cur_len = 0
            else:
                cur_len += 1
        p = sns.distplot(traj_lens)
        p.set_title(
            f"{len(traj_lens)} trajs, {len(self.trajs['obs'])} transitions, taking {args.traj_frac}, val {args.traj_val_ratio}"
        )
        p.set_xlabel("Trajectory Lengths")
        save_path = rutils.plt_save(rutils.get_save_dir(args), "traj_len_dist.png")
        print(f"Saved expert data visualization to {save_path}")

    def _load_npz(self, load_path):
        x = np.load(load_path)
        n_trajs, traj_len, _ = x["obs"].shape
        N = n_trajs * traj_len
        done = torch.zeros(N)
        for i in range(n_trajs):
            done[(i + 1) * traj_len - 1] = 1

        obs = x["obs"].reshape(N, -1)
        this_obs = obs[:-1]
        next_obs = obs[1:]
        return {
            "obs": torch.tensor(this_obs),
            "rewards": torch.tensor(x["rews"].reshape(N, -1))[:-1],
            "actions": torch.tensor(x["acs"].reshape(N, -1))[:-1],
            # 1 if the transition resulted in termination
            "done": done[1:],
            "next_obs": torch.tensor(next_obs),
        }

    def _load_pt(self, load_path):
        return torch.load(load_path)

    def get_num_trajs(self):
        return int((self.trajs["done"] == 1).sum())

    def _compute_action_stats(self):
        self.action_mean = torch.mean(self.trajs["actions"], dim=0)
        self.action_std = torch.std(self.trajs["actions"], dim=0)

    def clip_actions(self, low_val, high_val):
        self.trajs["actions"] = rutils.multi_dim_clip(
            self.trajs["actions"], low_val, high_val
        )
        self._compute_action_stats()

    def to(self, device):
        for k in self.trajs:
            self.trajs[k] = self.trajs[k].to(device)
        return self

    def get_expert_stats(self, device):
        return {
            "state": (self.state_mean.to(device), self.state_std.to(device)),
            "action": (self.action_mean.to(device), self.action_std.to(device)),
        }

    def __len__(self):
        return len(self.trajs["obs"])

    def __getitem__(self, i):
        return {
            "state": self.trajs["obs"][i],
            "next_state": self.trajs["next_obs"][i],
            "done": self.trajs["done"][i],
            "actions": self.trajs["actions"][i],
        }

    def group_into_trajs(self) -> List[DatasetTrajectory]:
        idxs = range(self.trajs["obs"].shape[0])

        trajs = []
        cur_traj = DatasetTrajectory()
        Transition = namedtuple("Transition", "index obs done actions next_obs")

        for i, obs, done, actions, next_obs in zip(
            idxs,
            self.trajs["obs"],
            self.trajs["done"],
            self.trajs["actions"],
            self.trajs["next_obs"],
        ):
            cur_traj.append(DatasetTransition(i, obs, done, actions, next_obs))
            if done.item():
                trajs.append(cur_traj)
                cur_traj = DatasetTrajectory()
        if len(cur_traj) != 0:
            raise ValueError("Trajectory from dataset does not end in termination")
        return trajs

    def compute_split(self, traj_frac, rnd_seed):
        # Need to split by trajectories, not transitions
        trajs = self.group_into_trajs()

        use_count = int(len(trajs) * traj_frac)

        rng = np.random.default_rng(rnd_seed)
        rng.shuffle(trajs)
        trajs = trajs[:use_count]

        idxs = [step.index for traj in trajs for step in traj]

        return torch.utils.data.Subset(self, idxs)
