import os
import os.path as osp
from collections import defaultdict

import numpy as np
import rlf.rl.utils as rutils
import torch


def decode_gw_state(state):
    state = state.view(8, 8, 3)
    found_idx = None
    for x in range(state.shape[0]):
        for y in range(state.shape[1]):
            if state[x][y][0] == 10:
                found_idx = (x, y)
                break
        if found_idx is not None:
            break
    return found_idx


class TrajSaver(object):
    """
    A helper class to accumulate and save transition pairs. Note that obs
    refers to RAW observations not normalized observations. This to preserve
    any observation information for a downstream task.
    """

    def __init__(self, save_dir, is_stochastic_policy=False):
        self.save_dir = save_dir
        self.all_obs = []
        self.all_next_obs = []
        self.all_done = []
        self.all_actions = []
        self.all_info = []
        if is_stochastic_policy:
            add_txt = "_stochastic"
        else:
            add_txt = ""
        self.save_name = f"trajs{add_txt}.pt"

        self.traj_buffer = defaultdict(list)
        if not osp.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def should_save_traj(self, traj):
        """
        Can override in child class to filter some trajectories from being
        saved.
        """
        return True

    def _collect_traj(self, trajs):
        obs, next_obs_cp, done, action, info = list(zip(*trajs))
        self.all_obs.extend(obs)
        self.all_next_obs.extend(next_obs_cp)
        self.all_done.extend(done)
        self.all_actions.extend(action)
        self.all_info.extend(info)

    def collect(self, obs, next_obs, done, action, info):
        """
        Collects a transition to be later saved. Note that only full
        trajectories are saved.
        - obs (tensor[n_processes, *obs_dim])
        - done (list(bool)[n_processes])
        """
        next_obs_cp = rutils.obs_op(next_obs, lambda x: x.clone())
        # Only count trajectories that satisfy the `self.should_save_traj` condition
        num_done = 0
        batch_size = action.shape[0]
        for i in range(batch_size):
            if done[i]:
                next_obs_cp[i] = torch.tensor(info[i]["final_obs"]).to(action.device)
            self.traj_buffer[i].append(
                (
                    rutils.obs_select(obs, i),
                    rutils.obs_select(next_obs_cp, i),
                    done[i],
                    action[i],
                    info[i],
                )
            )
            if done[i]:
                if self.should_save_traj(self.traj_buffer[i]):
                    self._collect_traj(self.traj_buffer[i])
                    num_done += 1
                else:
                    print("Skipping trajectory")
                self.traj_buffer[i] = []
        return num_done

    def save(self):
        """
        Saves all observations, actions, and masks from trajectories. Saves all
        data in info starting with key `ep_`.
        """
        info_tensors = defaultdict(lambda: torch.zeros(len(self.all_info)))

        def add_info(info, i):
            for k, v in info.items():
                if isinstance(v, dict):
                    add_info(v, i)
                elif k.startswith("ep_"):
                    # Convert to a float because saving bools or ints won't work
                    # in the torch tensor.
                    info_tensors[k][i] = float(v)

        for i, info in enumerate(self.all_info):
            add_info(info, i)

        for k, v in info_tensors.items():
            info_tensors[k] = info_tensors[k].view(-1)

        if len(self.all_obs) == 0:
            raise ValueError("There is no data to save")

        if isinstance(self.all_obs[0], torch.Tensor):
            save_obs = torch.stack(self.all_obs).cpu().detach()
            save_next_obs = torch.stack(self.all_next_obs).cpu().detach()
        else:
            save_obs = rutils.transpose_arr_dict(self.all_obs)
            save_next_obs = rutils.transpose_arr_dict(self.all_next_obs)
            save_obs = rutils.obs_op(save_obs, lambda x: x.cpu().detach())
            save_next_obs = rutils.obs_op(save_next_obs, lambda x: x.cpu().detach())

        save_done = (
            torch.tensor(np.array(self.all_done).astype(np.float32)).cpu().detach()
        )
        save_actions = torch.stack(self.all_actions).cpu().detach()

        save_name = osp.join(self.save_dir, self.save_name)
        n_steps = len(save_actions)
        print("Saving %i transitions to %s" % (n_steps, save_name))

        torch.save(
            {
                "obs": save_obs,
                "next_obs": save_next_obs,
                "done": save_done,
                "actions": save_actions,
                **info_tensors,
            },
            save_name,
        )
        print("Successfully saved trajectories to %s" % save_name)
        return save_name


class GoalTrajSaver(TrajSaver):
    """
    Only saves trajectories if they successfully accomplish the goal.
    """

    def __init__(
        self,
        save_dir,
        look_for_subkey: str,
        assert_saved: bool = False,
        is_stochastic_policy=False,
    ):
        """
        :param assert_saved: If true, will raise an exception if the trajectory did not end in success.
        :param look_for_subkey: A substring of a key in the info dictionary
            that will be used to determine if the trajectory ended in success or
            not.

        """
        self.assert_saved = assert_saved
        self._look_for_key = look_for_subkey
        super().__init__(save_dir, is_stochastic_policy)

    def should_save_traj(self, traj):
        last_info = traj[-1][-1]
        succ_key = [x for x in last_info.keys() if self._look_for_key in x]
        if len(succ_key) != 1:
            raise ValueError(f"Cannot find success key: {succ_key}")
        ret = last_info[succ_key[0]] == 1.0

        if self.assert_saved and not ret:
            raise ValueError("Trajectory did not end successfully")
        return ret
