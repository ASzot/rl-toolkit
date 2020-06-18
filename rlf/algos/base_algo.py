from rlf.storage.rollout_storage import RolloutStorage
from dataclasses import dataclass
from typing import Callable
import numpy as np

@dataclass
class AlgorithmSettings:
    """
    - ret_raw_obs: If this algorithm should have access to
        the raw observations in the `update` (via the storage object).
        A raw observation is before any preprocessing or normalization
        is applied. This raw observation is returned in the info
        dictionary of the environment.
    - mod_render_frames_fn: passes the RAW observation.
    """
    ret_raw_obs: bool
    state_fn: Callable[[np.ndarray], np.ndarray]
    action_fn: Callable[[np.ndarray], np.ndarray]
    include_info_keys: list
    mod_render_frames_fn: Callable



class BaseAlgo(object):
    """
    Base class for all algorithms to derive from. Use this class as an empty
    updater (for policies that need no learning!).
    """

    def __init__(self):
        pass

    def init(self, policy, args):
        self.update_i = 0
        self.policy = policy
        self.args = args

    def set_env_ref(self, get_env_ob_filt, env_norm):
        self.get_env_ob_filt = get_env_ob_filt

    def pre_main(self, log, env_interface):
        pass

    def first_train(self, log, eval_policy):
        """
        Called before any RL training loop starts.
        - log: logger object to log any statistics.
        - eval_policy: (policy: BasePolicy, total_num_steps: int, args) -> None
          function that evaluates the given policy with the args at timestep
          `total_num_steps`.
        """
        pass

    def get_num_updates(self):
        """
        Allows overridding the number of updates performed in the RL
        loop.Setting this is useful if an algorithm should
        dynamically calculate how many steps to take.
        """
        return int(self.args.num_env_steps) // self.args.num_steps // self.args.num_processes

    def get_completed_update_steps(self, num_updates):
        """
        num_updates: the number of times this updater has been called.
        Returns: (int) the number of environment frames processed.
        """
        return num_updates * self.args.num_processes * self.args.num_steps

    def get_env_settings(self, args):
        """
        Some updaters require specific things from the environment.
        """
        return AlgorithmSettings(False, None, None, [],
                lambda cur_frame, obs: cur_frame)

    def set_get_policy(self, get_policy_fn, policy_args):
        """
        - get_policy_fn: (None -> rlf.BasePolicy)
        Sets the factory object for creating the policy.
        """
        self._get_policy_fn = get_policy_fn
        self._policy_args = policy_args

    def _copy_policy(self):
        """
        Creates a copy of the current policy.

        returns: (rlf.BasePolicy) with same params as `self.policy`
        """
        cp_policy = self._get_policy_fn()
        cp_policy.init(*self._policy_args)
        return cp_policy

    def load_resume(self, checkpointer):
        pass

    def load(self, checkpointer):
        pass

    def save(self, checkpointer):
        pass

    def pre_update(self, cur_update):
        pass

    def update(self, storage):
        self.update_i += 1

    def get_storage_buffer(self, policy, envs, args):
        pass

    def on_traj_finished(self, traj):
        pass

    def get_add_args(self, parser):
        pass

    def get_storage_buffer(self, policy, envs, args):
        return RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space, envs.action_space, args)

    def get_requested_obs_keys(self):
        return ['observation']

