from rlf.storage.rollout_storage import RolloutStorage
from typing import Callable
import numpy as np
import rlf.rl.utils as rutils
from rlf.rl.envs import get_vec_normalize
import attr

@attr.s(auto_attribs=True, slots=True)
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
    Updater classes typically look like:
    ```
    class MyUpdater([OnPolicy or OffPolicy]):
        def update(self, rollouts):
            # Calculate loss and then update:
            loss = ...
            self._standard_step(loss)
            return {
                    'logging_stat': 0.0
                    }

        def get_add_args(self, parser):
            super().get_add_args(parser)
            # Add additional arguments.
    ```
    """

    def __init__(self):
        pass

    def init(self, policy, args):
        self.update_i = 0
        self.policy = policy
        self.args = args

    def set_env_ref(self, envs):
        env_norm = get_vec_normalize(envs)
        def get_vec_normalize_fn():
            if env_norm is not None:
                obfilt = get_vec_normalize(envs)._obfilt

                def mod_env_ob_filt(state, update=True):
                    state = obfilt(state, update)
                    state = rutils.get_def_obs(state)
                    return state
                return mod_env_ob_filt
            return None
        self.get_env_ob_filt = get_vec_normalize_fn

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
        if self.args.num_steps == 0:
            return 0
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

    def on_traj_finished(self, traj):
        """
        done_trajs: A list of transitions where each transition is a tuple of form:
            (state,action,mask,info_dict,reward). The data is a bit confusing.
            mask[t] is technically the mask at t+1. The mask at t=0 is always
            1. The final state is NOT included and must be included through the
            info_dict if needed.
        """
        pass

    def get_add_args(self, parser):
        pass

    def get_storage_buffer(self, policy, envs, args):
        return RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space, envs.action_space, args)

    def get_requested_obs_keys(self):
        return ['observation']

