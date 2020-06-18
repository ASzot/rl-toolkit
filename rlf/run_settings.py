from rlf.rl.checkpointer import Checkpointer
from rlf.rl.envs import make_vec_envs
from rlf.rl.evaluation import full_eval
from rlf.args import get_default_parser
from rlf.envs.env_interface import get_env_interface
import argparse
from rlf.rl.loggers.base_logger import BaseLogger
import rlf.rl.utils as rutils
import torch
from rlf.exp_mgr import config_mgr
from rlf.il.traj_mgr import TrajSaver
import rlf.rl.utils as rutils
from rlf.rl.runner import Runner
import numpy as np
import random
import os.path as osp
from rlf.rl.envs import get_vec_normalize



def init_torch(args):
    # Set all seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.set_num_threads(1)


def load_from_checkpoint(policy, envs, checkpointer, updater):
    policy.load_state_dict(checkpointer.get_key('policy'))

    if checkpointer.has_load_key('ob_rms'):
        ob_rms_dict = checkpointer.get_key('ob_rms')
        vec_norm = get_vec_normalize(envs)
        if vec_norm is not None:
            # vec_norm.eval()
            vec_norm.ob_rms_dict = ob_rms_dict
    updater.load(checkpointer)


class RunSettings(object):
    def __init__(self, args_str=None):
        self.args = None
        self.args_str = args_str
        self.eval_result = None

        base_parser = argparse.ArgumentParser()
        self.get_add_args(base_parser)
        if self.args_str is None:
            self.base_args, _ = base_parser.parse_known_args()
        else:
            self.base_args, _ = base_parser.parse_known_args(self.args_str)
        self.base_parser = base_parser

        self.policy = self.get_policy()
        self.algo = self.get_algo()

        self.args = self.get_args()

    def get_config_file(self):
        """
        - Returns (string)
        Returns the location to a config file that holds whatever information
        about the project.
        """

        config_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
        return osp.join(config_dir, 'config.yaml')

    def create_traj_saver(self, save_path):
        return TrajSaver(save_path)

    def get_add_args(self, parser):
        pass

    def get_logger(self):
        return BaseLogger()

    def get_policy(self):
        """
        Return: rlf.base_policy.BasePolicy
        """
        raise NotImplemented('Must return policy to be used.')

    def get_algo(self):
        """
        Return: rlf.base_algo.BaseAlgo
        """
        raise NotImplemented('Must return algorithm to be used')

    def get_env_interface(self):
        return self._get_env_interface(self.get_args())

    def _get_env_interface(self, args, task_id=None):
        env_interface = get_env_interface(args.env_name)(args)
        env_interface.setup(args, task_id)
        return env_interface

    def get_parser(self):
        return get_default_parser()

    def get_args(self):
        if self.args is not None:
            # If cached don't get them again
            return self.args

        parser = self.get_parser()
        self.algo.get_add_args(parser)
        self.policy.get_add_args(parser)

        if self.args_str is None:
            args, rest = parser.parse_known_args()
        else:
            args, rest = parser.parse_known_args(self.args_str)

        env_parser = argparse.ArgumentParser()
        get_env_interface(args.env_name)(args).get_add_args(env_parser)
        env_args, rest = env_parser.parse_known_args(rest)
        # Assign the env args to the main args namespace.
        rutils.update_args(args, vars(env_args))

        # Check that there are no arguments not accounted for in `base_args`
        _, rest_of_args = self.base_parser.parse_known_args(rest)
        if len(rest_of_args) != 0:
            raise ValueError('Unrecognized arguments %s' % str(rest_of_args))

        # Convert the types of some of the standard types that don't allow the
        # scientific notation when expecting integer inputs.
        args.num_env_steps = int(args.num_env_steps)
        return args

    def get_num_updates(self):
        args = self.get_args()
        config_mgr.init(self.get_config_file())

        args.device = torch.device("cuda:0" if args.cuda else "cpu")
        init_torch(args)

        env_interface = self.get_env_interface()

        checkpointer = Checkpointer(args)

        policy = self.policy
        updater = self.algo

        alg_env_settings = updater.get_env_settings(args)

        # Create the environment
        envs = make_vec_envs(args.env_name, args.seed, 1,
                             args.gamma, args.env_log_dir, args.device,
                             False, env_interface, args,
                             alg_env_settings)

        policy_args = (envs.observation_space, envs.action_space, args)

        policy.init(*policy_args)
        policy = policy.to(args.device)

        updater.set_get_policy(self.get_policy, policy_args)
        updater.init(policy, args)
        envs.close()
        return updater.get_num_updates()

    def setup(self):
        # Set up args used for training
        args = self.get_args()
        config_mgr.init(self.get_config_file())
        log = self.get_logger()
        log.init(args)
        log.set_prefix(args)

        args.device = torch.device("cuda:0" if args.cuda else "cpu")
        init_torch(args)

        env_interface = self.get_env_interface()

        checkpointer = Checkpointer(args)

        policy = self.policy
        updater = self.algo

        alg_env_settings = updater.get_env_settings(args)

        # Create the environment
        envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                             args.gamma, args.env_log_dir, args.device,
                             False, env_interface, args,
                             alg_env_settings)

        policy_args = (envs.observation_space, envs.action_space, args)

        policy.init(*policy_args)
        policy = policy.to(args.device)

        updater.set_get_policy(self.get_policy, policy_args)
        updater.init(policy, args)

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
        updater.set_env_ref(get_vec_normalize_fn, env_norm)

        if checkpointer.should_load():
            load_from_checkpoint(policy, envs, checkpointer, updater)

        storage = updater.get_storage_buffer(policy, envs, args)
        for ik, get_shape in alg_env_settings.include_info_keys:
            storage.add_info_key(ik, get_shape(envs))
        storage.to(args.device)
        storage.init_storage(envs.reset())
        storage.set_traj_done_callback(updater.on_traj_finished)

        if args.eval_only:
            self.eval_result = full_eval(envs, policy, log, checkpointer, env_interface, args,
                      alg_env_settings, self.create_traj_saver)
            envs.close()
            return None

        policy.watch(log)

        start_update = 0
        if args.resume:
            updater.load_resume(checkpointer)
            policy.load_resume(checkpointer)
            start_update = checkpointer.get_key('step')

        updater.pre_main(log, env_interface)

        num_updates = updater.get_num_updates()
        print('Updater requested to update %i times' % num_updates)

        return Runner(envs, storage, policy, log, start_update, num_updates,
                      env_interface, checkpointer, args, updater)

    def import_add(self):
        pass
