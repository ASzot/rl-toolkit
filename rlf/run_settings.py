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
from gym.spaces import Box
import os
from ray import tune

# Import the env interfaces
import rlf.envs.minigrid_interface
import rlf.envs.bit_flip
import rlf.envs.blackjack


def init_seeds(args):
    # Set all seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.set_num_threads(1)

class RunSettings(tune.Trainable):
    def __init__(self, args_str=None, config=None, logger_creator=None):
        self.args_str = args_str
        self.working_dir = os.getcwd()

        base_parser = self._get_base_parser()
        if self.args_str is None:
            self.base_args, _ = base_parser.parse_known_args()
        else:
            self.base_args, _ = base_parser.parse_known_args(self.args_str)
        super().__init__(config, logger_creator)

    def _get_base_parser(self):
        base_parser = argparse.ArgumentParser()
        self.get_add_args(base_parser)
        return base_parser

    def get_config_file(self):
        """
        - Returns (string)
        Returns the location to a config file that holds whatever information
        about the project.
        """
        return osp.join(self.working_dir, 'config.yaml')

    def create_traj_saver(self, save_path):
        return TrajSaver(save_path)

    def get_add_args(self, parser):
        pass

    def get_logger(self):
        return BaseLogger()

    def get_add_ray_config(self, config):
        return config

    def get_add_ray_kwargs(self):
        return {}

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

    def _get_env_interface(self, args, task_id=None):
        env_interface = get_env_interface(args.env_name)(args)
        env_interface.setup(args, task_id)
        return env_interface

    def get_parser(self):
        return get_default_parser()

    def get_args(self, algo, policy):
        parser = self.get_parser()
        algo.get_add_args(parser)
        policy.get_add_args(parser)

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
        _, rest_of_args = self._get_base_parser().parse_known_args(rest)
        if len(rest_of_args) != 0:
            raise ValueError('Unrecognized arguments %s' % str(rest_of_args))

        # Convert the types of some of the standard types that don't allow the
        # scientific notation when expecting integer inputs.
        args.num_env_steps = int(args.num_env_steps)
        return args

    def stop(self):
        self.ray_runner.close()
        del self.ray_runner
        del self.ray_args

    def create_runner(self, add_args={}):
        policy = self.get_policy()
        algo = self.get_algo()

        # Set up args used for training
        args = self.get_args(algo, policy)
        args.cwd = self.working_dir
        if 'wandb' in add_args:
            del add_args['wandb']
        rutils.update_args(args, add_args, True)
        if 'cwd' in add_args:
            self.working_dir = add_args['cwd']

        config_mgr.init(self.get_config_file())
        if args.ray:
            # No logger when ray is tuning
            log = BaseLogger()
        else:
            log = self.get_logger()
        log.init(args)
        log.set_prefix(args)

        args.device = torch.device("cuda:0" if args.cuda else "cpu")
        init_seeds(args)

        env_interface = self._get_env_interface(args)

        checkpointer = Checkpointer(args)

        alg_env_settings = algo.get_env_settings(args)

        # Setup environment
        envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                             args.gamma, args.device,
                             False, env_interface, args,
                             alg_env_settings)

        rutils.pstart_sep()
        print('Action space:', envs.action_space)
        if isinstance(envs.action_space, Box):
            print('Action range:', (envs.action_space.low, envs.action_space.high))
        print('Observation space', envs.observation_space)
        rutils.pend_sep()

        # Setup policy
        policy_args = (envs.observation_space, envs.action_space, args)
        policy.init(*policy_args)
        policy = policy.to(args.device)
        policy.watch(log)

        # Setup algo
        algo.set_get_policy(self.get_policy, policy_args)
        algo.init(policy, args)
        algo.set_env_ref(envs)

        # Setup storage buffer
        storage = algo.get_storage_buffer(policy, envs, args)
        for ik, get_shape in alg_env_settings.include_info_keys:
            storage.add_info_key(ik, get_shape(envs))
        storage.to(args.device)
        storage.init_storage(envs.reset())
        storage.set_traj_done_callback(algo.on_traj_finished)

        runner = Runner(envs, storage, policy, log, env_interface, checkpointer, args, algo)
        return runner

    def import_add(self):
        pass

    def setup(self, config):
        self.import_add()
        self.ray_runner = self.create_runner(config)
        self.ray_runner.setup()
        self.ray_args = self.ray_runner.args

        if not self.ray_args.ray_debug:
            self.ray_runner.log.disable_print()

    def step(self):
        updater_log_vals = self.ray_runner.training_iter(self.training_iteration)
        if (self.training_iteration+1) % self.ray_args.log_interval == 0:
            log_dict = self.ray_runner.log_vals(updater_log_vals, self.training_iteration)
        if (self.training_iteration+1) % self.ray_args.save_interval == 0:
            self.ray_runner.save()
        if (self.training_iteration+1) % self.ray_args.eval_interval == 0:
            self.ray_runner.eval()

        return log_dict
