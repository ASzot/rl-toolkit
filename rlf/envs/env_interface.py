import os.path as osp
import re
from collections import OrderedDict

import gym
import numpy as np
from rlf.baselines.vec_env.vec_env import VecEnv


class EnvInterface(object):
    def __init__(self, args):
        self.args = args

    def setup(self, args, task_id):
        self.task_id = task_id

    def requires_tensor_wrap(self) -> bool:
        """
        If False, there will be no numpy to torch conversion wrapper.
        """
        return True

    def env_trans_fn(self, env, set_eval):
        return env

    def final_trans_fn(self, env):
        return env

    def get_special_stat_names(self):
        return []

    def get_render_args(self):
        return {"mode": "rgb_array"}

    def mod_render_frames(self, cur_frame, **kwargs):
        return cur_frame

    def create_from_id(self, env_id: str) -> gym.Env:
        """
        Return the environment object. By default, the standard gym.make is
        used. This will work with any environments that are registered.
        """
        pass_args = self.args.env_custom_args.split(",")
        pass_kwargs = {}
        for pass_arg in pass_args:
            if pass_arg == "":
                continue
            arg_name, assign_val = pass_arg.split("=")
            if assign_val.lower() == "false":
                assign_val = False
            elif assign_val.lower() == "true":
                assign_val = True
            else:
                try:
                    assign_val = float(assign_val)
                except:
                    pass

            pass_kwargs[arg_name] = assign_val

        return gym.make(env_id, **pass_kwargs)

    def get_setup_multiproc_fn(
        self,
        make_env,
        env_id,
        seed,
        allow_early_resets,
        env_interface,
        set_eval,
        alg_env_settings,
        args,
    ) -> VecEnv:
        """
        If returns None, the default multiprocess worker is used.
        - make_env: ((seed_rank) -> gym.Env)
        """
        return None

    def get_add_args(self, parser):
        """
        Add additional command line arguments which will be available in
        `env_trans_fn`
        """
        parser.add_argument("--env-custom-args", type=str, default="")


class EnvInterfaceWrapper(EnvInterface):
    def __init__(self, args, wrapped_env_cls):
        self.args = args
        self.env_int = wrapped_env_cls(args)

    def setup(self, args, task_id):
        super().setup(args, task_id)
        self.env_int.setup(args, task_id)

    def env_trans_fn(self, env, set_eval):
        return self.env_int.env_trans_fn(env, set_eval)

    def final_trans_fn(self, env):
        return self.env_int.final_trans_fn(env)

    def get_special_stat_names(self):
        return self.env_int.get_special_stat_names()

    def get_render_args(self):
        return self.env_int.get_render_args()

    def mod_render_frames(self, cur_frame, **kwargs):
        return self.env_int.mod_render_frames(cur_frame, **kwargs)

    def create_from_id(self, env_id):
        return self.env_int.create_from_id(env_id)

    def get_setup_multiproc_fn(
        self,
        make_env,
        env_id,
        seed,
        allow_early_resets,
        env_interface,
        set_eval,
        alg_env_settings,
        args,
    ):
        return self.env_int.get_setup_multiproc_fn(
            make_env,
            env_id,
            seed,
            allow_early_resets,
            env_interface,
            set_eval,
            alg_env_settings,
            args,
        )

    def get_add_args(self, parser):
        self.env_int.get_add_args(parser)


g_env_interface = OrderedDict()


def get_module(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def register_env_interface(name, env_interface):
    global g_env_interface
    g_env_interface[name] = env_interface


def get_env_interface(name, verbose=True):
    global g_env_interface
    for k, class_ in reversed(list(g_env_interface.items())):
        if re.match(re.compile(k), name):
            if verbose:
                print("Found env interface %s for %s" % (class_, name))
            return class_

    return EnvInterface
