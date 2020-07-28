import numpy as np
import os.path as osp
import re
from collections import OrderedDict
import gym

class EnvInterface(object):
    def __init__(self, args):
        self.args = args

    def setup(self, args, task_id):
        self.task_id = task_id

    def env_trans_fn(self, env, set_eval):
        return env

    def get_special_stat_names(self):
        return []

    def get_render_args(self):
        return { 'mode': 'rgb_array' }

    def mod_render_frames(self, cur_frame, **kwargs):
        return cur_frame

    def create_from_id(self, env_id):
        """
        Return the environment object. By default, the standard gym.make is
        used. This will work with any environments that are registered.
        """
        return gym.make(env_id)

    def get_add_args(self, parser):
        """
        Add additional command line arguments which will be available in
        `env_trans_fn`
        """
        pass


g_env_interface = OrderedDict()

def get_module(name):
    components = name.split('.')
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
                print('Found env interface %s for %s' % (class_, name))
            return class_

    return EnvInterface


