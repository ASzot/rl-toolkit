import os.path as osp

import yaml

g_settings = None


def get_cached_settings():
    return g_settings


def get_prop(name, def_val=None):
    if def_val is None:
        return get_cached_settings()[name]
    else:
        return get_cached_settings().get(name, def_val)


def set_prop(name, set_val):
    g_settings[name] = set_val


def init(cfg_path):
    global g_settings
    with open(cfg_path) as f:
        g_settings = yaml.load(f, Loader=yaml.BaseLoader)
