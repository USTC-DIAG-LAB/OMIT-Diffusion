"""Configuration helpers and the lazily imported runner factory."""

import argparse
from collections.abc import Mapping

from Register import Registers


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        new_value = dict2namespace(value) if isinstance(value, Mapping) else value
        setattr(namespace, key, new_value)
    return namespace


def namespace2dict(config):
    conf_dict = {}
    for key, value in vars(config).items():
        if isinstance(value, argparse.Namespace):
            conf_dict[key] = namespace2dict(value)
        else:
            conf_dict[key] = value
    return conf_dict


def get_runner(runner_name, config):
    # Importing this module registers OMITRunner. Keep this side effect before
    # the lookup, but avoid loading the training stack for config-only users.
    from runners.OMIT import OMITRunner  # noqa: F401

    return Registers.runners[runner_name](config)
