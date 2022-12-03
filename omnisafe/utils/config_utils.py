"""config_utils"""

from collections import namedtuple, OrderedDict


def recursive_update(args: dict, update_args: dict):
    """recursively update args"""
    if update_args is not None:
        for key, value in args.items():
            if key in update_args:
                if isinstance(update_args[key], dict):
                    print(f'{key}:')
                    recursive_update(args[key], update_args[key])
                else:
                    # f-strings:
                    # https://pylint.pycqa.org/en/latest/user_guide/messages/convention/consider-using-f-string.html
                    args[key] = update_args[key]
                    menus = (key, update_args[key])
                    print(f'- {menus[0]}: {menus[1]} is update!')
            elif isinstance(value, dict):
                recursive_update(value, update_args)

    return create_namedtuple_from_dict(args)


def create_namedtuple_from_dict(obj):
    """Create namedtuple from dict"""
    if isinstance(obj, dict):
        fields = sorted(obj.keys())
        namedtuple_type = namedtuple(
            typename='GenericObject',
            field_names=fields,
            rename=True,
        )
        field_value_pairs = OrderedDict(
            (str(field), create_namedtuple_from_dict(obj[field])) for field in fields
        )
        try:
            return namedtuple_type(**field_value_pairs)
        except TypeError:
            # Cannot create namedtuple instance so fallback to dict (invalid attribute names)
            return dict(**field_value_pairs)
    elif isinstance(obj, (list, set, tuple, frozenset)):
        return [create_namedtuple_from_dict(item) for item in obj]
    else:
        return obj


def check_all_configs(configs):
    """Check all configs"""
    assert (
        configs.actor_iters > 0 and configs.critic_iters > 0
    ), "pi_iters and critic_iters must be greater than 0"
    assert (
        configs.actor_lr > 0 and configs.critic_lr > 0
    ), "actor_lr and critic_lr must be greater than 0"
    assert (
        configs.buffer_cfgs.gamma >= 0 and configs.buffer_cfgs.gamma < 1.0
    ), "gamma must be in [0, 1)"
    assert (
        configs.use_cost is False and configs.cost_gamma == 1.0
    ), "if use_cost is False, cost_gamma must be 1.0"
