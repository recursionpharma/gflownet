from foliconf import config_class, config_from_dict, config_to_dict, make_config, update_config, set_Config

__all__ = [
    "Config",
    "config_class",
    "config_from_dict",
    "config_to_dict",
    "make_config",
    "update_config",
]


class Config:
    pass


set_Config(Config)
