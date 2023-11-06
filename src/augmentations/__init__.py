from collections.abc import Callable
from typing import List

import src.augmentations.spectrogram_augmentations
import src.augmentations.wave_augmentations
from src.augmentations.random_choice import RandomChoice
from src.augmentations.sequential_random_apply import SequentialRandomApply

# from src.augmentations.sequential import SequentialAugmentation
# from src.augmentations.random_apply import RandomApply
from src.utils.parse_config import ConfigParser


def from_configs(configs: ConfigParser):
    wave_augs = []
    if "augmentations" in configs.config and "wave" in configs.config["augmentations"]:
        for aug_dict in configs.config["augmentations"]["wave"]:
            wave_augs.append(configs.init_obj(aug_dict, src.augmentations.wave_augmentations))

    spec_augs = []
    if "augmentations" in configs.config and "spectrogram" in configs.config["augmentations"]:
        for aug_dict in configs.config["augmentations"]["spectrogram"]:
            spec_augs.append(configs.init_obj(aug_dict, src.augmentations.spectrogram_augmentations))
    return _to_function(RandomChoice, wave_augs, configs.config["augmentations"]["random_apply_p"]), _to_function(
        SequentialRandomApply, spec_augs, configs.config["augmentations"]["random_apply_p"]
    )


def _to_function(random_type, augs_list: List[Callable], p: float):
    if len(augs_list) == 0:
        return None
    elif len(augs_list) == 1:
        return augs_list[0]
    else:
        return random_type(augs_list, p)
