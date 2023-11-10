from typing import List, Callable
from torch import Tensor
import random
from hw_asr.augmentations.base import AugmentationBase


class RandomChoice(AugmentationBase):
    def __init__(self, augmentation_list: List[Callable], p: float):
        self.augmentation_list = augmentation_list
        self.p = p

    def __call__(self, data: Tensor) -> Tensor:
        x = data
        if random.random() < self.p:
            augmentation = random.choice(self.augmentation_list)
            x = augmentation(x)
        return x