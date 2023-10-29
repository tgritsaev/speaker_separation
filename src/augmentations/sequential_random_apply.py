from typing import List, Callable
from torch import Tensor
import random
from hw_asr.augmentations.base import AugmentationBase


class SequentialRandomApply(AugmentationBase):
    def __init__(self, augmentation_list: List[Callable], p: float = 0.5):
        self.augmentation_list = augmentation_list
        self.p = p

    def __call__(self, data: Tensor) -> Tensor:
        x = data
        for augmentation in self.augmentation_list:
            if random.random() < self.p:
                x = augmentation(x)
        return x