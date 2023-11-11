import logging
from pathlib import Path

from src.base.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomAudioDataset(BaseDataset):
    def __init__(self, index, *args, **kwargs):
        for item in index:
            assert Path(item).exists(), f"Path {item} doesn't exist"
        super().__init__(index, *args, **kwargs)
