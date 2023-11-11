import logging
from glob import glob
import os

from src.datasets.custom_audio_dataset import CustomAudioDataset

logger = logging.getLogger(__name__)


def id_to_path(files):
    return {os.path.basename(path).split("-")[0]: path for path in files}


class CustomDirAudioDataset(CustomAudioDataset):
    def __init__(self, mix_dir, ref_dir, target_dir, *args, **kwargs):
        mixes = id_to_path(glob(os.path.join(mix_dir, "*-mixed.wav")))
        refs = id_to_path(glob(os.path.join(ref_dir, "*-ref.wav")))
        targets = id_to_path(glob(os.path.join(target_dir, "*-target.wav")))

        keys_intersection = mixes.keys() & refs.keys() & targets.keys()
        index = []
        for id in keys_intersection:
            index += [mixes[id], refs[id], targets[id]]
        super().__init__(index, *args, **kwargs)

    def __len__(self):
        return len(self._index) // 3

    def __getitem__(self, ind):
        idx = 3 * ind
        y_wav = self.load_audio(self._index[idx])
        x_wav = self.load_audio(self._index[idx + 1])
        target_wav = self.load_audio(self._index[idx + 2])
        return {"y_wav": y_wav, "x_wav": x_wav, "target_wav": target_wav, "speaker_id": 0}
