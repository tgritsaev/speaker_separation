import logging
import os
from tqdm import tqdm
from pathlib import Path

from src.base.base_dataset import BaseDataset


logger = logging.getLogger(__name__)


def get_speaker_id_by_path(path):
    return int(os.path.basename(path).split("_")[0])


class MixtureDataset(BaseDataset):
    def __init__(self, part: str, cut_mix, *args, **kwargs):
        if part in ["train-clean-100-mixed", "test-clean-mixed"]:
            self.path = Path("/kaggle/input/librispeech-mixed/librispeech-mixed/" + part)
        else:
            self.path = Path("data/mixture/" + part)
        index = sorted(list(os.listdir(self.path)))
        super().__init__(index, *args, **kwargs)
        self._map_speakers()
        self.len = len(self._index) // 3
        self.cut_mix = cut_mix

    def _map_speakers(self):
        logging.info("speakers mapping is started...")
        speakers_cnt = 0
        self.speaker_mapping = {}
        for i in tqdm(range(0, len(self._index), 3)):
            speaker_id = get_speaker_id_by_path(self._index[i])
            if speaker_id not in self.speaker_mapping.keys():
                self.speaker_mapping[speaker_id] = speakers_cnt
                speakers_cnt += 1
        self.speakers_cnt = speakers_cnt
        logging.info(f"speakers mapping has finished, speakers_cnt: {speakers_cnt}")

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        y_wav = self.load_audio(os.path.join(self.path, self._index[ind * 3]))
        x_wav = self.load_audio(os.path.join(self.path, self._index[ind * 3 + 1]))
        if self.cut_mix:
            x_wav = x_wav[: self.cut_mix]
        target_wav = self.load_audio(os.path.join(self.path, self._index[ind * 3 + 2]))
        mapped_speaker_id = self.speaker_mapping[get_speaker_id_by_path(self._index[ind * 3])]
        return {"y_wav": y_wav, "x_wav": x_wav, "target_wav": target_wav, "speaker_id": mapped_speaker_id}
