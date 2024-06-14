import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CORAADisfluencyDataset(Dataset):
    def __init__(self, metadata,data_dir,
                 duration: float = 7,
                 resample_freq: int = 16000,
                 device=DEVICE
    ):
        super().__init__()
        self.num_samples = int(duration * resample_freq)
        self.resample_freq = resample_freq
        self.device = device
        self.data = None

        self.data_dir = data_dir
        self.metadata = metadata.copy()
        self.label_names = ['votes_for_hesitation', 'votes_for_filled_pause']
        self.filter_data()
        self.file_paths = list(self.data.file_path)
        # Problema ta de uma label só assim
        self.labels = list(self.data.label)

    def filter_data(self):
        self.metadata = self.metadata.dropna()
        self.metadata['label'] = 0
        self.metadata.loc[:, 'label'] = self.metadata.apply(
            lambda row: 1 if any(row[col] != 0 for col in self.label_names) else 0, axis=1)

        df = self.metadata[self.metadata['label'] == 1]
        df_complement = self.metadata.drop(df.index)
        df_complement = df_complement.sample(n=len(df), random_state=1)

        self.data = pd.concat([df, df_complement])

    def _to_mono(self, waveform):
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def _cut_if_necessary(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[1] > self.num_samples:
            waveform = waveform[:, :self.num_samples]
        return waveform

    def _right_pad_if_necessary(self, waveform: torch.Tensor) -> torch.Tensor:
        num_samples = waveform.shape[1]
        if num_samples < self.num_samples:
            num_missing_samples = self.num_samples - num_samples
            last_dim_padding = (0, num_missing_samples)
            waveform = torch.nn.functional.pad(waveform, last_dim_padding)
        return waveform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fp = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(self.data_dir+fp)

        if sample_rate != self.resample_freq:
            resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.resample_freq)
            waveform = resample(waveform)

        waveform = self._to_mono(waveform)
        waveform = self._cut_if_necessary(waveform)
        waveform = self._right_pad_if_necessary(waveform)
        return waveform, self.labels[idx]
