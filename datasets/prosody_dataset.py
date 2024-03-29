import random
from os import path

import librosa
import torch
import torchaudio
from torch.nn import functional as F
from torch.utils.data import Dataset
from speech_utils.preprocessing.feature_extraction import extract_features

__SAMPLE_RATE = 22050


class ProsodyDataset(Dataset):
    def __init__(
        self,
        filenames,
        base_dir,
        sample_rate=22050,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        f_min=0.0,
        f_max=8000.0,
        n_mels=80,
        power=1,
        trim=True,
        spectrogram_segment_size=64,
    ):
        super().__init__()

        # Simple assignments
        self.filenames = filenames
        self.trim = trim
        self.base_dir = base_dir
        self.spectrogram_segment_size = spectrogram_segment_size

        # Create a Torchaudio MelSpectrogram generator
        self.melspectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=power,
            mel_scale="slaney",
            norm="slaney",
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        # Audio preprocessing -----------------------------------------------------------
        # Load the audio file and squeeze it to a 1D Tensor
        wav, _ = torchaudio.load(path.join(self.base_dir, self.filenames[i]))
        wav = wav.squeeze(0)

        if self.trim:
            wav, _ = librosa.effects.trim(wav.numpy(), frame_length=512)
            wav = torch.tensor(wav)

        # Create the Mel spectrogram and save its length
        mel_spectrogram = self.melspectrogram(wav)
        mel_spectrogram = torch.log(torch.clamp(mel_spectrogram, min=1e-5)).T
        mel_spectrogram_len = torch.IntTensor([len(mel_spectrogram)])

        # Pick a random segment of the Mel spectrogram
        last_mel_idx = len(mel_spectrogram) - self.spectrogram_segment_size
        mel_start_idx = random.randint(0, last_mel_idx)
        mel_end_idx = mel_start_idx + self.spectrogram_segment_size
        mel_spectrogram_segment = mel_spectrogram[mel_start_idx:mel_end_idx]

        wav_start_idx = mel_start_idx * 256
        wav_end_idx = mel_end_idx * 256
        wav_segment = F.pad(wav, (128, 128))[wav_start_idx:wav_end_idx]

        wav_segment_features = extract_features(wav_segment)

        out_data = {
            "mel_spectrogram": mel_spectrogram,
            "mel_spectrogram_segment": mel_spectrogram_segment,
            "wav_segment": wav_segment,
        }

        out_metadata = {
            "mel_spectrogram_len": mel_spectrogram_len,
        }

        out_data["wav"] = wav
        out_metadata["wav_len"] = torch.IntTensor([len(wav)])

        if self.features is not None:
            out_metadata["features"] = torch.Tensor([self.features[i]])
            if self.feature_override is not None:
                out_metadata["features"] = torch.Tensor([self.feature_override])

        return out_data, out_metadata
