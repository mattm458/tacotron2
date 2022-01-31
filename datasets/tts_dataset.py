import re

import librosa
import torch
import torchaudio
import unidecode
from sklearn.preprocessing import OrdinalEncoder
from torch.nn import functional as F
from torch.utils.data import Dataset

ALLOWED_CHARS = "!'(),.:;? \\-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


class TTSDataset(Dataset):
    """A class implementing a text-to-speech PyTorch Dataset. It supplies Mel spectrograms and
    textual data for a text-to-speech model."""

    def __init__(
        self,
        filenames,
        texts,
        allowed_chars=ALLOWED_CHARS,
        end_token="^",
        # sample_rate=22050,
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        f_min=0.0,
        f_max=8000.0,
        n_mels=80,
        power=1,
        silence=0.1,
        trim=True,
    ):
        """Create a TTSDataset object.

        Args:
            filenames -- a list of wav filenames containing speech
            texts -- a list of transcriptions associated with the wav files
            allowed_chars -- a list of characters allowed in transcriptions
                            (default "!'(),.:;? \\-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
            end_token -- a special character appended to the end of every transcript.
                        If None, no token will be appended (default "^")
            sample_rate -- the sample rate of the wav files (default 22050)
            n_fft -- size of FFT for mel spectrogram (default 1024)
            win_length -- window size (default 1024)
            hop_length -- length of hop between STFT windows (default 256)
            f_min -- minimum frequencey (default 0.0)
            f_max -- maximum frequency (default 8000.0)
            n_mels -- number of Mel filterbanks (default 80)
            power -- exponent for the magnitude spectrogram (must be >0) i.e., 1 for energy, 2 for power, etc.
                    If None, then the complex spectrum is returned instead (default 1)
            silence -- seconds of silence to append to the end of the audio.
                    If None, no silence is appended (default None)
            trim -- If True, audio data will be trimmed to remove silence from the beginning and
                    end. Defaults to True.
        """
        super(TTSDataset).__init__()

        if end_token is not None and end_token in allowed_chars:
            raise Exception("end_token cannot be in allowed_chars!")

        # Simple assignments
        self.filenames = filenames
        self.end_token = end_token
        self.sample_rate = sample_rate
        self.trim = trim

        # Preprocessing step - ensure textual data only contains allowed characters
        allowed_chars_re = re.compile(f"[^{allowed_chars}]+")
        self.texts = [allowed_chars_re.sub("", unidecode.unidecode(t)) for t in texts]

        # Preprocessing step - calculate the length of the silence vector
        self.silence_len = int(silence * sample_rate) if silence is not None else None

        # Preprocessing step - create an ordinal encoder to transform textual data to a
        # tensor of integers
        self.encoder = OrdinalEncoder()
        if end_token is None:
            self.encoder.fit([[x] for x in list(allowed_chars)])
        else:
            self.encoder.fit([[x] for x in list(allowed_chars) + [end_token]])

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
        wav, _ = torchaudio.load(self.filenames[i])
        wav = wav.squeeze(0)

        if self.trim:
            wav, _ = librosa.effects.trim(wav.numpy(), frame_length=512)
            wav = torch.tensor(wav)

        # Add silence if necessary
        if self.silence_len is not None:
            wav = torch.concat([wav, torch.zeros(self.silence_len)], 0)

        # Compute the Mel spectrogram, and translate it to a log scale
        mel_spectrogram = self.melspectrogram(wav)
        mel_spectrogram = torch.log(torch.clamp(mel_spectrogram, min=1e-5)).T
        mel_spectrogram_len = torch.IntTensor([len(mel_spectrogram)])

        # Create gate output indicating whether the TTS model should continue producing Mel
        # spectrogram frames
        gate = torch.ones(len(mel_spectrogram), 1)
        gate[-1] = 0.0
        gate_len = torch.IntTensor([len(gate)])

        # Text preprocessing ------------------------------------------------------------
        # Append the end token
        text = self.texts[i] + (self.end_token if self.end_token is not None else "")

        # Encode the text
        chars_idx = self.encoder.transform([[x] for x in text])

        # Index 0 is for padding, so increment all characters by 1
        if self.end_token:
            chars_idx += 1

        # Transform to a LongTensor and remove the extra dimension necessary for the OrdinalEncoder
        chars_idx = torch.LongTensor(chars_idx).squeeze(-1)
        chars_idx_len = torch.IntTensor([len(chars_idx)])

        return {
            "chars_idx": chars_idx,
            "mel_spectrogram": mel_spectrogram,
            "gate": gate,
        }, {
            "chars_idx_len": chars_idx_len,
            "mel_spectrogram_len": mel_spectrogram_len,
            "gate_len": gate_len,
        }
