import re
from os import path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import torch
import torchaudio
import unidecode
from sklearn.preprocessing import OrdinalEncoder
from speech_utils.audio.transforms import TacotronMelSpectrogram
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

ALLOWED_CHARS = "!'(),.:;? \\-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def _expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


class TTSDataset(Dataset):
    def __init__(
        self,
        filenames: List[str],
        texts: List[str],
        base_dir: str,
        speaker_ids: Optional[List[str]] = None,
        features=None,
        allowed_chars: str = ALLOWED_CHARS,
        end_token: Optional[str] = "^",
        silence: int = 0,
        trim: bool = True,
        feature_override=None,
        expand_abbreviations=False,
        include_wav=False,
        include_text=False,
        num_frames_per_step: int = 1,
        num_mels: int = 80,
        longest_text: Optional[int] = None,
    ):
        super().__init__()

        self.longest_text = longest_text

        if end_token is not None and end_token in allowed_chars:
            raise Exception("end_token cannot be in allowed_chars!")

        self.include_wav = include_wav
        self.include_text = include_text

        self.num_frames_per_step = num_frames_per_step

        # Simple assignments
        self.filenames = filenames
        self.end_token = end_token
        if end_token is None:
            print("Dataset: Not using an end token")
        else:
            print(f"Dataset: Using end token {end_token}")

        self.trim = trim
        if trim:
            print("Dataset: Trimming silence from input audio files")
        else:
            print("Dataset: Not trimming silence from input audio files")

        print(f"Dataset: Adding {silence} frames of silence to the end of each clip")

        self.silence_frames = silence

        self.base_dir = base_dir

        self.features = features
        self.feature_override = feature_override

        print(f"Dataset: Allowed characters {allowed_chars}")

        # Preprocessing step - ensure textual data only contains allowed characters
        allowed_chars_re = re.compile(f"[^{allowed_chars}]+")
        texts = [
            allowed_chars_re.sub("", unidecode.unidecode(t)).lower() for t in texts
        ]
        if expand_abbreviations:
            print("Dataset: Expanding abbreviations in input text...")
            texts = [_expand_abbreviations(t) for t in texts]

        self.texts = texts

        self.speaker_ids = speaker_ids

        # Preprocessing step - create an ordinal encoder to transform textual data to a
        # tensor of integers
        self.encoder = OrdinalEncoder()
        if end_token is None:
            self.encoder.fit([[x] for x in list(allowed_chars)])
        else:
            self.encoder.fit([[x] for x in list(allowed_chars) + [end_token]])

        # Create a Torchaudio MelSpectrogram generator
        self.melspectrogram = TacotronMelSpectrogram(
            n_mels=num_mels,
            cache=False,
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(
        self, i: int
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Any]]:
        # Audio preprocessing -----------------------------------------------------------
        # Load the audio file and squeeze it to a 1D Tensor
        wav, _ = torchaudio.load(path.join(self.base_dir, self.filenames[i]))
        wav = wav.squeeze(0)

        if self.trim:
            wav_np, _ = librosa.effects.trim(wav.numpy(), frame_length=512)
            wav = torch.tensor(wav_np)
        wav = F.pad(wav, (0, self.silence_frames))

        # Create the Mel spectrogram and save its length
        mel_spectrogram = self.melspectrogram(wav, id=str(i))
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
        chars_idx += 1

        # Transform to a tensor and remove the extra dimension necessary for the OrdinalEncoder
        chars_idx = torch.tensor(chars_idx, dtype=torch.int64).squeeze(-1)

        if self.longest_text is not None:
            chars_idx = F.pad(chars_idx, (0, self.longest_text - len(chars_idx)))

        chars_idx_len = torch.tensor([len(chars_idx)], dtype=torch.int64)

        out_data: Dict[str, Tensor] = {
            "chars_idx": chars_idx,
            "mel_spectrogram": mel_spectrogram,
            "gate": gate,
        }

        out_metadata: Dict[str, Tensor] = {
            "chars_idx_len": chars_idx_len,
            "mel_spectrogram_len": mel_spectrogram_len,
            "gate_len": gate_len,
        }

        out_extra: Dict[str, Any] = {}

        if self.include_text:
            out_extra["text"] = text

        if self.speaker_ids is not None:
            out_metadata["speaker_id"] = torch.IntTensor([self.speaker_ids[i]])

        # If we're including speech features, include them in output
        if self.features is not None:
            # We can optionally override features from the dataset.
            # This is done to evaluate the controllability of a trained model
            if self.feature_override is not None:
                out_metadata["features"] = torch.Tensor([self.feature_override])
            else:
                out_metadata["features"] = torch.Tensor([self.features[i]])

        return out_data, out_metadata, out_extra
