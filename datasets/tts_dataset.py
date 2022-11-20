import re
from os import path

import librosa
import torch
import torchaudio
import unidecode
from sklearn.preprocessing import OrdinalEncoder
from speech_utils.audio.transforms import TacotronMelSpectrogram
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


__SAMPLE_RATE = 22050


class TTSDataset(Dataset):
    """A class implementing a text-to-speech PyTorch Dataset. It supplies Mel spectrograms and
    textual data for a text-to-speech model."""

    def __init__(
        self,
        filenames,
        texts,
        base_dir,
        speaker_ids=None,
        features=None,
        allowed_chars=ALLOWED_CHARS,
        end_token="^",
        trim=True,
        feature_override=None,
        expand_abbreviations=False,
        include_wav=False,
        include_text=False,
        num_frames_per_step: int = 1,
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
            silence -- seconds of silence to append to the end of the audio.
                    If None, no silence is appended (default None)
            trim -- If True, audio data will be trimmed to remove silence from the beginning and
                    end. Defaults to True.
        """
        super().__init__()

        if end_token is not None and end_token in allowed_chars:
            raise Exception("end_token cannot be in allowed_chars!")

        self.include_wav = include_wav
        self.include_text = include_text

        self.num_frames_per_step = num_frames_per_step

        # Simple assignments
        self.filenames = filenames
        self.end_token = end_token
        self.trim = trim
        self.base_dir = base_dir

        self.features = features
        self.feature_override = feature_override

        # Preprocessing step - ensure textual data only contains allowed characters
        allowed_chars_re = re.compile(f"[^{allowed_chars}]+")
        texts = [
            allowed_chars_re.sub("", unidecode.unidecode(t)).lower() for t in texts
        ]
        if expand_abbreviations:
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
        self.melspectrogram = TacotronMelSpectrogram()

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

        # Transform to a LongTensor and remove the extra dimension necessary for the OrdinalEncoder
        chars_idx = torch.LongTensor(chars_idx).squeeze(-1)
        chars_idx_len = torch.IntTensor([len(chars_idx)])

        out_data = {
            "chars_idx": chars_idx,
            "mel_spectrogram": mel_spectrogram,
            "gate": gate,
        }

        out_metadata = {
            "chars_idx_len": chars_idx_len,
            "mel_spectrogram_len": mel_spectrogram_len,
            "gate_len": gate_len,
        }

        out_extra = {}

        # Optional additions to the output
        if self.include_wav:
            out_data["wav"] = wav
            out_metadata["wav_len"] = torch.IntTensor([len(wav)])

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
