import librosa
import numpy as np
import soundfile as sf
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

from model.tts_model import TTSModel


def do_say(
    dataset_config: dict,
    training_config: dict,
    model_config: dict,
    device: int,
    checkpoint: str,
    text: str,
    output: str,
):
    end_token = dataset_config["preprocessing"]["end_token"]
    allowed_chars = dataset_config["preprocessing"]["allowed_chars"]

    encoder = OrdinalEncoder()
    if end_token is None:
        encoder.fit([[x] for x in list(allowed_chars)])
    else:
        encoder.fit([[x] for x in list(allowed_chars) + [end_token]])

    chars_idx = encoder.transform([[x] for x in text.lower()]) + 1
    chars_idx = torch.tensor(chars_idx, dtype=torch.int64).unsqueeze(0).squeeze(-1)
    chars_idx_len = torch.tensor([chars_idx.shape[1]], dtype=torch.int64)

    model = TTSModel.load_from_checkpoint(
        checkpoint,
        lr=training_config["lr"],
        weight_decay=training_config["weight_decay"],
        num_chars=len(dataset_config["preprocessing"]["allowed_chars"])
        + (dataset_config["preprocessing"]["end_token"] is not None),
        num_mels=dataset_config["preprocessing"]["num_mels"],
        **model_config["args"],
    )

    model.eval()
    with torch.no_grad():
        mel_spectrogram, mel_spectrogram_post, gate, alignment = model(
            chars_idx=chars_idx, chars_idx_len=chars_idx_len, teacher_forcing=False
        )

    mel_spectrogram_post = np.exp(mel_spectrogram_post[0].numpy())
    wav = librosa.feature.inverse.mel_to_audio(
        mel_spectrogram_post.T,
        sr=22050,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        center=True,
        power=1.0,
        fmin=0,
        fmax=8000,
    )

    sf.write(output, wav, 22050)

    #     sample_rate=22050,
    # n_fft=1024,
    # win_length=1024,
    # hop_length=256,
    # f_min=0.0,
    # f_max=None,
    # n_mels=80,
    # power=1.0,
    # mel_scale="slaney",
    # norm="slaney",
    # center=False,
