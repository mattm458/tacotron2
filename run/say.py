from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from hifi_gan.model.generator import Generator
from matplotlib import pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

from model.tts_model import TTSModel


def do_say(
    dataset_config: dict,
    training_config: dict,
    model_config: dict,
    extensions_config: dict,
    device: int,
    checkpoint: str,
    text: str,
    output: str,
    speaker_id: Optional[int],
    hifi_gan_checkpoint: Optional[str],
):
    end_token = dataset_config["preprocessing"]["end_token"]
    allowed_chars = dataset_config["preprocessing"]["allowed_chars"]

    encoder = OrdinalEncoder()
    if end_token is None:
        encoder.fit([[x] for x in list(allowed_chars)])
    else:
        encoder.fit([[x] for x in list(allowed_chars) + [end_token]])
        text = text + end_token

    chars_idx = encoder.transform([[x] for x in text.lower()]) + 1
    chars_idx = torch.tensor(chars_idx, dtype=torch.int64).unsqueeze(0).squeeze(-1)
    chars_idx_len = torch.tensor([chars_idx.shape[1]], dtype=torch.int64)

    use_hifi_gan = hifi_gan_checkpoint is not None
    generator = None

    if use_hifi_gan:
        assert hifi_gan_checkpoint, "You must give a checkpoint if using HiFi-GAN"

        print(f"Loading HiFi-GAN checkpoint {hifi_gan_checkpoint}...")
        hifi_gan_states = torch.load(hifi_gan_checkpoint, map_location="cpu")[
            "state_dict"
        ]
        hifi_gan_states = dict(
            [(k[10:], v) for k, v in hifi_gan_states.items() if "generator" in k]
        )

        generator = Generator()
        generator.weight_norm()
        generator.load_state_dict(hifi_gan_states)
        generator.remove_weight_norm()
        generator.eval()
        generator = generator.cuda(0)

    controls = False
    controls_dim = 0

    if extensions_config["controls"]["active"]:
        controls = True
        controls_dim = len(extensions_config["controls"]["features"])

    model = TTSModel.load_from_checkpoint(
        checkpoint,
        lr=training_config["lr"],
        weight_decay=training_config["weight_decay"],
        num_chars=len(dataset_config["preprocessing"]["allowed_chars"])
        + (dataset_config["preprocessing"]["end_token"] is not None),
        num_mels=dataset_config["preprocessing"]["num_mels"],
        controls=controls,
        controls_dim=controls_dim,
        **model_config["args"],
        map_location="cpu",
    ).cpu()

    model.eval()
    with torch.no_grad():
        _, mel_spectrogram_post, _, _ = model(
            chars_idx=chars_idx,
            chars_idx_len=chars_idx_len,
            teacher_forcing=False,
            #            speaker_id=torch.LongTensor([speaker_id]),
            #            controls=torch.tensor([[0, 0, 0, 0, 0, 0, 0]]),
            max_len_override=5000,
        )

    wav = None

    if generator:
        with torch.no_grad():
            wav = generator(mel_spectrogram_post[:, :-1].cuda(0))[0].cpu()
    else:
        this_mel_spectrogram = mel_spectrogram_post[0, :-1]
        mel_spectrogram_post = np.exp(this_mel_spectrogram.numpy())

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
