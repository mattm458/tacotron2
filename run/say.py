import json
import re
from typing import List, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import unidecode
from sklearn.preprocessing import OrdinalEncoder
from torch import Tensor
from transformers import AutoTokenizer, BertModel

from model.hifi_gan import Generator
from model.tts_model import TTSModel


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def do_say(
    dataset_config: dict,
    training_config: dict,
    model_config: dict,
    extensions_config: dict,
    device: int,
    checkpoint: str,
    text: str,
    output: str,
    hifi_gan_checkpoint: Optional[str],
    random_seed: Optional[int],
    speaker_id: Optional[int],
    controls: Optional[str],
    export_mel: bool = True,
    description: Optional[str] = None,
):
    if random_seed is not None:
        torch.manual_seed(random_seed)

    end_token = dataset_config["preprocessing"]["end_token"]
    allowed_chars = dataset_config["preprocessing"]["allowed_chars"]

    encoder = OrdinalEncoder()
    if end_token is None:
        encoder.fit([[x] for x in list(allowed_chars)])
    else:
        encoder.fit([[x] for x in list(allowed_chars) + [end_token]])

    allowed_chars_re = re.compile(f"[^{allowed_chars}]+")
    text = allowed_chars_re.sub("", unidecode.unidecode(text).lower())

    if end_token is not None:
        text += end_token

    chars_idx = encoder.transform([[x] for x in text]) + 1
    chars_idx = torch.tensor(chars_idx, dtype=torch.int64).unsqueeze(0).squeeze(-1)
    chars_idx_len = torch.tensor([chars_idx.shape[1]], dtype=torch.int64)

    use_hifi_gan = hifi_gan_checkpoint is not None
    generator = None

    if use_hifi_gan:
        assert hifi_gan_checkpoint, "You must give a checkpoint if using HiFi-GAN"

        print(f"Loading HiFi-GAN checkpoint {hifi_gan_checkpoint}...")
        # hifi_gan_states = torch.load(hifi_gan_checkpoint, map_location="cpu")[
        #     "state_dict"
        # ]
        # hifi_gan_states = dict(
        #     [(k[10:], v) for k, v in hifi_gan_states.items() if "generator" in k]
        # )

        with open("web_checkpoints/hifi-gan/UNIVERSAL_V1/config.json", "r") as infile:
            generator_config = AttrDict(json.load(infile))

        generator_state_dict = torch.load(
            "web_checkpoints/hifi-gan/UNIVERSAL_V1/g_02500000",
            map_location=f"cuda:{device}",
        )

        print(generator_config)
        generator = Generator(generator_config)
        generator.load_state_dict(generator_state_dict["generator"])

        generator.remove_weight_norm()
        generator.eval()
        generator = generator.cuda(device)

    # Handle description text
    if model_config["args"]["description_embeddings"]:
        if description is None:
            description_embeddings = torch.zeros(
                (1, model_config["args"]["description_embeddings_dim"])
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")
            model = BertModel.from_pretrained("bert-large-uncased").cuda()
            tokenized = tokenizer(description, return_tensors="pt")
            description_embeddings = model(
                input_ids=tokenized["input_ids"].cuda(),
                token_type_ids=tokenized["token_type_ids"].cuda(),
                attention_mask=tokenized["attention_mask"].cuda(),
            ).pooler_output.cpu()

    # Handle optional speaker ID
    speaker_tensor: Optional[Tensor] = None
    if extensions_config["speaker_tokens"]["active"]:
        speaker_tensor = torch.LongTensor([speaker_id])

    # Handle optional model controls
    controls_dim = 0
    controls_tensor: Optional[Tensor] = None
    if extensions_config["controls"]["active"] and controls:
        controls_dim = len(extensions_config["controls"]["features"])
        controls_tensor = torch.Tensor([[float(x) for x in controls.split(",")]])

    scheduler_milestones = [
        int(x * training_config["args"]["max_steps"])
        for x in model_config["scheduler_milestones"]
    ]

    model = TTSModel.load_from_checkpoint(
        checkpoint,
        lr=training_config["lr"],
        weight_decay=training_config["weight_decay"],
        num_chars=len(dataset_config["preprocessing"]["allowed_chars"])
        + (dataset_config["preprocessing"]["end_token"] is not None),
        num_mels=dataset_config["preprocessing"]["num_mels"],
        controls=controls,
        controls_dim=controls_dim,
        scheduler_milestones=scheduler_milestones,
        **model_config["args"],
        map_location="cpu",
    ).cpu()

    model.eval()
    with torch.no_grad():
        _, mel_spectrogram_post, _, _ = model(
            chars_idx=chars_idx,
            chars_idx_len=chars_idx_len,
            teacher_forcing=False,
            speaker_id=speaker_tensor,
            controls=controls_tensor,
            max_len_override=5000,
            description_embeddings=description_embeddings,
        )

    wav = None

    if generator:
        with torch.no_grad():
            wav = generator(mel_spectrogram_post[:, :-1].cuda(device).swapaxes(1, 2))[
                0
            ].cpu()[0]
    else:
        this_mel_spectrogram = mel_spectrogram_post[0, :-1]

        wav = librosa.feature.inverse.mel_to_audio(
            np.exp(this_mel_spectrogram.numpy()).T,
            sr=dataset_config["preprocessing"]["sample_rate"],
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            center=True,
            power=1.0,
            fmin=0,
            fmax=8000,
        )

    sf.write(output, wav, samplerate=dataset_config["preprocessing"]["sample_rate"])

    if export_mel:
        if generator:
            np.save(output, mel_spectrogram_post.numpy().T)
        else:
            np.save(output, mel_spectrogram_post.T)
