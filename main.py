import csv
import os
from os import path

import librosa
import numpy as np
import pandas as pd
import soundfile
import torch
import yaml
from hifi_gan.model.generator import Generator
from pytorch_lightning import Trainer
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm

from datasets.tts_dataloader import TTSDataLoader
from datasets.tts_dataset import TTSDataset
from model.prosodic_features import prosody_detector
from model.tacotron2 import Tacotron2


def load_dataset(df, config, dataset_dir, feature_override=None, **args):
    args["filenames"] = list(df.wav)
    args["texts"] = list(df.text)

    if "expand_abbreviations" in config["preprocessing"]:
        args["expand_abbreviations"] = config["preprocessing"]["expand_abbreviations"]

    if "end_token" in config["preprocessing"]:
        args["end_token"] = config["preprocessing"]["end_token"]

    if "silence" in config["preprocessing"]:
        args["silence"] = config["preprocessing"]["silence"]

    if "model" in config and "extensions" in config["model"]:
        if "features" in config["model"]["extensions"]:
            features = df[config["extensions"]["features"]["allowed_features"]]
            args["features"] = features.values.tolist()

    return TTSDataset(
        **args,
        base_dir=dataset_dir,
        feature_override=feature_override,
        max_mel_len=None,
        max_text_len=None,
    )


def get_tacotron_args(config):
    num_chars = len(config["preprocessing"]["valid_chars"])
    if "end_token" in config["preprocessing"] and config["preprocessing"]["end_token"]:
        num_chars += 1
    elif "end_token" not in config["preprocessing"]:
        num_chars += 1

    tacotron_args = {
        "lr": config["training"]["lr"],
        "weight_decay": config["training"]["weight_decay"],
        "num_chars": num_chars,
        "encoder_kernel_size": config["encoder"]["encoder_kernel_size"],
        "num_mels": config["audio"]["num_mels"],
        "char_embedding_dim": config["encoder"]["char_embedding_dim"],
        "prenet_dim": config["decoder"]["prenet_dim"],
        "att_rnn_dim": config["attention"]["att_rnn_dim"],
        "att_dim": config["attention"]["att_dim"],
        "rnn_hidden_dim": config["decoder"]["rnn_hidden_dim"],
        "postnet_dim": config["decoder"]["postnet_dim"],
        "dropout": config["tacotron2"]["dropout"],
    }

    if "features" in config["model"]["extensions"]:
        tacotron_args["speech_features"] = True
        tacotron_args["speech_feature_dim"] = len(
            config["extensions"]["features"]["allowed_features"]
        )

    return tacotron_args


if __name__ == "__main__":
    from utils.args import args

    with open(args.config, "r") as infile:
        config = yaml.load(infile, Loader=yaml.Loader)

    tacotron_args = get_tacotron_args(config)

    if args.mode == "train":
        dataset_dir = args.dataset_dir

        train_dataset = load_dataset(
            df=pd.read_csv(
                config["data"]["train"],
                delimiter="|",
                quoting=csv.QUOTE_NONE,
            ),
            config=config,
            dataset_dir=dataset_dir,
        )
        val_dataset = load_dataset(
            df=pd.read_csv(
                config["data"]["val"],
                delimiter="|",
                quoting=csv.QUOTE_NONE,
            ),
            config=config,
            dataset_dir=dataset_dir,
        )

        train_dataloader = TTSDataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            prefetch_factor=config["data"]["prefetch_factor"],
            pin_memory=True,
            shuffle=True,
            persistent_workers=True,
            drop_last=True,
        )
        val_dataloader = TTSDataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
        )

        tacotron2 = Tacotron2(teacher_forcing=True, **tacotron_args)

        trainer = Trainer(
            devices=config["training"]["devices"],
            accelerator=config["training"]["accelerator"],
            precision=config["training"]["precision"],
            gradient_clip_val=1.0,
            max_epochs=config["training"]["max_epochs"],
            check_val_every_n_epoch=3,
            log_every_n_steps=40,
        )

        trainer.fit(
            tacotron2,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=args.checkpoint,
        )

    if args.mode == "finetune":
        dataset_dir = args.dataset_dir

        train_dataset = load_dataset(
            df=pd.read_csv(
                config["data"]["train"],
                delimiter="|",
                quoting=csv.QUOTE_NONE,
            ),
            config=config,
            dataset_dir=dataset_dir,
        )
        val_dataset = load_dataset(
            df=pd.read_csv(
                config["data"]["val"],
                delimiter="|",
                quoting=csv.QUOTE_NONE,
            ),
            config=config,
            dataset_dir=dataset_dir,
        )

        train_dataloader = TTSDataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            prefetch_factor=config["data"]["prefetch_factor"],
            pin_memory=True,
            shuffle=True,
            persistent_workers=True,
            drop_last=True,
        )
        val_dataloader = TTSDataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
        )

        prosody_model = prosody_detector.ProsodyPredictorLightning.load_from_checkpoint(
            args.prosody_model_checkpoint,
            use_lstm=True,
            rnn_layers=2,
            rnn_dropout=0.0,
        )

        if not args.fine_tune_prosody_model:
            prosody_model.requires_grad_(False)

        if args.fine_tune_lr is not None:
            tacotron_args["lr"] = args.fine_tune_lr

        tacotron2 = Tacotron2.load_from_checkpoint(
            args.tacotron_checkpoint,
            strict=False,
            prosody_model=prosody_model.prosody_predictor,
            fine_tune_prosody_model=args.fine_tune_prosody_model,
            fine_tune_style=args.fine_tune_tacotron_style,
            fine_tune_features=args.fine_tune_tacotron_features,
            teacher_forcing=True,
            **tacotron_args,
        )

        trainer = Trainer(
            devices=config["training"]["devices"],
            accelerator=config["training"]["accelerator"],
            precision=config["training"]["precision"],
            gradient_clip_val=1.0,
            max_epochs=config["training"]["max_epochs"],
            check_val_every_n_epoch=3,
            log_every_n_steps=40,
        )

        trainer.fit(
            tacotron2,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
    elif args.mode == "torchscript":
        generator = None

        if args.hifi_gan_checkpoint is not None:
            hifi_gan_states = torch.load(
                args.hifi_gan_checkpoint, map_location="cuda:1"
            )["state_dict"]
            hifi_gan_states = dict(
                [(k[10:], v) for k, v in hifi_gan_states.items() if "generator" in k]
            )

            generator = Generator()
            generator.load_state_dict(hifi_gan_states)

        tacotron2 = Tacotron2.load_from_checkpoint(
            checkpoint_path=args.checkpoint,
            teacher_forcing=False,
            **tacotron_args,
        )

        tacotron2.generator = generator

        tacotron2.cuda("cuda:1").to_torchscript(args.filename)

    elif args.mode == "say":
        generator = None

        if args.hifi_gan_checkpoint is not None:
            hifi_gan_states = torch.load(
                args.hifi_gan_checkpoint, map_location="cuda:1"
            )["state_dict"]
            hifi_gan_states = dict(
                [(k[10:], v) for k, v in hifi_gan_states.items() if "generator" in k]
            )

            generator = Generator()
            generator.weight_norm()
            generator.load_state_dict(hifi_gan_states)

        tacotron2 = Tacotron2.load_from_checkpoint(
            checkpoint_path=args.checkpoint,
            strict=False,
            teacher_forcing=False,
            **tacotron_args,
        )

        tacotron2.generator = generator

        ALLOWED_CHARS = (
            "!'(),.:;? \\-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        )
        end_token = "^"
        encoder = OrdinalEncoder()
        encoder.fit([[x] for x in list(ALLOWED_CHARS)])  # + [end_token]])

        encoded = (
            torch.LongTensor(
                encoder.transform([[x] for x in args.text.lower()])  # + [[end_token]])
            )
            .squeeze(1)
            .unsqueeze(0)
        ) + 1

        tts_data = {
            "chars_idx": torch.LongTensor(encoded),
            "mel_spectrogram": torch.zeros((1, 1000, 80)),
        }
        tts_metadata = {
            "chars_idx_len": torch.IntTensor([encoded.shape[1]]),
            "mel_spectrogram_len": torch.IntTensor([1000]),
            "features": torch.Tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        }

        if generator is None:
            with torch.no_grad():
                tacotron2.eval()
                mels, mels_post, gates, alignments = tacotron2.predict_step(
                    (tts_data, tts_metadata), 0, 0
                )

            mels = mels.cpu()
            mels_post = mels_post.cpu()
            gates = gates.cpu()
            alignments = alignments.cpu()

            gates = gates[0]
            gates = torch.sigmoid(gates)
            end = -1
            for i in range(gates.shape[0]):
                if gates[i][0] < 0.5:
                    end = i
                    break

            del mels
            del gates
            del alignments

            mels_exp = torch.exp(mels_post[0])[:end]
            wav = librosa.feature.inverse.mel_to_audio(
                mels_exp.numpy().T,
                sr=22050,
                n_fft=1024,
                win_length=1024,
                hop_length=256,
                power=1,
                fmin=0,
                fmax=8000,
            )
            print("Hello world")
            soundfile.write("output.wav", wav, samplerate=22050)
            np.save("output.npy", mels_post[0][:end].numpy())

        else:
            with torch.no_grad():
                tacotron2.eval()
                mels, wavs, gates, alignments = tacotron2.predict_step(
                    (tts_data, tts_metadata), 0, 0
                )
            soundfile.write("output2.wav", wavs[0].detach().numpy(), samplerate=22050)

    elif args.mode == "test":
        if not path.exists(args.dir_out):
            os.mkdir(args.dir_out)

        test_dataset = load_dataset(
            df=pd.read_csv(
                config["data"]["test"],
                delimiter="|",
                quoting=csv.QUOTE_NONE,
            ),
            config=config,
            dataset_dir=args.dataset_dir,
            feature_override=args.with_speech_features,
        )

        test_dataloader = TTSDataLoader(
            test_dataset,
            batch_size=config["training"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
        )

        tacotron2 = Tacotron2.load_from_checkpoint(
            checkpoint_path=args.checkpoint,
            teacher_forcing=False,
            **tacotron_args,
        )

        trainer = Trainer(
            devices=[1],  # config["training"]["devices"],
            accelerator=config["training"]["accelerator"],
            gradient_clip_val=1.0,
            enable_checkpointing=False,
            logger=False,
        )

        out = trainer.predict(tacotron2, dataloaders=test_dataloader)

        out_cpu = []

        for mel_spectrogram, mel_spectrogram_post, gate, alignment in out:
            out_cpu.append((mel_spectrogram_post.cpu(), gate.cpu()))
            del mel_spectrogram_post
            del gate
            del mel_spectrogram
            del alignment

        out = out_cpu
        del tacotron2

        torch.cuda.empty_cache()

        i = 0
        for mel_spectrogram_post, gate in out:
            for m in tqdm(range(len(mel_spectrogram_post))):
                stop_idx = 0
                g = torch.sigmoid(gate[m])

                for j in range(len(g)):
                    stop_idx = j
                    if g[j] <= 0.5:
                        break

                mel = torch.exp(mel_spectrogram_post[m, :stop_idx, :])

                wav = librosa.feature.inverse.mel_to_audio(
                    mel.numpy().T,
                    sr=22050,
                    n_fft=1024,
                    win_length=1024,
                    hop_length=256,
                    power=1,
                    pad_mode="reflect",
                )

                soundfile.write(
                    path.join(args.dir_out, f"{i}.wav"), wav, samplerate=22050
                )
                np.save(path.join(args.dir_out, f"{i}"), mel.numpy())
                i += 1
