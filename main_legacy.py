import csv
import os
import warnings
from os import path

import librosa
import matplotlib
import numpy as np
import pandas as pd
import soundfile
import soundfile as sf
import torch
import yaml
from hifi_gan.model import HifiGan
from hifi_gan.model.generator import Generator
from lightning import Trainer
from sklearn.preprocessing import OrdinalEncoder
from speech_utils.audio.transforms import HifiGanMelSpectrogram
from tqdm import tqdm

from datasets.tts_dataloader import TTSDataLoader
from datasets.tts_dataset import TTSDataset
from model.prosodic_features import prosody_detector
from model.tacotron2_legacy import Tacotron2

omit = set(
    [
        "LJ037-0076.wav",
        "LJ049-0075.wav",
        "LJ021-0165.wav",
        "LJ045-0047.wav",
        "LJ011-0152.wav",
        "LJ011-0028.wav",
        "LJ012-0199.wav",
        "LJ006-0281.wav",
        "LJ009-0074.wav",
        "LJ030-0137.wav",
        "LJ037-0003.wav",
    ]
)
# omit = set()


def load_dataset(df, config, dataset_dir, feature_override=None, **args):
    filenames = []
    texts = []
    features = []

    for filename, text, (_, feature) in zip(
        df.wav,
        df.text,
        df[config["extensions"]["features"]["allowed_features"]].iterrows(),
    ):
        if filename in omit:
            continue

        filenames.append(filename)
        texts.append(text)
        features.append(feature)

    args["filenames"] = filenames
    args["texts"] = texts

    if "allowed_chars" in config["preprocessing"]:
        args["allowed_chars"] = config["preprocessing"]["allowed_chars"]

    if "expand_abbreviations" in config["preprocessing"]:
        args["expand_abbreviations"] = config["preprocessing"]["expand_abbreviations"]

    if "end_token" in config["preprocessing"]:
        args["end_token"] = config["preprocessing"]["end_token"]

    if "silence" in config["preprocessing"]:
        args["silence"] = config["preprocessing"]["silence"]

    if "model" in config and "extensions" in config["model"]:
        if "features" in config["model"]["extensions"]:
            # features = df[config["extensions"]["features"]["allowed_features"]]
            args["features"] = features

    return TTSDataset(**args, base_dir=dataset_dir, feature_override=feature_override)


def get_tacotron_args(config):
    num_chars = len(config["preprocessing"]["allowed_chars"])
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

        tacotron2 = torch.compile(
            Tacotron2(teacher_forcing=True, **tacotron_args), fullgraph=True
        )

        trainer = Trainer(
            devices=config["training"]["devices"],
            accelerator=config["training"]["accelerator"],
            precision=config["training"]["precision"],
            gradient_clip_val=1.0,
            max_epochs=config["training"]["max_epochs"],
            # check_val_every_n_epoch=5,
        )

        trainer.fit(
            tacotron2,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=args.checkpoint,
        )

    if args.mode == "finetune":
        dataset_dir = args.dataset_dir

        print("Finetune: Loading datasets...")
        train_dataset = load_dataset(
            df=pd.read_csv(
                config["data"]["train"],
                delimiter="|",
                quoting=csv.QUOTE_NONE,
            ),
            config=config,
            dataset_dir=dataset_dir,
            include_wav=True,
        )
        val_dataset = load_dataset(
            df=pd.read_csv(
                config["data"]["val"],
                delimiter="|",
                quoting=csv.QUOTE_NONE,
            ),
            config=config,
            dataset_dir=dataset_dir,
            include_wav=True,
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

        hifi_gan_generator = None
        hifi_gan_mpd = None
        hifi_gan_msd = None
        hifi_gan_spectrogram = None

        if args.hifi_gan_checkpoint is not None:
            print("Finetune: Loading HiFi-GAN checkpoint...")
            hifi_gan = HifiGan.load_from_checkpoint(args.hifi_gan_checkpoint)
            hifi_gan_generator = hifi_gan.generator

            if args.fine_tune_hifi_gan:
                print(
                    "Finetune: Applying fine-tuning to HiFi-GAN and its discriminators"
                )
                hifi_gan_mpd = hifi_gan.multi_period_discriminator
                hifi_gan_msd = hifi_gan.multi_scale_discriminator
                hifi_gan_spectrogram = HifiGanMelSpectrogram()
            else:
                print("Finetune: Not fine tuning HiFi-GAN, disabling grad")
                hifi_gan_generator.requires_grad_(False)
        elif args.fine_tune_hifi_gan:
            if args.resume_finetune:
                print(
                    "Finetune: Resuming finetuning, expecting HiFi-GAN model in the checkpoint"
                )
                hifi_gan = HifiGan()
                hifi_gan_generator = hifi_gan.generator
                hifi_gan_mpd = hifi_gan.multi_period_discriminator
                hifi_gan_msd = hifi_gan.multi_scale_discriminator
                hifi_gan_spectrogram = HifiGanMelSpectrogram()
            else:
                raise Exception("Cannot fine-tune HiFi-GAN without a checkpoint!")

        prosody_model = None
        post_prosody_model = None

        if args.prosody_model_checkpoint is not None:
            print(
                f"Finetune: Loading pre-vocoder prosody model {args.prosody_model_checkpoint}"
            )

            if args.resume_finetune:
                prosody_model = prosody_detector.ProsodyPredictorLightning(
                    rnn_layers=2,
                    rnn_dropout=0.0,
                    features=[
                        "pitch_mean_norm_clip",
                        "pitch_range_norm_clip",
                        "intensity_mean_vcd_norm_clip",
                        "jitter_norm_clip",
                    ],
                )
            else:
                print("Finetune: Loading pre-vocoder prosody model...")
                prosody_model = (
                    prosody_detector.ProsodyPredictorLightning.load_from_checkpoint(
                        args.prosody_model_checkpoint,
                        rnn_layers=2,
                        rnn_dropout=0.0,
                        features=[
                            "pitch_mean_norm_clip",
                            "pitch_range_norm_clip",
                            "intensity_mean_vcd_norm_clip",
                            "jitter_norm_clip",
                            "rate_norm_clip",
                        ],
                    )
                )

            print(f"Finetune:Pre-vocoder prosody model: Disabling grad...")
            prosody_model.requires_grad_(False)

            prosody_model = prosody_model.prosody_predictor

        if args.prosody_model_post_checkpoint is not None:
            print(
                f"Finetune: Loading post-HiFi-GAN prosody model {args.prosody_model_post_checkpoint}"
            )

            if args.resume_finetune:
                post_prosody_model = prosody_detector.ProsodyPredictorLightning(
                    rnn_layers=2, rnn_dropout=0.0, features=["jitter_norm_clip"]
                )
            else:
                print("Finetune: Loading post-vocoder prosody model...")
                post_prosody_model = (
                    prosody_detector.ProsodyPredictorLightning.load_from_checkpoint(
                        args.prosody_model_post_checkpoint,
                        rnn_layers=2,
                        rnn_dropout=0.0,
                        features=["jitter_norm_clip"],
                    )
                )

            print(f"Finetune: Post-vocoder prosody model: Disabling grad...")
            post_prosody_model.requires_grad_(False)
            post_prosody_model = post_prosody_model.prosody_predictor

        if args.fine_tune_lr is not None:
            print(f"Finetune: Setting learning rate to {args.fine_tune_lr}")
            tacotron_args["lr"] = args.fine_tune_lr

        if not args.fine_tune_tacotron:
            print("Finetune: Freezing Tacotron model")
            tacotron_args["freeze_tacotron"] = True

        if args.resume_finetune:
            tacotron2 = Tacotron2(
                prosody_model=prosody_model,
                post_prosody_model=post_prosody_model,
                hifi_gan_generator=hifi_gan_generator,
                hifi_gan_mpd=hifi_gan_mpd,
                hifi_gan_msd=hifi_gan_msd,
                fine_tune_style=args.fine_tune_tacotron_style,
                fine_tune_features=args.fine_tune_tacotron_features,
                fine_tune_hifi_gan=args.fine_tune_hifi_gan,
                hifi_gan_spectrogram=hifi_gan_spectrogram,
                fine_tune_tacotron_limit=args.fine_tune_tacotron_limit,
                fine_tune_tacotron=args.fine_tune_tacotron,
                fine_tune_hifi_gan_style=args.fine_tune_hifi_gan_style,
                teacher_forcing=True,
                teacher_forcing_prob=0.5,
                **tacotron_args,
            )
        else:
            print("Finetune: Loading Tacotron checkpoint...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                tacotron2 = Tacotron2.load_from_checkpoint(
                    args.tacotron_checkpoint,
                    strict=False,
                    prosody_model=prosody_model,
                    post_prosody_model=post_prosody_model,
                    hifi_gan_generator=hifi_gan_generator,
                    hifi_gan_mpd=hifi_gan_mpd,
                    hifi_gan_msd=hifi_gan_msd,
                    fine_tune_style=args.fine_tune_tacotron_style,
                    fine_tune_features=args.fine_tune_tacotron_features,
                    fine_tune_hifi_gan=args.fine_tune_hifi_gan,
                    hifi_gan_spectrogram=hifi_gan_spectrogram,
                    fine_tune_tacotron_limit=args.fine_tune_tacotron_limit,
                    fine_tune_tacotron=args.fine_tune_tacotron,
                    teacher_forcing=True,
                    teacher_forcing_prob=0.5,
                    **tacotron_args,
                )

        trainer = Trainer(
            devices=config["training"]["devices"],
            accelerator=config["training"]["accelerator"],
            precision=config["training"]["precision"],
            gradient_clip_val=None if args.fine_tune_hifi_gan else 1.0,
            max_epochs=10,  # config["training"]["max_epochs"],
            check_val_every_n_epoch=1,
        )

        if args.resume_finetune:
            print("Finetune: Resuming training from checkpoint...")
            trainer.fit(
                tacotron2,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path=args.tacotron_checkpoint,
            )
        else:
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
                args.hifi_gan_checkpoint, map_location="cuda:0"
            )["state_dict"]
            hifi_gan_states = dict(
                [(k[10:], v) for k, v in hifi_gan_states.items() if "generator" in k]
            )

            generator = Generator()
            generator.weight_norm()
            generator.load_state_dict(hifi_gan_states)

        else:
            generator = Generator()
            generator.weight_norm()

        tacotron2 = Tacotron2.load_from_checkpoint(
            checkpoint_path=args.checkpoint,
            strict=False,
            teacher_forcing=False,
            hifi_gan_generator=generator,
            **tacotron_args,
        )
        generator = tacotron2.hifi_gan_generator

        ALLOWED_CHARS = (
            "!'(),.:;? \\-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            # "!'(),.:;? \\-abcdefghijklmnopqrstuvwxyz"
        )
        # end_token = "^"
        end_token = None
        encoder = OrdinalEncoder()
        # encoder.fit([[x] for x in list(ALLOWED_CHARS)] + [[end_token]])
        encoder.fit([[x] for x in list(ALLOWED_CHARS)])

        encoded = (
            torch.LongTensor(
                # encoder.transform([[x] for x in args.text.lower()] + [[end_token]])
                encoder.transform([[x] for x in args.text.lower()])
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
            # "features": torch.Tensor([[0.0, 0.0, 0.0, 1., 0.0, 0.0, 0.0]]),
            "features": torch.Tensor([[0.0, 0.0, 0.0, -1.0, 0.0]]),
        }

        if generator is None:
            with torch.no_grad():
                tacotron2.eval()
                mels, mels_post, gates, alignments = tacotron2.predict_step(
                    (tts_data, tts_metadata, {}), 0, 0
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

            print(mels_post.shape, end)
            mels_exp = torch.exp(mels_post[0])  # [:end]
            matplotlib.use("TkAgg")

            matplotlib.pyplot.imshow(mels[0].numpy())
            matplotlib.pyplot.show()

            print(mels_exp.shape)
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
                mels, mel_spectrogram_post, gates, alignments = tacotron2.predict_step(
                    (tts_data, tts_metadata, {}), 0, 0
                )

                gates = gates[0]
                gates = torch.sigmoid(gates)

                end = -1
                for i in range(gates.shape[0]):
                    if gates[i][0] < 0.5:
                        end = i
                        break

                print(end)
                wavs = generator(mel_spectrogram_post[:, :end])

            soundfile.write("output2.wav", wavs[0].detach().numpy(), samplerate=22050)

    elif args.mode == "test":
        if not path.exists(args.dir_out):
            os.mkdir(args.dir_out)

        torch.set_float32_matmul_precision("high")

        hifi_gan_checkpoint = args.hifi_gan_checkpoint

        use_hifi_gan = hifi_gan_checkpoint is not None
        generator = None

        if use_hifi_gan:
            print(f"Loading HiFi-GAN checkpoint {hifi_gan_checkpoint}...")
            hifi_gan_states = torch.load(hifi_gan_checkpoint, map_location="cuda:0")[
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
            generator = generator.cuda()

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

        results = trainer.predict(tacotron2, dataloaders=test_dataloader)
        i = 0

        if use_hifi_gan:
            with torch.no_grad():
                for _, mel_spectrogram_post, gate, _ in tqdm(
                    results, desc="Saving WAVs"
                ):
                    if mel_spectrogram_post.shape[1] == 2000:
                        print(
                            "Warning: Batch contains max-length Mel spectrogram! Skipping..."
                        )
                        continue

                    mel_lengths = (gate < 0).type(torch.long).squeeze(-1).argmax(dim=-1)
                    wav_lengths = mel_lengths * 256

                    wavs = generator(mel_spectrogram_post.to("cuda:0"))
                    wavs = wavs.cpu().numpy()

                    for wav, wav_length in zip(wavs, wav_lengths):
                        i += 1
                        wav = wav[:wav_length]
                        if len(wav) == 0:
                            continue

                        sf.write(
                            path.join(args.dir_out, f"{i}.wav"), wav[:wav_length], 22050
                        )
        else:
            for _, mel_spectrogram_post, gate, _ in tqdm(results, desc="Saving WAVs"):
                if mel_spectrogram_post.shape[1] == 2000:
                    print(
                        "Warning: Batch contains max-length Mel spectrogram! Skipping..."
                    )
                    continue
                mel_lengths = (gate < 0).type(torch.long).squeeze(-1).argmax(dim=-1)
                for this_mel_spectrogram, mel_length in zip(
                    mel_spectrogram_post, mel_lengths
                ):
                    i += 1

                    this_mel_spectrogram = this_mel_spectrogram[:mel_length]
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

                    sf.write(path.join(args.dir_out, f"{i}.wav"), wav, 22050)
