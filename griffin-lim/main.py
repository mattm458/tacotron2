import sys

from librosa.feature.inverse import mel_to_audio
from soundfile import write
from torch import load, exp

if __name__ == "__main__":
    print(sys.argv)
    mels = load(sys.argv[1])[0].swapaxes(0,1)
    mels = exp(mels)
    print(mels.shape)
    mels = mels[:,:int(sys.argv[2])]
    print(mels.shape)

    wav = mel_to_audio(
        mels.numpy(),
        sr=22040,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        power=1.0,
        pad_mode="reflect",
        fmin=0,
        fmax=8000,
    )

    write(sys.argv[3], wav, 22040)
