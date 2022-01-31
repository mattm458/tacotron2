from collections import namedtuple

TTSData = namedtuple("TTSData", ["chars_idx", "mel_spectrogram", "gate"])

TTSDataLength = namedtuple(
    "TTSDataLength", ["chars_idx_len", "mel_spectrogram_len", "gate_len"]
)
