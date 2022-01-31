import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import tts_types


def _collate(data):
    # Transform a list of TTSData/TTSDataLength instances into a single
    # TTSData/TTSDataLength instance containing data batches

    # Separate the TTSData and TTSDataLength instances into separate lists
    tts_data, tts_data_len = [], []
    for tts_data_i, tts_data_len_i in data:
        tts_data.append(tts_data_i)
        tts_data_len.append(tts_data_len_i)

    # Pad all tensors in the list of TTSData objects, then create a new TTSData
    # instance containing the batch
    tts_data = tts_types.TTSData(
        *[nn.utils.rnn.pad_sequence(i, batch_first=True) for i in zip(*tts_data)]
    )

    # Stack all tensors in the list of TTSDataLength objects, then create a new
    # TTSDataLength instance containing the batch
    tts_data_len = tts_types.TTSDataLength(
        *[torch.stack(i).squeeze(1) for i in zip(*tts_data_len)]
    )

    return tts_data, tts_data_len


def TTSDataLoader(
    dataset,
    batch_size=1,
    num_workers=0,
    pin_memory=None,
    shuffle=None,
    prefetch_factor=2,
):
    """Create a new DataLoader for TTS data.

    Args:
        dataset -- a TTSDataset object
        batch_size -- how many samples per batch to load (default 1)
        num_workers -- how many subprocesses to use for data loading. 0 means that the data will be
                       loaded in the main process (default 0)
        pin_memory -- if True, the data loader will copy Tensors into CUDA pinned memory before
                      returning them (default False)
        shuffle -- set to True to have data reshuffled at every epoch (default False)
        prefetch_factor -- number of samples loaded in advance by each worker. 2 means that there
                           will be a total of 2 * num_workers samples prefetched across all
                           workers (default 2)
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=_collate if batch_size > 1 else None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        prefetch_factor=prefetch_factor,
    )
