import torch
from torch import nn
from torch.utils.data import DataLoader
from collections import defaultdict


def _collate(data):
    # Transform a list of TTSData/TTSDataLength instances into a single
    # TTSData/TTSDataLength instance containing data batches

    # Separate the TTSData and TTSDataLength instances into separate collated dictionaries
    tts_data_collated, tts_data_len_collated = defaultdict(list), defaultdict(list)
    for tts_data_i, tts_data_len_i in data:
        for k, v in tts_data_i.items():
            tts_data_collated[k].append(v)
        for k, v in tts_data_len_i.items():
            tts_data_len_collated[k].append(v)
    
    tts_data = dict()
    for k, v in tts_data_collated.items():
        tts_data[k] = nn.utils.rnn.pad_sequence(v, batch_first=True)
    
    tts_data_len = dict()
    for k, v in tts_data_len_collated.items():
        tts_data_len[k] = torch.stack(v).squeeze(1)

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
