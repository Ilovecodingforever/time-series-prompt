"""
get data loaders
"""

import math
import glob
from typing import Dict, Any, Optional, List, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader

from momentfm.utils.data import load_from_tsfile

from ts_datasets import InformerDataset, ClassificationDataset





def get_data(batch_size: int, task: str,
             filename: str = None,
             forecast_horizon: int = None,):
    """
    get data
    """

    if task == 'classify':
        train_dataset = ClassificationDataset(batch_size=batch_size,
                                              data_split='train', filename=filename)

        val_dataset = ClassificationDataset(batch_size=batch_size,
                                            data_split='val', filename=filename)

        test_dataset = ClassificationDataset(batch_size=batch_size,
                                             data_split='test', filename=filename)

    elif task == 'forecasting_long':
        train_dataset = InformerDataset(batch_size=batch_size, data_split='train',
                                        forecast_horizon=forecast_horizon,
                                        data_stride_len=1, filename=filename)

        val_dataset = InformerDataset(batch_size=batch_size, data_split='val',
                                      forecast_horizon=forecast_horizon,
                                      data_stride_len=1, filename=filename)

        test_dataset = InformerDataset(batch_size=batch_size, data_split='test',
                                       forecast_horizon=forecast_horizon,
                                       data_stride_len=1, filename=filename)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,)

    return train_loader, val_loader, test_loader







def load_mimic(equal_length=False, small_part=True, benchmark='mortality', ordinal=False):
    from ts_datasets import MIMIC_mortality  # TODO: this changes visible devices, why?
    from ts_datasets import MIMIC_phenotyping

    # batch_size = 1
    # batch_size = 16
    batch_size = 4
    shuffle = True


    if benchmark == 'mortality':
        train_data = MIMIC_mortality(data_split="train", equal_length=equal_length, small_part=small_part, ordinal=ordinal)
        val_data = MIMIC_mortality(data_split="val", equal_length=equal_length, small_part=small_part, ordinal=ordinal)
        test_data = MIMIC_mortality(data_split="test", equal_length=equal_length, small_part=small_part, ordinal=ordinal)

        # if ordinal:
        #     n_channels = 18
        # else:
        #     n_channels = 60

        # n_classes = 1

    elif benchmark == 'phenotyping':
        train_data = MIMIC_phenotyping(data_split="train", equal_length=equal_length, small_part=small_part, ordinal=ordinal)
        val_data = MIMIC_phenotyping(data_split="val", equal_length=equal_length, small_part=small_part, ordinal=ordinal)
        test_data = MIMIC_phenotyping(data_split="test", equal_length=equal_length, small_part=small_part, ordinal=ordinal)

        # if ordinal:
        #     n_channels = 18
        # else:
        #     n_channels = 60

        # n_classes = 25

    else:
        raise ValueError('benchmark not supported')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader





