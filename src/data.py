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
from momentfm.data.informer_dataset import InformerDataset
from momentfm.data.classification_dataset import ClassificationDataset
from momentfm.data.anomaly_detection_dataset import AnomalyDetectionDataset

from main import RANDOM_SEED
from ts_datasets import InformerDatasetMultiFile, ClassificationDatasetMultiFile, AnomalyDetectionDatasetMultiFile, MonashDatasetMultiFile




class CollectedDataset(torch.utils.data.Dataset):
    """
    combine multiple datasets
    """

    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, idx):
        return {key: dataset.__getitem__(idx) if idx < len(dataset) else None \
                for key, dataset in self.datasets.items()}

    def __len__(self):
        return max(len(d) for d in self.datasets.values())



class CollectedDatasetMultiFile(torch.utils.data.IterableDataset):
    """
    combine multiple datasets
    """

    def __init__(self, datasets):
        self.datasets = datasets
        self.tasks = list(datasets.keys())
        self.reset()


    def __iter__(self):

        while True:
            vals = {}
            for key, dataset in self.datasets_iter.items():
                vals[key] = next(dataset, None)
                if vals[key] is not None:
                    self.counts[key] += 1

            if all(val is None for val in vals.values()):
                break


            self.filenames = {key: dataset.filename for key, dataset in self.datasets.items()}

            yield vals

    def reset(self):
        self.datasets_iter = {key: iter(dataset) for key, dataset in self.datasets.items()}
        self.counts = {key: 0 for key in self.tasks}






def collate_fn(batch):
    """
    collate function
    """
    data = {}
    for key in batch[0].keys():
        batch_ = [batch[i][key] for i in range(len(batch)) if batch[i][key] is not None]
        if len(batch_) == 0:
            continue
        data[key] = torch.utils.data.dataloader.default_collate(batch_)

    return data


def get_data(batch_size, dataset_names,
             all=False, files=None, forecast_horizon=None):
    """
    get data
    """

    # from datasets import get_dataset_config_names
    # configs = get_dataset_config_names("AutonLab/Timeseries-PILE")
    # /zfsauton/project/public/Mononito/TimeseriesDatasets

    train_datasets = {}
    val_datasets = {}
    test_datasets = {}
    if 'imputation' in dataset_names:

        if all:
            f = files
            if files is not None and isinstance(files, dict):
                f = files['imputation']
            train_dataset_impute = InformerDatasetMultiFile(batch_size=batch_size,
                                                            data_split='train',
                                                            random_seed=RANDOM_SEED,
                                                            task_name='imputation',
                                                            data_stride_len=1,
                                                            files=f,
                                                            )
            
            val_dataset_impute = InformerDatasetMultiFile(batch_size=batch_size,
                                                            data_split='val',
                                                            random_seed=RANDOM_SEED,
                                                            task_name='imputation',
                                                            data_stride_len=1,
                                                            files=f,
                                                            )
            
            test_dataset_impute = InformerDatasetMultiFile(batch_size=batch_size,
                                                            data_split='test',
                                                            random_seed=RANDOM_SEED,
                                                            task_name='imputation',
                                                            data_stride_len=1,
                                                            files=f,
                                                            )
        else:
            train_dataset_impute = InformerDataset(data_split='train', random_seed=RANDOM_SEED,
                                                    task_name='imputation',
                                                    data_stride_len=15
                                                    # data_stride_len=512
                                                    )
            
            val_dataset_impute = InformerDataset(data_split='val', random_seed=RANDOM_SEED,
                                                    task_name='imputation',
                                                    data_stride_len=15
                                                    # data_stride_len=512
                                                    )
            
            test_dataset_impute = InformerDataset(data_split='test', random_seed=RANDOM_SEED,
                                                    task_name='imputation',
                                                    data_stride_len=15
                                                    # data_stride_len=512
                                                    )


        train_datasets['imputation'] = train_dataset_impute
        val_datasets['imputation'] = val_dataset_impute
        test_datasets['imputation'] = test_dataset_impute

    if 'anomaly' in dataset_names:

        if all:
            f = files
            if files is not None and isinstance(files, dict):
                f = files['anomaly']
            train_dataset_anomaly = AnomalyDetectionDatasetMultiFile(batch_size=batch_size,
                                                                data_split='train',
                                                                random_seed=RANDOM_SEED,
                                                                data_stride_len=1,
                                                                files=f
                                                                )
            
            val_dataset_anomaly = AnomalyDetectionDatasetMultiFile(batch_size=batch_size,
                                                                data_split='val',
                                                                random_seed=RANDOM_SEED,
                                                                data_stride_len=1,
                                                                files=f
                                                                )
            
            test_dataset_anomaly = AnomalyDetectionDatasetMultiFile(batch_size=batch_size,
                                                                data_split='test',
                                                                random_seed=RANDOM_SEED,
                                                                data_stride_len=1,
                                                                files=f
                                                                )
        else:
            train_dataset_anomaly = AnomalyDetectionDataset(data_split='train', random_seed=RANDOM_SEED,
                                                            data_stride_len=100
                                                            # data_stride_len=512
                                                            )
            
            val_dataset_anomaly = AnomalyDetectionDataset(data_split='val', random_seed=RANDOM_SEED,
                                                            data_stride_len=100
                                                            # data_stride_len=512
                                                            )
            
            test_dataset_anomaly = AnomalyDetectionDataset(data_split='test', random_seed=RANDOM_SEED,
                                                            data_stride_len=100
                                                            # data_stride_len=512
                                                            )

        train_datasets['anomaly'] = train_dataset_anomaly
        val_datasets['anomaly'] = val_dataset_anomaly
        test_datasets['anomaly'] = test_dataset_anomaly

    if 'classify' in dataset_names:

        if all:
            f = files
            if files is not None and isinstance(files, dict):
                f = files['classify']
            train_dataset_classify = ClassificationDatasetMultiFile(batch_size=batch_size,
                                                                    data_split='train',
                                                                    files=f
                                                                    )
            
            val_dataset_classify = ClassificationDatasetMultiFile(batch_size=batch_size,
                                                                    data_split='val',
                                                                    files=f
                                                                    )
            
            test_dataset_classify = ClassificationDatasetMultiFile(batch_size=batch_size,
                                                                    data_split='test',
                                                                    files=f
                                                                    )
        else:
            train_dataset_classify = ClassificationDataset(data_split='train')
            val_dataset_classify = ClassificationDataset(data_split='val')
            test_dataset_classify = ClassificationDataset(data_split='test')

        train_datasets['classify'] = train_dataset_classify
        val_datasets['classify'] = val_dataset_classify
        test_datasets['classify'] = test_dataset_classify

    if 'forecasting_long' in dataset_names:

        if all:
            f = files
            if files is not None and isinstance(files, dict):
                f = files['forecasting_long']
            train_dataset_forecast_long = InformerDatasetMultiFile(batch_size=batch_size,
                                                                data_split='train',
                                                                random_seed=RANDOM_SEED,
                                                                forecast_horizon=forecast_horizon,
                                                                data_stride_len=1,
                                                                files=f
                                                                )
            
            val_dataset_forecast_long = InformerDatasetMultiFile(batch_size=batch_size,
                                                                data_split='val',
                                                                random_seed=RANDOM_SEED,
                                                                forecast_horizon=forecast_horizon,
                                                                data_stride_len=1,
                                                                files=f
                                                                )
            
            test_dataset_forecast_long = InformerDatasetMultiFile(batch_size=batch_size,
                                                                data_split='test',
                                                                random_seed=RANDOM_SEED,
                                                                forecast_horizon=forecast_horizon,
                                                                data_stride_len=1,
                                                                files=f
                                                                )
        else:
            train_dataset_forecast_long = InformerDataset(data_split="train", random_seed=RANDOM_SEED,
                                                            forecast_horizon=forecast_horizon,
                                                            data_stride_len=15
                                                            )
            
            val_dataset_forecast_long = InformerDataset(data_split="val", random_seed=RANDOM_SEED,
                                                            forecast_horizon=forecast_horizon,
                                                            data_stride_len=15
                                                            )
            
            test_dataset_forecast_long = InformerDataset(data_split="test", random_seed=RANDOM_SEED,
                                                            forecast_horizon=forecast_horizon,
                                                            data_stride_len=15
                                                            )

        train_datasets['forecasting_long'] = train_dataset_forecast_long
        val_datasets['forecasting_long'] = val_dataset_forecast_long
        test_datasets['forecasting_long'] = test_dataset_forecast_long

    if 'forecasting_short' in dataset_names:

        if all:
            f = files
            if files is not None and isinstance(files, dict):
                f = files['forecasting_short']
            train_dataset_forecast_short = MonashDatasetMultiFile(batch_size=batch_size,
                                                                data_split='train',
                                                                random_seed=RANDOM_SEED,
                                                                forecast_horizon=8,
                                                                data_stride_len=1,
                                                                files=f
                                                                )
            
            val_dataset_forecast_short = MonashDatasetMultiFile(batch_size=batch_size,
                                                                data_split='val',
                                                                random_seed=RANDOM_SEED,
                                                                forecast_horizon=8,
                                                                data_stride_len=1,
                                                                files=f
                                                                )
            
            test_dataset_forecast_short = MonashDatasetMultiFile(batch_size=batch_size,
                                                                data_split='test',
                                                                random_seed=RANDOM_SEED,
                                                                forecast_horizon=8,
                                                                data_stride_len=1,
                                                                files=f
                                                                )
        else:
            train_dataset_forecast_short = InformerDataset(data_split="train", random_seed=RANDOM_SEED,
                                                            forecast_horizon=8,
                                                            data_stride_len=15
                                                            )
            
            val_dataset_forecast_short = InformerDataset(data_split="val", random_seed=RANDOM_SEED, 
                                                            forecast_horizon=8,
                                                            data_stride_len=15
                                                            )
            
            test_dataset_forecast_short = InformerDataset(data_split="test", random_seed=RANDOM_SEED,
                                                            forecast_horizon=8,
                                                            data_stride_len=15
                                                            )

        train_datasets['forecasting_short'] = train_dataset_forecast_short
        val_datasets['forecasting_short'] = val_dataset_forecast_short
        test_datasets['forecasting_short'] = test_dataset_forecast_short


    if all:
        data = CollectedDatasetMultiFile(train_datasets)
        # data = train_datasets
    else:
        data = CollectedDataset(train_datasets)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn
                            )
    
    if all:
        data = CollectedDatasetMultiFile(val_datasets)
        # data = val_datasets
    else:
        data = CollectedDataset(val_datasets)
    val_loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn
                            )
    
    if all:
        data = CollectedDatasetMultiFile(test_datasets)
        # data = test_datasets
    else:
        data = CollectedDataset(test_datasets)
    test_loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn
                             )

    return train_loader, val_loader, test_loader






def func(d):
    yield next(d)


if __name__ == '__main__':

    batch_size = 8
    dataset = InformerDatasetMultiFile(batch_size)

    dataset = iter(dataset)
    print(next(func(dataset)))
    print(next(func(dataset)))

    dataset = CollectedDatasetMultiFile({'imputation': dataset})

    for d in dataset:
        print(d)

    print(next(dataset))
    print(next(dataset))

    loader = DataLoader(dataset, batch_size=batch_size)

    for timeseries, input_mask, labels in loader:
        print(timeseries.shape, input_mask.shape, labels.shape)




