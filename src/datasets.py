"""
get data sets
"""

import math
import glob
from typing import Dict, Any, Optional, List, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader

from main import RANDOM_SEED
from momentfm.utils.data import load_from_tsfile, convert_tsf_to_dataframe






# https://discuss.pytorch.org/t/dataloader-for-a-folder-with-multiple-files-pytorch-solutions-that-is-equivalent-to-tfrecorddataset-in-tf2-0/70512/4
class AnomalyDetectionDatasetMultiFile(torch.utils.data.IterableDataset):

    def __init__(
        self,
        batch_size,
        data_split: str = "train",
        data_stride_len: int = 512,
        random_seed: int = 42,
        dir="/zfsauton/project/public/Mononito/TimeseriesDatasets/anomaly_detection/TSB-UAD-Public",
        files = None
    ):
        """
        Parameters
        ----------
        data_split : str
            Split of the dataset, 'train', or 'test'
        data_stride_len : int
            Stride length for the data.
        random_seed : int
            Random seed for reproducibility.
        """

        # TODO: need to init?

        self.batch_size = batch_size

        self.dir = dir

        # TODO: shuffle this?
        if files is None:
            self.files = glob.glob(self.dir + '/**/*.*', recursive=True)
        else:
            self.files = files

        self.data_split = data_split
        self.data_stride_len = data_stride_len
        self.random_seed = random_seed
        self.seq_len = 512

        # Downsampling for experiments. Refer
        # https://github.com/mononitogoswami/tsad-model-selection for more details
        self.downsampling_factor = 10
        self.min_length = (
            2560  # Minimum length of time-series after downsampling for experiments
        )


    def _get_borders(self):

        n_train = math.floor(self.length_timeseries * 0.6)
        n_val = math.floor(self.length_timeseries * 0.1)
        n_test = math.floor(self.length_timeseries * 0.3)

        train_end = n_train
        val_end = n_train + n_val
        test_start = val_end - self.seq_len
        test_end = test_start + n_test + self.seq_len

        # TODO: is the indexing correct?
        train = slice(0, train_end)
        test = slice(test_start, test_end)
        val = slice(train_end - self.seq_len, val_end)

        return train, val, test


    def _read_data(self, file):
        self.scaler = StandardScaler()
        df = pd.read_csv(file)
        
        # TODO: too much data
        df = df.iloc[-int(len(df) * 0.05):]
        
        df.interpolate(inplace=True, method="cubic")

        self.length_timeseries = len(df)
        labels = df.iloc[:, -1].values
        timeseries = df.iloc[:, 0].values.reshape(-1, 1)
        self.n_channels = 1

        data_splits = self._get_borders()

        self.scaler.fit(timeseries[data_splits[0]])
        timeseries = self.scaler.transform(timeseries)
        timeseries = timeseries.squeeze()

        if self.data_split == "train":
            self.data, self.labels = timeseries[data_splits[0]], labels[data_splits[0]]
        elif self.data_split == "val":
            self.data, self.labels = timeseries[data_splits[1]], labels[data_splits[1]]
        elif self.data_split == "test":
            self.data, self.labels = timeseries[data_splits[2]], labels[data_splits[2]]

        self.length_timeseries = self.data.shape[0]


    def __iter__(self):

        for dir in self.files:
            for filename in glob.glob(dir + '/**/*.*', recursive=True):
                self.filename = filename
                self._read_data(filename)

                # make sure that one batch don't contain different datasets (channel size mixup)
                num = (self.length_timeseries // self.data_stride_len) + 1
                num = num // self.batch_size * self.batch_size

                for index in range(num):

                    seq_start = self.data_stride_len * index
                    seq_end = seq_start + self.seq_len
                    input_mask = np.ones(self.seq_len)

                    if seq_end > self.length_timeseries:
                        seq_start = self.length_timeseries - self.seq_len
                        seq_end = None

                    timeseries = self.data[seq_start:seq_end].reshape(
                        (self.n_channels, self.seq_len)
                    )
                    labels = (
                        self.labels[seq_start:seq_end]
                        .astype(int)
                        .reshape((self.n_channels, self.seq_len))
                    )

                    yield timeseries, input_mask, labels



class ClassificationDatasetMultiFile(torch.utils.data.IterableDataset):

    def __init__(self, batch_size, data_split="train",
                 dir="/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR",
                 files=None):
        """
        Parameters
        ----------
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        """
        # TODO: need to init?

        self.seq_len = 512
        self.batch_size = batch_size

        self.dir =  dir
        if files is None:
            # TODO: shuffle this?
            self.files = glob.glob(self.dir + '/**/*.*', recursive=True)
        else:
            self.files = files

        self.data_split = data_split  # 'train' or 'test'


    def _transform_labels(self, labels: np.ndarray):
        lbls = np.unique(self.train_labels)  # Move the labels to {0, ..., L-1}
        transform = {}
        for i, l in enumerate(lbls):
            transform[l] = i

        labels = np.vectorize(transform.get)(labels)

        return labels


    def _read_data(self, filename):
        self.scaler = StandardScaler()

        self.data, self.labels = load_from_tsfile(filename)
        self.labels = self._transform_labels(self.labels)

        temp = self.data

        self.original_length = None
        if isinstance(self.data, list):

            self.original_length = [s.shape[1] for s in self.data]

            # TODO: pad or trim or split?
            # TODO: change input_mask to not include padding
            self.data = [s[:, -self.seq_len:] if s.shape[1] >= self.seq_len else np.pad(s, ((0, 0), (self.seq_len - s.shape[1], 0))) \
                for s in self.data]
            self.data = np.stack(self.data, axis=0)


        self.num_timeseries = self.data.shape[0]
        self.len_timeseries = self.data.shape[2]
        self.num_dims = self.data.shape[1]


        self.data = self.data.transpose(0, 2, 1).reshape(-1, self.num_dims)
        self.scaler.fit(self.data)
        self.data = self.scaler.transform(self.data)
        # num x dim x len
        self.data = self.data.reshape(self.num_timeseries, self.len_timeseries, self.num_dims).transpose(0, 2, 1)

        # TODO: is this right?
        # self.data = self.data.reshape(-1, self.len_timeseries)
        # self.scaler.fit(self.data)
        # self.data = self.scaler.transform(self.data)
        # self.data = self.data.reshape(self.num_timeseries, self.len_timeseries)

        # self.data = self.data.T


    def __iter__(self):
        for filename in self.files:
            if self.data_split in filename.lower():
                self.filename = filename
                
                # get labels
                _, self.train_labels = load_from_tsfile(filename.replace(self.data_split.upper(), "TRAIN"))

                # get data
                self._read_data(filename)

                num = self.num_timeseries // self.batch_size * self.batch_size
                for index in range(num):

                    timeseries = self.data[index]
                    timeseries_len = timeseries.shape[1]
                    labels = self.labels[index,].astype(int)
                    input_mask = np.ones(self.seq_len)

                    if timeseries_len <= self.seq_len:
                        timeseries = np.pad(timeseries, ((0, 0), (self.seq_len - timeseries_len, 0)))
                        input_mask[: self.seq_len - timeseries_len] = 0
                    else:
                        # TODO: split or use last numbers?
                        timeseries = timeseries[:, -self.seq_len:]

                    if self.original_length is not None and self.original_length[index] < self.seq_len:
                        input_mask[: self.seq_len - self.original_length[index]] = 0

                    yield timeseries, input_mask, labels





class InformerDatasetMultiFile(torch.utils.data.IterableDataset):
    def __init__(
        self,
        batch_size,
        forecast_horizon: Optional[int] = 192,
        data_split: str = "train",
        data_stride_len: int = 1,
        task_name: str = "forecasting",
        random_seed: int = 42,
        dir = "/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer",
        files = None
    ):
        """
        Parameters
        ----------
        forecast_horizon : int
            Length of the prediction sequence.
        data_split : str
            Split of the dataset, 'train' or 'test'.
        data_stride_len : int
            Stride length when generating consecutive
            time series windows.
        task_name : str
            The task that the dataset is used for. One of
            'forecasting', or  'imputation'.
        random_seed : int
            Random seed for reproducibility.
        """
        # TODO: need to init?

        self.batch_size = batch_size

        self.dir = dir
        # TODO: shuffle this?
        if files is None:
            self.files = glob.glob(self.dir + '/**/*.*', recursive=True)
        else:
            self.files = files


        self.seq_len = 512
        self.forecast_horizon = forecast_horizon
        self.data_split = data_split
        self.data_stride_len = data_stride_len
        self.task_name = task_name
        self.random_seed = random_seed


    def _get_borders(self):

        n_train = math.floor(self.length_timeseries_original * 0.6)
        n_val = math.floor(self.length_timeseries_original * 0.1)
        n_test = math.floor(self.length_timeseries_original * 0.3)

        train_end = n_train
        val_end = n_train + n_val
        test_start = val_end - self.seq_len
        test_end = test_start + n_test + self.seq_len

        # TODO: is the indexing correct?
        train = slice(0, train_end)
        val = slice(train_end - self.seq_len, val_end)
        test = slice(test_start, test_end)

        assert train_end > 0 and val_end > 0 and test_start > 0 and test_end > 0 and train_end - self.seq_len > 0

        return train, val, test


    def _read_data(self, filename):
        self.scaler = StandardScaler()
        df = pd.read_csv(filename)
        self.length_timeseries_original = df.shape[0]
        self.n_channels = df.shape[1] - 1

        df.drop(columns=["date"], inplace=True)
        df = df.infer_objects(copy=False).interpolate(method="cubic")

        data_splits = self._get_borders()

        train_data = df[data_splits[0]]
        self.scaler.fit(train_data.values)
        df = self.scaler.transform(df.values)

        if self.data_split == "train":
            self.data = df[data_splits[0], :]
        elif self.data_split == "val":
            self.data = df[data_splits[1], :]
        elif self.data_split == "test":
            self.data = df[data_splits[2], :]

        self.length_timeseries = self.data.shape[0]


    def __iter__(self):
        for filename in self.files:
            self.filename = filename
            # get data
            self._read_data(filename)

            if self.task_name == "imputation":
                num = (self.length_timeseries - self.seq_len) // self.data_stride_len + 1
            elif self.task_name == "forecasting":
                num = (
                    self.length_timeseries - self.seq_len - self.forecast_horizon
                ) // self.data_stride_len + 1
            else:
                raise ValueError("Unknown task name")

            if num < 0:
                raise ValueError("data too short")

            num = num // self.batch_size * self.batch_size
            for index in range(num):

                seq_start = self.data_stride_len * index
                seq_end = seq_start + self.seq_len
                input_mask = np.ones(self.seq_len)

                if self.task_name == "imputation":
                    if seq_end > self.length_timeseries:
                        seq_end = self.length_timeseries
                        seq_end = seq_end - self.seq_len

                    timeseries = self.data[seq_start:seq_end, :].T

                    yield timeseries, input_mask

                else:
                    pred_end = seq_end + self.forecast_horizon

                    if pred_end > self.length_timeseries:
                        pred_end = self.length_timeseries
                        seq_end = seq_end - self.forecast_horizon
                        seq_start = seq_end - self.seq_len

                    assert(seq_start >=0 and seq_end >= 0 and pred_end >= 0)
                    timeseries = self.data[seq_start:seq_end, :].T
                    forecast = self.data[seq_end:pred_end, :].T

                    yield timeseries, forecast, input_mask



class MonashDatasetMultiFile(InformerDatasetMultiFile):
    def __init__(
        self,
        batch_size,
        forecast_horizon: Optional[int] = 10,
        data_split: str = "train",
        data_stride_len: int = 1,
        task_name: str = "forecasting",
        random_seed: int = 42,
        files = None,
        dir = "/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/monash"
    ):
        """
        Parameters
        ----------
        forecast_horizon : int
            Length of the prediction sequence.
        data_split : str
            Split of the dataset, 'train' or 'test'.
        data_stride_len : int
            Stride length when generating consecutive
            time series windows.
        task_name : str
            The task that the dataset is used for. One of
            'forecasting' or  'imputation'.
        random_seed : int
            Random seed for reproducibility.
        """

        super().__init__(batch_size, forecast_horizon, data_split, data_stride_len, 'forecasting_short', random_seed)

        self.dir = dir

        if files is None:
            # TODO: shuffle this?
            self.files = glob.glob(self.dir + '/**/*.*', recursive=True)
        else:
            self.files = files


    def _read_data(self, filename):

        self.scaler = StandardScaler()

        df = convert_tsf_to_dataframe(filename)[['series_value']]

        # shuffle
        df = df.sample(frac=1, random_state=RANDOM_SEED)
        
        # TODO: too much data
        df = df.iloc[:int(len(df) * 0.5)]

        # TODO: shuffle?
        if self.data_split == "train":
            df = df.iloc[:int(len(df) * 0.6)]
        elif self.data_split == "val":
            df = df.iloc[int(len(df) * 0.6):int(len(df) * 0.7)]
        elif self.data_split == "test":
            df = df.iloc[int(len(df) * 0.7):]


        # self.data = [np.array(s).reshape(1, -1) for s in df['series_value'].values]

        # if isinstance(self.data, list):

        #     self.original_length = [s.shape[1] for s in self.data]

        #     # TODO: pad or trim or split?
        #     self.data = [s[:, -self.seq_len:] if s.shape[1] >= self.seq_len else np.pad(s, ((0, 0), (self.seq_len - s.shape[1], 0))) \
        #         for s in self.data]
        #     self.data = np.stack(self.data, axis=0)


        # # TODO: do scaler only on training
        # data = self.data.reshape(-1, self.seq_len).T
        # self.scaler.fit(data)
        # data = self.scaler.transform(data)
        # self.data = data.T.reshape(self.data.shape[0], self.data.shape[1], self.seq_len)

        # self.n_channels = self.data.shape[1]

        self.data = [np.array(s).reshape(-1, 1) for s in df['series_value'].values]



    def __iter__(self):
        # TODO: debug this
        for filename in self.files:
            # get data
            self._read_data(filename)

            self.filename = filename

            for d in self.data:
                # TODO: do scaler only on training
                self.scaler.fit(d)
                d = self.scaler.transform(d)

                num = (
                    len(d) - self.seq_len - self.forecast_horizon
                ) // self.data_stride_len + 1

                if num < 0:
                    # print("data too short")
                    num = 1

                num = num // self.batch_size * self.batch_size
                for index in range(num):

                    seq_start = self.data_stride_len * index
                    seq_end = seq_start + self.seq_len
                    input_mask = np.ones(self.seq_len)

                    pred_end = seq_end + self.forecast_horizon

                    # TODO: check if this is correct
                    if len(d) < self.seq_len:
                        d = np.pad(d, ((self.seq_len + pred_end - len(d), 0), (0, 0)))
                        input_mask[: self.seq_len + pred_end - len(d)] = 0

                    if pred_end > len(d):
                        pred_end = len(d)
                        seq_end = seq_end - self.forecast_horizon
                        seq_start = seq_end - self.seq_len

                    assert(seq_start >=0 and seq_end >= 0 and pred_end >= 0)
                    timeseries = d[seq_start:seq_end, :].T
                    forecast = d[seq_end:pred_end, :].T

                    yield timeseries, forecast, input_mask
