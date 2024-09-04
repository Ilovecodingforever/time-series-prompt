"""
get data sets
"""
import os
import math
import glob
import logging
from typing import Dict, Any, Optional, List, Callable

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

import torch
from torch.utils.data import DataLoader

from main import RANDOM_SEED
from momentfm.utils.data import load_from_tsfile, convert_tsf_to_dataframe

import sys
sys.path.append('/zfsauton2/home/mingzhul/time-series-prompt')
sys.path.append('/zfsauton2/home/mingzhul/time-series-prompt/mimic3_benchmarks')

from mimic3_benchmarks.mimic3benchmark.readers import InHospitalMortalityReader
from mimic3_benchmarks.mimic3models.preprocessing import Discretizer, Normalizer
from mimic3_benchmarks.mimic3models.in_hospital_mortality import utils



# UCR: https://timeseriesclassification.com/dataset.php




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

        self.train_ratio = 0.6
        self.val_ratio = 0.1
        self.test_ratio = 0.3

    def _get_borders(self):

        n_train = int(self.train_ratio * self.length_timeseries)
        n_test = int(self.test_ratio * self.length_timeseries)
        n_val = self.length_timeseries - n_train - n_test

        # We use reconstruction based anomaly detection
        # so we do not need "context"
        train_end = n_train
        val_start = train_end  # - self.seq_len
        val_end = val_start + n_val
        test_start = val_end  # - self.seq_len

        train = slice(0, train_end)
        val = slice(val_start, val_end)
        test = slice(test_start, -1)

        return train, val, test


    def _get_borders_train_val(self):
        train_ratio = self.train_ratio / (self.train_ratio + self.val_ratio)
        n_train = int(train_ratio * self.length_timeseries)
        n_val = self.length_timeseries - n_train

        # We use reconstruction based anomaly detection
        # so we do not need "context"
        train_end = n_train
        val_start = train_end  # - self.seq_len
        val_end = val_start + n_val

        return slice(0, train_end), slice(val_start, val_end)


    def _get_borders_KDD21(self):
        train_ratio = self.train_ratio / (self.train_ratio + self.val_ratio)
        details = self.series.split("_")
        n_train = int(details[4])
        n_test = self.length_timeseries - n_train
        n_train = int(train_ratio * n_train)
        n_val = self.length_timeseries - n_train - n_test

        # We use reconstruction based anomaly detection
        # so we do not need "context"
        train_end = n_train
        val_start = train_end
        val_end = val_start + n_val
        test_start = val_end

        return slice(0, train_end), slice(val_start, val_end), slice(test_start, None)


    def _read_and_process_NASA(self):
        def _load_one_split(data_split: str = "train"):
            data_split = "test" if data_split == "test" else "train"
            root_path = self.full_file_path_and_name.split("/")[:-1]
            path = os.path.join("/".join(root_path), self.series + f".{data_split}.out")
            df = pd.read_csv(path).infer_objects(copy=False).interpolate(method="cubic")
            return df.iloc[:, 0].values, df.iloc[:, -1].values.astype(int)

        self.n_channels = 1  # All TSB-UAD datasets are univariate
        timeseries, labels = _load_one_split(data_split=self.data_split)
        timeseries = timeseries.reshape(-1, 1)


        if self.data_split == "train":
            self.scaler.fit(timeseries)
        else:
            train_timeseries, _ = _load_one_split(data_split="train")
            train_timeseries = train_timeseries.reshape(-1, 1)
            self.scaler.fit(train_timeseries)
            timeseries = self.scaler.transform(timeseries).squeeze()

        self.length_timeseries = len(timeseries)

        data_splits = self._get_borders_train_val()

        # Normalize train and validation ratios
        if self.data_split == "train":
            data_splits = self._get_borders_train_val()
            self.data, self.labels = self._downsample_timeseries(
                timeseries[data_splits[0]], labels[data_splits[0]]
            )
        elif self.data_split == "val":
            data_splits = self._get_borders_train_val()
            self.data, self.labels = self._downsample_timeseries(
                timeseries[data_splits[1]], labels[data_splits[1]]
            )
        elif self.data_split == "test":
            self.data, self.labels = self._downsample_timeseries(timeseries, labels)

        self.length_timeseries = self.data.shape[0]

    def _downsample_timeseries(self, timeseries, labels):
        # Downsampling code taken from TSADAMS: https://github.com/mononitogoswami/tsad-model-selection/blob/src/tsadams/datasets/load.py#L100
        if (
            (self.downsampling_factor is not None)
            and (self.min_length is not None)
            and (len(timeseries) // self.downsampling_factor > self.min_length)
        ):
            padding = (
                self.downsampling_factor - len(timeseries) % self.downsampling_factor
            )
            timeseries = np.pad(timeseries, ((padding, 0)))
            labels = np.pad(labels, (padding, 0))

            timeseries = timeseries.reshape(
                timeseries.shape[-1] // self.downsampling_factor,
                self.downsampling_factor,
            ).max(axis=1)
            labels = labels.reshape(
                labels.shape[0] // self.downsampling_factor, self.downsampling_factor
            ).max(axis=1)

        return timeseries, labels

    def _read_and_process_KDD21(self):
        df = pd.read_csv(self.full_file_path_and_name)
        df.interpolate(inplace=True, method="cubic")

        self.length_timeseries = len(df)
        self.n_channels = 1
        labels = df.iloc[:, -1].values
        timeseries = df.iloc[:, 0].values.reshape(-1, 1)

        data_splits = self._get_borders_KDD21()


        self.scaler.fit(timeseries[data_splits[0]])
        timeseries = self.scaler.transform(timeseries)
        timeseries = timeseries.squeeze()

        if self.data_split == "train":
            self.data, self.labels = self._downsample_timeseries(
                timeseries[data_splits[0]], labels[data_splits[0]]
            )
        elif self.data_split == "val":
            self.data, self.labels = self._downsample_timeseries(
                timeseries[data_splits[1]], labels[data_splits[1]]
            )
        elif self.data_split == "test":
            self.data, self.labels = self._downsample_timeseries(
                timeseries[data_splits[2]], labels[data_splits[2]]
            )

        self.length_timeseries = self.data.shape[0]

    def _read_and_process_general(self):
        df = pd.read_csv(self.full_file_path_and_name)
        df.interpolate(inplace=True, method="cubic")

        self.length_timeseries = df.shape[0]
        self.n_channels = 1
        data_splits = self._get_borders()

        labels = df.iloc[:, -1].values
        timeseries = df.iloc[:, 0].values.reshape(-1, 1)


        self.scaler.fit(timeseries[data_splits[0]])
        timeseries = self.scaler.transform(timeseries)
        timeseries = timeseries.squeeze()

        if self.data_split == "train":
            self.data, self.labels = self._downsample_timeseries(
                timeseries[data_splits[0]], labels[data_splits[0]]
            )
        elif self.data_split == "val":
            self.data, self.labels = self._downsample_timeseries(
                timeseries[data_splits[1]], labels[data_splits[1]]
            )
        elif self.data_split == "test":
            self.data, self.labels = self._downsample_timeseries(
                timeseries[data_splits[2]], labels[data_splits[2]]
            )

        self.length_timeseries = self.data.shape[0]



    def _read_data(self, file):
        self.scaler = StandardScaler()
        self.full_file_path_and_name = file
        if self.dataset_name in ["NASA-SMAP", "NASA-MSL"]:
            self._read_and_process_NASA()
        elif self.dataset_name in ["KDD21"]:
            self._read_and_process_KDD21()
        else:
            self._read_and_process_general()



    def __iter__(self):

        for dir in self.files:
            for filename in glob.glob(dir + '/**/*.*', recursive=True):

                # if not filename == '/zfsauton/project/public/Mononito/TimeseriesDatasets/anomaly_detection/TSB-UAD-Public/MITDB/234.test.csv@1.out':
                #     continue

                self.filename = filename
                self.series = filename.split("/")[-1].split(".")[0]
                self.dataset_name = filename.split("/")[-2]

                self._read_data(filename)

                if self.data.shape[-1] < self.seq_len:
                    continue

                # make sure that one batch don't contain different datasets (channel size mixup)
                # TODO: append nan to make it same size?
                num = ((self.length_timeseries - self.seq_len) // self.data_stride_len) + 1
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



DATASETS_WITHOUT_NORMALIZATION = [
    "AllGestureWiimoteX",
    "AllGestureWiimoteY",
    "AllGestureWiimoteZ",
    "BME",
    "Chinatown",
    "Crop",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "Fungi",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "GesturePebbleZ1",
    "GesturePebbleZ2",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "HouseTwenty",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "MelbournePedestrian",
    "PickupGestureWiimoteZ",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "PLAID",
    "PowerCons",
    "Rock",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ",
    "SmoothSubspace",
    "UMD",
]


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

        self.train_ratio = 0.6
        self.val_ratio = 0.1


    def _transform_labels(self, labels: np.ndarray):
        lbls = np.unique(self.train_labels)  # Move the labels to {0, ..., L-1}
        transform = {}
        for i, l in enumerate(lbls):
            transform[l] = i

        labels = np.vectorize(transform.get)(labels)

        return labels


    def _get_borders(self):
        train_end = 0
        val_start = 0
        val_end = 0
        if self.data_split in ["train", "val"]:
            train_ratio, val_ratio = (
                self.train_ratio / (self.train_ratio + self.val_ratio),
                self.val_ratio / (self.train_ratio + self.val_ratio),
            )
            train_end = int(train_ratio * self.n_timeseries)

        return slice(0, train_end), slice(train_end, None), slice(0, None)


    def _check_if_equal_length(self):
        if isinstance(self.data, list):
            n_timeseries = len(self.data)
            self.n_channels = self.data[0].shape[0]
            # Assume all time-series have the same number of channels
            # Then we have time-series of unequal lengths
            max_len = max([ts.shape[-1] for ts in self.data])
            for i, ts in enumerate(self.data):
                self.data[i] = interpolate_timeseries(
                    timeseries=ts, interp_length=max_len
                )
            self.data = np.asarray(self.data)
            logging.info(
                f"Time-series have unequal lengths. Reshaping to {self.data.shape}"
            )

    def _check_and_remove_nans(self):
        if np.isnan(self.data).any():
            logging.info("NaNs detected. Imputing values...")
            self.data = interpolate_timeseries(
                timeseries=self.data, interp_length=self.data.shape[-1]
            )
            self.data = np.nan_to_num(self.data)


    def _read_data(self, filename):
        self.scaler = StandardScaler()

        self.data, self.labels = load_from_tsfile(filename)
        self.labels = self._transform_labels(self.labels)

        self.num_classes = len(np.unique(self.labels))

        # temp = self.data

        self.original_length = None

        self._check_if_equal_length()
        self._check_and_remove_nans()



        if self.data.ndim == 3:
            self.n_timeseries, self.n_channels, self.length_timeseries = self.data.shape
        else:
            self.n_timeseries, self.length_timeseries = self.data.shape
            self.n_channels = 1

        if self.dataset_name not in DATASETS_WITHOUT_NORMALIZATION:
            length_timeseries = self.data.shape[0]
            self.data = self.data.reshape(-1, self.data.shape[-1])
            self.scaler.fit(self.data)
            self.data = self.scaler.transform(self.data)
            self.data = self.data.reshape(length_timeseries, -1, self.data.shape[-1])


        # TODO: shuffle? there's no shuffle in the research code
        # TODO: shuffle the data?
        idx = np.random.permutation(len(self.data))

        self.data = self.data[idx]
        self.labels = self.labels[idx]



        data_splits = self._get_borders()

        if self.data_split == "train":
            self.data = self.data[data_splits[0],]
            self.labels = self.labels[data_splits[0]]
        elif self.data_split == "val":
            self.data = self.data[data_splits[1],]
            self.labels = self.labels[data_splits[1]]
        elif self.data_split == "test":
            self.data = self.data[data_splits[2],]
            self.labels = self.labels[data_splits[2]]

        self.n_timeseries = self.data.shape[0]
        self.data = self.data.transpose(0, 2, 1)



    def __iter__(self):
        for filename in self.files:
            if self.data_split in filename.lower() or (self.data_split == "val" and "train" in filename.lower()):
                self.filename = filename

                self.dataset_name = filename.split("/")[-1].split("_")[0]

                # get labels
                _, self.train_labels = load_from_tsfile(filename.replace(self.data_split.upper(), "TRAIN"))

                # get data
                self._read_data(filename)

                num = self.n_timeseries // self.batch_size * self.batch_size
                for index in range(num):

                    timeseries = self.data[index]
                    timeseries_len = timeseries.shape[0]
                    labels = self.labels[index,].astype(int)

                    if timeseries_len <= self.seq_len:
                        timeseries, input_mask = upsample_timeseries(
                            timeseries,
                            self.seq_len,
                            direction="backward",
                            sampling_type="pad",
                            mode="constant",
                        )

                    elif timeseries_len > self.seq_len:
                        timeseries, input_mask = downsample_timeseries(
                            timeseries, self.seq_len, sampling_type="interpolate"
                        )


                    # input_mask = np.ones(self.seq_len)

                    # if timeseries_len <= self.seq_len:
                    #     timeseries = np.pad(timeseries, ((0, 0), (self.seq_len - timeseries_len, 0)))
                    #     input_mask[: self.seq_len - timeseries_len] = 0
                    # else:
                    #     # TODO: split or use last numbers?
                    #     timeseries = timeseries[:, -self.seq_len:]

                    # if self.original_length is not None and self.original_length[index] < self.seq_len:
                    #     input_mask[: self.seq_len - self.original_length[index]] = 0

                    yield timeseries.T, input_mask, labels





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


    def _get_borders(self, filename):
        self.train_ratio = 0.6
        self.val_ratio = 0.1
        self.test_ratio = 0.3

        # n_train = math.floor(self.length_timeseries_original * 0.6)
        # n_val = math.floor(self.length_timeseries_original * 0.1)
        # n_test = math.floor(self.length_timeseries_original * 0.3)

        if "ETTm" in filename:
            n_train = 12 * 30 * 24 * 4
            n_val = 4 * 30 * 24 * 4
            n_test = 4 * 30 * 24 * 4

        elif "ETTh" in filename:
            n_train = 12 * 30 * 24
            n_val = 4 * 30 * 24
            n_test = 4 * 30 * 24

        else:
            n_train = int(self.train_ratio * self.length_timeseries_original)
            n_test = int(self.test_ratio * self.length_timeseries_original)
            n_val = self.length_timeseries_original - n_train - n_test


        # num_train = int(self.length_timeseries_original * 0.7)
        # num_test = int(self.length_timeseries_original * 0.2)
        # num_vali = self.length_timeseries_original - num_train - num_test
        # border1s = [0, num_train - self.seq_len, self.length_timeseries_original - num_test - self.seq_len]
        # border2s = [num_train, num_train + num_vali, self.length_timeseries_original]

        # train = slice(border1s[0], border2s[0])
        # val = slice(border1s[1], border2s[1])
        # test = slice(border1s[2], border2s[2])

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

        data_splits = self._get_borders(filename)

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

        # TODO: why does research code do this? https://github.com/moment-timeseries-foundation-model/moment-research/blob/main/moment/data/forecasting_datasets.py
        # if self.data_split == "train":
        #     self.data = df.iloc[data_splits.train, :]
        # elif self.data_split == "val":
        #     self.data = df.iloc[data_splits.val, :]
        # elif self.data_split == "test":
        #     self.data = df.iloc[data_splits.test, :]



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
                # even the research code does it on whole dataset

                # TODO: this is wrong, this is fitting on the split, not the whole dataset
                self.scaler.fit(d)
                timeseries = self.scaler.transform(d).flatten()

                if len(timeseries) <= self.forecast_horizon:
                    continue

                input_mask = np.ones(self.seq_len)
                forecast = timeseries[-self.forecast_horizon :]
                timeseries = timeseries[: -self.forecast_horizon]

                timeseries_len = len(timeseries)

                if timeseries_len <= self.seq_len:
                    timeseries, input_mask = upsample_timeseries(
                        timeseries,
                        self.seq_len,
                        direction='backward',
                        sampling_type='pad',
                        mode='constant',
                    )

                elif timeseries_len > self.seq_len:
                    timeseries, input_mask = downsample_timeseries(
                        timeseries, self.seq_len, sampling_type='last'
                    )

                yield timeseries.reshape((1, self.seq_len)), forecast.reshape((1, self.forecast_horizon)), input_mask


import numpy.typing as npt

def upsample_timeseries(
    timeseries: npt.NDArray,
    seq_len: int,
    sampling_type: str = "pad",
    direction: str = "backward",
    **kwargs,
) -> npt.NDArray:
    timeseries_len = len(timeseries)
    input_mask = np.ones(seq_len)

    if timeseries_len >= seq_len:
        return timeseries, input_mask

    if sampling_type == "interpolate":
        timeseries = interpolate_timeseries(timeseries, seq_len)
    elif sampling_type == "pad" and direction == "forward":
        timeseries = np.pad(timeseries, (0, seq_len - timeseries_len), **kwargs)
        input_mask[: seq_len - timeseries_len] = 0
    elif sampling_type == "pad" and direction == "backward":
        timeseries = np.pad(timeseries, ((seq_len - timeseries_len, 0), (0,0)*(len(timeseries.shape)-1)), **kwargs)
        input_mask[: seq_len - timeseries_len] = 0
    else:
        error_msg = "Direction must be one of 'forward' or 'backward'"
        raise ValueError(error_msg)

    assert len(timeseries) == seq_len, "Padding failed"
    return timeseries, input_mask


def downsample_timeseries(
    timeseries: npt.NDArray, seq_len: int, sampling_type: str = "interpolate"
):
    input_mask = np.ones(seq_len)
    if sampling_type == "last":
        timeseries = timeseries[:seq_len]
    elif sampling_type == "first":
        timeseries = timeseries[seq_len:]
    elif sampling_type == "random":
        idx = np.random.randint(0, timeseries.shape[0] - seq_len)
        timeseries = timeseries[idx : idx + seq_len]
    elif sampling_type == "interpolate":
        timeseries = interpolate_timeseries(timeseries, seq_len)
    elif sampling_type == "subsample":
        factor = len(timeseries) // seq_len
        timeseries = timeseries[::factor]
        timeseries, input_mask = upsample_timeseries(
            timeseries, seq_len, sampling_type="pad", direction="forward"
        )
    else:
        error_msg = "Mode must be one of 'last', 'random',\
                'first', 'interpolate' or 'subsample'"
        raise ValueError(error_msg)
    return timeseries, input_mask


def interpolate_timeseries(
    timeseries: npt.NDArray, interp_length: int = 512
) -> npt.NDArray:
    x = np.linspace(0, 1, timeseries.shape[0])  # changed from timeseries.shape[-1]
    f = interp1d(x, timeseries, axis=0)
    x_new = np.linspace(0, 1, interp_length)
    timeseries = f(x_new)
    return timeseries






class MIMIC_mortality(torch.utils.data.Dataset):

    def __init__(self, data_split="train",
                 dir="/zfsauton2/home/mingzhul/time-series-prompt/mimic3_benchmarks/processed/in-hospital-mortality",
                 equal_length=False, small_part=True):
        """
        Parameters
        ----------
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        """

        self.seq_len = 128
        if equal_length:
            self.seq_len = 48

        self.data_split = data_split
        self.dir =  dir
        self.equal_length = equal_length
        self.small_part = small_part

        self._read_data()


    def _read_data(self):

        # Build readers, discretizers, normalizers
        folder = 'test' if self.data_split == 'test' else 'train'
        reader = InHospitalMortalityReader(dataset_dir=os.path.join(self.dir, folder),
                                            listfile=os.path.join(self.dir,
                                                                    f'{self.data_split}_listfile.csv'),
                                            period_length=48.0)

        self.discretizer = Discretizer(timestep=1.0,
                                        store_masks=True,
                                        impute_strategy='previous',
                                        start_time='zero',
                                        same_length=self.equal_length)

        self.discretizer_header = self.discretizer.transform(reader.read_example(0)["X"])[1].split(',')
        cont_channels = [i for (i, x) in enumerate(self.discretizer_header) if x.find("->") == -1]

        self.normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
        if self.equal_length:
            normalizer_state = '/zfsauton2/home/mingzhul/time-series-prompt/mimic3_benchmarks/notimestamp_ihm_ts-1.00_impute-previous_start-zero_masks-True_n-17903.normalizer'
        else:
            normalizer_state = '/zfsauton2/home/mingzhul/time-series-prompt/mimic3_benchmarks/timestamp_ihm_ts-1.00_impute-previous_start-zero_masks-True_n-17903.normalizer'
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
        self.normalizer.load_params(normalizer_state)

        # Read data
        # dir = os.path.join(self.dir, f'mimic_mortality_{self.data_split}_small_{self.small_part}_48hr_{self.equal_length}.pkl')
        # if os.path.exists(dir):
        #     with open(dir, 'rb') as f:
        #         self.raw = pickle.load(f)

        # else:
        self.raw = utils.load_data(reader, self.discretizer, self.normalizer, self.small_part, return_names=True)

            # np.max([len(x) for x in self.raw['data'][0]])
            # np.min([len(x) for x in self.raw['data'][0]])
            # np.mean([len(x) for x in self.raw['data'][0]])
            # np.quantile([len(x) for x in self.raw['data'][0]], 0.95) # 120.0

            # # save to file
            # with open(dir, 'wb') as f:
            #     pickle.dump(self.raw, f)










    def __getitem__(self, idx):

        data = self.raw['data'][0][idx]
        label = self.raw['data'][1][idx]

        # include mask as features
        # TODO: expand mask to match the number of features
        # timeseries = data

        data_indices = [i for (i, x) in enumerate(self.discretizer_header) if x.find("mask") == -1]
        columns = [x for (i, x) in enumerate(self.discretizer_header) if x.find("mask") == -1]
        timeseries = data[:, data_indices]
        mask = np.delete(data, data_indices, axis=1)

        column_mapping = [x.split('->')[0] for x in columns]
        ids = [self.discretizer._channel_to_id[x] for x in column_mapping]
        mask = mask[:, ids]

        timeseries_len = timeseries.shape[0]
        if timeseries_len <= self.seq_len:
            timeseries, input_mask = upsample_timeseries(
                timeseries,
                self.seq_len,
                direction='backward',
                sampling_type='pad',
                mode='constant',
            )
            # input_mask = np.repeat(input_mask.reshape(-1, 1), timeseries.shape[1], axis=1)
            # input_mask[-len(mask):] = mask

        elif timeseries_len > self.seq_len:
            timeseries, input_mask = downsample_timeseries(
                timeseries, self.seq_len, sampling_type='last'
            )

            # TODO: maybe mask the patch when the entire patch is missing
            # input_mask = mask[-len(input_mask):]


        # TODO: probably shouldn't use patches (patch size too big, and missing value mask probably doesn't work anymore). so repeat everything 8 times
        # timeseries = np.repeat(timeseries, 8, axis=0)
        # input_mask = np.repeat(input_mask, 8, axis=0)

        input_mask = np.repeat(input_mask.reshape(-1, 1), timeseries.shape[1], axis=1)

        return timeseries.T, input_mask.T, label


    def __len__(self):
        return len(self.raw['data'][0])


if __name__ == '__main__':
    data = MIMIC_mortality()


    for x in data:
        print(x)
        break
