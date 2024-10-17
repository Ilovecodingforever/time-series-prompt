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
import numpy.typing as npt

import torch
from torch.utils.data import DataLoader

from momentfm.utils.data import load_from_tsfile, convert_tsf_to_dataframe

import sys
sys.path.append('/zfsauton2/home/mingzhul/time-series-prompt')
sys.path.append('/zfsauton2/home/mingzhul/time-series-prompt/mimic3_benchmarks')

from mimic3_benchmarks.mimic3benchmark.readers import InHospitalMortalityReader
from mimic3_benchmarks.mimic3benchmark.readers import PhenotypingReader
from mimic3_benchmarks.mimic3models.preprocessing import Discretizer, Normalizer
from mimic3_benchmarks.mimic3models.in_hospital_mortality import utils

from typing import Optional


# UCR: https://timeseriesclassification.com/dataset.php



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


class ClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, batch_size: int,
                 filename: str,
                 data_split: str = "train",
                 ):
        """
        Parameters
        ----------
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        """
        self.seq_len = 512
        self.batch_size = batch_size

        self.data_split = data_split  # 'train' or 'test'
        self.filename =  filename
        if data_split == "train" or data_split == "val":
            self.filename += "_TRAIN.ts"
        elif data_split == "test":
            self.filename += "_TEST.ts"

        self.train_ratio = 0.6
        self.val_ratio = 0.1

        self.dataset_name = self.filename.split("/")[-1].split("_")[0]
        # get labels
        _, self.train_labels = load_from_tsfile(self.filename.replace(self.data_split.upper(), "TRAIN"))
        # get data
        self._read_data(self.filename)


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
                    timeseries=ts, interp_length=max_len, channel_first=True
                )
            self.data = np.asarray(self.data)
            logging.info(
                f"Time-series have unequal lengths. Reshaping to {self.data.shape}"
            )

    def _check_and_remove_nans(self):
        if np.isnan(self.data).any():
            logging.info("NaNs detected. Imputing values...")
            self.data = interpolate_timeseries(
                timeseries=self.data, interp_length=self.data.shape[-1], channel_first=True
            )
            self.data = np.nan_to_num(self.data)


    def _read_data(self, filename: str):
        self.scaler = StandardScaler()

        self.data, self.labels = load_from_tsfile(filename)
        self.labels = self._transform_labels(self.labels)

        self.num_classes = len(np.unique(self.labels))

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

        if isinstance(self.data, np.ndarray):
            self.data = self.data[idx]
        else:
            self.data = [self.data[i] for i in idx]
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



    def __getitem__(self, index: int):

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

        return timeseries.T, input_mask, labels


    def __len__(self):
        return self.n_timeseries // self.batch_size * self.batch_size





class InformerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        batch_size: int,
        filename: str,
        forecast_horizon: int = 192,
        data_split: str = "train",
        data_stride_len: int = 1,
        task_name: str = "forecasting",
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
        """
        self.batch_size = batch_size

        self.filename = filename
        self.seq_len = 512
        self.forecast_horizon = forecast_horizon
        self.data_split = data_split
        self.data_stride_len = data_stride_len
        self.task_name = task_name

        # get data
        self._read_data(filename)


    def _get_borders(self, filename: str):
        self.train_ratio = 0.6
        self.val_ratio = 0.1
        self.test_ratio = 0.3

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


    def _read_data(self, filename: str):
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


    def __getitem__(self, index: int):

        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)

        pred_end = seq_end + self.forecast_horizon

        if pred_end > self.length_timeseries:
            pred_end = self.length_timeseries
            seq_end = seq_end - self.forecast_horizon
            seq_start = seq_end - self.seq_len

        assert(seq_start >=0 and seq_end >= 0 and pred_end >= 0)
        timeseries = self.data[seq_start:seq_end, :].T
        forecast = self.data[seq_end:pred_end, :].T

        return timeseries, forecast, input_mask


    def __len__(self):
        return (
            self.length_timeseries - self.seq_len - self.forecast_horizon
        ) // self.data_stride_len + 1




def upsample_timeseries(
    timeseries: npt.NDArray,
    seq_len: int,
    sampling_type: str = "pad",
    direction: str = "backward",
    **kwargs,
) -> (npt.NDArray, npt.NDArray):
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
    timeseries: npt.NDArray, interp_length: int = 512, channel_first=False,
) -> npt.NDArray:
    # TODO: change this to match research code
    if channel_first:
        x = np.linspace(0, 1, timeseries.shape[-1])
        f = interp1d(x, timeseries)
    else:
        x = np.linspace(0, 1, timeseries.shape[0])  # changed from timeseries.shape[-1]
        f = interp1d(x, timeseries, axis=0)
    x_new = np.linspace(0, 1, interp_length)
    timeseries = f(x_new)
    return timeseries






class MIMIC_mortality(torch.utils.data.Dataset):

    def __init__(self,
                 data_split: str = "train",
                 dir: str = "/zfsauton2/home/mingzhul/time-series-prompt/mimic3_benchmarks/processed/in-hospital-mortality",
                 equal_length: bool = False,
                 small_part: bool = True,
                 ordinal: bool = False, keep_categorical: bool = False,
                 seed: int = 0,
                 normalize: bool = True):
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
        self.dir = dir
        self.equal_length = equal_length
        self.small_part = small_part
        self.ordinal = ordinal
        self.seed = seed
        self.keep_categorical = keep_categorical
        self.normalize = normalize

        self.load_size = None
        if self.small_part:
            if data_split == 'test':
                split = 0.3
            elif data_split == 'train':
                split = 0.6
            else:
                split = 0.1
            # self.load_size = int((1000/0.6) * split)
            self.load_size = int(1000 * split)  # NOTE: if this is too small, the data will be too similar

        self._read_data()
        self.filename = dir


    def _read_data(self):

        # Build readers, discretizers, normalizers
        folder = 'test' if self.data_split == 'test' else 'train'
        reader = InHospitalMortalityReader(dataset_dir=os.path.join(self.dir, folder),
                                            listfile=os.path.join(self.dir,
                                                                    f'{self.data_split}_listfile.csv'),
                                            period_length=48.0)

        reader.random_shuffle(self.seed)

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


        if not self.normalize:
            self.normalizer = None

        # else:
        self.raw = utils.load_data(reader, self.discretizer, self.normalizer, return_names=True, load_size=self.load_size)

        self.data_indices = [i for (i, x) in enumerate(self.discretizer_header) if x.find("mask") == -1]
        self.columns = [x for (i, x) in enumerate(self.discretizer_header) if x.find("mask") == -1]
        self.n_channels = 60
        self.num_classes = 1

        # one hot to ordinal
        if self.ordinal:
            self.ordinal_data = []

            used = set()
            mylist =[x.split('->')[0] for x in self.columns]
            unique = [x for x in mylist if x not in used and (used.add(x) or True)]
            vars = [x.split('->')[0] for x in self.columns]

            # self.categories = {i: sum([True for v in vars if v == x]) for i, x in enumerate(unique)}
            # self.categories = {k: v for k, v in self.categories if v > 1}


            for i in range(len(self.raw['data'][0])):
                row = []
                for var in unique:
                    idx = [i for i, x in enumerate(vars) if x == var]
                    if len(idx) == 1:
                        row.append(self.raw['data'][0][i][:, idx[0]])
                    else:
                        row.append(np.where(self.raw['data'][0][i][:, idx] == 1)[1])
                row = np.array(row).T
                self.ordinal_data.append(row)

            # get normalizer
            all = np.concatenate(self.ordinal_data, axis=0)
            need_normalize = [x.split('->')[0] for i, x in enumerate(self.columns) if len(x.split('->'))>1 ]
            idxs = set([i for i, x in enumerate(unique) if x in need_normalize])

            if not self.keep_categorical:
                # normalize
                for i, d in enumerate(self.ordinal_data):
                    for j in idxs:
                        if np.std(all[:, j]) != 0:
                            self.ordinal_data[i][:, j] = (d[:, j] - np.mean(all[:, j])) / np.std(all[:, j])

            self.raw['data'] = list(self.raw['data'])
            self.raw['data'][0] = self.ordinal_data

        self.n_channels = 18


    def __getitem__(self, idx):

        data = self.raw['data'][0][idx]
        label = np.array(self.raw['data'][1][idx]).reshape(-1)

        # include mask as features
        # TODO: expand mask to match the number of features
        if self.ordinal:
            timeseries = data
        else:
            timeseries = data[:, self.data_indices]
            mask = np.delete(data, self.data_indices, axis=1)

            column_mapping = [x.split('->')[0] for x in self.columns]
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

        # input_mask = np.repeat(input_mask.reshape(-1, 1), timeseries.shape[1], axis=1)

        # return timeseries.T, input_mask.T, label
        return timeseries.T, input_mask, label


    def __len__(self):
        return len(self.raw['data'][0])




class MIMIC_mortality_STraTS(MIMIC_mortality):

    def __init__(self,
                 train_batch_size: int,
                 dir: str = "/zfsauton2/home/mingzhul/time-series-prompt/mimic3_benchmarks/processed/in-hospital-mortality",
                 equal_length: bool = False,
                 small_part: bool = True,
                 ordinal: bool = False, keep_categorical: bool = False,
                 seed: int = 0,
                 normalize: bool = True,):
        """
        Parameters
        ----------
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        """

        super().__init__('train', dir, equal_length, small_part, ordinal, seed, normalize)
        train_data = self.raw['data']

        super().__init__('val', dir, equal_length, small_part, ordinal, seed, normalize)
        val_data = self.raw['data']

        super().__init__('test', dir, equal_length, small_part, ordinal, seed, normalize)
        test_data = self.raw['data']

        self.raw = {'data': [train_data[0] + val_data[0] + test_data[0],
                             train_data[1] + val_data[1] + test_data[1]],}

        # NOTE:
        # did strats normalize categorical variables?
        #   - they just normalized the ordinal values
        # how did they pad
        #   - pad at the end

        self.splits = {'train': np.arange(0, len(train_data[0])),
                       'val': np.arange(len(train_data[0]), len(train_data[0]) + len(val_data[0])),
                       'test': np.arange(len(train_data[0]) + len(val_data[0]), len(train_data[0]) + len(val_data[0]) + len(test_data[0]))}
        self.splits['eval_train'] = self.splits['train'][:int(len(train_data[0])*0.4)]  # if too small, all in same class

        from baselines.STraTS.src.utils import CycleIndex
        self.train_cycler = CycleIndex(self.splits['train'], train_batch_size)



    def get_batch(self, idx=None):
        # for strats
        # TODO: remove first padding by normal values?
        # TODO: range of normalized timestamps dont match strats, how did they do it?
        #   - data['minute']/max_minute*2-1, so it is normalized to [-1, 1]

        max_len = 880  # TODO: use 512?

        values_list = []
        times_list = []
        varis_list = []
        obs_mask_list = []
        label_list = []

        if idx is None:
            idx = self.train_cycler.get_batch_ind()

        for i in idx:
            timeseries, input_mask, label = self[i]
            timeseries = timeseries.T

            # remove padding from timeseries
            variables = timeseries[input_mask == 1][:, 1:]
            variables_df = pd.DataFrame(variables)
            # remove duplicates for each variable
            for var in variables_df.columns:
                variables_df[var] = variables_df[var].drop_duplicates()

            # flatten the dataframe, ignore nan
            variables = variables_df.values.flatten()
            variables = variables[~np.isnan(variables)]

            times = timeseries[input_mask == 1][:, 0]
            # repeat by number of non-nan values of each row
            times = np.repeat(times, variables_df.count(axis=1).values)

            # var codes
            varis = variables_df.copy()
            varis[~np.isnan(varis)] = 1
            varis *= varis.columns
            varis = varis.values.flatten()
            varis = varis[~np.isnan(varis)].astype('int32')

            # mask
            obs_mask = np.ones(max_len)
            obs_mask[len(times):] = 0
            times = np.pad(times, (0, max_len - len(times)))
            varis = np.pad(varis, (0, max_len - len(varis)))
            values = np.pad(variables, (0, max_len - len(variables)))

            values_list.append(values)
            times_list.append(times)
            varis_list.append(varis)
            obs_mask_list.append(obs_mask)
            label_list.append(label)

        values = torch.tensor(np.stack(values_list)).to(torch.float32)
        times = torch.tensor(np.stack(times_list)).to(torch.float32)
        varis = torch.tensor(np.stack(varis_list)).to(torch.int32)
        obs_mask = torch.tensor(np.stack(obs_mask_list)).to(torch.float32)
        label = torch.tensor(np.stack(label_list)).to(torch.float32)

        # place holder for demo
        demo = torch.zeros_like(values).to(torch.float32)

        return {'values':values, 'times':times, 'varis':varis,
                'obs_mask':obs_mask, 'demo':demo,
                'labels':label}






class MIMIC_phenotyping(torch.utils.data.Dataset):

    def __init__(self,
                 data_split: str = "train",
                 dir: str = "/zfsauton2/home/mingzhul/time-series-prompt/mimic3_benchmarks/processed/phenotyping",
                 equal_length: bool = False,
                 small_part: bool = True,
                 ordinal: bool = False,
                 seed: int = 0,
                 normalize: bool = True,
                 ):
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
        self.dir = dir
        self.equal_length = equal_length
        self.small_part = small_part
        self.ordinal = ordinal
        self.seed = seed
        self.normalize = normalize

        self.load_size = None
        if self.small_part:
            if data_split == 'test':
                split = 0.3
            elif data_split == 'train':
                split = 0.6
            else:
                split = 0.1
            # self.load_size = int((1000/0.6) * split)
            self.load_size = int(1000 * split)

        self._read_data()
        self.filename = dir

        self.is_strats = False


    def _read_data(self):

        # Build readers, discretizers, normalizers
        folder = 'test' if self.data_split == 'test' else 'train'
        reader = PhenotypingReader(dataset_dir=os.path.join(self.dir, folder),
                                            listfile=os.path.join(self.dir,
                                                                    f'{self.data_split}_listfile.csv'),)
        reader.random_shuffle(self.seed)

        self.discretizer = Discretizer(timestep=1.0,
                                        store_masks=True,
                                        impute_strategy='previous',
                                        start_time='zero',
                                        same_length=self.equal_length)

        self.discretizer_header = self.discretizer.transform(reader.read_example(0)["X"])[1].split(',')
        cont_channels = [i for (i, x) in enumerate(self.discretizer_header) if x.find("->") == -1]

        self.normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
        if self.equal_length:
            normalizer_state = None
        else:
            normalizer_state = '/zfsauton2/home/mingzhul/time-series-prompt/mimic3_benchmarks/timestamp_pheno_ts-1.00_impute-previous_start-zero_masks-True_n-35621.normalizer'
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
        self.normalizer.load_params(normalizer_state)


        if not self.normalize:
            self.normalizer = None

        self.raw = utils.load_data(reader, self.discretizer, self.normalizer, return_names=True, load_size=self.load_size)

        self.data_indices = [i for (i, x) in enumerate(self.discretizer_header) if x.find("mask") == -1]
        self.columns = [x for (i, x) in enumerate(self.discretizer_header) if x.find("mask") == -1]
        self.n_channels = 60
        self.num_classes = 25


        # one hot to ordinal
        if self.ordinal:
            self.ordinal_data = []

            used = set()
            mylist =[x.split('->')[0] for x in self.columns]
            unique = [x for x in mylist if x not in used and (used.add(x) or True)]
            vars = [x.split('->')[0] for x in self.columns]

            for i in range(len(self.raw['data'][0])):
                row = []
                for var in unique:
                    idx = [i for i, x in enumerate(vars) if x == var]
                    if len(idx) == 1:
                        row.append(self.raw['data'][0][i][:, idx[0]])
                    else:
                        row.append(np.where(self.raw['data'][0][i][:, idx] == 1)[1])
                row = np.array(row).T
                self.ordinal_data.append(row)

            # get normalizer
            all = np.concatenate(self.ordinal_data, axis=0)
            need_normalize = [x.split('->')[0] for i, x in enumerate(self.columns) if len(x.split('->'))>1 ]
            idxs = set([i for i, x in enumerate(unique) if x in need_normalize])

            # normalize
            for i, d in enumerate(self.ordinal_data):
                for j in idxs:
                    if np.std(all[:, j]) != 0:
                        self.ordinal_data[i][:, j] = (d[:, j] - np.mean(all[:, j])) / np.std(all[:, j])

            self.raw['data'] = list(self.raw['data'])
            self.raw['data'][0] = self.ordinal_data

        self.n_channels = 18


    def __getitem__(self, idx):

        data = self.raw['data'][0][idx]
        label = np.array(self.raw['data'][1][idx])

        # include mask as features
        # TODO: expand mask to match the number of features

        if self.ordinal:
            timeseries = data
        else:
            timeseries = data[:, self.data_indices]
            mask = np.delete(data, self.data_indices, axis=1)

            column_mapping = [x.split('->')[0] for x in self.columns]
            ids = [self.discretizer._channel_to_id[x] for x in column_mapping]
            mask = mask[:, ids]


        if self.is_strats:
            return timeseries.T, None, label


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

        # input_mask = np.repeat(input_mask.reshape(-1, 1), timeseries.shape[1], axis=1)

        return timeseries.T, input_mask, label


    def __len__(self):
        return len(self.raw['data'][0])






class MIMIC_phenotyping_STraTS(MIMIC_phenotyping):

    def __init__(self,
                 train_batch_size: int,
                 dir: str = "/zfsauton2/home/mingzhul/time-series-prompt/mimic3_benchmarks/processed/phenotyping",
                 equal_length: bool = False,
                 small_part: bool = True,
                 ordinal: bool = False,
                 seed: int = 0,
                 normalize: bool = True,):
        """
        Parameters
        ----------
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        """

        self.is_strats = True

        super().__init__('train', dir, equal_length, small_part, ordinal, seed, normalize)
        train_data = self.raw['data']

        super().__init__('val', dir, equal_length, small_part, ordinal, seed, normalize)
        val_data = self.raw['data']

        super().__init__('test', dir, equal_length, small_part, ordinal, seed, normalize)
        test_data = self.raw['data']

        self.raw = {'data': [train_data[0] + val_data[0] + test_data[0],
                             train_data[1] + val_data[1] + test_data[1]],}

        # TODO:
        # did strats normalize categorical variables?
        # how did they pad

        self.splits = {'train': np.arange(0, len(train_data[0])),
                       'val': np.arange(len(train_data[0]), len(train_data[0]) + len(val_data[0])),
                       'test': np.arange(len(train_data[0]) + len(val_data[0]), len(train_data[0]) + len(val_data[0]) + len(test_data[0]))}
        self.splits['eval_train'] = self.splits['train'][:int(len(train_data[0])*0.4)]  # if too small, all in same class

        from baselines.STraTS.src.utils import CycleIndex
        self.train_cycler = CycleIndex(self.splits['train'], train_batch_size)



    def get_batch(self, idx=None):
        # for strats
        # TODO: remove first padding by normal values?
        # TODO: range of normalized timestamps dont match strats, how did they do it?
        max_len = 880  # TODO: use 512?

        values_list = []
        times_list = []
        varis_list = []
        obs_mask_list = []
        label_list = []

        if idx is None:
            idx = self.train_cycler.get_batch_ind()


        for i in idx:
            timeseries, input_mask, label = self[i]

            timeseries = timeseries.T

            # remove padding from timeseries
            variables = timeseries[input_mask == 1][:, 1:]
            variables_df = pd.DataFrame(variables)
            # remove duplicates for each variable
            for var in variables_df.columns:
                variables_df[var] = variables_df[var].drop_duplicates()

            # flatten the dataframe, ignore nan
            variables = variables_df.values.flatten()
            variables = variables[~np.isnan(variables)]

            times = timeseries[input_mask == 1][:, 0]
            # repeat by number of non-nan values of each row
            times = np.repeat(times, variables_df.count(axis=1).values)

            # var codes
            varis = variables_df.copy()
            varis[~np.isnan(varis)] = 1
            varis *= varis.columns
            varis = varis.values.flatten()
            varis = varis[~np.isnan(varis)].astype('int32')

            # mask
            obs_mask = np.ones(max_len)
            obs_mask[len(times):] = 0
            times = np.pad(times, (0, max_len - len(times)))
            varis = np.pad(varis, (0, max_len - len(varis)))
            values = np.pad(variables, (0, max_len - len(variables)))

            values_list.append(values)
            times_list.append(times)
            varis_list.append(varis)
            obs_mask_list.append(obs_mask)
            label_list.append(label)

        values = torch.tensor(np.stack(values_list)).to(torch.float32)
        times = torch.tensor(np.stack(times_list)).to(torch.float32)
        varis = torch.tensor(np.stack(varis_list)).to(torch.int32)
        obs_mask = torch.tensor(np.stack(obs_mask_list)).to(torch.float32)
        label = torch.tensor(np.stack(label_list)).to(torch.float32)

        # place holder for demo
        demo = torch.zeros_like(values).to(torch.float32)

        return {'values':values, 'times':times, 'varis':varis,
                'obs_mask':obs_mask, 'demo':demo,
                'labels':label}





class MIMIC_pretrain_pheno_STraTS(MIMIC_phenotyping):

    def __init__(self,
                 train_batch_size: int,
                 dir: str = "/zfsauton2/home/mingzhul/time-series-prompt/mimic3_benchmarks/processed/phenotyping",
                 equal_length: bool = False,
                 small_part: bool = True,
                 ordinal: bool = False,
                 seed: int = 0,
                 normalize: bool = True,):
        """
        Parameters
        ----------
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        """

        super().__init__('train', dir, equal_length, small_part, ordinal, seed, normalize)
        train_data = self.raw['data']

        super().__init__('val', dir, equal_length, small_part, ordinal, seed, normalize)
        val_data = self.raw['data']

        self.raw = {'data': [train_data[0] + val_data[0], train_data[1] + val_data[1]],}

        self.timestamps = [d[:, 0]*self.normalizer._stds[0] + self.normalizer._means[0] for d in self.raw['data'][0]]
        # TODO: change to 12
        self.timestamps_12hrs = [x[x>=12] for x in self.timestamps] # atleast 12 hrs

        # TODO:
        # did strats normalize categorical variables?
        # how did they pad

        self.splits = {'train': np.arange(0, len(train_data[0])),
                       'val': np.arange(len(train_data[0]), len(train_data[0]) + len(val_data[0])),}

        from baselines.STraTS.src.utils import CycleIndex
        self.train_cycler = CycleIndex(self.splits['train'], train_batch_size)

        self.is_strats = True


    def get_batch(self, idx=None):
        # for strats
        # TODO: remove first padding by normal values?
        # TODO: range of normalized timestamps dont match strats, how did they do it?
        max_len = 880  # TODO: use 512?

        values_list = []
        times_list = []
        varis_list = []
        obs_mask_list = []
        label_list = []

        if idx is None:
            idx = self.train_cycler.get_batch_ind()

        forecast_values_list = []
        forecast_mask_list = []

        for _, i in enumerate(idx):
            # timeseries, _, _ = self[i]
            timeseries = self.raw['data'][0][i]
        
            # if len(self.timestamps_12hrs[i]) == 0:
            #     continue


            # remove padding from timeseries
            # variables = timeseries[input_mask == 1][:, 1:]
            variables = timeseries[:, 1:]
            variables_df = pd.DataFrame(variables)
            # remove duplicates for each variable
            for var in variables_df.columns:
                variables_df[var] = variables_df[var].drop_duplicates()

            # flatten the dataframe, ignore nan
            variables = variables_df.values.flatten()
            variables = variables[~np.isnan(variables)]

            # times = timeseries[input_mask == 1][:, 0]
            times = timeseries[:, 0]
            # repeat by number of non-nan values of each row
            times = np.repeat(times, variables_df.count(axis=1).values)

            # var codes
            varis = variables_df.copy()
            varis[~np.isnan(varis)] = 1
            varis *= varis.columns
            varis = varis.values.flatten()
            varis = varis[~np.isnan(varis)].astype('int32')


        # self.timestamps = [d[:, 0]*self.normalizer._stds[0] + self.normalizer._means[0] for d in self.raw['data'][0]]

            if len(times[times >= (12-self.normalizer._means[0])/self.normalizer._stds[0]]) == 0:
                continue

            t1 = np.random.choice(times[times > (12-self.normalizer._means[0])/self.normalizer._stds[0]]) # start of prediction window
            t1_ = t1 * self.normalizer._stds[0] + self.normalizer._means[0]

            t2 = (t1_+2-self.normalizer._means[0])/self.normalizer._stds[0]
            t0 = max((0-self.normalizer._means[0])/self.normalizer._stds[0],(t1_-24-self.normalizer._means[0])/self.normalizer._stds[0])

            t2 = times[times < t2][-1]
            t0 = times[times > t0][0]

            t1_idx = np.where(times == t1)[0][-1]
            t0_idx = np.where(times == t0)[0][0]
            t2_idx = np.where(times == t2)[0][-1]

            # ts = times * self.normalizer._stds[0] + self.normalizer._means[0]
            # # t1 = np.random.choice(self.timestamps_12hrs[i]) # start of prediction window
            # if len(ts[ts > 12]) == 0:
            #     continue

            # t1 = np.random.choice(ts[ts > 12]) # start of prediction window
            # times = self.timestamps[i]

            # max window 24hrs
            # t2 = t1+2
            # t0 = max(0,t1-24)
            # # find the index of t0, t1, and t2
            # # t2 = times[times < t2][-1]
            # # t0 = times[times > t0][0]

            # t2 = ts[ts < t2][-1]
            # t0 = ts[ts > t0][0]

            # t1 = (t1 - self.normalizer._means[0]) / self.normalizer._stds[0]
            # t2 = (t2 - self.normalizer._means[0]) / self.normalizer._stds[0]
            # t0 = (t0 - self.normalizer._means[0]) / self.normalizer._stds[0]



            # try: # TODO
            #     t1_idx = np.where(times == t1)[0][-1]
            #     t0_idx = np.where(times == t0)[0][0]
            #     t2_idx = np.where(times == t2)[0][-1]
            # except:
            #     continue

            # TODO: if duplicate, take the last one
            forecast_vals = variables[t1_idx+1:t2_idx+1]
            forecast_vars = varis[t1_idx+1:t2_idx+1]

            forecast_values = torch.zeros(self.n_channels)
            forecast_values[forecast_vars] = torch.tensor(forecast_vals).to(torch.float32)
            forecast_mask = torch.zeros(self.n_channels).to(torch.float32)
            forecast_mask[forecast_vars] = 1

            values = variables[t0_idx:t1_idx+1]
            times = times[t0_idx:t1_idx+1]
            varis = varis[t0_idx:t1_idx+1]

            # mask
            obs_mask = np.ones(max_len)
            if len(times) < max_len:
                obs_mask[len(times):] = 0
                times = np.pad(times, (0, max_len - len(times)))
                varis = np.pad(varis, (0, max_len - len(varis)))
                values = np.pad(values, (0, max_len - len(values)))
            else:
                # TODO: tuncate the other direction?
                obs_mask = np.ones_like(times)
                times = times[:max_len]
                varis = varis[:max_len]
                values = values[:max_len]
                

            values_list.append(values)
            times_list.append(times)
            varis_list.append(varis)
            obs_mask_list.append(obs_mask)
            forecast_values_list.append(forecast_values)
            forecast_mask_list.append(forecast_mask)

        values = torch.tensor(np.stack(values_list)).to(torch.float32)
        times = torch.tensor(np.stack(times_list)).to(torch.float32)
        varis = torch.tensor(np.stack(varis_list)).to(torch.int32)
        obs_mask = torch.tensor(np.stack(obs_mask_list)).to(torch.float32)
        forecast_values_list = torch.stack(forecast_values_list)
        forecast_mask_list = torch.stack(forecast_mask_list)


        return {'values':values, 'times':times, 'varis':varis,
                'obs_mask':obs_mask,
                'demo':torch.zeros_like(values).to(torch.float32),
                'forecast_values':forecast_values,
                'forecast_mask':forecast_mask}





class MIMIC_pretrain_mortality_STraTS(MIMIC_mortality):

    def __init__(self,
                 train_batch_size: int,
                 dir: str = "/zfsauton2/home/mingzhul/time-series-prompt/mimic3_benchmarks/processed/in-hospital-mortality",
                 equal_length: bool = False,
                 small_part: bool = True,
                 ordinal: bool = False,
                 seed: int = 0,
                 normalize: bool = True,):
        """
        Parameters
        ----------
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        """

        super().__init__('train', dir, equal_length, small_part, ordinal, seed, normalize)
        train_data = self.raw['data']

        super().__init__('val', dir, equal_length, small_part, ordinal, seed, normalize)
        val_data = self.raw['data']

        self.raw = {'data': [train_data[0] + val_data[0], train_data[1] + val_data[1]],}

        self.timestamps = [d[:, 0]*self.normalizer._stds[0] + self.normalizer._means[0] for d in self.raw['data'][0]]
        # TODO: change to 12
        self.timestamps_12hrs = [x[x>=12] for x in self.timestamps] # atleast 12 hrs

        # TODO:
        # did strats normalize categorical variables?
        # how did they pad

        self.splits = {'train': np.arange(0, len(train_data[0])),
                       'val': np.arange(len(train_data[0]), len(train_data[0]) + len(val_data[0])),}

        from baselines.STraTS.src.utils import CycleIndex
        self.train_cycler = CycleIndex(self.splits['train'], train_batch_size)

        self.is_strats = True


    def get_batch(self, idx=None):
        # for strats
        # TODO: remove first padding by normal values?
        # TODO: range of normalized timestamps dont match strats, how did they do it?
        max_len = 880  # TODO: use 512?

        values_list = []
        times_list = []
        varis_list = []
        obs_mask_list = []
        label_list = []

        if idx is None:
            idx = self.train_cycler.get_batch_ind()

        forecast_values_list = []
        forecast_mask_list = []

        for _, i in enumerate(idx):
            # timeseries, _, _ = self[i]
            timeseries = self.raw['data'][0][i]
        
            # if len(self.timestamps_12hrs[i]) == 0:
            #     continue


            # remove padding from timeseries
            # variables = timeseries[input_mask == 1][:, 1:]
            variables = timeseries[:, 1:]
            variables_df = pd.DataFrame(variables)
            # remove duplicates for each variable
            for var in variables_df.columns:
                variables_df[var] = variables_df[var].drop_duplicates()

            # flatten the dataframe, ignore nan
            variables = variables_df.values.flatten()
            variables = variables[~np.isnan(variables)]

            # times = timeseries[input_mask == 1][:, 0]
            times = timeseries[:, 0]
            # repeat by number of non-nan values of each row
            times = np.repeat(times, variables_df.count(axis=1).values)

            # var codes
            varis = variables_df.copy()
            varis[~np.isnan(varis)] = 1
            varis *= varis.columns
            varis = varis.values.flatten()
            varis = varis[~np.isnan(varis)].astype('int32')


        # self.timestamps = [d[:, 0]*self.normalizer._stds[0] + self.normalizer._means[0] for d in self.raw['data'][0]]

            if len(times[times >= (12-self.normalizer._means[0])/self.normalizer._stds[0]]) == 0:
                continue

            t1 = np.random.choice(times[times > (12-self.normalizer._means[0])/self.normalizer._stds[0]]) # start of prediction window
            t1_ = t1 * self.normalizer._stds[0] + self.normalizer._means[0]

            t2 = (t1_+2-self.normalizer._means[0])/self.normalizer._stds[0]
            t0 = max((0-self.normalizer._means[0])/self.normalizer._stds[0],(t1_-24-self.normalizer._means[0])/self.normalizer._stds[0])

            t2 = times[times < t2][-1]
            t0 = times[times > t0][0]

            t1_idx = np.where(times == t1)[0][-1]
            t0_idx = np.where(times == t0)[0][0]
            t2_idx = np.where(times == t2)[0][-1]

            # ts = times * self.normalizer._stds[0] + self.normalizer._means[0]
            # # t1 = np.random.choice(self.timestamps_12hrs[i]) # start of prediction window
            # if len(ts[ts > 12]) == 0:
            #     continue

            # t1 = np.random.choice(ts[ts > 12]) # start of prediction window
            # times = self.timestamps[i]

            # max window 24hrs
            # t2 = t1+2
            # t0 = max(0,t1-24)
            # # find the index of t0, t1, and t2
            # # t2 = times[times < t2][-1]
            # # t0 = times[times > t0][0]

            # t2 = ts[ts < t2][-1]
            # t0 = ts[ts > t0][0]

            # t1 = (t1 - self.normalizer._means[0]) / self.normalizer._stds[0]
            # t2 = (t2 - self.normalizer._means[0]) / self.normalizer._stds[0]
            # t0 = (t0 - self.normalizer._means[0]) / self.normalizer._stds[0]



            # try: # TODO
            #     t1_idx = np.where(times == t1)[0][-1]
            #     t0_idx = np.where(times == t0)[0][0]
            #     t2_idx = np.where(times == t2)[0][-1]
            # except:
            #     continue

            # TODO: if duplicate, take the last one
            forecast_vals = variables[t1_idx+1:t2_idx+1]
            forecast_vars = varis[t1_idx+1:t2_idx+1]

            forecast_values = torch.zeros(self.n_channels)
            forecast_values[forecast_vars] = torch.tensor(forecast_vals).to(torch.float32)
            forecast_mask = torch.zeros(self.n_channels).to(torch.float32)
            forecast_mask[forecast_vars] = 1

            values = variables[t0_idx:t1_idx+1]
            times = times[t0_idx:t1_idx+1]
            varis = varis[t0_idx:t1_idx+1]

            # mask
            obs_mask = np.ones(max_len)
            if len(times) < max_len:
                obs_mask[len(times):] = 0
                times = np.pad(times, (0, max_len - len(times)))
                varis = np.pad(varis, (0, max_len - len(varis)))
                values = np.pad(values, (0, max_len - len(values)))
            else:
                # TODO: tuncate the other direction?
                obs_mask = np.ones_like(times)
                times = times[:max_len]
                varis = varis[:max_len]
                values = values[:max_len]
                

            values_list.append(values)
            times_list.append(times)
            varis_list.append(varis)
            obs_mask_list.append(obs_mask)
            forecast_values_list.append(forecast_values)
            forecast_mask_list.append(forecast_mask)

        values = torch.tensor(np.stack(values_list)).to(torch.float32)
        times = torch.tensor(np.stack(times_list)).to(torch.float32)
        varis = torch.tensor(np.stack(varis_list)).to(torch.int32)
        obs_mask = torch.tensor(np.stack(obs_mask_list)).to(torch.float32)
        forecast_values_list = torch.stack(forecast_values_list)
        forecast_mask_list = torch.stack(forecast_mask_list)


        return {'values':values, 'times':times, 'varis':varis,
                'obs_mask':obs_mask,
                'demo':torch.zeros_like(values).to(torch.float32),
                'forecast_values':forecast_values,
                'forecast_mask':forecast_mask}




if __name__ == '__main__':
    data = MIMIC_mortality()

    for x in data:
        print(x)
        break



