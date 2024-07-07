"""
get data loaders
"""


import torch
from torch.utils.data import DataLoader

from momentfm.data.informer_dataset import InformerDataset
from momentfm.data.classification_dataset import ClassificationDataset
from momentfm.data.anomaly_detection_dataset import AnomalyDetectionDataset

from main import RANDOM_SEED




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




def get_data(batch_size, dataset_names):
    """
    get data
    """

    train_datasets = {}
    test_datasets = {}
    if 'imputation' in dataset_names:
        train_dataset_impute = InformerDataset(data_split='train', random_seed=RANDOM_SEED,
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
        test_datasets['imputation'] = test_dataset_impute

    if 'anomaly' in dataset_names:
        train_dataset_anomaly = AnomalyDetectionDataset(data_split='train', random_seed=RANDOM_SEED,
                                                        data_stride_len=100
                                                        # data_stride_len=512
                                                        )
        test_dataset_anomaly = AnomalyDetectionDataset(data_split='test', random_seed=RANDOM_SEED,
                                                        data_stride_len=100
                                                        # data_stride_len=512
                                                        )
        train_datasets['anomaly'] = train_dataset_anomaly
        test_datasets['anomaly'] = test_dataset_anomaly

    if 'classify' in dataset_names:
        train_dataset_classify = ClassificationDataset(data_split='train')
        test_dataset_classify = ClassificationDataset(data_split='test')
        train_datasets['classify'] = train_dataset_classify
        test_datasets['classify'] = test_dataset_classify

    if 'forecasting_long' in dataset_names:
        train_dataset_forecast_long = InformerDataset(data_split="train", random_seed=RANDOM_SEED,
                                                        forecast_horizon=196,
                                                        data_stride_len=15
                                                        )
        test_dataset_forecast_long = InformerDataset(data_split="test", random_seed=RANDOM_SEED,
                                                        forecast_horizon=196,
                                                        data_stride_len=15
                                                        )
        train_datasets['forecasting_long'] = train_dataset_forecast_long
        test_datasets['forecasting_long'] = test_dataset_forecast_long

    data = CollectedDataset(train_datasets)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn
                              )

    data = CollectedDataset(test_datasets)
    test_loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn
                             )

    return train_loader, test_loader
