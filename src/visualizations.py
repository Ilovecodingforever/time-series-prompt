import datetime
import pickle
import os
from copy import deepcopy

import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

import torch
from sklearn.decomposition import PCA

from momentfm.common import TASKS
from momentfm import MOMENTPipeline
from momentfm.utils.masking import Masking
from momentfm.utils.utils import control_randomness

from data import get_data


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HF_HOME"] = "/home/scratch/mingzhul/.cache/huggingface"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 0
# Set random seeds for PyTorch, Numpy etc.
control_randomness(seed=RANDOM_SEED)






# https://github.com/thuml/iTransformer/issues/44
def multivariate_correlations(model, loader):
    pass







def get_representation(embeddings):
    """
    get representation
    """

    for task, embedding in embeddings.items():
        embeddings_manifold = PCA(n_components=2).fit_transform(embedding)

        plt.scatter(
            embeddings_manifold[:, 0],
            embeddings_manifold[:, 1],
            label=task,
            alpha=0.5,
            # c=test_labels.squeeze()
        )

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.savefig("embedding.png")



def visualize_mpt(model, loader):
    mask_generator = Masking(mask_ratio=0.3)
    model = model.to(DEVICE)
    model.task_name = TASKS.RECONSTRUCTION

    embeddings = {
        'imputation': [],
        'anomaly': [],
        'classify': [],
        'forecasting_short': [],
        'forecasting_long': []
    }

    for key in ['imputation', 'anomaly', 'classify', 'forecasting_short', 'forecasting_long']:
        for batch in loader:
            for _, data in batch.items():

                if data is None:
                    continue

                if key == 'forecasting_short':
                    model.fore_head = model.fore_head_8
                elif key == 'forecasting_long':
                    model.fore_head = model.fore_head_96

                if key in ('forecasting_short', 'forecasting_long'):

                    timeseries, forecast, input_mask = data
                    # Move the data to the GPU
                    timeseries = timeseries.float().to(DEVICE)
                    input_mask = input_mask.to(DEVICE)
                    forecast = forecast.float().to(DEVICE)

                    output = model(timeseries, input_mask, task_name=key)
                    embedding = output.embeddings.detach().cpu().numpy()

                else:
                    if key == 'anomaly':
                        batch_x, batch_masks, labels = data
                    elif key == 'imputation':
                        batch_x, batch_masks = data
                    elif key == 'classify':
                        batch_x, batch_masks, labels = data

                    n_channels = batch_x.shape[1]

                    batch_x = batch_x.to(DEVICE).float()

                    batch_masks = batch_masks.to(DEVICE).long()
                    batch_masks = batch_masks[:, None, :].repeat(1, n_channels, 1)

                    # Randomly mask some patches of data
                    # 0 is masked
                    mask = mask_generator.generate_mask(
                        x=batch_x.reshape(-1, 1, 512),
                        input_mask=batch_masks.reshape(-1, 512)).to(DEVICE).long()
                    # mask = mask.reshape(-1, n_channels, 512)
                    mask = mask.reshape(-1, n_channels, 512)[:, 0, :] # TODO: is mask 2D or 3D? different for each channel?
                    batch_masks = batch_masks[:, 0, :]

                    if key == 'classify':
                        mask = torch.ones_like(mask)

                    # Forward
                    output = model(batch_x, input_mask=batch_masks, mask=mask, task_name=key)

                    embedding = output.embeddings.detach().cpu().numpy()

                embeddings[key].append(embedding)


    get_representation(embeddings)



if __name__ == '__main__':
    
    
    prefix_tuning_multi = False
    MPT = False
    multivariate_projection = 'attention'

    loader, _ = get_data(batch_size=1, dataset_names=['imputation'], all=True,
                         files=("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv", ))
    
    
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        # output_loading_info=True,
        # For imputation, we will load MOMENT in `reconstruction` mode
        model_kwargs={
            'task_name': 'reconstruction',
            'freeze_encoder': False, # Freeze the patch embedding layer
            'freeze_embedder': False, # Freeze the transformer encoder
            'freeze_head': False, # The linear forecasting head must be trained
            'forecast_horizons': (96, 8),
            # 'prefix_tuning': True,
            'prefix_tuning_multi': prefix_tuning_multi,
            'MPT': MPT,
            'num_prefix': 16,
            'task_names': ['imputation', 'anomaly', 'classify', 'forecasting_short', 'forecasting_long'],
            'multivariate_projection': multivariate_projection,
            }
    ).to(DEVICE)
    model.init()
    
    visualize_mpt(model, loader)

