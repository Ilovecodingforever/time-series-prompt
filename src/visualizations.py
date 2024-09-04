import datetime
import pickle
import os
from copy import deepcopy
from tqdm import tqdm

import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from momentfm.common import TASKS
from momentfm import MOMENTPipeline
from momentfm.utils.masking import Masking
from momentfm.utils.utils import control_randomness
from momentfm.models.t5_multivariate_prefix import T5StackWithPrefixMulti

from data import get_data
from train import train


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HOME"] = "/home/scratch/mingzhul/.cache/huggingface"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 0
# Set random seeds for PyTorch, Numpy etc.
control_randomness(seed=RANDOM_SEED)






# https://github.com/thuml/iTransformer/issues/44
def multivariate_correlations(model, loader):
    assert isinstance(model.encoder, T5StackWithPrefixMulti)

    mask_generator = Masking(mask_ratio=0.3)
    model = model.to(DEVICE)
    model.task_name = TASKS.RECONSTRUCTION

    model.eval()
    with torch.no_grad():
        loader.dataset.reset()
        for i, batch in enumerate(tqdm(loader)):
            if i == 1:
                break
            for key, data in batch.items():

                if data is None:
                    continue

                if key == 'anomaly':
                    batch_x, batch_masks, labels = data
                elif key == 'imputation':
                    batch_x, batch_masks = data
                elif key == 'classify':
                    batch_x, batch_masks, labels = data
                else:
                    raise ValueError(f"Unknown task: {key}")

                n_channels = batch_x.shape[1]

                batch_x = torch.tensor(batch_x).to(DEVICE).float()

                batch_masks = torch.tensor(batch_masks).to(DEVICE).long()
                batch_masks = batch_masks[:, None, :].repeat(1, n_channels, 1)

                # Randomly mask some patches of data
                # 0 is masked
                mask = mask_generator.generate_mask(
                    x=batch_x.reshape(-1, 1, 512),
                    input_mask=batch_masks.reshape(-1, 512)).to(DEVICE).long()
                # mask = mask.reshape(-1, n_channels, 512)
                mask = mask.reshape(-1, n_channels, 512)[:, 0, :] # TODO: is mask 2D or 3D? different for each channel?
                batch_masks = batch_masks[:, 0, :]

                # Forward
                output = model(batch_x, input_mask=batch_masks, mask=mask, task_name='imputation')


                # TODO: correlation
                batch_x = batch_x[0]
                means = batch_x.mean(dim=1, keepdim=True)
                norm = torch.linalg.norm(batch_x - means, dim=1)[:, None]

                corr = (batch_x - means) @ (batch_x - means).T / (norm @ norm.T)
                plt.imshow(corr.cpu().numpy())
                plt.savefig('plots/correlation.png')

                print()







def get_representation(embeddings):
    """
    get representation
    """
    fig, axs = plt.subplots(1, 5, sharey=True, sharex=True, figsize=(8, 8))

    for i, (task, embedding) in enumerate(embeddings.items()):
        embeddings_manifold = PCA(n_components=2).fit_transform(embedding)
        # embeddings_manifold = TSNE(n_components=2).fit_transform(embedding)

        axs[i].scatter(
            embeddings_manifold[:, 0],
            embeddings_manifold[:, 1],
            label=task,
            # alpha=0.2,
            # c=test_labels.squeeze()
        )
        axs[i].set_title(task)

    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.legend()
    plt.tight_layout()
    plt.savefig("embedding_mpt.png")
    print()



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

    model.eval()
    with torch.no_grad():
        for key in [ 'forecasting_short', 'imputation', 'anomaly', 'classify','forecasting_long']:
            loader.dataset.reset()
            for i, batch in enumerate(tqdm(loader)):
                # if i == 10:
                #     break
                for k, data in batch.items():

                    if data is None:
                        continue

                    if key == 'forecasting_short':
                        model.fore_head = model.fore_head_8
                    elif key == 'forecasting_long':
                        model.fore_head = model.fore_head_96

                    if k == 'anomaly':
                        batch_x, batch_masks, labels = data
                    elif k == 'imputation':
                        batch_x, batch_masks = data
                    elif k == 'classify':
                        batch_x, batch_masks, labels = data
                    else:
                        batch_x, forecast, batch_masks = data

                    batch_x = batch_x.float().to(DEVICE)
                    batch_masks = batch_masks.to(DEVICE)

                    if key in ('forecasting_short', 'forecasting_long'):
                        # Move the data to the GPU
                        output = model(batch_x, batch_masks, task_name=key)
                        embedding = output.embeddings.detach().cpu().numpy()

                    else:
                        n_channels = batch_x.shape[1]

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

            embeddings[key] = np.concatenate(embeddings[key], axis=0)

    get_representation(embeddings)



def train_model(save_path="/home/scratch/mingzhul/classify_heartbeat"):
    experiment_files = (
        "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Heartbeat/Heartbeat_TEST.ts",
        "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Heartbeat/Heartbeat_TRAIN.ts")
    train_loader, test_loader = get_data(batch_size=1, dataset_names=['classify'], all=True,
                                        files=experiment_files)

    # imputation model
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
            'prefix_tuning_multi': True,
            'MPT': False,
            'num_prefix': 16,
            'task_names': list(next(iter(train_loader)).keys()),
            'multivariate_projection': 'attention',
            }
    )
    model.init()

    # need to freeze head manually
    for n, param in model.named_parameters():
        if 'prefix' not in n and 'prompt' not in n and 'fore_head' not in n and 'mpt' not in n:
            param.requires_grad = False

    # print frozen params
    for n, param in model.named_parameters():
        if param.requires_grad:
            print(n)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = train(model, train_loader, test_loader, max_epoch=20, identifier='testing', log=False)

    model.save_pretrained(save_path, from_pt=True)






if __name__ == '__main__':
    # train_model()

    prefix_tuning_multi = True
    MPT = False

    assert prefix_tuning_multi ^ MPT, "Only one of prefix_tuning_multi and MPT can be True"

    if prefix_tuning_multi:
        # path = "ml233/time-series-prompt_prompttune_attention_imputation_ETTh2_19"
        # path = "/home/scratch/mingzhul/classify_heartbeat"
        path = "ml233/prompttune_attention_classify_Heartbeat"

        loader, _, _ = get_data(batch_size=1, dataset_names=['classify'], all=True,
                            #  files=("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTh2.csv", )
                                files=("/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Heartbeat/Heartbeat_TRAIN.ts", )
                                )
    else:
        path = "ml233/MPT_prompttune"

        experiment_files = {}
        experiment_files['forecasting_short'] = (
                                            '/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/monash/m4_yearly_dataset.tsf',
                                            )

        experiment_files['imputation'] = ("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv",
                                        "/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTh2.csv",
                                        "/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTm1.csv",
                                        "/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTm2.csv",
                                        "/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/exchange_rate.csv", )

        experiment_files['forecasting_long'] = ("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv",
                                                "/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTh2.csv",
                                                "/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTm1.csv",
                                                "/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTm2.csv",
                                                "/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/exchange_rate.csv", )

        experiment_files['classify'] = (
            "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/AtrialFibrillation/AtrialFibrillation_TEST.ts",
            "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/AtrialFibrillation/AtrialFibrillation_TRAIN.ts",
            "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Epilepsy/Epilepsy_TEST.ts",
            "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Epilepsy/Epilepsy_TRAIN.ts",
            "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/ERing/ERing_TEST.ts",
            "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/ERing/ERing_TRAIN.ts",
            "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Cricket/Cricket_TEST.ts",
            "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Cricket/Cricket_TRAIN.ts",
            "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP1/SelfRegulationSCP1_TEST.ts",
            "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP1/SelfRegulationSCP1_TRAIN.ts",
            "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP2/SelfRegulationSCP2_TEST.ts",
            "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP2/SelfRegulationSCP2_TRAIN.ts",
            "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Heartbeat/Heartbeat_TEST.ts",
            "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Heartbeat/Heartbeat_TRAIN.ts")

        experiment_files['anomaly'] = (
                                    '/zfsauton/project/public/Mononito/TimeseriesDatasets/anomaly_detection/TSB-UAD-Public/MITDB',
                                    )

        loader, _ = get_data(batch_size=1, dataset_names=experiment_files.keys(), all=True,
                             files=experiment_files
                                )

    # path = "AutonLab/MOMENT-1-large"
    # loader, _ = get_data(batch_size=1, dataset_names=['imputation'], all=True,
    #                         files=("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv",)
    #                         )

    multivariate_projection = 'attention'

    model = MOMENTPipeline.from_pretrained(
        path,
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
            'visualize_attention': prefix_tuning_multi,
            # 'visualize_mpt': True,
            }
    ).to(DEVICE)
    model.init()

    if MPT:
        visualize_mpt(model, loader)
    else:
        multivariate_correlations(model, loader)


