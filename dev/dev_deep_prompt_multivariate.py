"""
prefix tuning to incorporate multiple variables

https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
https://github.com/allenai/better-promptability/blob/5cb1e33c9988f6f973e92a1c78da8291fe55df64/better_promptability/models/t5_with_prefix.py



- check how many parameters
    - 4310MiB vs 7120MiB for imputation

- classify and anomaly only 1 channel
    - do forecasting too


TODO:
- does it make sense to flatten the data to train deep prompt?
    - time dimension gets flattened too (#patches)
- should I exclude the MPT when training the deep prompt?


- need one deep prompt for each dataset, if #channels is different
    - what can you do if #channels is different?
    - just add them up?
    - train one layer for each dataset, project into the same space
    - lora-like: u_i * v
    - RNN, flip channel and time dimension
- solution: flip channel and time dimension, subsample time dimension to the same size (maybe 16)
    - do attention block




- did flattening and not flattening give similar performance? why?

- can we draw semantic meaning from the prompt?



- now only works with 1 batch size
- classify test data strangely long
- run experiments with original moment, single task, finetune
"""


import os

import numpy as np
from tqdm import tqdm

import torch
import torch.cuda.amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

import wandb

from momentfm.utils.utils import control_randomness
from momentfm.data.informer_dataset import InformerDataset
from momentfm.data.classification_dataset import ClassificationDataset
from momentfm.data.anomaly_detection_dataset import AnomalyDetectionDataset

from momentfm.common import TASKS
from momentfm import MOMENTPipeline
from momentfm.utils.masking import Masking
from momentfm.models.statistical_classifiers import fit_svm


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HOME"] = "/home/scratch/mingzhul/.cache/huggingface"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 0

# Set random seeds for PyTorch, Numpy etc.
control_randomness(seed=RANDOM_SEED)




def step(model, batch, criterion, mask_generator,):
    """
    one train / inference step
    """

    loss = 0
    losses = {}
    embedding = None

    for key, data in batch.items():

        if key in ('imputation', 'anomaly', 'classify'):

            model.task_name = TASKS.RECONSTRUCTION

            if data is None:
                continue
            if key not in losses:
                losses[key] = 0

            if key == 'anomaly':
                batch_x, batch_masks, _ = data
            elif key == 'imputation':
                batch_x, batch_masks = data
            elif key == 'classify':
                batch_x, batch_masks, _ = data
            else:
                raise NotImplementedError

            n_channels = batch_x.shape[1]


            # Reshape to [batch_size * n_channels, 1, window_size]
            # batch_x = batch_x.reshape((-1, 1, 512)).to(DEVICE).float()
            batch_x = batch_x.to(DEVICE).float()

            batch_masks = batch_masks.to(DEVICE).long()
            batch_masks = batch_masks[:, None, :].repeat(1, n_channels, 1)

            # Randomly mask some patches of data
            # 0 is masked
            mask = mask_generator.generate_mask(
                x=batch_x.reshape(-1, 1, 512),
                input_mask=batch_masks.reshape(-1, 512)).to(DEVICE).long()
            # mask = mask.reshape(-1, n_channels, 512)
            mask = mask.reshape(-1, n_channels, 512)[:, 0, :] # TODO: is mask 2D or 3D?
            batch_masks = batch_masks[:, 0, :]

            # Forward
            output = model(batch_x, input_mask=batch_masks, mask=mask, task_name=key)

            # Compute loss
            recon_loss = criterion(output.reconstruction, batch_x)
            if key in ('anomaly', 'classify'):
                val = recon_loss.mean()
                loss += val
                losses[key] += val.detach().cpu().numpy()

                if key == 'classify':
                    embedding = output.embeddings

            elif key == 'imputation':
                observed_mask = (batch_masks * (1 - mask))[:, None, :].repeat(1, n_channels, 1)
                assert observed_mask.shape == recon_loss.shape
                masked_loss = observed_mask * recon_loss

                val = (masked_loss.nansum() / (observed_mask.nansum() + 1e-7))
                loss += val
                losses[key] += val.detach().cpu().numpy()

            else:
                raise NotImplementedError


            y = batch_x


        elif key in ('forecasting_short', 'forecasting_long'):

            model.task_name = TASKS.FORECASTING

            timeseries, forecast, input_mask = data
            # Move the data to the GPU
            timeseries = timeseries.float().to(DEVICE)
            input_mask = input_mask.to(DEVICE)
            forecast = forecast.float().to(DEVICE)

            with torch.cuda.amp.autocast():
                output = model(timeseries, input_mask, task_name=key)

            val = criterion(output.forecast, forecast).mean()
            loss += val
            losses[key] = val.detach().cpu().numpy()


        else:
            raise NotImplementedError


    return loss, model, losses, embedding




def train(model, train_loader, test_loader,
          # Gradient clipping value
          max_norm = 5.0,
          max_epoch = 400, max_lr = 1e-2
          ):
    """
    prompt tuning
    """

    total_steps = len(train_loader) * max_epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Create a OneCycleLR scheduler
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3)

    # the tutourial is wrong
    # criterion = torch.nn.MSELoss().to(DEVICE)
    criterion = torch.nn.MSELoss(reduction='none').to(DEVICE)

    # Move the model to the GPU
    model = model.to(DEVICE)

    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    mask_generator = Masking(mask_ratio=0.3)

    for cur_epoch in range(max_epoch):
        model.train()

        losses = []
        loss_dicts = []

        for data in tqdm(train_loader, total=len(train_loader)):
            loss, model, loss_dict, _ = step(model, data, criterion, mask_generator,)

            # Scales the loss for mixed precision training
            scaler.scale(loss).backward()

            # Clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())
            loss_dicts.append(loss_dict)

        losses = np.array(losses)
        average_loss = np.average(losses)
        loss_dict = {key: np.nanmean(
            [d[key] if key in d.keys() else np.nan for d in loss_dicts]
            ) for key in loss_dicts[0].keys()}

        print(f"Epoch {cur_epoch}: Train loss: {average_loss:.3f}, Individual losses: {loss_dict}")

        wandb.log({
            'train_loss': average_loss} | {
                'train_'+key+'_loss': val for key, val in loss_dict.items()
                },
                  step=cur_epoch)


        # Step the learning rate scheduler
        scheduler.step()

        # Evaluate the model on the test split
        model = inference(model, test_loader, criterion, cur_epoch, mask_generator, train_loader,)

    return model




def inference(model, test_loader, criterion, cur_epoch, mask_generator, train_loader,
              log=True):
    """
    perform inference
    """

    trues, preds, losses, loss_dicts = [], [], [], []
    train_embeddings, train_labels = [], []

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader)):
            loss, model, loss_dict, embedding = step(model, data, criterion, mask_generator,)

            losses.append(loss.item())
            loss_dicts.append(loss_dict)

            # classification
            if embedding is not None:
                _, _, batch_labels = data['classify']
                train_labels.append(batch_labels)
                train_embeddings.append(embedding.detach().cpu().numpy())


    losses = np.array(losses)
    average_loss = np.average(losses)
    loss_dict = {key: np.nanmean([d[key] if key in d.keys() else np.nan for d in loss_dicts]) \
        for key in loss_dicts[0].keys()}

    print(f"Epoch {cur_epoch}: Test loss: {average_loss:.3f}, Individual losses: {loss_dict}")


    # classification
    train_accuracy = None
    test_accuracy = None
    if len(train_embeddings) > 0:

        test_embeddings, test_labels = [], []

        with torch.no_grad():
            for data in tqdm(train_loader, total=len(train_loader)):
                _, _, _, embedding = step(model, data, criterion, mask_generator)

                # classification
                if embedding is not None:
                    _, _, batch_labels = data['classify']
                    test_labels.append(batch_labels)
                    test_embeddings.append(embedding.detach().cpu().numpy())

        train_embeddings = np.concatenate(train_embeddings, axis=0)
        test_embeddings = np.concatenate(test_embeddings, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        clf = fit_svm(features=train_embeddings, y=train_labels)
        train_accuracy = clf.score(train_embeddings, train_labels)
        test_accuracy = clf.score(test_embeddings, test_labels)

        print(f"Train accuracy: {train_accuracy:.2f}, Test accuracy: {test_accuracy:.2f}")


    if log:
        wandb.log({'test_loss': average_loss,
                   'train_accuracy': train_accuracy,
                   'test_accuracy': test_accuracy,
                   } | {'test_'+key+'_loss': val for key, val in loss_dict.items()}, step=cur_epoch)


    model.train()

    return model






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




def zero_shot(train_loader, test_loader, name='', **kwargs):
    """
    zero shot
    TODO: cannot do this with forecasting, because head not pretrained
    """

    # imputation model
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        # For imputation, we will load MOMENT in `reconstruction` mode
        model_kwargs={
            'task_name': 'reconstruction',
            'freeze_encoder': False, # Freeze the patch embedding layer
            'freeze_embedder': False, # Freeze the transformer encoder
            'freeze_head': False, # The linear forecasting head must be trained
            'forecast_horizon': 196,
            # 'prefix_tuning': False,
            # 'prefix_tuning_multi': False,
            # 'MPT': True,
            'num_prefix': 2,
            'task_names': list(next(iter(train_loader)).keys()),
            }
    ).to(DEVICE)
    model.init()

    # need to freeze head manually
    for param in model.parameters():
        param.requires_grad = False

    # print frozen params
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    model = inference(model, test_loader,
                      torch.nn.MSELoss(reduction='none').to(DEVICE),
                      'zero-shot',
                      Masking(mask_ratio=0.3),
                      train_loader,
                      log=False)

    return model



def prompt_tuning(train_loader, test_loader, name='',
                  prefix_tuning_multi=False,
                  MPT=False,):
    """
    prompt tuning
    """

    wandb.init(
        project="ts-prompt",
        name="multivariate_deep_prompt_tuning" + name,
    )

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
            'forecast_horizon': 196,
            # 'prefix_tuning': True,
            'prefix_tuning_multi': prefix_tuning_multi,
            'MPT': MPT,
            'num_prefix': 16,
            'task_names': list(next(iter(train_loader)).keys()),
            'multivariate_projection': 'attention',
            }
    ).to(DEVICE)
    model.init()


    # need to freeze head manually
    for name, param in model.named_parameters():
        if 'prefix' not in name and 'prompt' not in name and 'fore_head' not in name and 'mpt' not in name:
            param.requires_grad = False

    # print frozen params
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    model = train(model, train_loader, test_loader)

    wandb.finish()

    return model





def finetune(train_loader, test_loader, name='', **kwargs):
    """
    finetune
    """
    wandb.init(
        project="ts-prompt",
        name="multivariate_deep_finetune",
    )

    # imputation model
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        # For imputation, we will load MOMENT in `reconstruction` mode
        model_kwargs={
            'task_name': 'reconstruction',
            'freeze_encoder': False, # Freeze the patch embedding layer
            'freeze_embedder': False, # Freeze the transformer encoder
            'freeze_head': False, # The linear forecasting head must be trained
            'forecast_horizon': 196,
            # 'prefix_tuning': False,
            # 'prefix_tuning_multi': False,
            # 'MPT': True,
            'num_prefix': 2,
            'task_names': list(next(iter(train_loader)).keys()),
            }
    ).to(DEVICE)
    model.init()


    for param in model.parameters():
        param.requires_grad = True

    # print frozen params
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    model = train(model, train_loader, test_loader)

    wandb.finish()

    return model



if __name__ == "__main__":
    # os.environ["WANDB_MODE"] = "offline"

    EXPERIMENT_NAME = 'prompt_tuning'

    multitask = True
    multivariable = True

    dataset_names = ['imputation', 'forecasting_long']

    if not multitask and not multivariable and EXPERIMENT_NAME == 'prompt_tuning':
        dataset_names = ['forecasting_long']

    name = ''
    if EXPERIMENT_NAME == 'zero_shot':
        experiment = zero_shot
    elif EXPERIMENT_NAME == 'finetune':
        experiment = finetune
    elif EXPERIMENT_NAME == 'prompt_tuning':
        experiment = prompt_tuning
        name = f'_multitask_{multitask}_multivariable_{multivariable}'
    else:
        raise NotImplementedError


    train_loader, test_loader = get_data(batch_size=1, dataset_names=dataset_names)

    model = experiment(train_loader, test_loader, name,
                       prefix_tuning_multi=multivariable,
                       MPT=multitask,
                       )



"""
experiments:
1. zero-shot
2. finetune
    - all tasks
    - (one for each task probably tasks a long time)
        - TODO: should compare to this, since this is the use case
3. prompt tuning
    - multitask only
    - multi variable only
    - multitask and multi variable

"""
