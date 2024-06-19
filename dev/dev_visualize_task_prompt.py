"""
visualize embedding for different tasks

TODO:
check if finetuning different recon tasks gives different embeddings. Try using the same dataset. Maybe synthetic data
    - need to check: are you really finetuning all layers?
anomaly and imputation use exactly the same code: so there's no difference between embeddings

try T-SNE. a lot of people use it 

"""

import os

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


import torch
import torch.cuda.amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from momentfm.utils.utils import control_randomness
from momentfm.data.informer_dataset import InformerDataset
from momentfm.data.classification_dataset import ClassificationDataset
from momentfm.data.anomaly_detection_dataset import AnomalyDetectionDataset

from momentfm.common import TASKS
from momentfm import MOMENTPipeline
from momentfm.utils.masking import Masking
# from momentfm.utils.forecasting_metrics import get_forecasting_metrics


os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
os.environ["HF_HOME"] = "/home/scratch/mingzhul/.cache/huggingface"




def fine_tune(model, train_loader, test_loader):
    """
    fine tune model
    """

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cur_epoch = 0
    max_epoch = 1

    # Move the model to the GPU
    model = model.to(device)

    # Move the loss function to the GPU
    criterion = criterion.to(device)

    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Create a OneCycleLR scheduler
    max_lr = 1e-4
    total_steps = len(train_loader) * max_epoch
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3)

    # Gradient clipping value
    max_norm = 5.0

    mask_generator = Masking(mask_ratio=0.3)

    while cur_epoch < max_epoch:
        model.train()
        
        losses = []
        for data in tqdm(train_loader, total=len(train_loader)):

            if model.task_name == TASKS.RECONSTRUCTION:
                if isinstance(train_loader.dataset, AnomalyDetectionDataset):
                    batch_x, batch_masks, batch_labels = data
                else:
                    batch_x, batch_masks = data

                n_channels = batch_x.shape[1]

                # Reshape to [batch_size * n_channels, 1, window_size]
                batch_x = batch_x.reshape((-1, 1, 512)).to(device).float()

                batch_masks = batch_masks.to(device).long()
                batch_masks = batch_masks.repeat_interleave(n_channels, axis=0)

                # Randomly mask some patches of data
                mask = mask_generator.generate_mask(
                    x=batch_x, input_mask=batch_masks).to(device).long()

                # Forward
                output = model(batch_x, input_mask=batch_masks, mask=mask)

                # Compute loss
                recon_loss = criterion(output.reconstruction, batch_x)
                if isinstance(train_loader.dataset, AnomalyDetectionDataset):
                    loss = recon_loss
                else:
                    observed_mask = batch_masks * (1 - mask)
                    masked_loss = observed_mask * recon_loss
                    loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)


            elif model.task_name == TASKS.FORECASTING:
                timeseries, forecast, input_mask = data
                # Move the data to the GPU
                timeseries = timeseries.float().to(device)
                input_mask = input_mask.to(device)
                forecast = forecast.float().to(device)

                with torch.cuda.amp.autocast():
                    output = model(timeseries, input_mask)

                loss = criterion(output.forecast, forecast)


            elif model.task_name == TASKS.CLASSIFICATION:
                criterion = torch.nn.CrossEntropyLoss()

                batch_x, batch_masks, batch_labels = data

                batch_x = batch_x.to(device).float()
                batch_masks = batch_masks.to(device)
                batch_labels = batch_labels.to(device)

                output = model(batch_x, input_mask=batch_masks) # [batch_size x d_model (=1024)]
                # backward
                loss = criterion(output.logits, batch_labels)


            else:
                raise ValueError(f"Task {model.task_name} not supported")


            # Scales the loss for mixed precision training
            scaler.scale(loss).backward()

            # Clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())

        losses = np.array(losses)
        average_loss = np.average(losses)
        print(f"Epoch {cur_epoch}: Train loss: {average_loss:.3f}")

        # Step the learning rate scheduler
        scheduler.step()
        cur_epoch += 1

        # Evaluate the model on the test split
        trues, preds, histories, losses = [], [], [], []
        model.eval()
        with torch.no_grad():
            for data in tqdm(test_loader, total=len(test_loader)):

                if model.task_name == TASKS.RECONSTRUCTION:
                    if isinstance(test_loader.dataset, AnomalyDetectionDataset):
                        batch_x, batch_masks, batch_labels = data
                    else:
                        batch_x, batch_masks = data

                    n_channels = batch_x.shape[1]

                    # Reshape to [batch_size * n_channels, 1, window_size]
                    batch_x = batch_x.reshape((-1, 1, 512)).to(device).float()

                    batch_masks = batch_masks.to(device).long()
                    batch_masks = batch_masks.repeat_interleave(n_channels, axis=0)

                    # Randomly mask some patches of data
                    mask = mask_generator.generate_mask(
                        x=batch_x, input_mask=batch_masks).to(device).long()

                    # Forward
                    output = model(batch_x, input_mask=batch_masks, mask=mask).reconstruction

                    # Compute loss
                    recon_loss = criterion(output, batch_x)

                    if isinstance(test_loader.dataset, AnomalyDetectionDataset):
                        loss = recon_loss
                    else:
                        observed_mask = batch_masks * (1 - mask)
                        masked_loss = observed_mask * recon_loss
                        loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)

                    y = batch_x


                elif model.task_name == TASKS.FORECASTING:
                    timeseries, y, input_mask = data

                    # Move the data to the GPU
                    timeseries = timeseries.float().to(device)
                    input_mask = input_mask.to(device)
                    y = y.float().to(device)

                    with torch.cuda.amp.autocast():
                        output = model(timeseries, input_mask).forecast

                    loss = criterion(output, y)


                elif model.task_name == TASKS.CLASSIFICATION:
                    criterion = torch.nn.CrossEntropyLoss()

                    batch_x, batch_masks, y = data
                    batch_x = batch_x.to(device).float()
                    batch_masks = batch_masks.to(device)
                    y = y.to(device)

                    output = model(batch_x, input_mask=batch_masks).logits # [batch_size x d_model (=1024)]

                    # backward
                    loss = criterion(output, y)


                else:
                    raise ValueError(f"Task {model.task_name} not supported")


                losses.append(loss.item())

                trues.append(y.detach().cpu().numpy())
                preds.append(output.detach().cpu().numpy())
                # histories.append(timeseries.detach().cpu().numpy())

        losses = np.array(losses)
        average_loss = np.average(losses)
        model.train()

        # trues = np.concatenate(trues, axis=0)
        # preds = np.concatenate(preds, axis=0)
        # # histories = np.concatenate(histories, axis=0)

        # if model.task_name == TASKS.RECONSTRUCTION:
        #     if isinstance(test_loader.dataset, AnomalyDetectionDataset):
        #         n_unique_timesteps = 512 - trues.shape[0] + test_dataset.length_timeseries
        #         trues = np.concatenate([trues[:512*(test_dataset.length_timeseries//512)], trues[-n_unique_timesteps:]])
        #         preds = np.concatenate([preds[:512*(test_dataset.length_timeseries//512)], preds[-n_unique_timesteps:]])
        #         labels = np.concatenate([labels[:512*(test_dataset.length_timeseries//512)], labels[-n_unique_timesteps:]])
        #         assert trues.shape[0] == test_dataset.length_timeseries

        #         # We will use the Mean Squared Error (MSE) between the observed values and MOMENT's predictions as the anomaly score
        #         anomaly_scores = (trues - preds)**2
        #         from momentfm.utils.anomaly_detection_metrics import adjbestf1

        #         print(f"Zero-shot Adjusted Best F1 Score: {adjbestf1(y_true=labels, y_scores=anomaly_scores)}")

        #     else:


        # elif model.task_name == TASKS.FORECASTING:
        #     metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction='mean')
        #     print(f"Epoch {cur_epoch}: Test MSE: {metrics.mse:.3f} | Test MAE: {metrics.mae:.3f}")

        # elif model.task_name == TASKS.CLASSIFICATION:

    return model



def get_embedding(model, dataloader):
    """
    get embedding
    """
    original_task_name = model.task_name

    model.task_name = TASKS.EMBED

    device = next(model.parameters()).device

    mask_generator = Masking(mask_ratio=0.3)

    embeddings = []
    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):

            if original_task_name == TASKS.RECONSTRUCTION:

                if isinstance(dataloader.dataset, AnomalyDetectionDataset):
                    batch_x, batch_masks, batch_labels = data
                else:
                    batch_x, batch_masks = data

                n_channels = batch_x.shape[1]

                # Reshape to [batch_size * n_channels, 1, window_size]
                batch_x = batch_x.reshape((-1, 1, 512)).to(device).float()

                batch_masks = batch_masks.to(device).long()
                batch_masks = batch_masks.repeat_interleave(n_channels, axis=0)

                # Randomly mask some patches of data
                mask = mask_generator.generate_mask(
                    x=batch_x, input_mask=batch_masks).to(device).long()

                # Forward
                output = model(batch_x, input_mask=batch_masks, mask=mask)


            elif original_task_name == TASKS.FORECASTING:

                batch_x, _, batch_masks = data
                batch_x = batch_x.to(device).float()
                batch_masks = batch_masks.to(device)

                output = model(batch_x, batch_masks) # [batch_size x d_model (=1024)]


            elif original_task_name == TASKS.CLASSIFICATION:

                batch_x, batch_masks, _ = data
                batch_x = batch_x.to(device).float()
                batch_masks = batch_masks.to(device)

                output = model(batch_x, input_mask=batch_masks) # [batch_size x d_model (=1024)]


            else:
                raise ValueError(f"Task {original_task_name} not supported")

            embedding = output.embeddings
            embeddings.append(embedding.detach().cpu().numpy())

    embeddings = np.concatenate(embeddings)
    return embeddings



def get_representation(models):
    """
    get representation
    """

    for name, (model, loader) in models.items():
        model.to('cuda')
        embeddings = get_embedding(model, loader)
        embeddings_manifold = PCA(n_components=2).fit_transform(embeddings)

        plt.scatter(
            embeddings_manifold[:, 0],
            embeddings_manifold[:, 1],
            label=name,
            alpha=0.5,
            # c=test_labels.squeeze()
        )

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.savefig("embedding.png")







if __name__ == '__main__':

    BATCH_SIZE = 2

    # Set random seeds for PyTorch, Numpy etc.
    RANDOM_SEED = 0
    control_randomness(seed=RANDOM_SEED)

    # forecasting model long
    HORIZON = 300
    forecasting_model_long = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            'task_name': 'forecasting',
            'forecast_horizon': HORIZON,
            'head_dropout': 0.1,
            'weight_decay': 0,
            'freeze_encoder': False, # Freeze the patch embedding layer
            'freeze_embedder': False, # Freeze the transformer encoder
            'freeze_head': False, # The linear forecasting head must be trained
        },
    )
    forecasting_model_long.init()

    train_dataset = InformerDataset(data_split="train", random_seed=RANDOM_SEED,
                                    forecast_horizon=HORIZON)
    train_loader_long = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = InformerDataset(data_split="test", random_seed=RANDOM_SEED,
                                   forecast_horizon=HORIZON)
    test_loader_long = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    forecasting_model_long = fine_tune(forecasting_model_long, train_loader_long, test_loader_long).to('cpu')


    # forecasting model short
    HORIZON = 1
    forecasting_model_short = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            'task_name': 'forecasting',
            'forecast_horizon': HORIZON,
            'head_dropout': 0.1,
            'weight_decay': 0,
            'freeze_encoder': False, # Freeze the patch embedding layer
            'freeze_embedder': False, # Freeze the transformer encoder
            'freeze_head': False, # The linear forecasting head must be trained
        },
    )
    forecasting_model_short.init()

    train_dataset = InformerDataset(data_split="train", random_seed=RANDOM_SEED,
                                    forecast_horizon=HORIZON)
    train_loader_short = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = InformerDataset(data_split="test", random_seed=RANDOM_SEED,
                                   forecast_horizon=HORIZON)
    test_loader_short = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    forecasting_model_short = fine_tune(forecasting_model_short, train_loader_short,
                                        test_loader_short).to('cpu')


    # imputation model
    impute_model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        # For imputation, we will load MOMENT in `reconstruction` mode
        model_kwargs={
            'task_name': 'reconstruction',
            'freeze_encoder': False, # Freeze the patch embedding layer
            'freeze_embedder': False, # Freeze the transformer encoder
            'freeze_head': False, # The linear forecasting head must be trained
            }
    )
    impute_model.init()

    train_dataset = InformerDataset(data_split='train', random_seed=RANDOM_SEED,
                                    task_name='imputation', data_stride_len=512)
    train_loader_impute = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_dataset = InformerDataset(data_split='test', random_seed=RANDOM_SEED,
                                   task_name='imputation', data_stride_len=512)
    test_loader_impute = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    impute_model = fine_tune(impute_model, train_loader_impute, test_loader_impute).to('cpu')


    # anomaly detection model
    anomaly_model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        # For anomaly detection, we will load MOMENT in `reconstruction` mode
        model_kwargs={
            "task_name": "reconstruction",
            'freeze_encoder': False, # Freeze the patch embedding layer
            'freeze_embedder': False, # Freeze the transformer encoder
            'freeze_head': False, # The linear forecasting head must be trained
            },
    )
    anomaly_model.init()

    train_dataset = AnomalyDetectionDataset(data_split='train', random_seed=RANDOM_SEED)
    train_loader_anomaly = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                          shuffle=False, drop_last=False)

    test_dataset = AnomalyDetectionDataset(data_split='test', random_seed=RANDOM_SEED)
    test_loader_anomaly = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                         shuffle=False, drop_last=False)

    anomaly_model = fine_tune(anomaly_model, train_loader_anomaly, test_loader_anomaly).to('cpu')


    # classification model
    classification_model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            'task_name': 'classification',
            'n_channels': 1,
            'num_class': 5,
            'freeze_encoder': False, # Freeze the patch embedding layer
            'freeze_embedder': False, # Freeze the transformer encoder
            'freeze_head': False, # The linear forecasting head must be trained
        }, # We are loading the model in `classification` mode
    )
    classification_model.init()

    train_dataset = ClassificationDataset(data_split='train')
    train_loader_classification = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                                 shuffle=False, drop_last=False)

    test_dataset = ClassificationDataset(data_split='test')
    test_loader_classification = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                                shuffle=False, drop_last=False)

    classification_model = fine_tune(classification_model, train_loader_classification,
                                     test_loader_classification).to('cpu')


    # get embeddings
    get_representation({'long': (forecasting_model_long, train_loader_long),
                        'short': (forecasting_model_short, train_loader_short),
                        'impute': (impute_model, train_loader_impute),
                        'anomaly': (anomaly_model, train_loader_anomaly),
                        'classification': (classification_model, train_loader_classification)
                        })


