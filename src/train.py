"""
train model
"""
from copy import deepcopy

import numpy as np
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import OneCycleLR

import wandb

from momentfm.common import TASKS
from momentfm.utils.masking import Masking
from momentfm.models.statistical_classifiers import fit_svm

from momentfm.utils.forecasting_metrics import get_forecasting_metrics
from momentfm.utils.anomaly_detection_metrics import adjbestf1

from main import DEVICE



def step(model, batch, criterion, mask_generator,):
    """
    one train / inference step
    """

    loss = 0
    losses = {}
    xs, ys = {}, {}
    embedding = None

    for key, data in batch.items():

        if key in ('imputation', 'anomaly', 'classify'):

            model.task_name = TASKS.RECONSTRUCTION

            if data is None:
                continue
            if key not in losses:
                losses[key] = 0

            if key == 'anomaly':
                batch_x, batch_masks, labels = data
            elif key == 'imputation':
                batch_x, batch_masks = data
            elif key == 'classify':
                batch_x, batch_masks, labels = data
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

                    x = embedding.detach().cpu().numpy()
                    y = labels.detach().cpu().numpy()

                elif key == 'anomaly':
                    x = output.reconstruction.detach().cpu().numpy()
                    y = labels.detach().cpu().numpy()

                else:
                    raise NotImplementedError


            elif key == 'imputation':
                observed_mask = (batch_masks * (1 - mask))[:, None, :].repeat(1, n_channels, 1)  # mask: 0 not observed, 1 observed?
                assert observed_mask.shape == recon_loss.shape
                masked_loss = observed_mask * recon_loss

                val = (masked_loss.nansum() / (observed_mask.nansum() + 1e-7))
                loss += val
                losses[key] += val.detach().cpu().numpy()

                x = output.reconstruction[observed_mask == 1].detach().cpu().numpy()
                y = batch_x[observed_mask == 1].detach().cpu().numpy()

            else:
                raise NotImplementedError


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

            x = output.forecast.detach().cpu().numpy()
            y = forecast.detach().cpu().numpy()

        else:
            raise NotImplementedError



    return loss, model, losses, xs, ys



def evaluate(model, loader, criterion, mask_generator, clf=None):

    model.eval()

    performances = {}

    for task in loader.dataset.tasks:
        if isinstance(loader.dataset, torch.utils.data.IterableDataset):
            loader.dataset.reset()
            it = tqdm(loader)
        else:
            it = tqdm(loader, total=len(loader))

        xs, ys = [], []

        with torch.no_grad():
            for data in it:
                _, _, _, x, y = step(model, data[task], criterion, mask_generator,)
                xs.append(x)
                ys.append(y)


        xs = np.concatenate(xs, axis=0)
        ys = np.concatenate(ys, axis=0)

        if task == 'classify':
            if clf is None:
                clf = fit_svm(features=xs, y=ys)
            acc = clf.score(xs, ys)
            performances[task] = {
                'accuracy': acc,
            }
        elif task == 'anomaly':
            # TODO: anomaly performance, VUS ROC
            # https://github.com/TheDatumOrg/VUS
            performances[task] = {
                'adj F1': adjbestf1(ys, xs) # TODO: is this correct?
            }
        elif task in ('imputation', 'forecasting_long'):
            p = get_forecasting_metrics(ys, xs)
            performances[task] = {
                'mse': p['mse'],
                'mae': p['mae'],
            }
        elif task == 'forecasting_short':
            p = get_forecasting_metrics(ys, xs)
            performances[task] = {
                'smape': p['smape'],
            }
        else:
            raise NotImplementedError

    return performances, clf






def train(model, train_loader, test_loader,
          # Gradient clipping value
          max_norm = 5.0,
          max_epoch = 400, max_lr = 1e-2
          ):
    """
    prompt tuning
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = None
    if not isinstance(train_loader.dataset, torch.utils.data.IterableDataset):
        # Create a OneCycleLR scheduler
        total_steps = len(train_loader) * max_epoch
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

        if isinstance(train_loader.dataset, torch.utils.data.IterableDataset):
            train_loader.dataset.reset()
            test_loader.dataset.reset()
            it = tqdm(train_loader)
        else:
            it = tqdm(train_loader, total=len(train_loader))

        n_b = 0
        for data in it:
            loss, model, loss_dict, _, _ = step(model, data, criterion, mask_generator,)

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

            n_b += 1

            # if n_b  == 10:
            #     break


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
        if scheduler is not None:
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
    test_embeddings, test_labels = [], []

    model.eval()

    if isinstance(test_loader.dataset, torch.utils.data.IterableDataset):
        train_loader.dataset.reset()
        test_loader.dataset.reset()
        it = tqdm(test_loader)
    else:
        it = tqdm(test_loader, total=len(test_loader))

    n_b = 0
    with torch.no_grad():
        for data in test_loader:
            n_b += 1
            loss, model, loss_dict, embedding = step(model, data, criterion, mask_generator,)

            losses.append(loss.item())
            loss_dicts.append(loss_dict)

            # classification
            if embedding is not None:
                _, _, batch_labels = data['classify']
                test_labels.append(batch_labels)
                test_embeddings.append(embedding.detach().cpu().numpy())

            # if n_b  == 10:
            #     break

    losses = np.array(losses)
    average_loss = np.average(losses)
    loss_dict = {key: np.nanmean([d[key] if key in d.keys() else np.nan for d in loss_dicts]) \
        for key in loss_dicts[0].keys()}

    print(f"Epoch {cur_epoch}: Test loss: {average_loss:.3f}, Individual losses: {loss_dict}")
    print(n_b, 'batches')


    # classification
    train_accuracy = None
    test_accuracy = None
    if len(test_embeddings) > 0:

        train_embeddings, train_labels = [], []
        if isinstance(train_loader.dataset, torch.utils.data.IterableDataset):
            train_loader.dataset.reset()
            test_loader.dataset.reset()
            it = tqdm(train_loader)
        else:
            it = tqdm(train_loader, total=len(train_loader))

        with torch.no_grad():
            for data in it:
                _, _, _, embedding = step(model, data, criterion, mask_generator)

                # classification
                if embedding is not None:
                    _, _, batch_labels = data['classify']
                    train_labels.append(batch_labels)
                    train_embeddings.append(embedding.detach().cpu().numpy())

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
