"""
train model
"""

import numpy as np
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import OneCycleLR

import wandb

from momentfm.common import TASKS
from momentfm.utils.masking import Masking
from momentfm.models.statistical_classifiers import fit_svm

from main import DEVICE



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
