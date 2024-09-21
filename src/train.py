"""
train model
"""
from main import DEVICE

import pickle
import os
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torch import Tensor

import peft
import wandb

import sys
sys.path.append('/zfsauton2/home/mingzhul/time-series-prompt/src/momentfm')

from momentfm.common import TASKS
from momentfm.utils.forecasting_metrics import get_forecasting_metrics




def step(model: torch.nn.Module,
         data: tuple[Tensor, Tensor, Tensor],
         scaler: torch.cuda.amp.GradScaler = None,
         max_norm: float = None,
         optimizer: torch.optim.Optimizer = None):
    """
    one train / inference step
    """

    task = model.task_name

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):

        if task == 'classification':
            batch_x, batch_masks, labels = data

            n_channels = batch_x.shape[1]

            batch_x = batch_x.to(DEVICE).float()
            labels = labels.to(DEVICE).long()

            batch_masks = batch_masks.to(DEVICE).long()
            batch_masks = batch_masks[:, None, :].repeat(1, n_channels, 1)

            batch_masks = batch_masks[:, 0, :]

            # Forward
            output = model(batch_x, input_mask=batch_masks, task_name=task)

            # Compute loss
            if len(labels.shape) == 1:
                criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
            else:
                criterion = torch.nn.BCEWithLogitsLoss().to(DEVICE)
            loss = criterion(output.logits, labels.to(output.logits.dtype))

            # x = torch.argmax(output.logits, dim=1).detach().cpu().numpy()
            x = output.logits.detach().cpu().numpy()
            y = labels.detach().cpu().numpy()

        elif task == 'forecasting_long':
            if isinstance(model, peft.PeftModel):
                model.base_model.model.task_name = TASKS.FORECASTING

                if isinstance(model, peft.PeftModel):
                    model.base_model.model.fore_head = model.fore_head_long
                else:
                    model.fore_head = model.fore_head_long

            timeseries, forecast, input_mask = data
            timeseries = timeseries.float().to(DEVICE)
            input_mask = input_mask.to(DEVICE)
            forecast = forecast.float().to(DEVICE)

            output = model(timeseries, input_mask, task_name=task)

            criterion = torch.nn.MSELoss().to(DEVICE)
            loss = criterion(output.forecast, forecast)

            x = output.forecast.float().detach().cpu().numpy()
            y = forecast.float().detach().cpu().numpy()

        else:
            raise NotImplementedError

        if optimizer is not None:
            # Scales the loss for mixed precision training
            scaler.scale(loss).backward()

            # Clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        model.float()

    if loss.isnan():
        print('nan')


    return loss, x, y, model, scaler, optimizer



def get_metrics(probs, ys):
    # TODO: is this correct for 1 class and multi-class?

    if len(ys.shape) == 1:
        probs = torch.nn.functional.softmax(torch.tensor(probs), dim=1).numpy()
        # accuracy
        acc = np.mean((probs > 0.5).argmax(axis=1) == ys)
        # # F1
        # from sklearn.metrics import f1_score
        # f1 = f1_score(ys, (probs > 0.5).argmax(axis=1), average='macro')
        return (acc, )

    else:
        probs = torch.sigmoid(torch.tensor(probs)).numpy()
        # accuracy
        acc = np.mean((probs > 0.5) == ys)
        # F1
        from sklearn.metrics import f1_score
        f1 = f1_score(ys, probs > 0.5, average='macro')
    
        # auc
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(ys, probs, average='macro', multi_class='ovo')

        # AUPRC
        if len(ys.shape) > 1 and ys.shape[1] == 1:
            from sklearn.metrics import precision_recall_curve, auc
            (precisions, recalls, thresholds) = precision_recall_curve(ys, probs)
            auprc = auc(recalls, precisions)
        else:
            ys = torch.from_numpy(ys)
            probs = torch.from_numpy(probs)
            from torcheval.metrics import MultilabelAUPRC
            metric = MultilabelAUPRC(num_labels=ys.shape[1], average='macro')
            metric.update(probs, ys)
            auprc = metric.compute().item()

        return acc, auroc, f1, auprc



def get_performance(ys: list[NDArray],
                    xs: list[NDArray],
                    losses: list[float],
                    task: str,
                    curr_filename: str,
                    split: str = 'train',
                    cur_epoch: int = 0,
                    log: bool = True):

    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    loss = np.average(losses)

    if task == 'classification':
        out = get_metrics(xs, ys)
        if len(out) != 1:
            acc, auroc, f1, auprc = out
            performance = {
                'accuracy': acc,
                'AUC': auroc,
                'F1': f1,
                'AUPRC': auprc,
                'loss': loss,
            }
        else:
            acc = out
            performance = {
                'accuracy': acc,
                'loss': loss,
            }    
    elif task == 'forecasting_long':
        p = get_forecasting_metrics(ys, xs)
        performance = {
            'mse': p.mse,
            'mae': p.mae,
            'loss': loss,
        }
    else:
        raise NotImplementedError

    for key in performance.keys():
        if key == 'loss' and split == 'train':
            continue

        logging = {split+ '_'+task+'_'+curr_filename.split('/')[-2]+'/'+curr_filename.split('/')[-1]+'_'+key: performance[key]}
        print(logging)
        if log:
            wandb.log(logging, step=cur_epoch)

    return performance



def evaluate(model: torch.nn.Module,
             loader: torch.utils.data.DataLoader,
             split: str,
             cur_epoch: int,
             log: bool = True,
             logging_dir: str = None,
             bootstrap: bool = False):

    task = model.task_name
    filename = loader.dataset.filename
    xs, ys, losses = [], [], []
    model.eval()

    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    with torch.no_grad():
        for data in loader:
            loss, x, y, _, _, _ = step(model, data)
            xs.append(x)
            ys.append(y)
            losses.append(loss.item())

        if split == 'test' and bootstrap:
            performance_lst = []
            for _ in range(100):
                idx = np.random.choice(len(ys), len(ys))
                performance = get_performance(ys[idx], xs[idx], losses, task, filename, split=split,
                                               cur_epoch=cur_epoch, log=False)
                performance_lst.append(performance)

        else:
            performance = get_performance(ys, xs, losses, task, filename, split=split, cur_epoch=cur_epoch,
                                                log=log)

        # performance to pickle
        if not os.path.exists(logging_dir+'/'+task):
            os.makedirs(logging_dir+'/'+task)
        with open(logging_dir+'/'+task+'/'+filename.split('/')[-2]+'_'+filename.split('/')[-1]+'.pkl', 'wb') as f:
            pickle.dump(performance, f)


    model.train()

    return np.average(losses)






def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          extra: str = '',
          max_norm: float = 5.0, max_epoch: int = 10,
          max_lr: float = 1e-2,
          identifier: str = None, log: bool = True
          ):
    """
    train model
    """

    print("train counts", len(train_loader.dataset))
    print("val counts", len(val_loader.dataset))
    print("test counts", len(test_loader.dataset))

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    total_steps = len(train_loader) * max_epoch
    scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=total_steps, pct_start=0.3)
    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Move the model to the GPU
    model = model.to(DEVICE)

    best_val_loss = np.inf
    best_model = None

    logging_dir = 'performance/' + identifier + '/' + extra
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)


    # TODO: data proportion
    # labels = np.array(train_loader.dataset.raw['data'][1])
    # proportion = torch.tensor((len(labels) - np.sum(labels, axis=0)) / np.sum(labels, axis=0), device=DEVICE)


    for cur_epoch in tqdm(range(max_epoch)):
        model.train()

        losses = []
        for data in train_loader:
            loss, _, _, model, scaler, optimizer = step(model, data, scaler, max_norm, optimizer)
            losses.append(loss.item())

        losses = np.array(losses)

        # TODO: nan in forecast
        average_loss = np.nanmean(losses)
        print(f"Epoch {cur_epoch}: Train loss: {average_loss:.3f}")
        if log:
            wandb.log({'train_loss': average_loss}, step=cur_epoch)

        # Step the learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        # Evaluate the model on the test split
        model, val_loss = inference(model, val_loader, cur_epoch, train_loader,
                                    'val', logging_dir=logging_dir, log=log)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model)


    _, _ = inference(best_model, test_loader, 0, train_loader,
                        'test', logging_dir=logging_dir, log=log)

    return best_model




def inference(model: torch.nn.Module,
              test_loader: torch.utils.data.DataLoader,
              cur_epoch: int,
              train_loader: torch.utils.data.DataLoader,
              split: str,
              log: bool = True,
              logging_dir: str = 'performance/testing'):
    """
    perform inference
    """

    _ = evaluate(model, train_loader, 'train', cur_epoch, log=log, logging_dir=logging_dir+'/train/'+str(cur_epoch))
    val_loss = evaluate(model, test_loader, split, cur_epoch, log=log, logging_dir=logging_dir+f'/{split}/'+str(cur_epoch))

    return model, val_loss

