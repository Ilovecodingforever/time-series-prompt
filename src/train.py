"""
train model
"""

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

from typing import Optional


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none' )
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss




def step(model: torch.nn.Module,
         data: tuple[Tensor, Tensor, Tensor],
         scaler: Optional[torch.cuda.amp.GradScaler] = None,
         max_norm: Optional[float] = None,
         optimizer: Optional[torch.optim.Optimizer] = None,
         criterion=None,):
    """
    one train / inference step
    """

    task = model.task_name

    with torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32):

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

            if criterion is None:
                # Compute loss
                if len(labels.shape) == 1:
                    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
                else:
                    criterion = torch.nn.BCEWithLogitsLoss().to(DEVICE)

            if len(labels.shape) != 1:
                labels = labels.to(output.logits.dtype)

            loss = criterion(output.logits, labels)

            # x = torch.argmax(output.logits, dim=1).detach().cpu().numpy()
            x = output.logits.detach().cpu().numpy()
            y = labels.detach().cpu().numpy()

        elif 'forecasting' in task:
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
            print(task)
            raise NotImplementedError

        if optimizer is not None and scaler is not None:
            # Scales the loss for mixed precision training
            scaler.scale(loss).backward()

            # Clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # model.float()

    if loss.isnan():
        # NOTE: if raise error, then the gpu will be lost for some reason
        print("Loss is NaN")
        # raise ValueError("Loss is NaN")
    # assert not loss.isnan()

    return loss, x, y, model, scaler, optimizer




def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          extra: str = '',
          max_norm: float = 5.0, max_epoch: int = 10,
          max_lr: float = 1e-2,
          identifier: str = '', log: bool = True
          ):
    """
    train model
    """

    print("train counts", len(train_loader.dataset))
    print("val counts", len(val_loader.dataset))
    print("test counts", len(test_loader.dataset))

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    total_steps = len(train_loader) * max_epoch
    scheduler = OneCycleLR(optimizer, max_lr=0.001, total_steps=total_steps, pct_start=0.3)
    # Enable mixed precision training
    scaler = torch.amp.GradScaler('cuda')

    # Move the model to the GPU
    model = model.to(DEVICE)

    best_val_loss = np.inf
    best_model = None

    logging_dir = 'performance/' + identifier + '/' + extra
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    criterion = None
    if model.task_name == 'classification':
        # data proportion
        labels = np.array(train_loader.dataset.raw['data'][1])
        # 0s / 1s
        proportion = torch.tensor((len(labels) - np.sum(labels, axis=0)) / np.sum(labels, axis=0), device=DEVICE)
        # https://machinelearningmastery.com/cost-sensitive-learning-for-imbalanced-classification/ just use inverse proportion
        cost_sensitive_loss = torch.nn.BCEWithLogitsLoss(pos_weight=proportion)

        # # https://github.com/AdeelH/pytorch-multi-class-focal-loss
        # focal_loss = torch.hub.load(
        #     'adeelh/pytorch-multi-class-focal-loss',
        #     model='FocalLoss',
        #     alpha=proportion,
        #     gamma=2,
        #     reduction='mean',
        #     force_reload=False
        # )
        focal_loss = FocalLoss(logits=True)
        

        # TODO:
        criterion = focal_loss
        # criterion = cost_sensitive_loss


    for cur_epoch in tqdm(range(max_epoch)):
        model.train()

        losses = []
        for i, data in enumerate(tqdm(train_loader)):
            loss, _, _, model, scaler, optimizer = step(model, data, scaler, max_norm, optimizer, criterion=criterion)
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
        train_loss = evaluate(model, val_loader, 'train', cur_epoch, logging_dir=logging_dir+'/train/'+str(cur_epoch), log=log, criterion=criterion)
        val_loss = evaluate(model, train_loader, 'val', cur_epoch, logging_dir=logging_dir+'/val/'+str(cur_epoch), log=log, criterion=criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model)

    val_loss = evaluate(best_model, test_loader, 'test', 0, logging_dir=logging_dir+'/test/'+str(cur_epoch), log=log)

    return best_model




def get_metrics(probs, ys, STraTS=False):
    if isinstance(ys, torch.Tensor):
        ys = ys.numpy()
    if isinstance(probs, torch.Tensor):
        probs = probs.numpy()

    if len(ys.shape) == 1:
        # NOTE: just do accuracy for non-mimic
        if STraTS:
            raise NotImplementedError("STraTS doesn't support multi-class, single label (needs softmax)")
        probs = torch.nn.functional.softmax(torch.tensor(probs), dim=1).numpy()
        # accuracy
        # TODO: is this correct for multi-class?
        acc = np.mean(probs.argmax(axis=1) == ys)

        return (acc, )

    else:
        if not STraTS:
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
            performance = {'accuracy': acc, 'AUC': auroc, 'F1': f1, 'AUPRC': auprc, 'loss': loss,}
        else:
            acc = out
            performance = {'accuracy': acc, 'loss': loss,}
    elif 'forecasting' in task:
        p = get_forecasting_metrics(ys, xs)
        performance = {'mse': p.mse, 'mae': p.mae, 'loss': loss,}
    else:
        raise NotImplementedError


    if split == 'test':
        metrics_table = wandb.Table(
            columns=list(performance.keys()),
            data=[list(performance.values())],
        )
        print('test:', performance)
        if log:
            wandb.log({split+ '_'+task+'_'+curr_filename.split('/')[-2]+'/'+curr_filename.split('/')[-1]: metrics_table})

    else:
        for key, value in performance.items():
            if key == 'loss' and split == 'train':
                continue

            logging = {split+ '_'+task+'_'+curr_filename.split('/')[-2]+'/'+curr_filename.split('/')[-1]+'_'+key: value}
            print(logging)
            if log:
                wandb.log(logging, step=cur_epoch)

    return performance



def evaluate(model: torch.nn.Module,
             loader: torch.utils.data.DataLoader,
             split: str,
             cur_epoch: int,
             log: bool = True,
             logging_dir: str = '',
             bootstrap: bool = False,
             criterion=None):

    task = model.task_name
    filename = loader.dataset.filename
    xs, ys, losses = [], [], []
    model.eval()

    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    with torch.no_grad():
        for data in loader:
            loss, x, y, _, _, _ = step(model, data, criterion=criterion)
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



