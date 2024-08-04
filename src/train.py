"""
train model
"""
import datetime
import pickle
import os
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



def step(model, batch, criterion, mask_generator,
         scaler=None, max_norm=None, optimizer=None):
    """
    one train / inference step
    """

    # print(torch.cuda.max_memory_allocated())

    loss = 0
    losses = {}
    xs, ys = {}, {}
    embedding = None

    for key, data in batch.items():

        # if key != 'forecasting_short' and key != 'forecasting_long':
        #     continue

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
            # batch_x = batch_x.to(DEVICE).double()

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

            # Compute loss
            recon_loss = criterion(output.reconstruction, batch_x)
            if key in ('anomaly', 'classify'):
                val = recon_loss.mean()
                loss += val
                losses[key] += val.detach().cpu().numpy()

                if key == 'classify':
                    xs[key] = output.embeddings.detach().cpu().numpy()
                    ys[key] = labels.detach().cpu().numpy()
                    
                    # del embedding

                elif key == 'anomaly':
                    # We will use the Mean Squared Error (MSE) between the observed values and MOMENT's predictions as the anomaly score
                    xs[key] = ((batch_x - output.reconstruction)**2).detach().cpu().numpy()
                    ys[key] = labels.detach().cpu().numpy()

                else:
                    raise NotImplementedError

                # del labels

            elif key == 'imputation':
                observed_mask = (batch_masks * (1 - mask))[:, None, :].repeat(1, n_channels, 1)  # mask: 0 not observed, 1 observed?
                assert observed_mask.shape == recon_loss.shape
                masked_loss = observed_mask * recon_loss

                val = (masked_loss.nansum() / (observed_mask.nansum() + 1e-7))
                loss += val
                losses[key] += val.detach().cpu().numpy()

                xs[key] = output.reconstruction[observed_mask == 1].detach().cpu().numpy()
                ys[key] = batch_x[observed_mask == 1].detach().cpu().numpy()

                # del observed_mask, masked_loss

            else:
                raise NotImplementedError

            # del batch_x, batch_masks, mask, recon_loss

        elif key in ('forecasting_short', 'forecasting_long'):
            # model.double()
            # torch.autograd.set_detect_anomaly(True)
            # with torch.autocast(device_type="cuda", enabled=False):

            model.task_name = TASKS.FORECASTING

            if key == 'forecasting_short':
                model.fore_head = model.fore_head_8
            elif key == 'forecasting_long':
                model.fore_head = model.fore_head_96

            timeseries, forecast, input_mask = data
            # Move the data to the GPU
            timeseries = timeseries.float().to(DEVICE)
            # timeseries = timeseries.double().to(DEVICE)
            input_mask = input_mask.to(DEVICE)
            forecast = forecast.float().to(DEVICE)
            # forecast = forecast.double().to(DEVICE)

            # with torch.cuda.amp.autocast():
            output = model(timeseries, input_mask, task_name=key)

            val = criterion(output.forecast, forecast).mean()

            if val.isnan():
                print('nan')
            # # check nan weights
            # for name, param in model.named_parameters():
            #     if torch.isnan(param).any():
            #         print(name)
            #         print(param)

            # val = val.float()
            
            loss += val
            losses[key] = val.detach().cpu().numpy()

            xs[key] = output.forecast.float().detach().cpu().numpy()
            ys[key] = forecast.float().detach().cpu().numpy()


            # del timeseries, forecast, input_mask

        else:
            raise NotImplementedError

        # del output
        # val = val.float()
        if optimizer is not None:
            # Scales the loss for mixed precision training
            scaler.scale(val).backward()

            # Clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        model.float()

    #     torch.cuda.empty_cache()
    #     print(torch.cuda.max_memory_allocated())

    # print(torch.cuda.max_memory_allocated())

    return loss, model, losses, xs, ys, scaler, optimizer



def get_performance(ys, xs, losses, task, curr_filename, clf=None, split='train', cur_epoch=0, log=True):
    performances = {}

    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    loss = np.average(losses)

    if task == 'classify':
        if clf is None:
            clf = fit_svm(features=xs, y=ys)
        acc = clf.score(xs, ys)
        performances[task] = {
            'accuracy': acc,
            'loss': loss,
        }
    elif task == 'anomaly':
        # TODO: anomaly performance, VUS ROC
        # https://github.com/TheDatumOrg/VUS

        performances[task] = {
            'adj F1': adjbestf1(ys.reshape(-1), xs.reshape(-1)),
            'loss': loss,
        }
    elif task in ('imputation', 'forecasting_long'):
        p = get_forecasting_metrics(ys, xs)
        performances[task] = {
            'mse': p.mse,
            'mae': p.mae,
            'loss': loss,
        }
    elif task == 'forecasting_short':
        p = get_forecasting_metrics(ys, xs)
        performances[task] = {
            'smape': p.smape,
            'loss': loss,
        }
    else:
        raise NotImplementedError

    for key1 in performances.keys():
        for key2 in performances[key1].keys():
            if key2 == 'loss' and split == 'train':
                continue

            logging = {split+ '_'+key1+'_'+curr_filename.split('/')[-2]+'/'+curr_filename.split('/')[-1]+'_'+key2: performances[key1][key2]}
            print(logging)
            if log:
                wandb.log(logging, step=cur_epoch)

    return performances, clf



def evaluate(model, loader, criterion, mask_generator, split, cur_epoch, clfs=None, log=True,
             logging_dir=None):

    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    model.eval()
    clfs = {} if clfs is None else clfs

    clf = None

    with torch.no_grad():
        for task in loader.dataset.tasks:
            if isinstance(loader.dataset, torch.utils.data.IterableDataset):
                loader.dataset.reset()
                it = tqdm(loader, total=loader.bs)
            else:
                it = tqdm(loader, total=len(loader))

            xs, ys, losses = [], [], []

            curr_filename = None

            for b, data in enumerate(it):
                if curr_filename is None:
                    curr_filename = loader.dataset.filenames[task]

                if curr_filename != loader.dataset.filenames[task]:
                    clf = None
                    name = curr_filename.split('/')[-2]
                    if task == 'classify' and name in clfs:
                        clf = clfs[name]
                        print('using trained clf')
                    performances, clf = get_performance(ys, xs, losses, task, curr_filename, clf=clf, split=split, cur_epoch=cur_epoch,
                                                        log=log)

                    # performance to pickle
                    if not os.path.exists(logging_dir+'/'+task):
                        os.makedirs(logging_dir+'/'+task)
                    with open(logging_dir+'/'+task+'/'+curr_filename.split('/')[-2]+'_'+curr_filename.split('/')[-1]+'.pkl', 'wb') as f:
                        pickle.dump(performances, f)

                    clfs[name] = clf
                    xs, ys, losses = [], [], []
                    curr_filename = loader.dataset.filenames[task]

                if task not in data:
                    break
                loss, _, _, x, y, _, _ = step(model, {task: data[task]}, criterion, mask_generator,)
                xs.append(x[task])
                ys.append(y[task])
                losses.append(loss.item())


            clf = None
            name = curr_filename.split('/')[-2]
            if task == 'classify' and name in clfs:
                clf = clfs[name]
                print('using trained clf')
            performances, clf = get_performance(ys, xs, losses, task, curr_filename, clf=clf, split=split, cur_epoch=cur_epoch,
                                                log=log)

            # performance to pickle
            if not os.path.exists(logging_dir+'/'+task):
                os.makedirs(logging_dir+'/'+task)
            with open(logging_dir+'/'+task+'/'+curr_filename.split('/')[-2]+'_'+curr_filename.split('/')[-1]+'.pkl', 'wb') as f:
                pickle.dump(performances, f)

            clfs[name] = clf


                # if b == 100:
                #     break


    model.train()

    return clfs






def train(model, train_loader, test_loader,
          # Gradient clipping value
          max_norm = 5.0,
          max_epoch = 400, max_lr = 1e-2,
          identifier=None, log=True
          ):
    """
    prompt tuning
    """

    train_bs = 0
    for data in train_loader:
        train_bs += 1
    train_loader.bs = train_bs

    test_bs = 0
    for data in test_loader:
        test_bs += 1
    test_loader.bs = test_bs
    
    print("train counts", train_loader.dataset.counts)
    print("test counts", test_loader.dataset.counts)

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


    logging_dir = 'performance/' + identifier + '/' + str(datetime.datetime.now())
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    # model.double()

    # https://discuss.pytorch.org/t/finding-source-of-nan-in-forward-pass/51153/3
    def nan_hook(self, inp, output):
        if not isinstance(output, tuple):
            outputs = [output]
        else:
            outputs = output

        for i, out in enumerate(outputs):
            if out is None:
                continue
            if not isinstance(out, torch.Tensor):

                for j, o in enumerate(out):
                    if torch.isnan(out[o]).any():
                        print("In", self.__class__.__name__)
                        raise RuntimeError(f"Found NAN in output {i} at indices: ", torch.isnan(out[o]).nonzero(), "where:", o[torch.isnan(out[o]).nonzero()[:, 0].unique(sorted=True)])

                    if torch.isinf(out[o]).any():
                        print("In", self.__class__.__name__)
                        raise RuntimeError(f"Found INF in output {i} at indices: ", torch.isinf(out[o]).nonzero(), "where:", o[torch.isinf(out[o]).nonzero()[:, 0].unique(sorted=True)])
            else:
                nan_mask = torch.isnan(out)
                if nan_mask.any():
                    print("In", self.__class__.__name__)
                    raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

                inf_mask = torch.isinf(out)
                if inf_mask.any():
                    print("In", self.__class__.__name__)
                    raise RuntimeError(f"Found INF in output {i} at indices: ", inf_mask.nonzero(), "where:", out[inf_mask.nonzero()[:, 0].unique(sorted=True)])

    # for submodule in model.modules():
    #     submodule.register_forward_hook(nan_hook)


    for cur_epoch in range(max_epoch):

        # model = inference(model, test_loader, criterion, cur_epoch, mask_generator, train_loader,
        #                   logging_dir=logging_dir)

        model.train()

        losses = []
        loss_dicts = []

        if isinstance(train_loader.dataset, torch.utils.data.IterableDataset):
            train_loader.dataset.reset()
            test_loader.dataset.reset()
            it = tqdm(train_loader, total=train_loader.bs)
        else:
            it = tqdm(train_loader, total=len(train_loader))

        n_b = 0
        for data in it:
            n_b += 1

            # if n_b < 1600:
            #     continue

            # continue
            # temp = deepcopy(model)

            loss, model, loss_dict, _, _, scaler, optimizer = step(model, data, criterion, mask_generator, scaler, max_norm, optimizer)
            # print(torch.cuda.max_memory_allocated())

            # # Scales the loss for mixed precision training
            # scaler.scale(loss).backward()

            # # Clip gradients
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # scaler.step(optimizer)
            # scaler.update()
            # optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())
            loss_dicts.append(loss_dict)
            # if n_b  == 10:
            #     break
        
        print(train_loader.dataset.counts)

        losses = np.array(losses)

        # TODO: nan in forecast
        average_loss = np.nanmean(losses)
        # average_loss = np.average(losses)
        loss_dict = {key: np.nanmean(
            [d[key] if key in d.keys() else np.nan for d in loss_dicts]
            ) for key in loss_dicts[0].keys()}

        print(f"Epoch {cur_epoch}: Train loss: {average_loss:.3f}, Individual losses: {loss_dict}")

        logging = {
            'train_loss': average_loss} | {
                'train_'+key+'_loss': val for key, val in loss_dict.items()
                }
        if log:
            wandb.log(logging, step=cur_epoch)
        print(logging)

        # try:
        #     model.push_to_hub('time-series-prompt_'+identifier.replace('/', '_')+'_'+str(cur_epoch), private=True)
        # except Exception as e:
        #     print(e)

        # Step the learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        # Evaluate the model on the test split
        model = inference(model, test_loader, criterion, cur_epoch, mask_generator, train_loader,
                          logging_dir=logging_dir, log=log)

    return model




def inference(model, test_loader, criterion, cur_epoch, mask_generator, train_loader,
              log=True,
              logging_dir=None):
    """
    perform inference
    """

    clfs = evaluate(model, train_loader, criterion, mask_generator, 'train', cur_epoch, clfs=None, log=log, logging_dir=logging_dir+'/train/'+str(cur_epoch))
    evaluate(model, test_loader, criterion, mask_generator, 'test', cur_epoch, clfs=clfs, log=log, logging_dir=logging_dir+'/test/'+str(cur_epoch))

    print(test_loader.dataset.counts)

    # test_performance, _ = evaluate(model, test_loader, criterion, mask_generator, clf=clf)


    # for key1 in test_performance.keys():
    #     for key2 in test_performance[key1].keys():
    #         log = {'test_'+key1+'_'+key2: test_performance[key1][key2]}
    #         print(log)
    # if log:
    #     wandb.log(log, step=cur_epoch)

    return model
