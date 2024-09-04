"""
mimic mortality prediction
"""

import sys
sys.path.append('/zfsauton2/home/mingzhul/time-series-prompt/src/momentfm')



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["HF_HOME"] = "/home/scratch/mingzhul/.cache/huggingface"



import torch
import wandb
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from peft import LoraConfig, get_peft_model

from momentfm import MOMENTPipeline
from momentfm.utils.utils import control_randomness

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def load_data(equal_length=False, small_part=True):
    from ts_datasets import MIMIC_mortality  # TODO: this changes visible devices, why?

    # batch_size = 1
    batch_size = 8
    shuffle = True

    train_data = MIMIC_mortality(data_split="train", equal_length=equal_length, small_part=small_part)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)

    val_data = MIMIC_mortality(data_split="val", equal_length=equal_length, small_part=small_part)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle)

    test_data = MIMIC_mortality(data_split="test", equal_length=equal_length, small_part=small_part)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader




def load_model(mode='finetune', equal_length=False,
               num_prefix=1, visualize_attention=False, prompt='attention'):

    n_channels = 60
    # n_channels = 78
    if equal_length:
        n_channels = 59
    num_classes = 1

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            'task_name': 'classification',
            'freeze_encoder': False, # Freeze the patch embedding layer
            'freeze_embedder': False, # Freeze the transformer encoder
            'freeze_head': False, # The linear forecasting head must be trained
            'forecast_horizons': (24, 8),
            # 'num_prefix': 16,
            'num_prefix': num_prefix,
            'task_names': ['classify'],
            'multivariate_projection': prompt,
            'n_channels': n_channels,
            'num_class': num_classes,
            'prefix_tuning_multi': (mode == 'prompt'),
            'seq_len': 48 if equal_length else 128,
            'visualize_attention': visualize_attention,
            }
    )

    model.init()

    for param in model.parameters():
        param.requires_grad = True

    if mode == 'prompt':
        for n, param in model.named_parameters():
            if 'prefix' not in n and 'prompt' not in n and 'head' not in n and 'mpt' not in n and 'value_embedding' not in n and 'layer_norm' not in n:
                param.requires_grad = False

    elif mode == 'lora':
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q", "v"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["fore_head_long", "fore_head_short", "head", "emb_head", "classification_head"],
        )
        model = get_peft_model(model, config)


    # print frozen params
    for n, param in model.named_parameters():
        if param.requires_grad:
            print(n)

    print('number of parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Move the model to the GPU
    model = model.to(DEVICE)

    return model





def step(model, batch, criterion,
         scaler=None, max_norm=None, optimizer=None):
    """
    one train / inference step
    """

    # with torch.autocast(device_type="cuda", enabled=False, dtype=torch.float16):

    batch_x, batch_masks, labels = batch
    n_channels = batch_x.shape[1]

    batch_x = batch_x.to(DEVICE).float()
    labels = labels.to(DEVICE).float()
    batch_masks = batch_masks.to(DEVICE).long()

    # Forward
    output = model(batch_x, input_mask=batch_masks, task_name='classify')

    # Compute loss
    loss = criterion(output.logits.flatten(), labels)

    xs = output.logits.detach().cpu().numpy()
    ys = labels.detach().cpu().numpy()

    if optimizer is not None:

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # # Scales the loss for mixed precision training
        # scaler.scale(loss).backward()

        # # Clip gradients
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # scaler.step(optimizer)
        # scaler.update()
        # optimizer.zero_grad(set_to_none=True)

    # model.float()


    return loss, model, xs, ys, scaler, optimizer








def train(model, train_loader, val_loader, test_loader,
          max_norm = 5.0,
          max_epoch = 10, max_lr = 1e-2,
          identifier=None, log=True
          ):


    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    # Create a OneCycleLR scheduler
    total_steps = len(train_loader) * max_epoch
    scheduler = OneCycleLR(optimizer, max_lr=5e-5, total_steps=total_steps, pct_start=0.3)


    # optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.05)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, eta_min=1e-7, last_epoch=-1, verbose='deprecated')

    proportion = torch.tensor((len(train_loader.dataset.raw['data'][1]) - np.array(train_loader.dataset.raw['data'][1]).sum()) / np.array(train_loader.dataset.raw['data'][1]).sum())
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=proportion)


    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler()


    best_val_loss = np.inf
    best_model = None

    for cur_epoch in range(max_epoch):
        model.train()
        losses = []

        for data in tqdm(train_loader, desc=f"Epoch {cur_epoch}"):
            loss, model, _, _, scaler, optimizer = step(model, data, criterion, scaler, max_norm, optimizer)
            losses.append(loss.item())

        average_loss = np.mean(losses)

        print(f"Epoch {cur_epoch}: Train loss: {average_loss:.3f}")

        # Step the learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        # Evaluate the model on the test split
        val_loss, _ = inference(model, val_loader, criterion, cur_epoch, 'val',)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model)


    average_loss, metrics = inference(best_model, test_loader, criterion, 0, 'test')

    return best_model, average_loss, metrics






def get_metrics(probs, ys):

    # accuracy
    acc = np.mean((probs > 0.5) == ys)
    # auc
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(ys, probs)
    # F1
    from sklearn.metrics import f1_score
    f1 = f1_score(ys, probs > 0.5)
    # AUPRC
    from sklearn.metrics import precision_recall_curve, auc
    (precisions, recalls, thresholds) = precision_recall_curve(ys, probs)
    auprc = auc(recalls, precisions)

    return [acc, auroc, f1, auprc]    



def inference(model, test_loader, criterion, cur_epoch, split):
    """
    perform inference
    """

    xs = []
    ys = []
    losses = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, desc=f"Epoch {cur_epoch}"):
            loss, _, x, y, _, _ = step(model, data, criterion)
            xs.append(x)
            ys.append(y)
            losses.append(loss.item())

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    losses = np.array(losses)
    average_loss = np.nanmean(losses)

    probs = torch.sigmoid(torch.tensor(xs)).numpy()

    acc, auroc, f1, auprc = get_metrics(probs, ys)

    print(f"Epoch {cur_epoch}: {split} loss: {average_loss:.3f}, acc: {acc:.3f}, auc: {auroc:.3f}, f1: {f1:.3f}, auprc: {auprc:.3f}")

    return average_loss, [acc, auroc, f1, auprc]






# https://github.com/thuml/iTransformer/issues/44
def multivariate_correlations(model, train_loader):
    from momentfm.models.t5_multivariate_prefix import T5StackWithPrefixMulti
    from matplotlib import pyplot as plt
    assert isinstance(model.encoder, T5StackWithPrefixMulti)

    proportion = torch.tensor((len(train_loader.dataset.raw['data'][1]) - np.array(train_loader.dataset.raw['data'][1]).sum()) / np.array(train_loader.dataset.raw['data'][1]).sum())
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=proportion)


    # correlation with label
    means = []
    data_indices = [i for (i, x) in enumerate(train_loader.dataset.discretizer_header) if x.find("mask") == -1]
    for d in train_loader.dataset.raw['data'][0]:
        means.append(np.mean(d[:, data_indices], axis=0))
    means = np.array(means)

    labels = train_loader.dataset.raw['data'][1]

    # get correation
    from scipy.stats import pearsonr
    corrs = []
    for i in range(means.shape[1]):
        corr, _ = pearsonr(means[:, i], labels)
        corrs.append(corr)

    # plot correlation
    import matplotlib.pyplot as plt
    plt.figure()
    plt.bar(range(len(corrs)), corrs)
    plt.savefig('plots/correlation_means.png')
    plt.close()



    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_loader)):
            if i == 1:
                break

            step(model, batch, criterion, scaler=None, max_norm=None, optimizer=None)

            # TODO: correlation
            batch_x, batch_masks, labels = batch
            batch_x = batch_x[0]
            means = batch_x.mean(dim=1, keepdim=True)
            norm = torch.linalg.norm(batch_x - means, dim=1)[:, None]

            corr = (batch_x - means) @ (batch_x - means).T / (norm @ norm.T)
            plt.imshow(corr.cpu().numpy())
            plt.savefig('plots/correlation.png')

            print()




def logistic_regression(train_loader, val_loader, test_loader, random_state=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
    from sklearn.neural_network import MLPClassifier


    train_x = []
    train_y = []
    for batch in train_loader:
        x, masks, y = batch
        train_x.append(x.numpy())
        train_y.append(y.numpy())
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)

    val_x = []
    val_y = []
    for batch in val_loader:
        x, masks, y = batch
        val_x.append(x.numpy())
        val_y.append(y.numpy())
    val_x = np.concatenate(val_x)
    val_y = np.concatenate(val_y)

    test_x = []
    test_y = []
    for batch in test_loader:
        x, masks, y = batch
        test_x.append(x.numpy())
        test_y.append(y.numpy())
    test_x = np.concatenate(test_x)
    test_y = np.concatenate(test_y)


    # flatten
    train_x = train_x.reshape(train_x.shape[0], -1)
    val_x = val_x.reshape(val_x.shape[0], -1)
    test_x = test_x.reshape(test_x.shape[0], -1)
    
    # use multilayer perceptron
    model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=random_state)
    model.fit(train_x, train_y)
    
    # predict
    probs = model.predict_proba(test_x)[:, 1]
    

    # # flatten
    # train_x = np.concatenate([np.mean(train_x, axis=2), np.min(train_x, axis=2), np.max(train_x, axis=2)], axis=1)
    # val_x = np.concatenate([np.mean(val_x, axis=2), np.min(val_x, axis=2), np.max(val_x, axis=2)], axis=1)
    # test_x = np.concatenate([np.mean(test_x, axis=2), np.min(test_x, axis=2), np.max(test_x, axis=2)], axis=1)

    # # train
    # clf = LogisticRegression(random_state=random_state, max_iter=1000).fit(train_x, train_y)

    # # predict
    # probs = clf.predict_proba(test_x)[:, 1]

    # metrics
    acc, auroc, f1, auprc = get_metrics(probs, test_y)

    print(f"acc: {acc:.3f}, auc: {auroc:.3f}, f1: {f1:.3f}, auprc: {auprc:.3f}")

    return [acc, auroc, f1, auprc]





if __name__ == "__main__":

    # TODO:
    # add time to equal length, use prompt?

    # linear prompts, ablation
    # more baselines


    equal_length = False
    small_part = True

    # mode = 'finetune'
    mode = 'prompt'
    # mode = 'lora'


    # prompt = 'attention'
    prompt = 'linear'




    for seed in range(5):
        control_randomness(seed=seed)
        train_loader, val_loader, test_loader = load_data(equal_length=equal_length, small_part=small_part)
        logistic_regression(train_loader, val_loader, test_loader, random_state=seed)



    num_prefix = 1


    losses = []
    metrics = []

    for seed in range(5):

        control_randomness(seed=seed)

        model = load_model(mode=mode, equal_length=equal_length, num_prefix=num_prefix, prompt=prompt)
        train_loader, val_loader, test_loader = load_data(equal_length=equal_length, small_part=small_part)


        logistic_regression(train_loader, val_loader, test_loader, random_state=seed)


        model_weights, average_loss, metrics = train(model, train_loader, val_loader, test_loader)
        losses.append(average_loss)
        metrics.append(metrics)

        # save model
        # make dir if not exist
        # if not os.path.exists("/home/scratch/mingzhul/time-series-prompt/models"):
        #     os.makedirs("/home/scratch/mingzhul/time-series-prompt/models")
        # torch.save(model_weights.state_dict(), f"/home/scratch/mingzhul/time-series-prompt/models/mimic_{mode}_{seed}.pt")

    # model = load_model(mode=mode, equal_length=equal_length, num_prefix=num_prefix, visualize_attention=True)
    # model.load_state_dict(model_weights.state_dict())
    # multivariate_correlations(model, train_loader)


    print(losses)
    print(metrics)

    metrics = np.array(metrics)

    print('mean', np.mean(losses))
    print('mean', np.mean(metrics, axis=1))
    print('std', np.std(losses))
    print('std', np.std(metrics, axis=1))







