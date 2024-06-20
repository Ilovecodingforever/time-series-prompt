"""
https://arxiv.org/pdf/2303.02861#page=5.10

"""


import os

import numpy as np
from tqdm import tqdm
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA


import torch
import torch.nn as nn
import torch.cuda.amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from momentfm.utils.utils import control_randomness
from momentfm.data.informer_dataset import InformerDataset
# from momentfm.data.classification_dataset import ClassificationDataset
from momentfm.data.anomaly_detection_dataset import AnomalyDetectionDataset

from momentfm.common import TASKS
from momentfm import MOMENTPipeline
from momentfm.utils.masking import Masking
# from momentfm.utils.forecasting_metrics import get_forecasting_metrics


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["HF_HOME"] = "/home/scratch/mingzhul/.cache/huggingface"



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
RANDOM_SEED = 0






def step(model, batch, criterion, mask_generator):
    """
    one train / inference step
    """

    loss = 0

    losses = {}

    if model.task_name == TASKS.RECONSTRUCTION:
        for key, data in batch.items():

            if data is None:
                continue

            if key not in losses:
                losses[key] = 0

            # initialize the task name
            if isinstance(model.patch_embedding.value_embedding, MPT):
                model.patch_embedding.value_embedding.task_name = key

            if key == 'anomaly':
                batch_x, batch_masks, _ = data
            elif key == 'imputation':
                batch_x, batch_masks = data
            else:
                raise NotImplementedError

            n_channels = batch_x.shape[1]

            # Reshape to [batch_size * n_channels, 1, window_size]
            batch_x = batch_x.reshape((-1, 1, 512)).to(DEVICE).float()

            batch_masks = batch_masks.to(DEVICE).long()
            batch_masks = batch_masks.repeat_interleave(n_channels, axis=0)

            # Randomly mask some patches of data
            # 0 is masked
            mask = mask_generator.generate_mask(
                x=batch_x, input_mask=batch_masks).to(DEVICE).long()

            # Forward
            output = model(batch_x, input_mask=batch_masks, mask=mask).reconstruction

            # Compute loss
            recon_loss = criterion(output, batch_x)
            if key == 'anomaly':
                val = recon_loss.mean()
                loss += val
                losses[key] += val.detach().cpu().numpy()

            elif key == 'imputation':
                observed_mask = (batch_masks * (1 - mask))[:, None, :]
                assert observed_mask.shape == recon_loss.shape
                masked_loss = observed_mask * recon_loss

                val = (masked_loss.nansum() / (observed_mask.nansum() + 1e-7))
                loss += val
                losses[key] += val.detach().cpu().numpy()

            else:
                raise NotImplementedError

            y = batch_x


    else:
        raise NotImplementedError


    return loss, model, y, output, losses




def train(model, train_loader, test_loader,
          # Gradient clipping value
          max_norm = 5.0,
          max_epoch = 10, max_lr = 1e-2
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
            loss, model, _, _, loss_dict = step(model, data, criterion, mask_generator)

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
        loss_dict = {key: np.nanmean([d[key] if key in d.keys() else np.nan for d in loss_dicts]) for key in loss_dicts[0].keys()}

        print(f"Epoch {cur_epoch}: Train loss: {average_loss:.3f}, Individual losses: {loss_dict}")

        # Step the learning rate scheduler
        scheduler.step()

        # Evaluate the model on the test split
        model = inference(model, test_loader, criterion, cur_epoch, mask_generator)


    return model




def inference(model, test_loader, criterion, cur_epoch, mask_generator):
    """
    perform inference
    """

    trues, preds, losses, loss_dicts = [], [], [], []
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader)):
            loss, model, y, output, loss_dict = step(model, data, criterion, mask_generator)

            losses.append(loss.item())
            trues.append(y.detach().cpu().numpy())
            preds.append(output.detach().cpu().numpy())
            loss_dicts.append(loss_dict)

    losses = np.array(losses)
    average_loss = np.average(losses)
    loss_dict = {key: np.nanmean([d[key] if key in d.keys() else np.nan for d in loss_dicts]) for key in loss_dicts[0].keys()}

    model.train()

    print(f"Epoch {cur_epoch}: Test loss: {average_loss:.3f}, Individual losses: {loss_dict}")

    trues = np.concatenate(trues, axis=0)
    preds = np.concatenate(preds, axis=0)

    return model






class MPT(nn.Module):
    """
    multitask prompt tuning
    """
    def __init__(self,
                wte: nn.Module,
                tasks: list,
                n_tokens: int = 10,
                random_range: float = 0.5,
                hidden_size: int = 64):
        """appends learned embedding to

        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab).
                            Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super().__init__()
        self.wte = wte
        self.n_tokens = n_tokens

        self.hidden_size = hidden_size

        self.mlp = nn.Linear(self.hidden_size, wte.weight.size(0))

        self.shared_prompt = nn.parameter.Parameter(self.initialize_embedding(n_tokens,
                                                                               self.hidden_size,
                                                                               random_range))

        self.tasks = tasks
        self.task_prompt = nn.ParameterDict({task: nn.ParameterDict({
            'u': nn.parameter.Parameter(self.initialize_embedding(n_tokens,
                                                                    1,
                                                                    random_range)),
            'v': nn.parameter.Parameter(self.initialize_embedding(1,
                                                                    self.hidden_size,
                                                                    random_range)),
            }) for task in tasks})

        self.task_name = None


    def initialize_embedding(self,
                             i: int,
                             j: int,
                             random_range: float = 0.5, ):
        """initializes learned embedding

        Args:
            same as __init__

        Returns:
            torch.float: initialized using original schemes
        """
        return torch.FloatTensor(i, j).uniform_(-random_range, random_range)


    def forward(self, tokens):
        """run forward pass

        Args:
            tokens (torch.long): input tokens before encoding

        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """

        task = self.task_name

        input_embedding = self.wte(tokens)

        u = self.task_prompt[task]['u']
        v = self.task_prompt[task]['v']

        learned_embedding = self.mlp(torch.matmul(u, v) * self.shared_prompt)

        # n_batches x features x n_patches x embedding_size
        learned_embedding = learned_embedding.repeat(input_embedding.size(0),
                                                     input_embedding.size(1), 1, 1)

        self.task_name = None

        return torch.cat([learned_embedding, input_embedding], 2)



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




def zero_shot(model, train_loader, test_loader):
    """
    zero shot
    """

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
                      Masking(mask_ratio=0.3))

    return model




def prompt_tuning(model, train_loader, test_loader, n_tokens=5):
    """
    prompt tuning
    """
    # need to freeze head manually
    for param in model.parameters():
        param.requires_grad = False

    setattr(model.patch_embedding, 'value_embedding',
            MPT(model.patch_embedding.value_embedding,
                ['imputation', 'anomaly'],
                n_tokens=n_tokens,))

    # print frozen params
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    model = train(model, train_loader, test_loader)

    return model


def finetune(model, train_loader, test_loader):
    """
    finetune
    """

    for param in model.parameters():
        param.requires_grad = True

    # print frozen params
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    model = train(model, train_loader, test_loader)

    return model




def get_data(batch_size):
    """
    get data
    """
    train_dataset_impute = InformerDataset(data_split='train', random_seed=RANDOM_SEED,
                                            task_name='imputation',
                                            data_stride_len=1
                                            # data_stride_len=512
                                            )
    test_dataset_impute = InformerDataset(data_split='test', random_seed=RANDOM_SEED,
                                            task_name='imputation',
                                            data_stride_len=1
                                            # data_stride_len=512
                                            )
    train_dataset_anomaly = AnomalyDetectionDataset(data_split='train', random_seed=RANDOM_SEED,
                                                    data_stride_len=10
                                                    # data_stride_len=512
                                                    )
    test_dataset_anomaly = AnomalyDetectionDataset(data_split='test', random_seed=RANDOM_SEED,
                                                    data_stride_len=10
                                                    # data_stride_len=512
                                                    )

    train_datasets = {'imputation':  train_dataset_impute, 'anomaly': train_dataset_anomaly}
    train_loader = DataLoader(CollectedDataset(train_datasets), batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)

    test_datasets = {'imputation':  test_dataset_impute, 'anomaly': test_dataset_anomaly}
    test_loader = DataLoader(CollectedDataset(test_datasets), batch_size=batch_size,
                             shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader



def run_mpt():
    """
    run MPT
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
            }
    ).to(DEVICE)
    model.init()


    experiment = prompt_tuning

    batch_size = 50
    if experiment == finetune:
        batch_size = 10

    train_loader, test_loader = get_data(batch_size)

    model = experiment(model, train_loader, test_loader)




if __name__ == '__main__':
    # Set random seeds for PyTorch, Numpy etc.
    control_randomness(seed=RANDOM_SEED)

    run_mpt()
