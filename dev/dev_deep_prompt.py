"""
https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py

"""
import os

import wandb

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
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
from momentfm.models.statistical_classifiers import fit_svm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HOME"] = "/home/scratch/mingzhul/.cache/huggingface"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 0

# Set random seeds for PyTorch, Numpy etc.
control_randomness(seed=RANDOM_SEED)






def step(model, batch, criterion, mask_generator,
         prefix_tokens
         ):
    """
    one train / inference step
    """

    loss = 0

    if model.task_name == TASKS.RECONSTRUCTION:

        batch_x, batch_masks = batch

        n_channels = batch_x.shape[1]

        # Reshape to [batch_size * n_channels, 1, window_size]
        batch_x = batch_x.reshape((-1, 1, 512)).to(DEVICE).float()

        batch_masks = batch_masks.to(DEVICE).long()
        batch_masks = batch_masks.repeat_interleave(n_channels, axis=0)

        # Randomly mask some patches of data
        # 0 is masked
        mask = mask_generator.generate_mask(
            x=batch_x, input_mask=batch_masks).to(DEVICE).long()

        # train prefix
        past_key_values = model.prefix_encoder(prefix_tokens.unsqueeze(0).expand(batch_x.shape[0], -1))


        # Forward
        output = model(batch_x, input_mask=batch_masks, mask=mask,
                       past_key_values=past_key_values
                       )

        recon_loss = criterion(output.reconstruction, batch_x)
        # TODO: this is just imputation
        observed_mask = (batch_masks * (1 - mask))[:, None, :]
        assert observed_mask.shape == recon_loss.shape
        masked_loss = observed_mask * recon_loss

        val = (masked_loss.nansum() / (observed_mask.nansum() + 1e-7))
        loss += val

        y = batch_x


    return loss, model, y, output.reconstruction




def train(model, train_loader, test_loader,
          prefix_tokens,
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

        for data in tqdm(train_loader, total=len(train_loader)):
            loss, model, _, _ = step(model, data, criterion, mask_generator,
                                     prefix_tokens
                                     )

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

        # Evaluate the model on the test split
        model = inference(model, test_loader, criterion, cur_epoch, mask_generator, train_loader, prefix_tokens)

    return model




def inference(model, test_loader, criterion, cur_epoch, mask_generator, train_loader, prefix_tokens):
    """
    perform inference
    """

    trues, preds, losses = [], [], []

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader)):
            loss, model, y, output = step(model, data, criterion, mask_generator, prefix_tokens)

            losses.append(loss.item())
            trues.append(y.detach().cpu().numpy())
            preds.append(output.detach().cpu().numpy())


    losses = np.array(losses)
    average_loss = np.average(losses)

    print(f"Epoch {cur_epoch}: Test loss: {average_loss:.3f}")


    model.train()


    return model





class PrefixEncoder(torch.nn.Module):
    """
    P-tuning v2
    """
    def __init__(self, num_heads, num_layer, dim_ebd=32, seq_len=10):
        super().__init__()

        self.num_layer = num_layer
        self.dim_ebd = dim_ebd
        self.seq_len = seq_len
        self.num_heads = num_heads

        # TODO: is this just having one trainable vector?
        self.embedding = torch.nn.Embedding(seq_len, dim_ebd)

        self.trans = torch.nn.Sequential(
            torch.nn.Linear(dim_ebd, dim_ebd),
            torch.nn.Tanh(),
            torch.nn.Linear(dim_ebd, num_layer * 2 * dim_ebd)
        )

        self.dropout = torch.nn.Dropout(0.3)


    def forward(self, prefix):
        prefix_tokens = self.embedding(prefix)
        past_key_values = self.trans(prefix_tokens)

        bsz, seqlen, _ = past_key_values.shape

        past_key_values = past_key_values.view(bsz, seqlen, self.num_layer * 2, self.num_heads, -1)
        past_key_values = self.dropout(past_key_values).permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values


def get_prefix(model):

    # this is for the num_layer to work. this only has encoder
    assert(model.config.transformer_type == 'encoder_only')
    num_layer = model.encoder.config.num_hidden_layers

    num_heads = model.encoder.config.num_attention_heads
    hidden_size = model.encoder.config.hidden_size

    n_tokens = 10
    prefix_tokens = torch.arange(n_tokens).long().to(DEVICE)

    # prefix encoder
    prefix_encoder = PrefixEncoder(num_heads,
                                   num_layer,
                                   seq_len=n_tokens,
                                   dim_ebd=hidden_size,
                                   ).to(DEVICE)

    setattr(model, 'prefix_encoder', prefix_encoder)

    return model, prefix_tokens




if __name__ == "__main__":

    # imputation model
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        # For imputation, we will load MOMENT in `reconstruction` mode
        model_kwargs={
            'task_name': 'reconstruction',
            'freeze_encoder': False, # Freeze the patch embedding layer
            'freeze_embedder': False, # Freeze the transformer encoder
            'freeze_head': False, # The linear forecasting head must be trained
            'prefix-tuning': True,
            'd_model': 768
            }
    ).to(DEVICE)
    model.init()

    model, prefix_tokens = get_prefix(model)

    batch_size = 50

    train_dataset_impute = InformerDataset(data_split='train', random_seed=RANDOM_SEED,
                                            task_name='imputation',
                                            # data_stride_len=1
                                            data_stride_len=512
                                            )
    test_dataset_impute = InformerDataset(data_split='test', random_seed=RANDOM_SEED,
                                            task_name='imputation',
                                            # data_stride_len=1
                                            data_stride_len=512
                                            )


    train_loader = DataLoader(train_dataset_impute, batch_size=batch_size,
                              shuffle=False, )

    test_loader = DataLoader(test_dataset_impute, batch_size=batch_size,
                             shuffle=False, )


    model = train(model, train_loader, test_loader,
                  prefix_tokens
                  )





