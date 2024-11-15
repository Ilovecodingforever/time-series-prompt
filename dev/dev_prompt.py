"""
add a learnable prompt, to the input time series
freeze all layers, do representation learning: https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/representation_learning.ipynb

goal: one model, prompt tuning on different tasks, all perform well: https://huggingface.co/docs/peft/en/conceptual_guides/prompting



sotries:
1. prompt tunine as a bridge:
    - different tasks
    - different variables






huggingface: https://huggingface.co/docs/peft/main/en/task_guides/clm-prompt-tuning
custom prompt:
- https://github.com/mkshing/Prompt-Tuning/tree/master
- https://github.com/kipgparker/soft-prompt-tuning/blob/main/soft_embedding.py


anomaly detection: 0 shot, reconstruction
imputation: 0 shot, reconstruction
classification: 0 shot or svm, classification
forcasting: train head, forecasting

-- if can use prompt for the 0 shot tasks, there is no need to fine tune the model at all. Just learn a prompt.


tokenizer: patching -- 3D to 4D
embedding: value_embedding + position embedding
-- TODO: should I add position embedding to prompt?




tasks:
1. for each task, train a model, do prompt tuning, see if perform well
    - it doesn't. Because the prediction head is not trained.
    - so only do reconstruction tasks
        - does reparametrization. now works better than 0 shot
2. Story 2: train one model for all tasks. use different prompts.
    - Multi-task Prompt Tuning



ideas:
1. check attention mask
2. add prompt instead of concatenate





"""


import os

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
from momentfm.utils.forecasting_metrics import get_forecasting_metrics


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/home/scratch/mingzhul/.cache/huggingface"



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 0





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

    # TODO: the tutourial is wrong
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

            if model.task_name == TASKS.RECONSTRUCTION:
                if isinstance(train_loader.dataset, AnomalyDetectionDataset):
                    batch_x, batch_masks, batch_labels = data
                else:
                    batch_x, batch_masks = data

                n_channels = batch_x.shape[1]

                # Reshape to [batch_size * n_channels, 1, window_size]
                batch_x = batch_x.reshape((-1, 1, 512)).to(DEVICE).float()

                batch_masks = batch_masks.to(DEVICE).long()
                batch_masks = batch_masks.repeat_interleave(n_channels, axis=0)

                # Randomly mask some patches of data
                # 0 is masked
                mask = mask_generator.generate_mask(
                    x=batch_x, input_mask=batch_masks).to(DEVICE).long()

                # TODO: initialize the task name
                if isinstance(model.patch_embedding.value_embedding, MPT):
                    model.patch_embedding.value_embedding.task_name = 'imputation'

                # Forward
                output = model(batch_x, input_mask=batch_masks, mask=mask)

                # Compute loss
                recon_loss = criterion(output.reconstruction, batch_x)
                if isinstance(train_loader.dataset, AnomalyDetectionDataset):
                    loss = recon_loss
                else:
                    observed_mask = (batch_masks * (1 - mask))[:, None, :]
                    assert observed_mask.shape == recon_loss.shape
                    masked_loss = observed_mask * recon_loss
                    loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)


            else:
                raise NotImplementedError


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
        model = inference(model, test_loader, criterion, cur_epoch)


    return model




def inference(model, test_loader, criterion, cur_epoch):
    """
    perform inference
    """
    mask_generator = Masking(mask_ratio=0.3)

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
                batch_x = batch_x.reshape((-1, 1, 512)).to(DEVICE).float()

                batch_masks = batch_masks.to(DEVICE).long()
                batch_masks = batch_masks.repeat_interleave(n_channels, axis=0)

                # Randomly mask some patches of data
                mask = mask_generator.generate_mask(
                    x=batch_x, input_mask=batch_masks).to(DEVICE).long()

                # TODO: initialize the task name
                if isinstance(model.patch_embedding.value_embedding, MPT):
                    model.patch_embedding.value_embedding.task_name = 'imputation'

                # Forward
                output = model(batch_x, input_mask=batch_masks, mask=mask).reconstruction

                # Compute loss
                recon_loss = criterion(output, batch_x)

                if isinstance(test_loader.dataset, AnomalyDetectionDataset):
                    loss = recon_loss
                else:
                    observed_mask = (batch_masks * (1 - mask))[:, None, :]
                    assert observed_mask.shape == recon_loss.shape
                    masked_loss = observed_mask * recon_loss
                    loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)

                y = batch_x


            else:
                raise NotImplementedError


            losses.append(loss.item())
            trues.append(y.detach().cpu().numpy())
            preds.append(output.detach().cpu().numpy())

    losses = np.array(losses)
    average_loss = np.average(losses)
    model.train()

    print(f"Epoch {cur_epoch}: Test loss: {average_loss:.3f}")

    trues = np.concatenate(trues, axis=0)
    preds = np.concatenate(preds, axis=0)
    # metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction='mean')
    # print(f"Epoch {cur_epoch}: Test MSE: {metrics.mse:.3f} | Test MAE: {metrics.mae:.3f}")

    return model




class SoftEmbedding(nn.Module):
    """
    reimplement nn.Embedding, with prompt tuning
    """
    def __init__(self,
                # wte: nn.Embedding,
                wte: nn.Module,
                n_tokens: int = 10,
                random_range: float = 0.5,
                hidden_size: int = 64):
        """appends learned embedding to

        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens

        self.hidden_size = hidden_size

        size = wte.weight.size(0)
        # reparametraize: https://arxiv.org/pdf/2101.00190
        self.mlp = nn.Linear(self.hidden_size, size)

        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(self.hidden_size,
                                                                                n_tokens,
                                                                                random_range))


    def initialize_embedding(self,
                            #  wte: nn.Embedding,
                            size: int,
                            #  wte: nn.Module,
                             n_tokens: int = 10,
                             random_range: float = 0.5, ):
        """initializes learned embedding

        Args:
            same as __init__

        Returns:
            torch.float: initialized using original schemes
        """
        # return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
        return torch.FloatTensor(n_tokens, size).uniform_(-random_range, random_range)


    def forward(self, tokens):
        """run forward pass

        Args:
            tokens (torch.long): input tokens before encoding

        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        # input_embedding = self.wte(tokens[:, self.n_tokens:])
        input_embedding = self.wte(tokens)
        learned_embedding = self.mlp(self.learned_embedding.repeat(input_embedding.size(0), input_embedding.size(1), 1, 1))
        # n_batches x features x n_patches x embedding_size
        # return torch.cat([learned_embedding, input_embedding], 1)
        return torch.cat([learned_embedding, input_embedding], 2)








def zero_shot(impute_model, train_loader_impute, test_loader_impute):

    # need to freeze head manually
    for param in impute_model.head.parameters():
        param.requires_grad = False

    inference(impute_model, test_loader_impute, torch.nn.MSELoss(reduction='none').to(DEVICE), 'zero-shot')




def prompt_tuning(impute_model, train_loader_impute, test_loader_impute):

    n_tokens = 5

    # need to freeze head manually
    for param in impute_model.parameters():
        param.requires_grad = False

    # n_tokens = 10
    # random_range = 0.5
    # forecasting_model_long.prompt = nn.parameter.Parameter(torch.FloatTensor(n_tokens, 1024).uniform_(-random_range, random_range))
    # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.get_input_embeddings
    setattr(impute_model.patch_embedding, 'value_embedding', SoftEmbedding(impute_model.patch_embedding.value_embedding,
                                                                    n_tokens=n_tokens,))
    
    # impute_model.patch_embedding.value_embedding = SoftEmbedding(impute_model.patch_embedding.value_embedding,
    #                                                                 n_tokens=n_tokens,)

    # print frozen params
    for name, param in impute_model.named_parameters():
        if param.requires_grad:
            print(name)

    train(impute_model, train_loader_impute, test_loader_impute)



def full_fine_tune(impute_model, train_loader_impute, test_loader_impute):

    for param in impute_model.parameters():
        param.requires_grad = True

    # print frozen params
    for name, param in impute_model.named_parameters():
        if param.requires_grad:
            print(name)

    train(impute_model, train_loader_impute, test_loader_impute)






def run_single_prompt():

    impute_model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        # For imputation, we will load MOMENT in `reconstruction` mode
        model_kwargs={
            'task_name': 'reconstruction',
            'freeze_encoder': True, # Freeze the patch embedding layer
            'freeze_embedder': True, # Freeze the transformer encoder
            'freeze_head': True, # The linear forecasting head must be trained
            }
    ).to(DEVICE)

    impute_model.init()

    # run experiment
    experiment = prompt_tuning

    batch_size = 50
    if experiment == full_fine_tune:
        batch_size = 20

    train_dataset = InformerDataset(data_split='train', random_seed=RANDOM_SEED,
                                    task_name='imputation',
                                    # data_stride_len=512
                                    data_stride_len=1
                                    )
    train_loader_impute = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = InformerDataset(data_split='test', random_seed=RANDOM_SEED,
                                   task_name='imputation',
                                #    data_stride_len=512
                                   data_stride_len=1
                                   )
    test_loader_impute = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    experiment(impute_model, train_loader_impute, test_loader_impute)











if __name__ == '__main__':

    # Set random seeds for PyTorch, Numpy etc.
    control_randomness(seed=RANDOM_SEED)

    run_single_prompt()

