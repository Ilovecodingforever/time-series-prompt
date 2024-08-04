"""
load experiments
"""

import torch
import wandb

from momentfm import MOMENTPipeline
# from momentfm.models.moment_original import MOMENTPipeline
from momentfm.utils.masking import Masking

from main import DEVICE
from train import train, inference


def zero_shot(train_loader, test_loader, name='', **kwargs):
    """
    zero shot
    TODO: cannot do this with forecasting, because head not pretrained
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
            'forecast_horizons': (96, 8),
            # 'prefix_tuning': False,
            # 'prefix_tuning_multi': False,
            # 'MPT': True,
            'num_prefix': 2,
            'task_names': list(next(iter(train_loader)).keys()),
            }
    ).to(DEVICE)
    model.init()

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
                      Masking(mask_ratio=0.3),
                      train_loader,
                      log=False)

    return model



def prompt_tuning(train_loader, test_loader, name='',
                  prefix_tuning_multi=False,
                  MPT=False,
                  no_train_forehead=False,
                  epochs=400,
                  multivariate_projection='attention'):
    """
    prompt tuning
    """

    wandb.init(
        project="ts-prompt",
        name=name,
    )

    # imputation model
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        # output_loading_info=True,
        # For imputation, we will load MOMENT in `reconstruction` mode
        model_kwargs={
            'task_name': 'reconstruction',
            'freeze_encoder': False, # Freeze the patch embedding layer
            'freeze_embedder': False, # Freeze the transformer encoder
            'freeze_head': False, # The linear forecasting head must be trained
            'forecast_horizons': (96, 8),
            # 'prefix_tuning': True,
            'prefix_tuning_multi': prefix_tuning_multi,
            'MPT': MPT,
            'num_prefix': 16,
            'task_names': list(next(iter(train_loader)).keys()),
            'multivariate_projection': multivariate_projection,
            }
    )
    model.init()



    # need to freeze head manually
    for n, param in model.named_parameters():
        if 'prefix' not in n and 'prompt' not in n and 'fore_head' not in n and 'mpt' not in n:
            param.requires_grad = False

    if no_train_forehead:
        for param in model.fore_head.parameters():
            param.requires_grad = False

    # print frozen params
    for n, param in model.named_parameters():
        if param.requires_grad:
            print(n)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = train(model, train_loader, test_loader, max_epoch=epochs, identifier=name)

    try:
        model.push_to_hub(name.replace('/', '_'))
    except Exception as e:
        print(e)

    wandb.finish()

    return model





def finetune(train_loader, test_loader, name='', epochs=400, **kwargs):
    """
    finetune
    """
    wandb.init(
        project="ts-prompt",
        name=name,
    )

    # imputation model
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        # For imputation, we will load MOMENT in `reconstruction` mode
        model_kwargs={
            'task_name': 'reconstruction',
            # 'task_name': 'forecasting',
            'freeze_encoder': False, # Freeze the patch embedding layer
            'freeze_embedder': False, # Freeze the transformer encoder
            'freeze_head': False, # The linear forecasting head must be trained
            'forecast_horizons': (96, 8),
            # 'forecast_horizon': 96,
            # 'prefix_tuning': False,
            # 'prefix_tuning_multi': False,
            # 'MPT': True,
            'num_prefix': 2,
            'task_names': list(next(iter(train_loader)).keys()),
            }
    )
    model.init()


    for param in model.parameters():
        param.requires_grad = True

    # print frozen params
    for n, param in model.named_parameters():
        if param.requires_grad:
            print(n)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = train(model, train_loader, test_loader, max_epoch=epochs, identifier=name)

    try:
        model.push_to_hub(name.replace('/', '_'))
    except Exception as e:
        print(e)

    wandb.finish()

    return model
