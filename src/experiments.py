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


def zero_shot(train_loader, val_loader, test_loader, name='', extra='', **kwargs):
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

    train_bs = 0
    for data in train_loader:
        train_bs += 1
    train_loader.bs = train_bs

    test_bs = 0
    for data in test_loader:
        test_bs += 1
    test_loader.bs = test_bs



    model = inference(model, test_loader,
                      torch.nn.MSELoss(reduction='none').to(DEVICE),
                      'zero-shot',
                      Masking(mask_ratio=0.3),
                      train_loader,
                      'test',
                      log=False)

    return model



def prompt_tuning(train_loader, val_loader, test_loader, 
                  num_prefix=16, name='', extra='',
                  prefix_tuning_multi=False,
                  MPT=False,
                  no_train_forehead=False,
                  epochs=400,
                  multivariate_projection='attention',
                  save_model=False,
                  forecast_horizon=96,
                  mask_ratio=0.3,):
    """
    prompt tuning
    """

    wandb.init(
        project="ts-prompt",
        name=name,
    )


    n_channels = 1

    n_channels = next(iter(train_loader.dataset.datasets['forecasting_long']))[0].shape[0]

    num_classes = 1
    if 'classify' in train_loader.dataset.datasets:
        next(iter(train_loader))
        n_channels = train_loader.dataset.datasets['classify'].n_channels # TODO: this is not elegant
        num_classes = train_loader.dataset.datasets['classify'].num_classes

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
            'forecast_horizons': (forecast_horizon, 8),
            # 'prefix_tuning': True,
            'prefix_tuning_multi': prefix_tuning_multi,
            'MPT': MPT,
            'num_prefix': num_prefix,
            'task_names': list(next(iter(train_loader)).keys()),
            'multivariate_projection': multivariate_projection,
            'n_channels': n_channels,
            'num_class': num_classes,
            }
    )
    model.init()

    if not 'finetune' in name:
        # need to freeze head manually
        for n, param in model.named_parameters():
            if 'prefix' not in n and 'prompt' not in n and 'head' not in n and 'mpt' not in n and 'value_embedding' not in n and 'layer_norm' not in n:
                param.requires_grad = False

    if no_train_forehead:
        for param in model.fore_head.parameters():
            param.requires_grad = False

    # print frozen params
    for n, param in model.named_parameters():
        if param.requires_grad:
            print(n)

    print('number of parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = train(model, train_loader, val_loader, test_loader, max_epoch=epochs, identifier=name,
                  extra=extra, mask_ratio=mask_ratio)

    if save_model:
        try:
            model.push_to_hub(name.replace('/', '_'))
        except Exception as e:
            print(e)

    wandb.finish()

    return model





def finetune(train_loader, val_loader, test_loader, name='', extra='',
             epochs=400, save_model=False,
             lora=False, linearprobe=False,
             forecast_horizon=96,
             mask_ratio=0.3,
             **kwargs):
    """
    finetune
    """
    wandb.init(
        project="ts-prompt",
        name=name,
    )

    n_channels = 1
    num_classes = 1
    if 'classify' in train_loader.dataset.datasets:
        next(iter(train_loader))
        n_channels = train_loader.dataset.datasets['classify'].n_channels # TODO: this is not elegant
        num_classes = train_loader.dataset.datasets['classify'].num_classes

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
            'forecast_horizons': (forecast_horizon, 8),
            # 'forecast_horizon': 96,
            # 'prefix_tuning': False,
            # 'prefix_tuning_multi': False,
            # 'MPT': True,
            'num_prefix': 16,
            'task_names': list(next(iter(train_loader)).keys()),
            'n_channels': n_channels,
            'num_class': num_classes,
            }
    )

    model.init()

    for param in model.parameters():
        param.requires_grad = True

    # # linear probe
    # for n, param in model.named_parameters():
    #     if 'prefix' not in n and 'prompt' not in n and 'head' not in n and 'mpt' not in n and 'value_embedding' not in n and 'layer_norm' not in n:
    #         param.requires_grad = False

    if lora:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["q", "v"], # https://github.com/huggingface/peft/blob/39ef2546d5d9b8f5f8a7016ec10657887a867041/src/peft/utils/other.py#L220
            lora_dropout=0.1,
            # bias="none",
            modules_to_save=["value_embedding", "layer_norm", "fore_head_long", "classification_head"],
        )
        model = get_peft_model(model, config)
    
    if linearprobe:
        for n, param in model.named_parameters():
            if 'head' not in n:
                param.requires_grad = False


    # print frozen params
    for n, param in model.named_parameters():
        if param.requires_grad:
            print(n)

    print('number of parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = train(model, train_loader, val_loader, test_loader, max_epoch=epochs, identifier=name,
                  extra=extra, mask_ratio=mask_ratio)

    if save_model:
        try:
            model.push_to_hub(name.replace('/', '_'))
        except Exception as e:
            print(e)

    wandb.finish()

    return model





def lora(train_loader, val_loader, test_loader, name='', extra='', epochs=400, save_model=False, forecast_horizon=96,
             **kwargs):


    return finetune(train_loader, val_loader, test_loader, name=name, extra=extra,
                    epochs=epochs, save_model=save_model, lora=True, linearprobe=False,
                    forecast_horizon=forecast_horizon, **kwargs)



def linearprobe(train_loader, val_loader, test_loader, name='', extra='', epochs=400, save_model=False, forecast_horizon=96,
             **kwargs):


    return finetune(train_loader, val_loader, test_loader, name=name, extra=extra,
                    epochs=epochs, save_model=save_model, lora=False, linearprobe=True,
                    forecast_horizon=forecast_horizon, **kwargs)


