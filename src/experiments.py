"""
load experiments
"""

import torch
import wandb

from momentfm import MOMENTPipeline

from train import train
from ts_datasets import InformerDataset, ClassificationDataset

import sys
sys.path.append('/zfsauton2/home/mingzhul/time-series-prompt/moment-research')
from moment.utils.config import Config
from moment.utils.utils import control_randomness, parse_config
from moment.models.gpt4ts_prompt import GPT4TS_prompt
from moment.models.gpt4ts import GPT4TS
from moment.models.timesnet import TimesNet

from typing import Optional



def load_gpt4ts(task_name: str, forecast_horizon: int,
                n_channels: int, num_classes: int,
                train_loader: torch.utils.data.DataLoader,
                num_prefix: Optional[int] = None,
                multivariate_projection: Optional[str] = None, agg: Optional[str] = None):

        if task_name == 'classification':
            config_path = "/zfsauton2/home/mingzhul/time-series-prompt/moment-research/configs/prompt/gpt4ts_classification.yaml"
        elif task_name == 'forecasting':
            config_path = "/zfsauton2/home/mingzhul/time-series-prompt/moment-research/configs/forecasting/gpt4ts_long_horizon.yaml"
        else:
            raise ValueError('task_name must be classification or forecasting')

        gpu_id = 0
        random_seed = 0

        config = Config(
            config_file_path=config_path, default_config_file_path="/zfsauton2/home/mingzhul/time-series-prompt/moment-research/configs/default.yaml"
        ).parse()

        config["device"] = gpu_id if torch.cuda.is_available() else "cpu"
        args = parse_config(config)
        args.shuffle = False
        args.finetuning_mode = "end-to-end"

        if multivariate_projection is not None:
            args.model_name = "GPT4TS_prompt"
            model = GPT4TS_prompt
        else:
            args.model_name = "GPT4TS"
            model = GPT4TS

        args.n_channels = n_channels
        args.num_prefix = num_prefix
        args.num_class = num_classes
        args.seq_len = train_loader.dataset.seq_len
        args.multivariate_projection = multivariate_projection
        args.agg = agg
        args.forecast_horizon = forecast_horizon

        model = model(configs=args)

        return model


def load_timesnet(task_name: str, forecast_horizon: int,
                n_channels: int, num_classes: int,
                train_loader: torch.utils.data.DataLoader):

    if task_name == 'classification':
        config_path = "moment-research/configs/classification/timesnet.yaml"
    elif task_name == 'forecasting':
        config_path = "moment-research/configs/forecasting/timesnet_long_horizon.yaml"
    else:
        raise ValueError('task_name must be classification or forecasting')

    gpu_id = 0
    random_seed = 0

    config = Config(
        config_file_path=config_path, default_config_file_path="moment-research/configs/default.yaml"
    ).parse()

    config["device"] = gpu_id if torch.cuda.is_available() else "cpu"
    args = parse_config(config)
    args.shuffle = False
    args.finetuning_mode = "end-to-end"

    args.n_channels = n_channels
    args.num_class = num_classes
    args.seq_len = train_loader.dataset.seq_len
    args.forecast_horizon = forecast_horizon

    args.d_model = 16
    args.d_ff = 16

    model = TimesNet(configs=args)

    return model



def load_patchtst(task_name: str, forecast_horizon: int,
                    n_channels: int, num_classes: int,
                    train_loader: torch.utils.data.DataLoader):


    from moment.models.patchtst import MOMENT

    assert task_name == 'forecasting'
    config_path = "moment-research/configs/forecasting/linear_probing.yaml"


    gpu_id = 0
    random_seed = 0

    config = Config(
        config_file_path=config_path, default_config_file_path="moment-research/configs/default.yaml"
    ).parse()

    config["device"] = gpu_id if torch.cuda.is_available() else "cpu"
    args = parse_config(config)
    args.shuffle = False
    args.finetuning_mode = "end-to-end"

    args.n_channels = n_channels
    args.num_class = num_classes
    args.seq_len = train_loader.dataset.seq_len
    args.forecast_horizon = forecast_horizon

    args.transformer_backbone = 'PatchTST'

    model = MOMENT(configs=args)

    return model







def prompt_tuning(train_loader: torch.utils.data.DataLoader,
                  val_loader: torch.utils.data.DataLoader,
                  test_loader: torch.utils.data.DataLoader,
                  model_name: str, task_name: str,
                  multivariate_projection: str,
                  agg: str,
                  num_prefix: int = 16,
                  name: str = '', extra: str = '',
                  epochs: int = 10,
                  save_model: bool = False,
                  forecast_horizon: int = 0,
                  flatten: bool = False,):
    """
    prompt tuning
    """

    wandb.init(
        project="ts-prompt",
        name=name,
    )

    num_classes = 1
    n_channels = train_loader.dataset.n_channels
    assert task_name in ['classification', 'forecasting']
    if task_name == 'classification':
        num_classes = train_loader.dataset.num_classes


    if model_name == 'moment':
        # imputation model
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                'task_name': task_name,
                'seq_len': train_loader.dataset.seq_len,
                'freeze_encoder': False, # Freeze the patch embedding layer
                'freeze_embedder': False, # Freeze the transformer encoder
                'freeze_head': False, # The linear forecasting head must be trained
                'prefix_tuning_multi': True,
                'forecast_horizon': forecast_horizon,
                'num_prefix': num_prefix,
                'multivariate_projection': multivariate_projection,
                'agg': agg,
                'n_channels': n_channels,
                'num_class': num_classes,
                }
        )
        model.init()

        if 'finetune' not in name:
            # need to freeze head manually
            for n, param in model.named_parameters():
                if 'prefix' not in n and 'prompt' not in n and 'head' not in n and 'value_embedding' not in n and 'layer_norm' not in n:
                    param.requires_grad = False

    elif model_name == 'gpt4ts':
        model = load_gpt4ts(task_name, forecast_horizon, n_channels, num_classes, train_loader, num_prefix, multivariate_projection, agg)

    # print frozen params
    for n, param in model.named_parameters():
        if param.requires_grad:
            print(n)

    print('number of parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))


    import time
    start_time = time.time()

    model = train(model, train_loader, val_loader, test_loader, max_epoch=epochs, identifier=name,
                  extra=extra)

    print("--- %s seconds ---" % (time.time() - start_time))


    if save_model:
        try:
            model.push_to_hub(name.replace('/', '_'))
        except Exception as e:
            print(e)

    wandb.finish()

    return model





def finetune(train_loader: torch.utils.data.DataLoader,
             val_loader: torch.utils.data.DataLoader,
             test_loader: torch.utils.data.DataLoader,
             model_name: str, task_name: str,
             lora: bool = False, linearprobe: bool = False,
             name: str = '', extra: str = '',
             epochs: int = 10,
             save_model: bool = False,
             forecast_horizon: int = 0,
             flatten: bool = False,
             **kwargs):
    """
    finetune
    """
    wandb.init(
        project="ts-prompt",
        name=name,
    )

    num_classes = 1
    n_channels = train_loader.dataset.n_channels
    assert task_name in ['classification', 'forecasting']
    if task_name == 'classification':
        num_classes = train_loader.dataset.num_classes

    if model_name == 'moment':
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                'task_name': task_name,
                'seq_len': train_loader.dataset.seq_len,
                'freeze_encoder': False, # Freeze the patch embedding layer
                'freeze_embedder': False, # Freeze the transformer encoder
                'freeze_head': False, # The linear forecasting head must be trained
                'forecast_horizon': forecast_horizon,
                'num_prefix': 16,
                'n_channels': n_channels,
                'num_class': num_classes,
                }
        )
        model.init()

        for param in model.parameters():
            param.requires_grad = True

        if lora:
            from peft import LoraConfig, get_peft_model
            config = LoraConfig(
                r=2,
                lora_alpha=16,
                target_modules=["q", "v"], # https://github.com/huggingface/peft/blob/39ef2546d5d9b8f5f8a7016ec10657887a867041/src/peft/utils/other.py#L220
                lora_dropout=0.1,
                # bias="none",
                modules_to_save=["value_embedding", "layer_norm", "head"],
            )
            model = get_peft_model(model, config)

        if linearprobe:
            for n, param in model.named_parameters():
                if 'head' not in n:
                    param.requires_grad = False

    elif model_name == 'gpt4ts':
        model = load_gpt4ts(task_name, forecast_horizon, n_channels, num_classes, train_loader)

        if lora:
            from peft import LoraConfig, get_peft_model

            config = LoraConfig(
                r=1,
                lora_alpha=16,
                lora_dropout=0.1,
                # bias="none",
                modules_to_save=["wpe", "enc_embedding", "ln", "predict_linear", "out_layer"],
            )
            model.gpt2 = get_peft_model(model.gpt2, config)

        if linearprobe:
            for n, param in model.named_parameters():
                if 'predict_linear' not in n and 'out_layer' not in n and "enc_embedding" not in n:
                    param.requires_grad = False


    # print frozen params
    for n, param in model.named_parameters():
        if param.requires_grad:
            print(n)

    print('number of parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))


    import time
    start_time = time.time()

    model = train(model, train_loader, val_loader, test_loader, max_epoch=epochs, identifier=name,
                  extra=extra)

    print("--- %s seconds ---" % (time.time() - start_time))


    if save_model:
        try:
            model.push_to_hub(name.replace('/', '_'))
        except Exception as e:
            print(e)

    wandb.finish()

    return model





def lora(train_loader: torch.utils.data.DataLoader,
             val_loader: torch.utils.data.DataLoader,
             test_loader: torch.utils.data.DataLoader,
             model_name: str, task_name: str,
             name: str = '', extra: str = '',
             epochs: int = 10, save_model: bool = False,
             forecast_horizon: int = 0,
             flatten: bool = False,
             **kwargs):

    return finetune(train_loader, val_loader, test_loader, model_name, task_name,
                    name=name, extra=extra,
                    epochs=epochs, save_model=save_model, lora=True, linearprobe=False,
                    forecast_horizon=forecast_horizon, flatten=flatten,
                    **kwargs)



def linearprobe(train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                model_name: str, task_name: str,
                name: str = '', extra: str = '',
                epochs: int = 10, save_model: bool = False,
                forecast_horizon: int = 0,
                flatten: bool = False,
                **kwargs):

    return finetune(train_loader, val_loader, test_loader, model_name, task_name,
                    name=name, extra=extra,
                    epochs=epochs, save_model=save_model, lora=False, linearprobe=True,
                    forecast_horizon=forecast_horizon, flatten=flatten,
                    **kwargs)


