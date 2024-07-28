import sys
from Time_Series_Library.models.TimesNet import Model
# import importlib
# ts_lib = importlib.import_module("Time-Series-Library")



if __name__ == '__main__':
    config = {
        'task_name': 'long_term_forecast',
        'pred_len': 96,
        'num_classes': 2,
        'freq': 'h',
        # same as moment
        'seq_len': 512,
        'd_model': 1024,
        # Time-Series-Library/scripts/classification/TimesNet.sh, first one
        'top_k': 3,
        'd_ff': 32,
        'e_layers': 3,
        # default
        'label_len': 48,
        'num_kernels': 6,
        'enc_in': 7,
        'embed': 'timeF',
        'dropout': 0.1,
        'c_out': 7,
    }


    model = Model(config)


