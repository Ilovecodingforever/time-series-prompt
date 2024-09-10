


strings = [
"Epoch 0: test loss: 1.084, acc: 0.678, auc: 0.712, f1: 0.544, auprc: 0.223",
"Epoch 0: test loss: 1.069, acc: 0.627, auc: 0.723, f1: 0.532, auprc: 0.257",
"Epoch 0: test loss: 1.107, acc: 0.590, auc: 0.693, f1: 0.501, auprc: 0.229",
"Epoch 0: test loss: 1.105, acc: 0.519, auc: 0.712, f1: 0.461, auprc: 0.244",
"Epoch 0: test loss: 1.076, acc: 0.720, auc: 0.717, f1: 0.582, auprc: 0.236",

"Epoch 0: test loss: 1.084, acc: 0.584, auc: 0.717, f1: 0.501, auprc: 0.253",
"Epoch 0: test loss: 1.074, acc: 0.670, auc: 0.731, f1: 0.560, auprc: 0.254",
"Epoch 0: test loss: 1.069, acc: 0.617, auc: 0.726, f1: 0.523, auprc: 0.240",
"Epoch 0: test loss: 1.067, acc: 0.689, auc: 0.728, f1: 0.568, auprc: 0.254",
"Epoch 0: test loss: 1.138, acc: 0.428, auc: 0.708, f1: 0.397, auprc: 0.236"
]

with open("Output.csv", "w") as text_file:
    text_file.write("mse, mae, loss\n")
    for s in strings:
        # get the numbers from the string
        numbers = [x.split(' ')[-1] for x in s.split(',')]

        # write the numbers to the file
        text_file.write(", ".join(map(str, numbers)) + "\n")











import os
os.environ["HF_HOME"] = "/home/scratch/mingzhul/.cache/huggingface"

from ts_datasets import get_dataset_config_names, load_dataset
import huggingface_hub


# from src.data import AnomalyDetectionDatasetMultiFile



from momentfm.utils.data import load_from_tsfile, convert_tsf_to_dataframe



path = "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/"

# list files recursively
for root, dirs, files in os.walk(path):
    for file in files:
        x, y, meta = load_from_tsfile(os.path.join(root, file), return_meta_data=True)
        if meta['missing'] or meta['timestamps']:
            print(file)







# huggingface_hub.list_files_info("AutonLab/Timeseries-PILE")


# dataset1 = load_dataset("AutonLab/Timeseries-PILE", data_files="classification/UCR/ACSF1/ACSF1_TEST.ts",)
dataset = load_dataset("AutonLab/Timeseries-PILE", data_files="anomaly_detection/TSB-UAD-Public/Daphnet/S01R02E0.test.csv@1.out",)
# dataset = load_dataset("AutonLab/Timeseries-PILE", data_files="forecasting/autoformer/ETTh1.csv",)




# configs = get_dataset_config_names("AutonLab/Timeseries-PILE")


# from huggingface_hub import HfFileSystem

# fs = HfFileSystem()
# fs.ls("AutonLab/Timeseries-PILE")

