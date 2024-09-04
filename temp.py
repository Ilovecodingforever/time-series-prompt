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

