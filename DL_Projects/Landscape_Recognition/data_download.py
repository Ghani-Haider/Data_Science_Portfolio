import argparse
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
from zipfile import ZipFile
"""
A module to download a kaggle dataset.
Use: python3 data_download.py --dataset avenn98/world-of-warcraft-demographics
"""

# create and download the dataset folder
def download_landscapedata(kaggle_dataset: str) -> None:
    """Creates and downloads given kaggle dataset folder
    Args:
        kaggle_dataset: kaggle dataset name
    Returns:
        None
    """
    datafolder = Path("data")
    datafolder.mkdir(parents=True, exist_ok=True)
    # create the dataset folder
    dataset_name = kaggle_dataset.split("/")[1]
    dataset_folder = datafolder / dataset_name

    # folder for the dataset
    if dataset_folder.is_dir():
        print(f"[INFO] Dataset already exists!")
    else:
        print(f"[INFO] Creating dataset folder...")
        dataset_folder.mkdir(parents=True, exist_ok=True)
        
        # data download
        data_file = Path(dataset_name + ".zip")
        if data_file.is_file():
            print(f"[INFO] Dataset already downloaded!")
        else:
            print(f"[INFO] Downloading the dataset...")
            api.dataset_download_files(kaggle_dataset)
            print(f"[INFO] Downloaded successfully!")
        
        # data extract
        print(f"[INFO] Extracting the dataset...")
        with ZipFile(data_file, "r") as zipfile:
            zipfile.extractall(dataset_folder)
        print(f"[INFO] Extracted successfully!")

if __name__ == "__main__":
    # kaggle dataset name parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        help="kaggle dataset to be downloaded, ex: 'avenn98/world-of-warcraft-demographics'")
    args = parser.parse_args()
    if args.dataset is not None:
        kaggle_dataset = args.dataset
    else:
        raise argparse.ArgumentError(argument=None, message="kaggle dataset info not given")

    # authenticating with the api server
    api = KaggleApi()
    api.authenticate()
    # create and download the given dataset folder
    download_landscapedata(kaggle_dataset)