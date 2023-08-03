# Landscape Recognition

This project is an implementation of landscape recognition using deep learning with PyTorch and torchvision. The goal of this project is to train a deep neural network to classify different types of landscapes images, such as mountains, sea, forests, streets etc, based on input images. The project consists of two main scripts: `train.py` and `predict.py`.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction](#prediction)
- [Dataset](#dataset)
- [Model](#model)
- [Contributing](#contributing)
- [License](#license)

## Requirements

- Python 3.x
- PyTorch
- torchvision
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/Ghani-Haider/Data_Science_Portfolio.git
   ```

2. Navigate to the project directory:

   ```bash
   cd ./DL_Projects/Landscape_Recognition/
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the landscape recognition model, you can use the `train.py` script. This script allows you to specify the number of epochs and other hyperparameters using command line arguments.

```bash
python train.py --epochs <num_epochs> --learning_rate <learning_rate> --batch_size <batch_size> --num_workers <num_workers> --hidden_units <hidden_units>
```

For example, to train the model for 50 epochs with learning_rate 0.01:

```bash
python train.py --epochs 50 --learning_rate 0.01
```

### Prediction

After training the model, you can use the `predict.py` script to make predictions on a given image and model name.

```bash
python predict.py --image_path <image_path> --model_name <model_name>
```

Replace `<image_path>` with the path to the image you want to classify and `<model_name>` with a model present in the model directory.

## Dataset

The dataset used for training and evaluating the landscape recognition model is not included in this repository. It was the kaggle dataset landscape-recognition-image-dataset-12k-images from utkarshsaxenadn.

In order to download a kaggle dataset, use `data_download.py`. This script allows you to specify the kaggle dataset using command line arguments.

```bash
python data_download.py --dataset <dataset>
```

For example, download the kaggle dataset 'landscape-recognition-image-dataset-12k-images' by utkarshsaxenadn:

```bash
python data_download.py --dataset utkarshsaxenadn/landscape-recognition-image-dataset-12k-images
```
For the given project, your dataset should be images in separate folders for each class (where folder name is the class label) within separate train and test folders. You can then specify the dataset path (both train and test) in the `train.py` script.

## Model

The landscape recognition model architecture, training process, and evaluation metrics can be found in the `model_builder.py`, `engine.py` and `utils.py` scripts. The default model architecture used is a Convolutional Neural Network (CNN) called `TinyVGG`, but you can modify and experiment with different architectures in the script.

## Contributing

Contributions to this project are welcome. Feel free to fork this repository, make improvements, and submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).

---