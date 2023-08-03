from data_setup import create_dataloaders
from engine import train
from model_builder import TinyVGG
from utils import save_model, plot_curves, simple_transform
from torch import nn, optim, cuda
from torchvision import transforms
import argparse
from os import cpu_count

# training and testing directories
TRAIN_DIR = "./data/landscape-recognition-image-dataset-12k-images/seg_train/seg_train"
TEST_DIR = "./data/landscape-recognition-image-dataset-12k-images/seg_test/seg_test"

# set hyperparameter
EPOCHS = 1 # epochs to train for
LEARNING_RATE = 0.01 # optimizer learning rate
BATCH_SIZE = 32 # dataloader batchsize
NUM_WORKERS = cpu_count() # dataloader number of workers
HIDDEN_UNITS = 10 #hidden units for the model

# device agnostic code
DEVICE = "cuda" if cuda.is_available() else "cpu"

if __name__ == "__main__":
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,
                        help="number of epochs to train the model")
    parser.add_argument("--learning_rate", type=float,
                        help="optimizer learning rate")
    parser.add_argument("--batch_size", type=int,
                        help="dataloader batchsize")
    parser.add_argument("--num_workers", type=int,
                        help="dataloader number of workers")
    parser.add_argument("--hidden_units", type=int,
                        help="hidden units for the model")
    args = parser.parse_args()
    if args.epochs:
        EPOCHS = args.epochs
    if args.learning_rate:
        LEARNING_RATE = args.learning_rate
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    if args.num_workers:
        NUM_WORKERS = args.num_workers
    if args.hidden_units:
        HIDDEN_UNITS = args.hidden_units

    # print(EPOCHS, LEARNING_RATE, BATCH_SIZE, NUM_WORKERS, HIDDEN_UNITS)
    # simple image transformation
    simple_transform = simple_transform()
    
    # loading data
    train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=TRAIN_DIR,
                                                           test_dir=TEST_DIR,
                                                           transform=simple_transform,
                                                           batch_size=BATCH_SIZE,
                                                           num_workers=NUM_WORKERS)
    
    # loading model, optimizer and loss function
    model = TinyVGG(in_shape=3, hidden_unit=HIDDEN_UNITS, out_shape=len(class_names))
    optimizer = optim.SGD(params=model.parameters(),
                                lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    # train
    print(f"[INFO] training model")
    result = train(model=model,
                   train_dl=train_dataloader,
                   test_dl=test_dataloader,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   device=DEVICE,
                   epochs=EPOCHS)
    
    # save model
    print(f"[INFO] plotting loss")
    save_model(model=model, target_dir="./model", model_name="landscape_classification.pth")

    # plot loss and accuracy curves
    plot_curves(result)