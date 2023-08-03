from model_builder import TinyVGG
from torch import nn, inference_mode, cuda, softmax
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from utils import load_model, simple_transform

# Model directory
MODEL_DIR = "./model"
MODEL_NAME = "landscape_classification.pth"
CLASS_IDX = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# hyper-parameters
HIDDEN_UNIT = 10 # same as saved model!
LABEL = 6 # same as saved model!

# device agnostic code
DEVICE = "cuda" if cuda.is_available() else "cpu"

def get_model():
    # create model instance
    model_0 = TinyVGG(in_shape=3,
                    hidden_unit=HIDDEN_UNIT,
                    out_shape=LABEL)
    # load model with saved weights
    model_0.load_state_dict(load_model(target_dir=MODEL_DIR,
                                            model_name=MODEL_NAME))
    # print(f"model type {type(model_0)}")
    return model_0

def get_prediction(model: nn.Module, image_file: str):
    # load
    img = Image.open(image_file)
    # image transformation
    img = simple_transform()(img=img)

    # model prediction
    model.eval()
    model.to(device=DEVICE)
    with inference_mode():
        y_logit = model(img.unsqueeze(dim=0).to(DEVICE))
        y_pred = y_logit.argmax(dim=1).item()
        confidence = softmax(y_logit, dim=1).max(dim=1)[0].item()
    
    # plot prediction
    plt.imshow(img.permute(1,2,0))
    plt.title(f"{CLASS_IDX[y_pred]} | {confidence:.2f}", fontsize=12, c='g')
    plt.axis(False)
    plt.show()

if __name__ == "__main__":
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str,
                        required=True,
                        help="Path of the image to be predicted")
    parser.add_argument("--model_name", type=str,
                        help="Model Name")
    args = parser.parse_args()
    
    IMAGE_DIR = args.image_path
    
    if args.model_name:
        MODEL_NAME = args.model_name

    model = get_model() # create model
    get_prediction(model=model, image_file=IMAGE_DIR) # get prediction

