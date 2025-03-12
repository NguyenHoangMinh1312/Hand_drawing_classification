"""Do the inference of the hand drawing classification model"""
from argparse import ArgumentParser
import torch
from torchvision.models import resnet18
import torch.nn as nn
import cv2


def get_args():
    parser = ArgumentParser(description = "Hand drawing classification")

    parser.add_argument("--model_path", "-m", type = str, default = "./hand_drawing_model/checkpoint/best.pt", help = "path to the model")
    parser.add_argument("--image_path", "-i", type = str, default = "./hand_drawing_model/test_img/alarm.png", help = "path to the image")

    args = parser.parse_args()
    return args

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_data = torch.load(args.model_path, weights_only = False)
    categories = saved_data["categories"]
    
    #set up model
    model = resnet18()
    model.fc = nn.Linear(in_features = model.fc.in_features,
                         out_features = len(categories),
                         bias = True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias = False)
    model.load_state_dict(saved_data["model"])
    model.to(device)

    #preprocess the image
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224, 224))
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)    #change to dark background and white foreground

    image = image / 255
    mean = 0.1800765688410995
    std = 0.30052850582901786
    image = (image - mean) / std
    image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

    #inference
    softmax = nn.Softmax()
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        probs = softmax(output)
        
        prediction = torch.argmax(probs, dim = 1).item()
        prob = torch.max(probs).item()

        cv2.imshow(f"{categories[prediction]}: {prob * 100:.2f}", cv2.imread(args.image_path))
        cv2.waitKey(0)
    
if __name__ == "__main__":
    args = get_args()
    inference(args)


