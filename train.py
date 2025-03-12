"""This script is to train the hand drawing classification model"""
from dataset import HandDrawingDataset
from argparse import ArgumentParser
import torch
import os
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score

#modify the command line arguments (Eg: python3 train.py -n 10 -b 64 -l 1e-4 -o ./tensorboard -c)
def get_args():
    parser = ArgumentParser(description = "Hand drawing classification")

    parser.add_argument("--num_epochs", "-n", type = int, default = 100, help = "Number of epochs")
    parser.add_argument("--batch_size", "-b", type = int, default = 32, help = "Number of images in a batch")
    parser.add_argument("--lr", "-l", type = float, default = 1e-3, help = "learning rate")
    parser.add_argument("--log_path", "-o", type = str, default = "./hand_drawing_model/tensorboard", help = "place to save the tensorboard")
    parser.add_argument("--checkpoint_path", "-c", type = str, default = "./hand_drawing_model/checkpoint", help = "place to save the model")
    parser.add_argument("--data_path", "-d", type = str, default = "./datasets/hand_drawing", help = "path to the dataset")  
    parser.add_argument("--image_size", "-i", type = int, default = 224, help = "image size (i x i) to crop")
    parser.add_argument("--resume_training", "-r", type = bool, default = False, help = "continue training from previous epoch or not")
    
    return parser.parse_args()

#train the model
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isdir(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    writer = SummaryWriter(args.log_path)
    
    #preprocess the data
    mean = 0.1800765688410995   #mean and std are precomputed. You should compute on your own dataset
    std = 0.30052850582901786
    train_set = HandDrawingDataset(root = args.data_path,
                                   train = True,
                                   mean = mean,
                                   std = std,
                                   image_size = args.image_size)
    train_loader = DataLoader(train_set,
                              batch_size = args.batch_size,
                              shuffle = True,
                              drop_last = True,
                              num_workers = 8)
    eval_set = HandDrawingDataset(root = args.data_path,
                                  train = False,
                                  mean = mean,
                                  std = std,
                                  image_size = args.image_size)
    eval_loader = DataLoader(eval_set,
                             batch_size = args.batch_size,
                             shuffle = False,
                             drop_last = False,
                             num_workers = 8)

    #set up model, cvhange the first and last layer to fit the dataset
    model = resnet18(weights = ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(in_features = model.fc.in_features,
                         out_features = len(train_set.categories),
                         bias = True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias = False)
    
    # model = efficientnet_b3(weights = EfficientNet_B3_Weights.DEFAULT)
    # model.features[0][0] = nn.Conv2d(1, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)    
    # model.classifier[1] = nn.Linear(in_features = model.classifier[1].in_features, 
    #                                 out_features = len(train_set.categories), 
    #                                 bias = True)

    # model = mobilenet_v2(weights = None)
    # model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    # model.classifier[1] = nn.Linear(1280, len(train_set.categories))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model.parameters(),
                                 lr = args.lr)
        
    if args.resume_training:
        checkpoint = os.path.join(args.checkpoint_path, "last.pt")
        saved_data = torch.load(checkpoint, weights_only = False)
        model.load_state_dict(saved_data["model"])
        optimizer.load_state_dict(saved_data["optimizer"])
        start_epoch = saved_data["start_epoch"]
        best_acc = saved_data["best_acc"]
    else:
        best_acc = 0    #to store the best model, for deployment
        start_epoch = 0

    for epoch in range(start_epoch, args.num_epochs):
        #training
        total_loss = 0
        prrogress_bar = tqdm(train_loader, colour = "green")
        model.train()
        for iter, (images, labels) in enumerate(prrogress_bar):
            #forward pass
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)

            #calculate loss
            loss = criterion(output, labels)
            total_loss += loss.item()
            avg_loss = total_loss/(iter + 1)
            prrogress_bar.set_description(f"Epoch:{epoch + 1}/{args.num_epochs}, Loss:{avg_loss:.2f}, Device:{device}")
            writer.add_scalar(tag = "Train/Loss",
                              scalar_value = avg_loss, 
                              global_step = epoch * len(train_loader) + iter)

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #validtion
        model.eval()
        total_loss = 0
        progress_bar = tqdm(eval_loader, colour = "yellow")
        y_true = []
        y_pred = []
        with torch.no_grad():
            for iter, (images, labels) in enumerate(progress_bar):
                #forward pass
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)

                #calculate loss
                total_loss += criterion(output, labels).item()
                prediction = torch.argmax(output, dim = 1)
                y_true.extend(labels.tolist())
                y_pred.extend(prediction.tolist())
        
        avg_loss = total_loss/len(eval_loader)
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Epoch:{epoch + 1}/{args.num_epochs}, Loss:{avg_loss:.2f}, Accuracy:{accuracy:.2f}")
        writer.add_scalar(tag = "Eval/Loss",
                              scalar_value = avg_loss, 
                              global_step = epoch)
        writer.add_scalar(tag = "Eval/Accuarcy",
                              scalar_value = accuracy, 
                              global_step = epoch)
        
        #save the model
        saved_data = {
            "start_epoch": epoch + 1,
            "model" : model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": max(best_acc, accuracy),
            "categories": train_set
        }

        checkpoint  = os.path.join(args.checkpoint_path, "last.pt")
        torch.save(saved_data, checkpoint)

        if accuracy > best_acc:
            checkpoint  = os.path.join(args.checkpoint_path, "best.pt")
            torch.save(saved_data, checkpoint)
            best_acc = accuracy

if __name__ == "__main__":
    args = get_args()
    train(args)