# ----- Import libraries -----
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import MNISTDataset, VinaFood21, collate_fn
from Model.LeNet import LeNet
from Model.GoogLeNet import GoogLeNet
from Model.ResNet18 import ResNet18
from Model.pretrained_resnet import PretrainedResnet
from train_eval import train, evaluate

# ----- Define parameters -----
num_epochs = 5
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Parse arguments -----
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="mnist",
                    choices=["mnist", "vinafood"],
                    help="Choose dataset")
parser.add_argument("--model", type=str, default="LeNet",
                    choices=["LeNet", "GoogLeNet", "ResNet18", "ResNet50"],
                    help="Choose model architecture")
parser.add_argument("--lr", type=float, default=0.001,
                    help="Choose learning rate")
parser.add_argument("--num_classes", type=int, default=10,
                    help="Choose num classes")
args = parser.parse_args()

lr = args.lr
data_name = args.data
model_name = args.model
num_classes = args.num_classes

# ----- Load data -----
if data_name == "mnist":
    train_dataset = MNISTDataset(
        images_path="Data/Mnist/train-images.idx3-ubyte",
        labels_path="Data/Mnist/train-labels.idx1-ubyte"
    )

    test_dataset = MNISTDataset(
        images_path="Data/Mnist/t10k-images.idx3-ubyte",
        labels_path="Data/Mnist/t10k-labels.idx1-ubyte"
    )
elif data_name == "vinafood":
    train_dataset = VinaFood21(path="VinaFood21/train")
    test_dataset = VinaFood21(path="VinaFood21/test")
else:
    print("Wrong dataset")
    exit

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

# ----- Modeling -----
if model_name == "LeNet":
    model = LeNet(num_classes=num_classes)
elif model_name == "GoogLeNet":
    model = GoogLeNet(num_classes=num_classes)
elif model_name == "ResNet18":
    model = ResNet18(num_classes=num_classes)
elif model_name == "ResNet50":
    model = PretrainedResnet(num_classes=num_classes)
else:
    print("Wrong model")
    exit

model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=lr)

# ----- Training -----
train(model, model_name, train_dataloader, num_epochs, optimizer, loss_fn, device)

# ----- Evaluation -----
evaluate(model, model_name, test_dataloader, loss_fn, device)