import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import wandb
import random
import math

from model_factory import ModelFactory


def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
    parser.add_argument(
        "--data",
        type=str,
        default="data_sketches",
        metavar="D",
        help="folder where data is located. train_images/ and val_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="basic_cnn",
        metavar="MOD",
        help="Name of the model for model and transform instantiation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment",
        metavar="E",
        help="folder where experiment outputs are located.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        metavar="NW",
        help="number of workers for data loading",
    )
    args = parser.parse_args()
    return args


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def log_confusion_matrix(model, val_loader, use_cuda):
    all_preds = []
    all_targets = []
    model.eval()
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True).view(-1)
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix")
    wandb.log({"Confusion Matrix": wandb.Image(fig)})
    plt.close(fig)
    
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        """
        Multi-Class Focal Loss.
        Args:
            alpha (list, tensor, or None): Class weights (size = num_classes). If None, equal weights are applied.
            gamma (float): Focusing parameter gamma.
        """
        super(WeightedFocalLoss, self).__init__()
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)  # Convert to tensor
        else:
            self.alpha = None  # No class weighting
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits of shape (batch_size, num_classes).
            targets: Ground truth labels of shape (batch_size), with values in [0, num_classes-1].
        Returns:
            Scalar focal loss.
        """
        # Convert logits to probabilities
        probs = F.softmax(inputs, dim=1)  # Shape: (batch_size, num_classes)
        
        # Gather probabilities of the true class
        targets = targets.view(-1, 1)  # Shape: (batch_size, 1)
        p_t = probs.gather(1, targets).squeeze(1)  # Shape: (batch_size)
        
        # Compute the cross-entropy loss per sample
        CE_loss = -torch.log(p_t)  # Shape: (batch_size)
        
        # Compute the focal loss factor
        focal_factor = (1 - p_t) ** self.gamma  # Shape: (batch_size)
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if inputs.is_cuda:
                self.alpha = self.alpha.cuda()  # Ensure alpha is on the same device
            alpha_t = self.alpha.gather(0, targets.squeeze(1))  # Shape: (batch_size)
            CE_loss = CE_loss * alpha_t  # Scale the loss by alpha

        # Compute final focal loss
        F_loss = focal_factor * CE_loss
        return F_loss.mean()


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
    epoch: int,
    args: argparse.ArgumentParser,
) -> None:
    """Default Training Loop.

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optimizer): Optimizer to use
        train_loader (torch.utils.data.DataLoader): Training data loader
        use_cuda (bool): Whether to use cuda or not
        epoch (int): Current epoch
        args (argparse.ArgumentParser): Arguments parsed from command line
    """
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        #criterion = WeightedFocalLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        metrics = {"Train : loss": loss.item()}
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )
            wandb.log(metrics)
    print(
        "\nTrain set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset),
        )
    )


def validation(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
) -> float:
    """Default Validation Loop.

    Args:
        model (nn.Module): Model to train
        val_loader (torch.utils.data.DataLoader): Validation data loader
        use_cuda (bool): Whether to use cuda or not

    Returns:
        float: Validation loss
    """
    model.eval()
    validation_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        #criterion = WeightedFocalLoss()
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        
    validation_loss /= len(val_loader.dataset)
    accuracy_loss = 100.0 * correct / len(val_loader.dataset)
    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            validation_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )
    
    # Log confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    wandb.log({"Confusion Matrix": wandb.Image(fig)})
    plt.close(fig)
    
    return validation_loss, accuracy_loss



def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    # Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", "pred", "target"]+[f"score_{i}" for i in range(10)])
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
    wandb.log({"predictions_table":table}, commit=False)

def main():
    """Default Main Function."""
    # options
    args = opts()
    
    # Initialize wandb
    wandb.init(project="recvis24_a3", 
               config=vars(args))

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set the seed (for reproducibility)
    torch.manual_seed(args.seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # load model and transform
    model, data_transforms = ModelFactory(args.model_name).get_all()
    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    # Data initialization and loading
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + "/train_images", transform=data_transforms),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + "/val_images", transform=data_transforms),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Loop over the epochs
    best_val_loss = 1e8
    for epoch in range(1, args.epochs + 1):
        # training loop
        train(model, optimizer, train_loader, use_cuda, epoch, args)
        # validation loop
        val_loss, val_accuracy = validation(model, val_loader, use_cuda)
        val_metrics = {"Validation : loss": val_loss, "Validation : accuracy": val_accuracy}
        wandb.log(val_metrics)
        if val_loss < best_val_loss:
            # save the best model for validation
            best_val_loss = val_loss
            best_model_file = args.experiment + "/model_best.pth"
            torch.save(model.state_dict(), best_model_file)
            wandb.log_model(best_model_file, aliases="model_best")
        # also save the model every epoch
        model_file = args.experiment + "/model_" + str(epoch) + ".pth"
        torch.save(model.state_dict(), model_file)
        wandb.log_model(model_file, aliases="model_epoch_"+str(epoch))
        print(
            "Saved model to "
            + model_file
            + f". You can run `python evaluate.py --model_name {args.model_name} --model "
            + best_model_file
            + "` to generate the Kaggle formatted csv file\n"
        )
    wandb.finish()

if __name__ == "__main__":
    main()
