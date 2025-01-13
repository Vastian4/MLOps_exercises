import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import typer
import os
import wandb

from model import my_model #virker fint selvom der er gul streg


def plot_accuracy_and_loss(train_loss, train_acc):
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(train_loss)
    axs[0].set_title("Train loss")
    axs[1].plot(train_acc)
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/Training loss and accuracy.png")



def train(lr: float=1e-3):

    batch_size = 32
    epochs = 3

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = torch.load("data/processed/train_images.pt")
    train_labels = torch.load("data/processed/train_target.pt")

    train_set = torch.utils.data.TensorDataset(train_data, train_labels)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    #initialize logging with wandb
    run = wandb.init(
        project="corrupt_mnist",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )
    
    model = my_model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()

    print("Training")
    train_loss = []
    train_accuracy = []
    for epoch in range(epochs):
        
        counter = 0
        for images, labels in tqdm(iter(train_dataloader)):
            
            counter += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)

            loss = criterion(output, labels)
            train_loss.append(loss.item())

            predictions = output.argmax(dim=1)
            accuracy = (predictions == labels).float().mean().item()
            train_accuracy.append(accuracy)

            loss.backward()
            optimizer.step()

            #logging loss, accuracy and some input images with wandb
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})


            if counter % 200 == 0:
                images = wandb.Image(images[:5].detach().cpu(), caption="Input images")
                wandb.log({"images": images})


    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    torch.save(model.state_dict(), "models/model.pth")


    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images"
    )
    artifact.add_file("models/model.pth")
    run.log_artifact(artifact)
    
    plot_accuracy_and_loss(train_loss, train_accuracy)

if __name__ == "__main__":
    typer.run(train)

    #docker build og run commands:
    #build: docker build -f dockerfiles\train.dockerfile . -t train:latest
    #run: docker run --rm --name experiment1 train:latest