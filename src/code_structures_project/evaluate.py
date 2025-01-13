import torch
from tqdm import tqdm
import typer

from model import my_model


def evaluate(model_checkpoint: str):
    print("Evaluating model")

    batch_size = 32

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = my_model().to(device)
    model.load_state_dict(torch.load(model_checkpoint, weights_only=True, map_location=torch.device('cpu')))

    test_data = torch.load("data/processed/test_images.pt")
    test_labels = torch.load("data/processed/test_target.pt")

    test_set = torch.utils.data.TensorDataset(test_data, test_labels)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    model.eval()
    correct = 0
    total = 0

    for images, labels in tqdm(iter(test_dataloader)):
        images, labels = images.to(device), labels.to(device)

        output = model(images)

        predictions = output.argmax(dim=1)
        correct += (predictions == labels).float().sum().item()
        total += labels.size(0)

    test_accuracy = correct / total
    print(f"Test accuracy: {test_accuracy}")


if __name__ == "__main__":
    typer.run(evaluate)

    #docker build og run commands:
    #build: docker build -f dockerfiles\evaluate.dockerfile . -t evaluate:latest
    #run: docker run --rm --name experiment2 evaluate:latest
