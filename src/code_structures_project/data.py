from pathlib import Path

import typer
from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        
        self.data_path = raw_data_path

        self.train_data = []
        self.train_labels = []

        #load train data
        for i in range(6):
            self.train_data.append(torch.load(f"{self.data_path}/train_images_{i}.pt", weights_only=False))
            self.train_labels.append(torch.load(f"{self.data_path}/train_target_{i}.pt", weights_only=False))
        
        self.train_data = torch.cat(self.train_data)
        self.train_labels = torch.cat(self.train_labels)

        #load test data
        self.test_data = torch.load(f"{self.data_path}/test_images.pt", weights_only=False)
        self.test_labels = torch.load(f"{self.data_path}/test_target.pt", weights_only=False)

        #add dimension
        self.train_data = self.train_data.unsqueeze(1).float()
        self.test_data = self.test_data.unsqueeze(1).float()

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def len(self, true_if_train_false_if_test = True):

        if true_if_train_false_if_test:

            return self.train_data.shape[0]
        else:
            return self.test_data.shape[0]
    
    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        
        #normalize
        self.train_data = (self.train_data - self.train_data.mean()) / self.train_data.std()
        self.test_data = (self.test_data - self.test_data.mean()) / self.test_data.std()

        #save processed data
        torch.save(self.train_data, f"{output_folder}/train_images.pt")
        torch.save(self.train_labels, f"{output_folder}/train_target.pt")
        torch.save(self.test_data, f"{output_folder}/test_images.pt")
        torch.save(self.test_labels, f"{output_folder}/test_target.pt")



def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    # typer.run(preprocess)
    # preprocess("data/raw", "data/processed")
    test_dataset= MyDataset("data/raw")
    print(test_dataset.train_labels[42])
