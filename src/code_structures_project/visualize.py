import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import typer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from model import my_model

def visualize(model_checkpoint: str):

    batch_size = 32

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    model = my_model().to(device)
    model.load_state_dict(torch.load(model_checkpoint, weights_only=False))

    test_data = torch.load("data/processed/test_images.pt")
    test_labels = torch.load("data/processed/test_target.pt")

    test_set = torch.utils.data.TensorDataset(test_data, test_labels)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    model.eval()

    #set last layer to identity layer so output is just the output from the second to
    #last layer of the original architecture
    model.out = torch.nn.Identity()
    # print(model)

    embeddings = []
    labels_arr = []
    with torch.no_grad():
        for images, labels in tqdm(iter(test_dataloader)):

            images, labels = images.to(device), labels.to(device)
            
            output = model(images)
            embeddings.append(output)
            labels_arr.append(labels)

        embeddings = torch.cat(embeddings).cpu().numpy()
        labels_arr = torch.cat(labels_arr).cpu().numpy()
    
    pca = PCA(n_components=100)
    embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = labels_arr == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))
    plt.legend()
    plt.savefig(f"reports/figures/embeddings_visualization.png")
    
    

if __name__ == "__main__":
    typer.run(visualize)