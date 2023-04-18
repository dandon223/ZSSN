
from dataset import BatchLoader
from torch_geometric_temporal.nn.recurrent import GConvLSTM, GConvGRU, DCRNN
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import torch
import torch.nn.functional as F
from tqdm import tqdm

class RGCNGRU(torch.nn.Module):
    def __init__(self, node_features):
        super(RGCNGRU, self).__init__()
        self.recurrent = GConvGRU(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

class RGCNLSTM(torch.nn.Module):
    def __init__(self, node_features):
        super(RGCNLSTM, self).__init__()
        self.recurrent = GConvLSTM(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h, _ = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


def demo(model_str: str):
    """Metoda demo, gdzie implementuje z innym przygotowanym juz datasetem przez torch_geometric_temporal
    Klasa RecurrentGCN wydaje sie dla mnie (DG) klasa ostateczna dla naszego problemu przy korzystaniu z GConvGRU, poniewaz
    ma podobna wartosc MSE jak przy wykorzystaniu DCRNN, ktory wzialem z przykladu https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html?highlight=temporal_signal_split#applications
    Nie wiem jeszcze co robic z C = 'Cell state matrix for all nodes' od GConvLSTM"""
    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset(lags=4)
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)

    if model_str == 'GRU':
        model = RGCNGRU(node_features = 4) # 4 poniewaz na podstawie 4 tygodniu przewidujemy kolejny
    if model_str == 'LSTM':
        model = RGCNLSTM(node_features = 4) # 4 poniewaz na podstawie 4 tygodniu przewidujemy kolejny

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()

    for epoch in tqdm(range(200)):
        cost = 0
        for time, snapshot in enumerate(train_dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            cost = cost + torch.mean((y_hat-snapshot.y)**2)
        cost = cost / (time+1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    cost = 0
    for time, snapshot in enumerate(test_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
    cost = cost / (time+1)
    cost = cost.item()
    print("MSE: {:.4f}".format(cost))

# TODO zaimplementowac cos takiego https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/dataset/chickenpox.html#ChickenpoxDatasetLoader
# Wtedy podpiac pod powyzszy model i powinno dzialac
# Ustalic co ma byc naszym wejsciem, wyjsciem, oraz ile mamy features i jakie tworzyc batche
def main():
    #batch_size = 1
    #seq_length = 50
    #batch_loader = BatchLoader(batch_size, seq_length)
    demo('LSTM')



if __name__ == "__main__":
    main()