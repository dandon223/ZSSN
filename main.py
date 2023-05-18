from dataset import BatchLoader, convert_to_one_hot
from torch_geometric_temporal.nn.recurrent import GConvLSTM, GConvGRU, DCRNN
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn

def demo(model_str: str):
    """Metoda demo, gdzie implementuje z innym przygotowanym juz datasetem przez torch_geometric_temporal
    Klasa RecurrentGCN wydaje sie dla mnie (DG) klasa ostateczna dla naszego problemu przy korzystaniu z GConvGRU, poniewaz
    ma podobna wartosc MSE jak przy wykorzystaniu DCRNN, ktory wzialem z przykladu https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html?highlight=temporal_signal_split#applications
    Nie wiem jeszcze co robic z C = 'Cell state matrix for all nodes' od GConvLSTM"""
    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset(lags=4)
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)

    if model_str =='GRU':
        model = GConvGRU(4, 1, 1) # 4 poniewaz na podstawie 4 tygodniu przewidujemy kolejny
    if model_str == 'LSTM':
        model = GConvLSTM(4, 1, 1) # 4 poniewaz na podstawie 4 tygodniu przewidujemy kolejny
    if model_str == 'DCRNN':
        model = DCRNN(4, 1 , 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()

    for epoch in tqdm(range(200)):
        cost = 0
        for time, snapshot in enumerate(train_dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            #print('snapshot.x', snapshot.x.shape)
            #print('snapshot.edge_index', snapshot.edge_index.shape)
            #print('snapshot.edge_attr', snapshot.edge_attr.shape)
            #print('snapshot.y', snapshot.y.shape)
            #print('y_hat', y_hat)
            #print('y_hat.shape', type(y_hat))
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

def ours(model_str: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 1
    batch_size = 1
    seq_length = 20
    num_nodes = 50
    batch_loader = BatchLoader(batch_size, seq_length)

    if model_str =='GRU':
        model = GConvGRU(seq_length, 1, 1).to(device)
    if model_str == 'LSTM':
        model = GConvLSTM(seq_length, 1, 1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()

    n_total_steps = batch_loader.sizes[0]
    for epoch in  range(num_epochs):
        #cost = 0
        batch_loader.reset_batch_pointer(0)
        for time, batch in enumerate(range(batch_loader.sizes[0])): # 0 = training
            # Fetch training data
            batch_x, batch_y = batch_loader.next_batch(0)
            #print('batch_x_pred', batch_x)
            batch_x_onehot = convert_to_one_hot(batch_x, num_nodes)
            #print('batch_x_onehot', batch_x_onehot)
            #print('batch_x.shape', batch_x.shape)
            #print('batch_x_onehot.shape', batch_x_onehot.shape)
            reshaped = batch_x_onehot.reshape([num_nodes, seq_length]) # nie wiem co z batch_size, oni maja [batch_size, num_nodes, seq_length]
            batch_x = reshaped.to(device)
            #print('batch_x', batch_x)
            #print('batch_x.shape', batch_x.shape)
            #print('batch_y.shape', batch_y.shape)
            #print('batch_y', batch_y)
            #print('batch_y.shape', batch_y.shape)
            #print('batch_y[:, -1]', batch_y[:, -1])
            #print('batch_y[:, -1].shape', batch_y[:, -1].reshape([1, 1]).shape)
            batch_y_onehot = convert_to_one_hot(batch_y[:, -1].reshape([1, 1]), num_nodes)
            reshaped = batch_y_onehot.reshape([num_nodes, 1])
            batch_y = reshaped.to(device)
            #print('batch_y', batch_y)
            if model_str == 'LSTM':
                y_hat, _ = model(batch_x, batch_loader.get_edge_index(), batch_loader.get_edge_attr())
            else:
                y_hat = model(batch_x, batch_loader.get_edge_index(), batch_loader.get_edge_attr())
            #print('y_hat', y_hat)
            #loss = criterion(y_hat, batch_y)
            
            loss = torch.mean((y_hat-batch_y)**2)
            #print('loss', loss)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (time+1) % 2000 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{time+1}/{n_total_steps}], Loss: {loss.item():.4f}')

        print (f'Epoch [{epoch+1}/{num_epochs}], Step [{time+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# TODO zaimplementowac cos takiego https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/dataset/chickenpox.html#ChickenpoxDatasetLoader
# Wtedy podpiac pod powyzszy model i powinno dzialac
# Ustalic co ma byc naszym wejsciem, wyjsciem, oraz ile mamy features i jakie tworzyc batche
def main():
    
    #demo('DCRNN')
    ours('LSTM')


if __name__ == "__main__":
    main()