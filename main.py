
from dataset import BatchLoader, convert_to_one_hot
from torch_geometric_temporal.nn.recurrent import GConvLSTM, GConvGRU
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

class OURLSTM(torch.nn.Module):
    def __init__(self, node_features, hidden_layer_size):
        super(OURLSTM, self).__init__()
        self.recurrent = GConvLSTM(node_features, hidden_layer_size, 4)
        #self.dropout = nn.Dropout(0.2)
        self.linear = torch.nn.Linear(hidden_layer_size, 1)

    def forward(self, x, edge_index, edge_weight):
        h, _ = self.recurrent(x, edge_index, edge_weight)
        #h = self.dropout(h)
        h = F.relu(h)
        #print("before", h)
        #print(h.shape)
        h = self.linear(h)
        #print(h)
        #print(h.shape)
        return h

def train(model_str: str):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 13
    batch_size = 20
    seq_length = 20
    num_nodes = 10000
    hidden_layer_size = 200
    batch_loader = BatchLoader(batch_size, seq_length)

    if 'GRU' in model_str:
        model = GConvGRU(seq_length, 1, 4).to(device)
    if 'LSTM' in model_str:
        model = OURLSTM(seq_length, hidden_layer_size).to(device)

    learning_rate = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()

    print(model)

    n_total_steps = batch_loader.sizes[0]
    for epoch in  range(num_epochs):

        batch_loader.reset_batch_pointer(0)

        for time in range(batch_loader.sizes[0]): # 0 = training

            batches_x, batches_y = batch_loader.next_batch(0)
            loss = 0
            for batch_id, batch in enumerate(batches_x):

                # Fetch training data
                batch_x = batch
                batch_y = batches_y[batch_id]
                batch_x_onehot = convert_to_one_hot(batch_x, num_nodes)
                reshaped = batch_x_onehot.reshape([num_nodes, seq_length])
                batch_x = reshaped.to(device)

                batch_y = batch_y[-1].reshape([1]).long().to(device)
                #batch_y_onehot = convert_to_one_hot(batch_y, num_nodes)
                #reshaped = batch_y_onehot.reshape([num_nodes, 1])
                #batch_y = reshaped.to(device)

                if 'LSTM' in model_str:
                    y_hat = model(batch_x, batch_loader.get_edge_index().to(device), batch_loader.get_edge_attr().to(device))
                else:
                    y_hat = model(batch_x, batch_loader.get_edge_index().to(device), batch_loader.get_edge_attr().to(device))

                y_hat = y_hat.reshape(1, -1)
                #print(y_hat)
                #print(y_hat.shape)
                #print(batch_y)
                #print(batch_y.shape)
                #y_pred = torch.sigmoid(y_hat)
                loss += criterion(y_hat, batch_y)

            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            optimizer.zero_grad()

            expo = max(0, epoch+ 1)
            learning_decay = 0.5**expo
            learning_rate *= learning_decay
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            if (time+1) % 20 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{time+1}/{n_total_steps}], Loss: {loss.item()/batch_size:.4f}')
            #break

        print (f'Epoch [{epoch+1}/{num_epochs}], Step [{time+1}/{n_total_steps}], Loss: {loss.item()/batch_size:.4f}, Learning rate: {learning_rate:.4f}')

    # Save model
    torch.save(model, model_str)

def test(model_str: str):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes = 10000
    seq_length = 20
    batch_size = 20
    batch_loader = BatchLoader(batch_size, seq_length)

    criterion = nn.CrossEntropyLoss()

    model = torch.load(model_str)
    model.eval()
    loss = 0
    for _ in range(batch_loader.sizes[2]): # 2 = test

        batches_x, batches_y = batch_loader.next_batch(2)
        for batch_id, batch in enumerate(batches_x):

            # Fetch training data
            batch_x = batch
            batch_y = batches_y[batch_id]
            batch_x_onehot = convert_to_one_hot(batch_x, num_nodes)
            reshaped = batch_x_onehot.reshape([num_nodes, seq_length])
            batch_x = reshaped.to(device)

            batch_y = batch_y[-1].reshape([1]).long().to(device)

            if 'LSTM' in model_str:
                y_hat, _ = model(batch_x, batch_loader.get_edge_index().to(device), batch_loader.get_edge_attr().to(device))
            else:
                y_hat = model(batch_x, batch_loader.get_edge_index().to(device), batch_loader.get_edge_attr().to(device))

            y_hat = y_hat.reshape(1, -1)
            #print(y_pred)
            loss += criterion(y_hat, batch_y)

    print("loss = ", loss.item()/batch_loader.sizes[2])

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ZSSN')

    parser.add_argument('--train', help='to run training', type=str)
    parser.add_argument('--test', help='to run evaluation', type=str)
    
    args = parser.parse_args()

    if args.train is not None:
        train(args.train)
    elif args.test is not None:
        test(args.test)