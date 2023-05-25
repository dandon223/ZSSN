
from dataset import BatchLoader, convert_to_one_hot
from torch_geometric_temporal.nn.recurrent import GConvLSTM, GConvGRU
import torch
import torch.nn as nn
import argparse

class OURLSTM(torch.nn.Module):
    def __init__(self, node_features, hidden_layer_size):
        super(OURLSTM, self).__init__()
        self.recurrent = GConvLSTM(node_features, hidden_layer_size, 4)
        self.dropout = nn.Dropout(0.25)
        self.linear = torch.nn.Linear(hidden_layer_size, 1)

    def forward(self, x, edge_index, edge_weight=None):
        h, _ = self.recurrent(x, edge_index, edge_weight)
        h = self.dropout(h)
        h = self.linear(h)
        return h

class OURGRU(torch.nn.Module):
    def __init__(self, node_features, hidden_layer_size):
        super(OURGRU, self).__init__()
        self.recurrent = GConvGRU(node_features, hidden_layer_size, 4)
        self.dropout = nn.Dropout(0.25)
        self.linear = torch.nn.Linear(hidden_layer_size, 1)

    def forward(self, x, edge_index, edge_weight=None):
        h, = self.recurrent(x, edge_index, edge_weight)
        h = self.dropout(h)
        h = self.linear(h)
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
        model = OURGRU(seq_length, hidden_layer_size).to(device)
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
            print(batches_x.shape)
            loss = 0
            new_batches_x = []
            new_batches_y = torch.LongTensor().to(device)
            print("new_batches_y", new_batches_y)
            print("new_batches_y", new_batches_y.shape)
            edge_index = torch.LongTensor().to(device)
            print("edge_index", edge_index.shape)
            for batch_id, batch in enumerate(batches_x):

                # Fetch training data
                batch_x = batch
                batch_y = batches_y[batch_id]
                batch_x_onehot = convert_to_one_hot(batch_x, num_nodes)
                reshaped = batch_x_onehot.reshape([num_nodes, seq_length])
                batch_x = reshaped.to(device)

                batch_y = batch_y[-1].reshape([1]).long().to(device)

                edge_index_temp = torch.clone(batch_loader.get_edge_index()).to(device)

                edge_index_temp += batch_id * num_nodes
                edge_index = torch.cat((edge_index, edge_index_temp), 1)
                new_batches_x.append(batch_x)
                new_batches_y = torch.cat((new_batches_y, batch_y), 0)
            
            b = torch.Tensor(batch_size * num_nodes, seq_length).to(device)
            torch.cat(new_batches_x, out=b)


            print("b.shape", b.shape)
            print("edge_index.shape", edge_index.shape)
            print("new_batches_y.shape", new_batches_y.shape)
            y_hat = model(b, edge_index.to(device))
            print("y_hat.shape", y_hat.shape)
            y_hat = y_hat.reshape(batch_size, -1)
            print("y_hat.shape", y_hat.shape)
            print("new_batches_y", new_batches_y)
            loss += criterion(y_hat, new_batches_y)
            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            optimizer.zero_grad()

            if (time+1) % 20 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{time+1}/{n_total_steps}], Loss: {loss.item()/batch_size:.4f}')
            break

        expo = max(0, epoch+1 - 4)
        learning_decay = 0.5**expo
        learning_rate *= learning_decay
        for g in optimizer.param_groups:
            g['lr'] = learning_rate

        print (f'Epoch [{epoch+1}/{num_epochs}], Step [{time+1}/{n_total_steps}], Loss: {loss.item()/batch_size:.4f}, Learning rate: {learning_rate}')

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
    parser = argparse.ArgumentParser(description='ZZSN')

    parser.add_argument('--train', help='to run training', type=str)
    parser.add_argument('--test', help='to run evaluation', type=str)
    
    args = parser.parse_args()

    if args.train is not None:
        train(args.train)
    elif args.test is not None:
        test(args.test)