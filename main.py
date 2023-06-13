
from dataset import BatchLoader, convert_to_one_hot
from results_visualiser import visualise_results
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
        h = self.recurrent(x, edge_index, edge_weight)
        h = self.dropout(h)
        h = self.linear(h)
        return h

def train(model_str: str):

    f = open("results.txt", "a")
    f.write("\n" + model_str + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 13
    batch_size = 8
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

    print(model)

    n_total_steps = batch_loader.sizes[0]
    for epoch in  range(num_epochs):
        model.train()

        expo = max(0, epoch+1 - 4)
        learning_decay = 0.5**expo
        learning_rate *= learning_decay
        for g in optimizer.param_groups:
            g['lr'] = learning_rate

        batch_loader.reset_batch_pointer(0)
        loss_sum = 0.0
        loops=0
        for time in range(batch_loader.sizes[0]): # 0 = training

            batches_x, batches_y = batch_loader.next_batch(0)
            new_batches_x = torch.Tensor().to(device)
            new_batches_y = torch.LongTensor().to(device)
            edge_index = torch.LongTensor().to(device)
            loss = 0
            perplexity = 0
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
                new_batches_x = torch.cat((new_batches_x, batch_x), 0)
                new_batches_y = torch.cat((new_batches_y, batch_y), 0)

            y_hat = model(new_batches_x, edge_index.to(device))
            y_hat = y_hat.reshape(batch_size, -1)

            loss = criterion(y_hat, new_batches_y)
            loss_sum += loss.item()
            loops += 1
            perplexity  = torch.exp(criterion(y_hat, new_batches_y))
            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            optimizer.zero_grad()

            if (time+1) % 1000 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{time+1}/{n_total_steps}], Loss: {loss.item():.4f}, Perplexity: {perplexity}')

        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{time+1}/{n_total_steps}], Loss: {loss_sum/loops:.4f}, Learning rate: {learning_rate}, Perplexity: {torch.exp(torch.tensor(loss_sum/loops))}')
        f.write(f'Epoch [{epoch+1}/{num_epochs}], Step [{time+1}/{n_total_steps}], Loss: {loss_sum/loops:.4f}, Learning rate: {learning_rate}, Perplexity: {torch.exp(torch.tensor(loss_sum/loops))}\n')
        f.write(test(model, batch_loader, epoch, num_epochs, batch_size, seq_length, num_nodes))
    # Save model
    torch.save(model, model_str)
    f.close()

def test(model, batch_loader, epoch, num_epochs, batch_size, seq_length, num_nodes):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()

    model.eval()
    batch_loader.reset_batch_pointer(2)
    loops = 0
    loss = 0
    for _ in range(batch_loader.sizes[2]): # 2 = test

        batches_x, batches_y = batch_loader.next_batch(2)
        new_batches_x = torch.Tensor().to(device)
        new_batches_y = torch.LongTensor().to(device)
        edge_index = torch.LongTensor().to(device)
        for batch_id, batch in enumerate(batches_x):

            # Fetch testing data
            batch_x = batch
            batch_y = batches_y[batch_id]
            batch_x_onehot = convert_to_one_hot(batch_x, num_nodes)
            reshaped = batch_x_onehot.reshape([num_nodes, seq_length])
            batch_x = reshaped.to(device)

            batch_y = batch_y[-1].reshape([1]).long().to(device)

            edge_index_temp = torch.clone(batch_loader.get_edge_index()).to(device)

            edge_index_temp += batch_id * num_nodes
            edge_index = torch.cat((edge_index, edge_index_temp), 1)
            new_batches_x = torch.cat((new_batches_x, batch_x), 0)
            new_batches_y = torch.cat((new_batches_y, batch_y), 0)
        
        y_hat = model(new_batches_x, edge_index.to(device))
        y_hat = y_hat.reshape(batch_size, -1)
        loss += criterion(y_hat, new_batches_y).item()
        loops += 1

    print (f'Test Epoch [{epoch+1}/{num_epochs}], Loss: {loss/loops:.4f}, Perplexity: {torch.exp(torch.tensor(loss/loops)).item()}')
    return f'Test Epoch [{epoch+1}/{num_epochs}], Loss: {loss/loops:.4f}, Perplexity: {torch.exp(torch.tensor(loss/loops)).item()}\n'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ZZSN')

    parser.add_argument('--train', help='to run training', type=str)
    parser.add_argument('--test', help='to run evaluation', type=str)
    
    args = parser.parse_args()

    if args.train is not None:
        train(args.train)
    elif args.test is not None:
        test(args.test)

    visualise_results()
