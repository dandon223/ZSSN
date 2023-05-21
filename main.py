
from dataset import BatchLoader, convert_to_one_hot
from torch_geometric_temporal.nn.recurrent import GConvLSTM, GConvGRU
import torch
import torch.nn as nn
import argparse

def train(model_str: str):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 13
    batch_size = 20
    seq_length = 20
    num_nodes = 10000
    batch_loader = BatchLoader(batch_size, seq_length)

    if 'GRU' in model_str:
        model = GConvGRU(seq_length, 1, 4).to(device)
    if 'LSTM' in model_str:
        model = GConvLSTM(seq_length, 1, 4).to(device)

    learning_rate = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    model.train()

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

                batch_y = batch_y[-1].reshape([1,1])
                batch_y_onehot = convert_to_one_hot(batch_y, num_nodes)
                reshaped = batch_y_onehot.reshape([num_nodes, 1])
                batch_y = reshaped.to(device)

                if 'LSTM' in model_str:
                    y_hat, _ = model(batch_x, batch_loader.get_edge_index().to(device), batch_loader.get_edge_attr().to(device))
                else:
                    y_hat = model(batch_x, batch_loader.get_edge_index().to(device), batch_loader.get_edge_attr().to(device))

                y_pred = torch.sigmoid(y_hat)
                loss += criterion(y_pred, batch_y)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            expo = max(0, epoch+1 - 4)
            learning_decay = 0.5**expo
            learning_rate *= learning_decay
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            if (time+1) % 20 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{time+1}/{n_total_steps}], Loss: {loss.item()/batch_size:.4f}')
            #break

        print (f'Epoch [{epoch+1}/{num_epochs}], Step [{time+1}/{n_total_steps}], Loss: {loss.item()/batch_size:.4f}')

    # Save model
    torch.save(model, model_str)

def test(model_str: str):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes = 10000
    seq_length = 20
    batch_size = 20
    batch_loader = BatchLoader(batch_size, seq_length)

    criterion = nn.BCELoss()

    model = torch.load(model_str)
    model.eval()

    for _ in range(batch_loader.sizes[2]): # 2 = test

        batches_x, batches_y = batch_loader.next_batch(2)
        loss = 0
        for batch_id, batch in enumerate(batches_x):

            # Fetch training data
            batch_x = batch
            batch_y = batches_y[batch_id]
            batch_x_onehot = convert_to_one_hot(batch_x, num_nodes)
            reshaped = batch_x_onehot.reshape([num_nodes, seq_length])
            batch_x = reshaped.to(device)

            batch_y = batch_y[-1].reshape([1,1])
            batch_y_onehot = convert_to_one_hot(batch_y, num_nodes)
            reshaped = batch_y_onehot.reshape([num_nodes, 1])
            batch_y = reshaped.to(device)

            if 'LSTM' in model_str:
                y_hat, _ = model(batch_x, batch_loader.get_edge_index().to(device), batch_loader.get_edge_attr().to(device))
            else:
                y_hat = model(batch_x, batch_loader.get_edge_index().to(device), batch_loader.get_edge_attr().to(device))

            y_pred = torch.sigmoid(y_hat)
            #print(y_pred)
            loss += criterion(y_pred, batch_y)

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