from dataset import convert_to_one_hot
import torch

def ours(model_str: str):
    num_epochs = 1

    # if model_str =='GRU':
    #     model = GConvGRU(seq_length, 1, 1).to(device)
    # if model_str == 'LSTM':
    #     model = GConvLSTM(seq_length, 1, 1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()

    n_total_steps = batch_loader.sizes[0]
    for epoch in  range(num_epochs):
        #cost = 0
        batch_loader.reset_batch_pointer(0)
        for time, batch in enumerate(range(batch_loader.sizes[0])): # 0 = training
            # Fetch training data
            batch_x, batch_y = batch_loader.next_batch(0)
            batch_x_onehot = convert_to_one_hot(batch_x, num_nodes)
            reshaped = batch_x_onehot.reshape([num_nodes, seq_length]) # nie wiem co z batch_size, oni maja [batch_size, num_nodes, seq_length]
            batch_x = reshaped.to(device)
            batch_y_onehot = convert_to_one_hot(batch_y[:, -1].reshape([1, 1]), num_nodes)
            reshaped = batch_y_onehot.reshape([num_nodes, 1])
            batch_y = reshaped.to(device)
            #print('batch_y', batch_y)
            if model_str == 'LSTM':
                y_hat, _ = model(batch_x, batch_loader.get_edge_index(), batch_loader.get_edge_attr())
            else:
                y_hat = model(batch_x, batch_loader.get_edge_index(), batch_loader.get_edge_attr())
            #loss = criterion(y_hat, batch_y)
            
            loss = torch.mean((y_hat-batch_y)**2)
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