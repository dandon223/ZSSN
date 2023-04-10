import torch
import torchtext
import numpy as np
import os
import pickle
import math
import matplotlib.pyplot as plt  
from torchtext import data
from torchtext import datasets
# proby dopasowania danych do tego co maja w githubie w https://github.com/youngjoo-epfl/gconvRNN

def pklSave(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

def pklLoad(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class BatchLoader():
    def __init__(self, batch_size, seq_length):
        self.vocab_fname = "data/vocab_char.pkl"
        self.tensor_fname = "data/data_char.pkl"
        self.Adj_fname = "data/adj_char.pkl"
        train, valid, test = datasets.PennTreebank()

        train_data = self.preprocess_data(train)
        test_data = self.preprocess_data(test)
        valid_data = self.preprocess_data(valid)

        if not os.path.exists(self.vocab_fname) or not os.path.exists(self.tensor_fname) or not os.path.exists(self.Adj_fname):
            print("Creating vocab...")
            self.text_to_tensor([train_data, valid_data, test_data], self.vocab_fname, self.tensor_fname, self.Adj_fname)
    
        print("Loading vocab...")
        adj = pklLoad(self.Adj_fname)
        all_data = pklLoad(self.tensor_fname)
        self.idx2char, self.char2idx = pklLoad(self.vocab_fname)
        vocab_size = len(self.idx2char)
        print(adj)

        print("Char vocab size: %d" % (len(self.idx2char)))
        self.sizes = []
        self.all_batches = []
        self.all_data = all_data
        self.adj = adj

        print("Reshaping tensors...")
        for split, data in enumerate(all_data):  # split = 0:train, 1:valid, 2:test
            
            length = data.shape[0]
            data = data[: batch_size * seq_length * int(math.floor(length / (batch_size * seq_length)))]
            ydata = np.zeros_like(data)
            ydata[:-1] = data[1:].copy()
            ydata[-1] = data[0].copy()
            print(data.shape)
            print(type(data))

            x_batches = torch.from_numpy(data.reshape([-1, batch_size, seq_length]))
            y_batches = torch.from_numpy(ydata.reshape([-1, batch_size, seq_length]))
            print(x_batches.shape)
            print(x_batches)
            print(y_batches.shape)
            print(y_batches)
            self.sizes.append(len(x_batches))

            self.all_batches.append([x_batches, y_batches])

        self.batch_idx = [0, 0, 0]
        print("data load done. Number of batches in train: %d, val: %d, test: %d" \
                % (self.sizes[0], self.sizes[1], self.sizes[2]))

    def next_batch(self, split_idx):
        # cycle around to beginning
        if self.batch_idx[split_idx] >= self.sizes[split_idx]:
            self.batch_idx[split_idx] = 0
        idx = self.batch_idx[split_idx]
        self.batch_idx[split_idx] = self.batch_idx[split_idx] + 1
        return self.all_batches[split_idx][0][idx], \
               self.all_batches[split_idx][1][idx]

    def reset_batch_pointer(self, split_idx, batch_idx=None):
        if batch_idx == None:
            batch_idx = 0
        self.batch_idx[split_idx] = batch_idx

    def text_to_tensor(input_texts: str, vocab_fname: str, tensor_fname: str, Adj_fname: str):

        counts = []
        char2idx = {}
        idx2char = []
        output = []

        for input_text in input_texts:
            count = 0
            output_chars = []
            for line in input_text: # w oryginale ostatnia wartosc line to lista pusta bo jest enter w pliku,a le tutaj nie zmieniam
                chars_in_line = list(line)
                chars_in_line.append('|')
                for char in chars_in_line:
                    if char not in char2idx:
                        idx2char.append(char)
                        char2idx[char] = len(idx2char) - 1
                    output_chars.append(char2idx[char])
                    count += 1
        
            counts.append(count)
            output.append(np.array(output_chars))

        print(char2idx)
        print(output_chars)
        print(len(output_chars))
        print(len(idx2char))
        print(idx2char)
        print(counts)

        train_data = output[0]
        train_data_shift = np.zeros_like(train_data)
        print(len(train_data_shift))
        train_data_shift[:-1] = train_data[1:].copy()
        train_data_shift[-1] = train_data[0].copy()
        print(train_data)
        print(train_data_shift)

        # Co-occurance
        Adj = np.zeros([len(idx2char), len(idx2char)])
        for x, y in zip(train_data, train_data_shift):
            Adj[x, y] += 1
        print(Adj)
        print("Number of chars : train %d, val %d, test %d" % (counts[0], counts[1], counts[2]))
        #plt.scatter(x, y, alpha=0.8) 
        #plt.show()

        print(vocab_fname)
        print(type(idx2char))
        pklSave(vocab_fname, [idx2char, char2idx])
        pklSave(tensor_fname, output)
        pklSave(Adj_fname, Adj)

    def preprocess_data(self, data: list) -> list:
        new_data = []
        for line in data:
            line = line.replace(' ', '_')
            new_data.append(line)
        return new_data
