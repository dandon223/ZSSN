import torch
import numpy as np
import os
import pickle
import math
import matplotlib.pyplot as plt  
from torchtext import data
from torchtext import datasets
from scipy.sparse import coo_matrix

import pytorch_lightning as pl
from torch.utils.data import DataLoader

class PennTreebankDataset(pl.LightningDataModule):
    def __init__(self, batch_size, n_nodes) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.n_nodes = n_nodes

    def prepare_data(self) -> None:
        self.vocab_fname = "data/vocab_char.pkl"
        self.tensor_fname = "data/data_char.pkl"
        self.adj_fname = "data/adj_char.pkl"
        train, valid, test = datasets.PennTreebank()
        train_data = self._preprocess_data(train)
        test_data = self._preprocess_data(test)
        valid_data = self._preprocess_data(valid)

        if not os.path.exists(self.vocab_fname) or not os.path.exists(self.tensor_fname) or not os.path.exists(self.adj_fname):
            print("Creating vocab...")
            self._text_to_tensor([train_data, valid_data, test_data])

        print("Loading vocab...")
        adj = self._pklLoad(self.adj_fname)
        self.all_data = self._pklLoad(self.tensor_fname)
        
        self.idx2char, self.char2idx = self._pklLoad(self.vocab_fname)
        print(adj)
        print("Char vocab size: %d" % (len(self.idx2char)))

        self.edge_index = None
        self.edge_attr = None
        for x in range(adj.shape[0]):
            for y in range(adj.shape[1]):
                if adj[x][y] > 0:
                    if self.edge_index == None:
                        self.edge_index = torch.tensor([[x],[y]])
                        self.edge_attr = torch.Tensor([1])
                    else:
                        self.edge_index = torch.cat((self.edge_index, torch.tensor([[x],[y]])), 1)
                        self.edge_attr = torch.cat((self.edge_attr, torch.Tensor([1])), 0)
        print('self.edge_index.shape', self.edge_index.shape)
        print('self.edge_attr.shape', self.edge_attr.shape)
    
    def train_dataloader(self):
        return DataLoader(self.all_data[0], batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.all_data[1], batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.all_data[2], batch_size=self.batch_size)
    
    def get_edge_index(self):
        return self.edge_index
    
    def get_edge_attr(self):
        return self.edge_attr
    
    @staticmethod
    def convert_to_one_hot(a, max_val=None):
        N = a.shape[0]*a.shape[1] #a.size
        data = np.ones(N,dtype=int)
        sparse_out = coo_matrix((data,(np.arange(N),a.ravel())), shape=(N,max_val)).T
        return torch.Tensor(np.array(sparse_out.todense()))
    
    def _preprocess_data(self, data: list) -> list:
        new_data = []
        for line in data:
            line = line.replace(' ', '_')
            new_data.append(line)
        return new_data
    
    def _text_to_tensor(self, input_texts):

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

        train_data = output[0]
        train_data_shift = np.zeros_like(train_data)
        train_data_shift[:-1] = train_data[1:].copy()
        train_data_shift[-1] = train_data[0].copy()

        # Co-occurance
        adj = np.zeros([len(idx2char), len(idx2char)])
        counter = 0
        for x, y in zip(train_data, train_data_shift):
            adj[x, y] += 1
            if adj[x,y] == 1:
                counter +=1
        print(adj)
        print('counter', counter)
        print("Number of chars : train %d, val %d, test %d" % (counts[0], counts[1], counts[2]))
        #plt.scatter(x, y, alpha=0.8) 
        #plt.show()

        self._pklSave(self.vocab_fname, [idx2char, char2idx])
        self._pklSave(self.tensor_fname, output)
        self._pklSave(self.adj_fname, adj)

    @staticmethod
    def _pklSave(fname, obj):
        with open(fname, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def _pklLoad(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)
