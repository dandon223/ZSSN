import torch
import torchtext
import numpy as np
import matplotlib.pyplot as plt  
from torchtext import data
from torchtext import datasets
# proby dopasowania danych do tego co maja w githubie w https://github.com/youngjoo-epfl/gconvRNN

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
    plt.scatter(x, y, alpha=0.8) 
    plt.show()

def main():
    train, valid, test = datasets.PennTreebank()

    test_data = []
    for element in test:
        element = element.replace(' ', '_')
        test_data.append(element)
    print(test_data[0])
    print(test_data[-1])

    train_data = []
    for element in train:
        element = element.replace(' ', '_')
        train_data.append(element)
    print(train_data[0])
    print(train_data[-1])

    valid_data = []
    for element in valid:
        element = element.replace(' ', '_')
        valid_data.append(element)
    print(valid_data[0])
    print(valid_data[-1])
    
    text_to_tensor([train_data, valid_data, test_data], "1", "1", "1")

if __name__ == "__main__":
    main()