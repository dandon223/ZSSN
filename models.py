import pytorch_lightning as pl
from torch_geometric_temporal.nn.recurrent import GConvLSTM, GConvGRU, DCRNN


class GCRN_GRU(pl.LightningModule):
    def __init__(self, seq_length) -> None:
        super(GCRN_GRU, self).__init__()
        self.model = GConvGRU(seq_length, 1, 1)

    def forward(self, x, edge_index, edge_attr):
        return self.model(x, edge_index, edge_attr)


class GCRN_LSTM(pl.LightningModule):
    def __init__(self, seq_length) -> None:
        super(GCRN_LSTM, self).__init__()
        self.model = GConvLSTM(seq_length, 1, 1)

    def forward(self, x, edge_index, edge_attr):
        y, _ = self.model(x, edge_index, edge_attr)
        return y


class DiffConvRNN(pl.LightningModule):
    def __init__(self, seq_length) -> None:
        super(DiffConvRNN, self).__init__()
        self.model = DCRNN(seq_length, 1, 1)

    def forward(self, x, edge_index, edge_attr):
        return self.model(x, edge_index, edge_attr)
