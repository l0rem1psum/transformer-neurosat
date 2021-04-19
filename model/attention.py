import math

import pytorch_lightning as pl
import torch
import torch_geometric as pyg
from pytorch_lightning.metrics import functional as FM
import matplotlib.pyplot as plt

from model import MLP, LayerNormBasicLSTMCell, compute_loss
from .layers import GraphAttentionLayer


class MHASublayerConnection(torch.nn.Module):
    def __init__(self, d):
        super(MHASublayerConnection, self).__init__()
        self.norm = torch.nn.LayerNorm(d)
        self.dropout = torch.nn.Dropout(0.1)
        self.mha = torch.nn.MultiheadAttention(d, 8)

    def forward(self, x):
        return x + self.dropout(self.mha(self.norm(x), self.norm(x), self.norm(x))[0])

class GCN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=8, dropout=True, **kwargs):
        super(GCN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = torch.nn.Linear(input_dim, feature_dim)
            self.conv_first = pyg.nn.GATConv(feature_dim, hidden_dim)
        else:
            self.conv_first = pyg.nn.GATConv(input_dim, hidden_dim)
        self.conv_hidden = torch.nn.ModuleList([pyg.nn.GATConv(hidden_dim, hidden_dim) for _ in range(layer_num - 2)])
        self.conv_out = pyg.nn.GATConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = torch.nn.functional.relu(x)
        if self.dropout:
            x = torch.nn.functional.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = torch.nn.functional.relu(x)
            if self.dropout:
                x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x

class AttLayer(torch.nn.Module):
    def __init__(self, d_in, d_h, d_out):
        super(AttLayer, self).__init__()
        self.fc = torch.nn.Linear(d_in, d_h)
        self.dropout = torch.nn.Dropout(0.2)
        self.attn = pyg.nn.TransformerConv(d_h, d_out, dropout=0.2, beta=False)

    def forward(self, x, edge_index):
        x = self.fc(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.attn(x, edge_index)
        return x

class SimpleAttentionSat(pl.LightningModule):
    def __init__(self, d):
        super(SimpleAttentionSat, self).__init__()
        # self.example_input_array = (torch.zeros(20, 35), torch.tensor(1))  # n_lits * n_clauses

        self.d = d

        self.attn = torch.nn.ModuleList(
            [AttLayer(3, d, d)] +
            [AttLayer(d, d, d) for _ in range(7)]
        )

        self.fc4 = torch.nn.Linear(d, 1)
        self.dropout4 = torch.nn.Dropout(0.2)

        self._init_weight()

        # Metrics
        self.train_accuracy = pl.metrics.Accuracy()

    def _init_weight(self):
        pass
        # torch.nn.init.zeros_(self.vote_bias)64

    def forward(self, x, edge_index, f, b, n_lits, n_batches):

        for a in self.attn:
            x = a(x, edge_index)
        x = self.fc4(x)
        logits = torch.mean(x, 0)
        print(logits)
        return logits.view(-1)

    def training_step(self, batch, batch_idx):
        x, edge_index, f, b, n_lits, y = batch
        outputs = self(x.float(), edge_index, f, b, n_lits, n_batches=len(y))
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        # loss = compute_loss(outputs, y, self.parameters())
        # print(outputs.shape)
        # print(y.shape)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, y.float())
        self.log_dict(
            {
                "train_loss": loss.item(),
                "train_acc": self.train_accuracy(outputs > 0, y),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        # plt.imshow(votes.detach().numpy(), cmap="bwr")
        # plt.savefig("votes/{}.jpg".format(batch_idx))
        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index, f, b, n_lits, y = batch
        outputs = self(x.float(), edge_index, f, b, n_lits, n_batches=len(y))
        loss = compute_loss(outputs, y, self.parameters())
        acc = FM.accuracy(outputs > 0, y)
        self.log_dict(
            {
                "validation_loss": loss.item(),
                "validation_acc": acc,
            },
            prog_bar=False,
            on_step=True,
            on_epoch=True,
        )
        return acc

    def test_step(self, batch, batch_idx):
        x, edge_index, f, b, n_lits, y = batch
        outputs = self(x.float(), edge_index, f, b, n_lits, n_batches=len(y))
        acc = FM.accuracy(outputs > 0, y)
        self.log_dict(
            {
                "test_acc": acc,
            },
            prog_bar=False,
            on_step=True,
            on_epoch=True,
        )
        return acc

    def configure_optimizers(self):
        # return torch.optim.RMSprop(self.parameters(), lr=1e-5)
        # return torch.optim.SGD(self.parameters(), lr=1e-5)
        return torch.optim.Adam(self.parameters(), lr=5e-3)
