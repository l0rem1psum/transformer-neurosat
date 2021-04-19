import math

import pytorch_lightning as pl
import torch
import torch_geometric as pyg
from pytorch_lightning.metrics import functional as FM

from model import MLP, LayerNormBasicLSTMCell, compute_loss
from .layers import GraphAttentionLayer


class GatSublayerConnection(torch.nn.Module):
    def __init__(self, d):
        super(GatSublayerConnection, self).__init__()
        self.norm = torch.nn.LayerNorm(d)
        self.dropout = torch.nn.Dropout(0.1)
        self.gat = GraphAttentionLayer(d, d, 0.1, 0.1)

    def forward(self, x, adj):
        return self.dropout(self.gat(self.norm(x), adj)) \
 \
                class GCN(torch.nn.Module):

    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GCN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = torch.nn.Linear(input_dim, feature_dim)
            self.conv_first = pyg.nn.GCNConv(feature_dim, hidden_dim)
        else:
            self.conv_first = pyg.nn.GCNConv(input_dim, hidden_dim)
        self.conv_hidden = torch.nn.ModuleList([pyg.nn.GCNConv(hidden_dim, hidden_dim) for _ in range(layer_num - 2)])
        self.conv_out = pyg.nn.GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = torch.nn.functional.relu(x)
        if self.dropout:
            x = torch.nn.functional.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, edge_index)
            x = torch.nn.functional.relu(x)
            if self.dropout:
                x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x


class SimpleAttentionSat(pl.LightningModule):
    def __init__(self, d):
        super(SimpleAttentionSat, self).__init__()
        # self.example_input_array = (torch.zeros(20, 35), torch.tensor(1))  # n_lits * n_clauses

        self.d = d

        self.lit_init = torch.nn.Parameter(torch.empty([1, d]))
        self.cls_init = torch.nn.Parameter(torch.empty([1, d]))
        self.slc_init = torch.nn.Parameter(torch.empty([1, d]))

        self.lit_proj = MLP(d, [d, d, d])
        self.cls_proj = MLP(d, [d, d, d])

        # self.adj_init = torch.nn.Parameter(torch.empty([1, d]))
        # self.adj_proj = MLP(d, [d, d, d])

        self.forward_gat = torch.nn.ModuleList(
            SAGEConv(d, d, flow="source_to_target", normalize=True) for _ in range(4))
        self.backward_gat = torch.nn.ModuleList(
            SAGEConv(d, d, flow="source_to_target", normalize=True) for _ in range(4))

        # self.forward_gat = TransformerConv((d, d), d, flow="source_to_target", heads=1)
        # self.backward_gat = TransformerConv((d, d), d, flow="source_to_target", heads=1)

        # self.forward_gat = GINConv(torch.nn.Sequential(torch.nn.Linear(d, d)), d, flow="source_to_target")
        # self.backward_gat = GINConv(torch.nn.Sequential(torch.nn.Linear(d, d)), d, flow="source_to_target")
        # self.gat_mlp = torch.nn.ModuleList(MLP(d, [d, d]) for _ in range(6))

        self.decode = MLP(2 * d, [d, d, 1])
        self.vote_bias = torch.nn.Parameter(torch.empty([]))

        self._init_weight()

        # Metrics
        self.train_accuracy = pl.metrics.Accuracy()

    def _init_weight(self):
        # torch.nn.init.normal_(self.adj_init)
        torch.nn.init.normal_(self.lit_init)
        torch.nn.init.normal_(self.slc_init)
        torch.nn.init.normal_(self.cls_init)
        torch.nn.init.zeros_(self.vote_bias)

    def forward(self, a, a1, a2, n_batches):
        n_lits, n_clauses = a.shape

        cls_emb_init = (self.lit_init / math.sqrt(self.d)).repeat([n_clauses, 1])
        lit_emb_init = (self.cls_init / math.sqrt(self.d)).repeat([n_lits // 2, 1])
        til_emb_init = (self.slc_init / math.sqrt(self.d)).repeat([n_lits // 2, 1])
        lit_emb_init = torch.cat([lit_emb_init, til_emb_init], 0)

        lit_emb = a @ self.lit_proj(cls_emb_init)  # n_lits * d
        cls_emb = a.t() @ self.cls_proj(lit_emb_init)  # n_clauses * d

        emb = torch.cat([lit_emb, cls_emb], 0)

        # for _ in range(16):
        for f, b in zip(self.forward_gat, self.backward_gat):
            #     cls_emb = self.forward_gat((lit_emb, cls_emb), a1, size=(lit_emb.size(0), cls_emb.size(0)))
            #     lit_emb = self.backward_gat((cls_emb, lit_emb), a2, size=(cls_emb.size(0), lit_emb.size(0)))
            emb = f(emb, a1)
            emb = b(emb, a2)
            # lit_emb = self.backward_gat((lit_emb, cls_emb), a2, size=(lit_emb.size(0), cls_emb.size(0)))

        lit_emb_total = torch.cat([emb[:n_lits // 2], emb[n_lits // 2:n_lits]], 1)
        out = self.decode(lit_emb_total) + self.vote_bias
        ret = torch.mean(out, 0)
        print(ret)
        return ret

    def training_step(self, batch, batch_idx):
        A, n_lits, n_clauses, y = batch
        outputs = self(A.float(), n_lits, n_clauses, n_batches=len(y))
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        loss = compute_loss(outputs, y, self.parameters())
        # loss = torch.nn.functional.binary_cross_entropy(outputs, y)
        self.log_dict(
            {
                "train_loss": loss.item(),
                "train_acc": self.train_accuracy(outputs > 0, y),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        A, n_lits, n_clauses, y = batch
        outputs = self(A.float(), n_lits, n_clauses, n_batches=len(y))
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
        A, n_lits, n_clauses, y = batch
        outputs = self(A.float(), n_lits, n_clauses, n_batches=len(y))
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
        return torch.optim.Adam(self.parameters(), lr=1e-4)
