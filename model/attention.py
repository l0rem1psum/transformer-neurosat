import math

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import functional as FM

from model import MLP, LayerNormBasicLSTMCell, compute_loss
from .layers import GraphAttentionLayer


class SublayerConnection(torch.nn.Module):
    def __init__(self, d, sublayer):
        super(SublayerConnection, self).__init__()
        self.norm = torch.nn.LayerNorm(d)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class SimpleAttentionSat(pl.LightningModule):
    def __init__(self, d, n_attn):
        super(SimpleAttentionSat, self).__init__()
        # self.example_input_array = (torch.zeros(20, 35), torch.tensor(1))  # n_lits * n_clauses

        self.d = d

        self.proj_lit = torch.nn.Parameter(torch.empty([1, d]))
        self.proj_cls = torch.nn.Parameter(torch.empty([1, d]))

        self.proj1 = MLP(d, [d, d, d])
        self.proj2 = MLP(d, [d, d, d])

        self.adj_init = torch.nn.Parameter(torch.empty([1, d]))
        self.adj_proj = MLP(d, [d, d, d])

        self.adj_gat = torch.nn.ModuleList([GraphAttentionLayer(d, d, 0.1, 0.1) for _ in range(6)])

        self.decode = MLP(d, [d, d, 1])
        self.vote_bias = torch.nn.Parameter(torch.empty([]))

        # self.n_rounds = n_rounds

        self._init_weight()

        # Metrics
        self.train_accuracy = pl.metrics.Accuracy()

    def _init_weight(self):
        torch.nn.init.normal_(self.adj_init)
        torch.nn.init.xavier_uniform_(self.proj_lit)
        torch.nn.init.xavier_uniform_(self.proj_cls)
        torch.nn.init.zeros_(self.vote_bias)

    def forward(self, A, n_lits, n_clauses, n_batches):
        N, _ = A.size() # where N = n_literals + n_clauses
        B = A[:n_lits,n_lits:] # where B is n_lits * n_clauses

        lit_init = B @ self.proj1((self.proj_lit / math.sqrt(self.d)).repeat([n_clauses, 1]))  # n_clauses * d
        cls_init = B.t() @ self.proj2((self.proj_cls / math.sqrt(self.d)).repeat([n_lits, 1]))  # n_lits * d
        x_emb = torch.cat([lit_init, cls_init], 0)

        for gat in self.adj_gat:
            x_emb = gat(x_emb, A)

        # attn = torch.cat([a(x_emb, A) for a in self.adj_gat])

        # for _ in range(1):
        #     cls = (x.t() @ self.lin2(lit_msg.squeeze())).unsqueeze(1)  # n_clauses * 1 * d
        #     cls_n = self.cls_norm(cls)
        #     lit = (x @ self.lin1(cls_msg.squeeze())).unsqueeze(1)  # n_lits * 1 * d
        #     lit_n = self.lit_norm(lit)
        #
        #     lit_msg, _ = self.cls_attn(lit, cls, cls)  # n_lits * 1 * d
        #     lit_msg = self.lit_drop(lit_n) + lit
        #     cls_msg, _ = self.lit_attn(cls, lit_msg, lit_msg)  # n_clauses * 1 * d
        #     cls_msg = self.cls_drop(cls_n) + cls

        out = self.decode(x_emb)
        ret = torch.mean(out, 0) + self.vote_bias
        print(ret)
        return ret

    def training_step(self, batch, batch_idx):
        A, n_lits, n_clauses, y = batch
        outputs = self(A.float(), n_lits, n_clauses, n_batches=len(y))
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        loss = compute_loss(outputs, y, self.parameters())
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
        # return torch.optim.SGD(self.parameters(), lr=1e-5)
        return torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-9)
