import math

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import functional as FM

from model import MLP, LayerNormBasicLSTMCell, compute_loss


class SimpleAttentionSat(pl.LightningModule):
    def __init__(self, d, n_rounds):
        super(SimpleAttentionSat, self).__init__()
        self.example_input_array = (torch.zeros(20, 35), torch.tensor(1))  # n_lits * n_clauses

        self.proj_lit = torch.nn.Parameter(torch.empty([1, d]))
        self.proj_cls = torch.nn.Parameter(torch.empty([1, d]))

        self.lit_attn = torch.nn.MultiheadAttention(embed_dim=d, num_heads=4)
        self.cls_attn = torch.nn.MultiheadAttention(embed_dim=d, num_heads=4)

        # self.lit_attn = torch.nn.ModuleList([torch.nn.MultiheadAttention(embed_dim=d, num_heads=4) for _ in range(n_attn)])
        # self.cls_attn = torch.nn.ModuleList([torch.nn.MultiheadAttention(embed_dim=d, num_heads=4) for _ in range(n_attn)])

        self.decode_1 = torch.nn.Linear(d, 1)
        # self.decode_2 = torch.nn.Linear(int(math.sqrt(d)), 1)

        self.n_rounds = n_rounds

        self._init_weight()

        # Metrics
        self.train_accuracy = pl.metrics.Accuracy()

    def _init_weight(self):
        torch.nn.init.xavier_uniform_(self.proj_lit)
        torch.nn.init.xavier_uniform_(self.proj_cls)

    def forward(self, x, n_batches):
        n_lits, n_clauses = x.size()

        lit_msg = (x @ self.proj_lit.repeat([n_clauses, 1])).unsqueeze(1)  # n_lits * 1 * d
        cls_msg = (x.t() @ self.proj_cls.repeat([n_lits, 1])).unsqueeze(1)   # n_clauses * 1 * d

        for _ in range(self.n_rounds):
            lit_attn_msg, _ = self.cls_attn(lit_msg, cls_msg, cls_msg)  # n_lits * 1 * d, ?
            flipped_lit_attn_msg = torch.cat([lit_msg[n_lits // 2:n_lits, :, :], lit_msg[0:n_lits // 2, :, :]], 0)
            combined_lit_attn_msg = torch.cat([lit_msg, flipped_lit_attn_msg], 0) # 2n_lits * 1 * d
            cls_msg, _ = self.lit_attn(cls_msg, combined_lit_attn_msg, combined_lit_attn_msg)  # n_clauses * 1 * d, ?

        # lit_attn_msg = lit_msg
        # for attn_layer in self.lit_attn:
        #     lit_attn_msg, _ = attn_layer(lit_attn_msg, lit_attn_msg, lit_attn_msg)  # 1 * n_lits * d, ?
        # flipped_lit_attn_msg = torch.cat([cls_attn_msg[:, n_lits//2:n_lits, :], cls_attn_msg[:, 0:n_lits//2, :]], 1)

        # cls_attn_msg = cls_msg
        # for attn_layer in self.cls_attn:
        #     cls_attn_msg, _ = attn_layer(cls_attn_msg, cls_attn_msg, cls_attn_msg)  # 1 * n_clauses * d, ?

        inter_decoded_msg = self.decode_1(cls_msg.squeeze())
        # final_msg = self.decode_2(inter_decoded_msg)

        return torch.mean(inter_decoded_msg, 0)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x.float(), n_batches=len(y))
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
        x, y = batch
        outputs = self(x.float(), n_batches=len(y))
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
        x, y = batch
        outputs = self(x.float(), n_batches=len(y))
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
        return torch.optim.SGD(self.parameters(), lr=1e-6)
        # return torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-10)
