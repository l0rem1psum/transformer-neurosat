import math

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import functional as FM

from model import MLP, LayerNormBasicLSTMCell, compute_loss


class SimpleAttentionSat(pl.LightningModule):
    def __init__(self, d, n_attn):
        super(SimpleAttentionSat, self).__init__()
        self.example_input_array = (torch.zeros(20, 35), torch.tensor(1))  # n_lits * n_clauses

        self.d = d

        self.proj_lit = torch.nn.Parameter(torch.empty([1, d]))
        self.proj_cls = torch.nn.Parameter(torch.empty([1, d]))

        self.lin1 = MLP(d, [d, d, d])
        self.lin2 = MLP(d, [d, d, d])

        # self.lit_attn = torch.nn.MultiheadAttention(embed_dim=d, num_heads=4)
        # self.cls_attn = torch.nn.MultiheadAttention(embed_dim=d, num_heads=4)

        self.lit_attn = torch.nn.ModuleList([torch.nn.MultiheadAttention(embed_dim=d, num_heads=8, dropout=0.1, add_bias_kv=True) for _ in range(n_attn)])
        self.cls_attn = torch.nn.ModuleList([torch.nn.MultiheadAttention(embed_dim=d, num_heads=8, dropout=0.1, add_bias_kv=True) for _ in range(n_attn)])

        self.lit_ln = torch.nn.LayerNorm(d)
        self.cls_ln = torch.nn.LayerNorm(d)

        self.decode1 = torch.nn.Linear(d, d)
        self.activation1 = torch.nn.LeakyReLU()
        self.decode2 = torch.nn.Linear(d, d)
        self.activation2 = torch.nn.LeakyReLU()
        self.decode3 = torch.nn.Linear(d, 1)

        # self.n_rounds = n_rounds

        self._init_weight()

        # Metrics
        self.train_accuracy = pl.metrics.Accuracy()

    def _init_weight(self):
        torch.nn.init.xavier_uniform_(self.proj_lit)
        torch.nn.init.xavier_uniform_(self.proj_cls)

        torch.nn.init.xavier_uniform_(self.decode1.weight)
        torch.nn.init.normal_(self.decode1.bias)
        torch.nn.init.xavier_uniform_(self.decode2.weight)
        torch.nn.init.zeros_(self.decode2.bias)
        torch.nn.init.xavier_uniform_(self.decode3.weight)
        torch.nn.init.zeros_(self.decode3.bias)

    def forward(self, x, n_batches):
        n_lits, n_clauses = x.size()

        lit_init = (self.proj_lit / math.sqrt(self.d)).repeat([n_clauses, 1])  # n_clauses * d
        cls_init = (self.proj_cls / math.sqrt(self.d)).repeat([n_lits, 1])     # n_lits * d
        #
        lit_msg = (x @ self.lin1(lit_init)).unsqueeze(1)      # n_lits * 1 * d
        cls_msg = (x.t() @ self.lin2(cls_init)).unsqueeze(1)  # n_clauses * 1 * d

        # lit_msg = self.lit_ln(lit_msg)
        # cls_msg = self.cls_ln(cls_msg)



        for attn_layer in self.lit_attn:
            lit_msg, _ = attn_layer(lit_msg, lit_msg, lit_msg)  # n_lits * 1 * d

        for attn_layer in self.cls_attn:
            cls_msg, _ = attn_layer(cls_msg, cls_msg, cls_msg)  # n_clauses * 1 * d

        combined_attn = torch.cat([lit_msg, cls_msg], 0).squeeze()  # (n_lits + n_clauses) * d
        result = torch.mean(self.decode3(self.activation2(self.decode2(self.activation1(self.decode1(combined_attn))))), 0)
        print(result)
        return result

        # # lit_msg = (x @ (self.proj_lit/math.sqrt(self.d)).repeat([n_clauses, 1])).unsqueeze(1)  # n_lits * 1 * d
        # # cls_msg = (x.t() @ (self.proj_cls/math.sqrt(self.d)).repeat([n_lits, 1])).unsqueeze(1)   # n_clauses * 1 * d
        # lit_init = (self.proj_lit/math.sqrt(self.d)).repeat([n_clauses, 1])  # n_clauses * d
        # cls_init = (self.proj_cls/math.sqrt(self.d)).repeat([n_lits, 1])     # n_lits * d
        #
        # lit_msg = (x @ self.lin1(lit_init)).unsqueeze(1)      # n_lits * 1 * d
        # cls_msg = (x.t() @ self.lin2(cls_init)).unsqueeze(1)  # n_clauses * 1 * d
        #
        # # print(lit_msg.squeeze())
        #
        # # for _ in range(self.n_rounds):
        # #     lit_msg, _ = self.cls_attn(lit_msg, cls_msg, cls_msg)  # n_lits * 1 * d, ?
        # #     flipped_lit_attn_msg = torch.cat([lit_msg[n_lits // 2:n_lits, :, :], lit_msg[0:n_lits // 2, :, :]], 0)
        # #     combined_lit_attn_msg = torch.cat([lit_msg, flipped_lit_attn_msg], 0) # 2n_lits * 1 * d
        # #     cls_msg, _ = self.lit_attn(cls_msg, combined_lit_attn_msg, combined_lit_attn_msg)  # n_clauses * 1 * d, ?
        #
        # for _ in range(8):
        #     for attn_layer in self.lit_attn:
        #         lit_msg = (x @ self.lin1(lit_init)).unsqueeze(1)
        #         # print(lit_msg.squeeze())
        #         lit_msg, w = attn_layer(self.lit_ln(lit_msg), cls_msg, cls_msg)  # n_lits * 1 * d, 1 * n_lits * n_clauses
        #         # print(1)
        #
        #     # print(lit_msg.squeeze())
        #
        #     for attn_layer in self.cls_attn:
        #         flipped_lit_attn_msg = torch.cat([lit_msg[n_lits // 2:n_lits, :, :], lit_msg[0:n_lits // 2, :, :]], 0)
        #         combined_lit_attn_msg = torch.cat([lit_msg, flipped_lit_attn_msg], 0)  # 2n_lits * 1 * d
        #         cls_msg = (x.t() @ self.lin2(cls_init)).unsqueeze(1)
        #         cls_msg, _ = attn_layer(self.cls_ln(cls_msg), combined_lit_attn_msg, combined_lit_attn_msg)  # n_clauses * 1 * d, ?
        #         # print(combined_lit_attn_msg.squeeze())
        # # print(list(self.lin2.parameters()))
        # # print(cls_msg.squeeze())
        #
        # # lit_attn_msg = lit_msg
        # # for attn_layer in self.lit_attn:
        # #     lit_attn_msg, _ = attn_layer(lit_attn_msg, lit_attn_msg, lit_attn_msg)  # 1 * n_lits * d, ?
        # # flipped_lit_attn_msg = torch.cat([cls_attn_msg[:, n_lits//2:n_lits, :], cls_attn_msg[:, 0:n_lits//2, :]], 1)
        #
        # # cls_attn_msg = cls_msg
        # # for attn_layer in self.cls_attn:
        # #     cls_attn_msg, _ = attn_layer(cls_attn_msg, cls_attn_msg, cls_attn_msg)  # 1 * n_clauses * d, ?
        #
        # inter_decoded_msg = self.decode2(self.activation1(self.decode1(cls_msg.squeeze())))
        # # final_msg = self.decode_2(inter_decoded_msg)
        # print(torch.mean(inter_decoded_msg, 0))
        # return torch.mean(inter_decoded_msg, 0)

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
        # return torch.optim.SGD(self.parameters(), lr=1e-6)
        return torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-9)
