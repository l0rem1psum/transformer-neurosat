import math

import pytorch_lightning as pl
import torch
from torch_geometric.nn import GATConv
from pytorch_lightning.metrics import functional as FM


from model import MLP, LayerNormBasicLSTMCell, compute_loss


class NeuroSAT(pl.LightningModule):
    def __init__(self, d, n_msg_layers, n_vote_layers, n_rounds):
        super(NeuroSAT, self).__init__()
        self.example_input_array = (torch.zeros(20, 35), torch.zeros(2, 10).long(), torch.zeros(2, 10).long(), torch.tensor(1)) # n_lits * n_clauses

        self.d = d
        self.n_rounds = n_rounds

        self.L_init = torch.nn.Parameter(torch.empty([1, d]))
        self.C_init = torch.nn.Parameter(torch.empty([1, d]))

        self.LC_msg = MLP(d, [d for _ in range(n_msg_layers)] + [d])
        self.CL_msg = MLP(d, [d for _ in range(n_msg_layers)] + [d])

        self.L_update = LayerNormBasicLSTMCell(2 * d, d)
        self.C_update = LayerNormBasicLSTMCell(d, d)

        self.forward_gat = GATConv((d, d), d, add_self_loops=False)
        self.backward_gat = GATConv((d, d), d, add_self_loops=False)

        self.L_vote = MLP(d, [d for _ in range(n_vote_layers)] + [1])
        self.vote_bias = torch.nn.Parameter(torch.empty([]))

        self._init_weight()

        # Metrics
        self.train_accuracy = pl.metrics.Accuracy()

    def _init_weight(self):
        torch.nn.init.normal_(self.L_init)
        torch.nn.init.normal_(self.C_init)
        torch.nn.init.zeros_(self.vote_bias)

    def forward(self, x, f, b, n_batches=1):
        n_lits, n_clauses = x.size()
        n_vars = n_lits // 2
        denom = math.sqrt(self.d)

        L_state_h = (self.L_init / denom).repeat([n_lits, 1])
        L_state_c = torch.zeros([n_lits, self.d])#.cuda()

        C_state_h = (self.C_init / denom).repeat([n_clauses, 1])
        C_state_c = torch.zeros([n_clauses, self.d])#.cuda()

        # LC_pre_msgs = self.LC_msg(L_state_h)
        CL_pre_msgs = self.CL_msg(C_state_h)

        for i in range(self.n_rounds):
            LC_pre_msgs = self.LC_msg(L_state_h)
            # LC_msgs = x.t() @ LC_pre_msgs
            LC_msgs = self.forward_gat((LC_pre_msgs, CL_pre_msgs), f)
            C_state_h, C_state_c = self.C_update(LC_msgs, (C_state_h, C_state_c))

            CL_pre_msgs = self.CL_msg(C_state_h)
            # CL_msgs = x @ CL_pre_msgs # n_lits * d
            CL_msgs = self.backward_gat((CL_pre_msgs, LC_pre_msgs), b)
            xx = torch.cat([L_state_h[n_vars:n_lits, :], L_state_h[0:n_vars, :]], 0)
            xxx = torch.cat([CL_msgs, xx], 1)
            L_state_h, L_state_c = self.L_update(xxx, (L_state_h, L_state_c))

        all_votes = self.L_vote(L_state_h)
        all_votes_join = torch.cat([all_votes[0:n_vars], all_votes[n_vars:n_lits]], 1)

        all_votes_batched = torch.reshape(all_votes_join, [n_batches, n_vars // n_batches, 2])
        logits = torch.mean(all_votes_batched, [1, 2]) + self.vote_bias
        print(logits)
        return logits

    def training_step(self, batch, batch_idx):
        x, f, b, y = batch
        outputs = self(x.float(), f.long(), b.long(), n_batches=len(y))
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.75)
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
        x, f, b, y = batch
        outputs = self(x.float(), f.long(), b.long(), n_batches=len(y))
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
        x, f, b, y = batch
        outputs = self(x.float(), f.long(), b.long(), n_batches=len(y))
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
        return torch.optim.Adam(self.parameters(), lr=5e-5, weight_decay=1e-10)
