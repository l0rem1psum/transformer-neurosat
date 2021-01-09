import math

import pytorch_lightning as pl
import torch

from model import MLP, LayerNormBasicLSTMCell, compute_loss


class NeuroSAT(pl.LightningModule):
    def __init__(self, d, n_msg_layers, n_vote_layers, n_rounds):
        super(NeuroSAT, self).__init__()

        self.d = d
        self.n_rounds = n_rounds

        self.L_init = torch.nn.Parameter(torch.empty([1, d]))
        self.C_init = torch.nn.Parameter(torch.empty([1, d]))

        self.LC_msg = MLP(d, [d for _ in range(n_msg_layers)] + [d])
        self.CL_msg = MLP(d, [d for _ in range(n_msg_layers)] + [d])

        self.L_update = LayerNormBasicLSTMCell(2 * d, d)
        self.C_update = LayerNormBasicLSTMCell(d, d)

        self.L_vote = MLP(d, [d for _ in range(n_vote_layers)] + [1])
        self.vote_bias = torch.nn.Parameter(torch.empty([]))

        self._init_weight()

        # Metrics
        self.accuracy = pl.metrics.Accuracy()

    def _init_weight(self):
        torch.nn.init.normal_(self.L_init)
        torch.nn.init.normal_(self.C_init)
        torch.nn.init.zeros_(self.vote_bias)

    def forward(self, x, n_batches):
        n_lits, n_clauses = x.size()
        n_vars = n_lits // 2
        denom = math.sqrt(self.d)

        L_state_h = (self.L_init / denom).repeat([n_lits, 1])
        L_state_c = torch.zeros([n_lits, self.d])

        C_state_h = (self.C_init / denom).repeat([n_clauses, 1])
        C_state_c = torch.zeros([n_clauses, self.d])

        for i in range(self.n_rounds):
            LC_pre_msgs = self.LC_msg(L_state_h)
            LC_msgs = x.t() @ LC_pre_msgs
            C_state_h, C_state_c = self.C_update(LC_msgs, (C_state_h, C_state_c))

            CL_pre_msgs = self.CL_msg(C_state_h)
            CL_msgs = x @ CL_pre_msgs
            xx = torch.cat([L_state_h[n_vars:n_lits, :], L_state_h[0:n_vars, :]], 0)
            xxx = torch.cat([CL_msgs, xx], 1)
            L_state_h, L_state_c = self.L_update(xxx, (L_state_h, L_state_c))

        all_votes = self.L_vote(L_state_h)
        all_votes_join = torch.cat([all_votes[0:n_vars], all_votes[n_vars:n_lits]], 1)

        all_votes_batched = torch.reshape(all_votes_join, [n_batches, n_vars // n_batches, 2])
        logits = torch.mean(all_votes_batched, [1, 2]) + self.vote_bias

        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x[0], y[0]
        outputs = self(x.float(), n_batches=len(y))
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        loss = compute_loss(outputs, y, self.parameters())
        self.log_dict(
            {
                "train_loss_step": loss.item(),
                "train_acc_step": self.accuracy(outputs > 0, y),
            },
            prog_bar=True,
            on_step=True,
        )
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.log('train_acc_epoch', self.accuracy.compute())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-10)
