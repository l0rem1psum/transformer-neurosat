import math
import torch
from utils import repeat_end


class MLP(torch.nn.Module):
    def __init__(self, d_in, d_outs):
        super(MLP, self).__init__()
        self.d_in = d_in
        self.d_outs = d_outs

        self.linears = torch.nn.ModuleList()
        self.activation_func = torch.nn.ReLU()

        self._initialize_layers()

    def _initialize_layers(self):
        d_in = self.d_in
        for d_out in self.d_outs:
            l = torch.nn.Linear(d_in, d_out)
            torch.nn.init.xavier_uniform_(l.weight)
            torch.nn.init.zeros_(l.bias)
            self.linears.append(l)
            d_in = d_out

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if i < len(self.linears) - 1:
                x = self.activation_func(x)
        return x


class LayerNormBasicLSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LayerNormBasicLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fiou_linear = torch.nn.Linear(input_size + hidden_size, hidden_size * 4, bias=False)
        self.fiou_ln_layers = torch.nn.ModuleList(torch.nn.LayerNorm(hidden_size) for _ in range(4))
        self.cell_ln = torch.nn.LayerNorm(hidden_size)

    def forward(self, input, state):
        hidden_tensor, cell_tensor = state
        fiou_linear = self.fiou_linear(torch.cat([input, hidden_tensor], dim=1))
        fiou_linear_tensors = fiou_linear.split(self.hidden_size, dim=1)
        fiou_linear_tensors = tuple(ln(tensor) for ln, tensor in zip(self.fiou_ln_layers, fiou_linear_tensors))

        f, i, o = tuple(torch.sigmoid(tensor) for tensor in fiou_linear_tensors[:3])
        u = torch.tanh(fiou_linear_tensors[3])

        new_cell = self.cell_ln(i * u + (f * cell_tensor))
        new_h = o * torch.tanh(new_cell)

        return new_h, new_cell


class NeuroSAT(torch.nn.Module):
    def __init__(self, d, n_msg_layers, n_vote_layers, n_rounds):
        super(NeuroSAT, self).__init__()

        self.d = d
        self.n_rounds = n_rounds

        self.L_init = torch.nn.Parameter(torch.empty([1, d]))
        self.C_init = torch.nn.Parameter(torch.empty([1, d]))

        self.LC_msg = MLP(d, repeat_end(d, n_msg_layers, d))
        self.CL_msg = MLP(d, repeat_end(d, n_msg_layers, d))

        self.L_update = LayerNormBasicLSTMCell(2 * d, d)
        self.C_update = LayerNormBasicLSTMCell(d, d)

        self.L_vote = MLP(d, repeat_end(d, n_vote_layers, 1))
        self.vote_bias = torch.nn.Parameter(torch.empty([]))

        self._init_weight()

    def _init_weight(self):
        torch.nn.init.normal_(self.L_init)
        torch.nn.init.normal_(self.C_init)
        torch.nn.init.zeros_(self.vote_bias)

    def forward(self, x, n_batches=1):
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