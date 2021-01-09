import torch


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
