import numpy as np


def ilit_to_var_sign(x):
    assert (abs(x) > 0)
    var = abs(x) - 1
    sign = x < 0
    return var, sign


def ilit_to_vlit(x, n_vars):
    assert (x != 0)
    var, sign = ilit_to_var_sign(x)
    if sign:
        return var + n_vars
    else:
        return var


class Problem(object):
    def __init__(self, n_vars, iclauses, is_sat, n_cells_per_batch, all_dimacs):
        self.n_vars = n_vars
        self.n_lits = 2 * n_vars
        self.n_clauses = len(iclauses)

        self.n_cells = sum(n_cells_per_batch)
        self.n_cells_per_batch = n_cells_per_batch

        self.is_sat = is_sat
        self.compute_L_unpack(iclauses)

        # will be a list of None for training problems
        self.dimacs = all_dimacs

    def compute_L_unpack(self, iclauses):
        self.L_unpack_indices = np.zeros([self.n_cells, 2], dtype=np.int)
        cell = 0
        for clause_idx, iclause in enumerate(iclauses):
            vlits = [ilit_to_vlit(x, self.n_vars) for x in iclause]
            for vlit in vlits:
                self.L_unpack_indices[cell, :] = [vlit, clause_idx]
                cell += 1

        assert (cell == self.n_cells)
