import os
import random
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath('.'))
import PyMiniSolvers.minisolvers as minisolvers


def generate_k_iclause(n, k):
    vs = np.random.choice(n, size=min(n, k), replace=False)
    return [v + 1 if random.random() < 0.5 else -(v + 1) for v in vs]


def gen_iclause_pair(min_n, max_n, p_k_2, p_geo):
    n = random.randint(min_n, max_n)

    solver = minisolvers.MinisatSolver()
    for i in range(n): solver.new_var(dvar=True)

    iclauses = []

    while True:
        k_base = 1 if random.random() < p_k_2 else 2
        k = k_base + np.random.geometric(p_geo)
        iclause = generate_k_iclause(n, k)

        solver.add_clause(iclause)
        is_sat = solver.solve()
        if is_sat:
            iclauses.append(iclause)
        else:
            break

    iclause_unsat = iclause
    iclause_sat = [- iclause_unsat[0]] + iclause_unsat[1:]
    return n, iclauses, iclause_unsat, iclause_sat


class CnfDataSet(torch.utils.data.IterableDataset):
    def __init__(self, data_dir, batch_size=1):
        super(CnfDataSet, self).__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
