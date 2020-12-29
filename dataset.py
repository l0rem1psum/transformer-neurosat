import torch
import os
import pickle
import numpy as np


class SatProblemDataSet(torch.utils.data.Dataset):
    def __init__(self, dirname):
        super(SatProblemDataSet, self).__init__()

        self.dirname = dirname
        self.filenames = [dirname + "/" + f for f in os.listdir(dirname)]
        self.problems = list()
        for filename in self.filenames:
            with open(filename, 'rb') as f:
                problems = pickle.load(f)
                self.problems.extend(problems)

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        problem = self.problems[idx]
        x = np.zeros([problem.n_lits, problem.n_clauses], dtype=np.float64)
        for i in problem.L_unpack_indices:
            x[i[0], i[1]] = 1
        y = list()
        for i in problem.is_sat:
            y.append(float(i))
        return torch.tensor(x), torch.tensor(y)
