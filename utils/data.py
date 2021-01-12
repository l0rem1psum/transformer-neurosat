import os
import random
import sys
import uuid
from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch

sys.path.append(os.path.abspath('.'))
import PyMiniSolvers.minisolvers as minisolvers


class CnfGenerator:
    def __init__(self, one: bool = True, min_n: int = 2, max_n: int = 4, p_k_2: float = 0.3, p_geo: float = 0.4,
                 max_nodes_per_batch: int = 20000):
        self.one = one
        self.min_n = min_n
        self.max_n = max_n
        self.p_k_2 = p_k_2
        self.p_geo = p_geo
        self.max_nodes_per_batch = max_nodes_per_batch

        self.n_pairs = None
        self.partition = None
        self.cached_problem = None

    @staticmethod
    def _generate_k_iclause(n, k):
        vs = np.random.choice(n, size=min(n, k), replace=False)
        return [v + 1 if random.random() < 0.5 else -(v + 1) for v in vs]

    @staticmethod
    def _shift_ilit(x, offset):
        assert (x != 0)
        if x > 0:
            return x + offset
        else:
            return x - offset

    @staticmethod
    def shift_iclauses(iclauses, offset):
        return [[CnfGenerator._shift_ilit(x, offset) for x in iclause] for iclause in iclauses]

    def _generate_iclause_pair(self, n_vars):
        solver = minisolvers.MinisatSolver()
        for i in range(n_vars): solver.new_var(dvar=True)

        iclauses = []

        while True:
            k_base = 1 if random.random() < self.p_k_2 else 2
            k = k_base + np.random.geometric(self.p_geo)
            iclause = self._generate_k_iclause(n_vars, k)

            solver.add_clause(iclause)
            is_sat = solver.solve()
            if is_sat:
                iclauses.append(iclause)
            else:
                break

        iclause_unsat = iclause
        iclause_sat = [- iclause_unsat[0]] + iclause_unsat[1:]

        unsat_clauses = iclauses.copy()
        unsat_clauses.append(iclause_unsat)

        iclauses.append(iclause_sat)

        return unsat_clauses, iclauses

    def _make_partition(self, n):
        rand_n_vars = np.random.choice(range(self.min_n, self.max_n + 1), n)
        unique, counts = np.unique(rand_n_vars, return_counts=True)
        self.partition = dict(zip(unique, counts))

    def _get_next_n_var(self):
        n_vars = self.min_n
        count = self.partition.get(n_vars, 0)
        while count == 0:
            n_vars += 1
            if n_vars > self.max_n:
                return None
            count = self.partition.get(n_vars, 0)
        self.partition[n_vars] -= 1
        return n_vars

    def set_n_pairs(self, n_pairs):
        if self.n_pairs is not None:
            raise ValueError("Value of n_pairs is already set")
        self.n_pairs = n_pairs
        self._make_partition(n_pairs)

    def generate_one_minibatch(self) -> List[Tuple[int, List[List[int]], bool]]:
        if self.n_pairs is None:
            raise ValueError("Give a value to n_pairs first")

        if self.cached_problem is not None:
            problem = self.cached_problem
            self.cached_problem = None
            return problem

        problems = []
        n_nodes_in_batch = 0
        prev_n_vars = None

        while True:
            n_vars = self._get_next_n_var()
            if n_vars is None:
                break  # all n_pairs are generated
            elif prev_n_vars and n_vars != prev_n_vars:
                self.partition[n_vars] += 1  # put back n_vars
                break

            unsat_clauses, sat_clauses = self._generate_iclause_pair(n_vars)
            n_clauses = len(unsat_clauses)
            n_nodes = 2 * (2 * n_vars + n_clauses)
            if n_nodes > self.max_nodes_per_batch:
                continue

            if self.one:
                self.cached_problem = [(n_vars, sat_clauses, True)]
                return [(n_vars, unsat_clauses, False)]

            if n_nodes_in_batch + n_nodes > self.max_nodes_per_batch:
                return problems

            problems.append((n_vars, unsat_clauses, False))
            problems.append((n_vars, sat_clauses, True))

            prev_n_vars = n_vars
            n_nodes_in_batch += n_nodes

        return problems


class CnfDataSet(torch.utils.data.IterableDataset):
    def __init__(self, data_dir: str, requires_generation: bool = False, one: bool = True, n_pairs: int = 100,
                 min_n: int = 2, max_n: int = 4,
                 p_k_2: float = 0.3, p_geo: float = 0.4,
                 max_nodes_per_batch: int = 20000):
        super(CnfDataSet, self).__init__()

        self.data_dir = data_dir
        self.requires_generation = requires_generation
        self.one = one
        self.n_pairs = n_pairs
        self.min_n = min_n
        self.max_n = max_n
        self.p_k_2 = p_k_2
        self.p_geo = p_geo
        self.max_nodes_per_batch = max_nodes_per_batch

        self.cnf_generator = None
        if requires_generation:
            self.cnf_generator = CnfGenerator(one, min_n, max_n, p_k_2, p_geo, max_nodes_per_batch)
            self.cnf_generator.set_n_pairs(self.n_pairs)

    @staticmethod
    def _read_dimacs(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        i = 0
        while lines[i].strip().split(" ")[0] == "c":
            i += 1
        header = lines[i].strip().split(" ")
        assert (header[0] == "p")
        n_vars = int(header[2])
        iclauses = [[int(s) for s in line.strip().split(" ")[:-1]] for line in lines[i + 1:]]
        is_sat = int(os.path.splitext(filename)[0][-1])
        return n_vars, iclauses, bool(is_sat)

    def _write_dimacs(self, problems: List[Tuple[int, List[List[int]], bool]], n_problems_so_far: int):
        for i, (n_vars, iclauses, is_sat) in enumerate(problems):
            out_filename = '{}/sr_n={:04d}_pk2={:.2f}_pg={:.2f}_t={}_sat={}.dimacs'.format(
                os.path.join(self.data_dir),
                n_vars,
                self.p_k_2,
                self.p_geo,
                n_problems_so_far + i,
                int(is_sat)
            )
            with open(out_filename, 'w') as f:
                f.write("p cnf %d %d\n" % (n_vars, len(iclauses)))
                for c in iclauses:
                    for x in c:
                        f.write("%d " % x)
                    f.write("0\n")

    def __len__(self):
        return self.n_pairs * 2

    def __iter__(self):
        if self.requires_generation and self.cnf_generator is not None:
            count = 0
            minibatch = self.cnf_generator.generate_one_minibatch()
            while minibatch:
                yield minibatch
                self._write_dimacs(minibatch, count)
                count += len(minibatch)
                minibatch = self.cnf_generator.generate_one_minibatch()
            self.cnf_generator = None
            self.requires_generation = False
            return

        problems = []
        n_nodes_in_batch = 0
        prev_n_vars = None

        filenames = os.listdir(self.data_dir)
        filenames = sorted(filenames)

        for filename in filenames:
            n_vars, iclauses, is_sat = self._read_dimacs("%s/%s" % (self.data_dir, filename))
            n_clauses = len(iclauses)

            n_nodes = 2 * n_vars + n_clauses
            if n_nodes > self.max_nodes_per_batch:
                continue

            if (
                    (self.one and len(problems) > 0) or
                    (prev_n_vars and n_vars != prev_n_vars) or
                    ((not self.one) and n_nodes_in_batch + n_nodes > self.max_nodes_per_batch)
            ):
                yield problems
                del problems[:]
                n_nodes_in_batch = 0

            problems.append((n_vars, iclauses, is_sat))

            prev_n_vars = n_vars
            n_nodes_in_batch += n_nodes


class CnfDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, one: bool = True, n_pairs: int = 100,
                 min_n: int = 2, max_n: int = 4,
                 p_k_2: float = 0.3, p_geo: float = 0.4,
                 max_nodes_per_batch: int = 20000):
        super(CnfDataModule, self).__init__()

        self.data_dir = data_dir

        self.one = one
        self.n_pairs = n_pairs
        self.min_n = min_n
        self.max_n = max_n
        self.p_k_2 = p_k_2
        self.p_geo = p_geo
        self.max_nodes_per_batch = max_nodes_per_batch

        self.requires_generation = None
        self.uuid = None

    def prepare_data(self):
        # check if data of same parameter already exists
        self.requires_generation = True
        self.uuid = uuid.uuid4().hex[:8]

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        data_dir = os.path.join(self.data_dir, self.uuid, "train")
        os.makedirs(data_dir)
        ds = CnfDataSet(
            data_dir=data_dir,
            requires_generation=self.requires_generation,
            one=self.one,
            n_pairs=self.n_pairs,
            min_n=self.min_n,
            max_n=self.max_n,
            p_k_2=self.p_k_2,
            p_geo=self.p_geo,
            max_nodes_per_batch=self.max_nodes_per_batch
        )
        return torch.utils.data.DataLoader(ds, batch_size=None, collate_fn=self.collate_fn)

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        data_dir = os.path.join(self.data_dir, self.uuid, "test")
        os.makedirs(data_dir)
        ds = CnfDataSet(
            data_dir=data_dir,
            requires_generation=self.requires_generation,
            one=self.one,
            n_pairs=self.n_pairs,
            min_n=self.min_n,
            max_n=self.max_n,
            p_k_2=self.p_k_2,
            p_geo=self.p_geo,
            max_nodes_per_batch=self.max_nodes_per_batch
        )
        return torch.utils.data.DataLoader(ds, batch_size=None, collate_fn=self.collate_fn)

    @staticmethod
    def _ilit_to_var_sign(x: int):
        assert (abs(x) > 0)
        var = abs(x) - 1
        sign = x < 0
        return var, sign

    @staticmethod
    def _ilit_to_vlit(x: int, n_vars: int):
        assert (x != 0)
        var, sign = CnfDataModule._ilit_to_var_sign(x)
        if sign:
            return var + n_vars
        else:
            return var

    @staticmethod
    def collate_fn(problems: List[Tuple[int, List[List[int]], bool]]):
        all_iclauses = []
        all_is_sat = []
        all_n_cells = []
        offset = 0

        prev_n_vars = None
        for n_vars, iclauses, is_sat in problems:
            assert (prev_n_vars is None or n_vars == prev_n_vars)
            prev_n_vars = n_vars

            all_iclauses.extend(CnfGenerator.shift_iclauses(iclauses, offset))
            all_is_sat.append(is_sat)
            all_n_cells.append(sum([len(iclause) for iclause in iclauses]))
            offset += n_vars

        n_cells = sum(all_n_cells)
        L_unpack_indices = np.zeros([n_cells, 2], dtype=np.int)
        cell = 0
        for clause_idx, iclause in enumerate(all_iclauses):
            vlits = [CnfDataModule._ilit_to_vlit(x, offset) for x in iclause]
            for vlit in vlits:
                L_unpack_indices[cell, :] = [vlit, clause_idx]
                cell += 1

        assert (cell == n_cells)

        x = np.zeros([2 * offset, len(all_iclauses)], dtype=np.float64)
        for i in L_unpack_indices:
            x[i[0], i[1]] = 1
        y = list()
        for i in all_is_sat:
            y.append(float(i))
        return torch.tensor(x), torch.tensor(y)
