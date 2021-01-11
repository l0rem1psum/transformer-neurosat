import os
import random
import sys
from typing import List, Tuple

import numpy as np
import torch

sys.path.append(os.path.abspath('.'))
import PyMiniSolvers.minisolvers as minisolvers
from utils.problem import Problem


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
    def _shift_iclauses(iclauses, offset):
        return [[CnfGenerator._shift_ilit(x, offset) for x in iclause] for iclause in iclauses]

    @staticmethod
    def _mk_batch_problem(problems):
        all_iclauses = []
        all_is_sat = []
        all_n_cells = []
        all_dimacs = []
        offset = 0

        prev_n_vars = None
        for dimacs, n_vars, iclauses, is_sat in problems:
            assert (prev_n_vars is None or n_vars == prev_n_vars)
            prev_n_vars = n_vars

            all_iclauses.extend(CnfGenerator._shift_iclauses(iclauses, offset))
            all_is_sat.append(is_sat)
            all_n_cells.append(sum([len(iclause) for iclause in iclauses]))
            all_dimacs.append(dimacs)
            offset += n_vars

        return Problem(offset, all_iclauses, all_is_sat, all_n_cells, all_dimacs)

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
        count = self.partition[n_vars]
        while count == 0:
            n_vars += 1
            if n_vars > self.max_n:
                return None
            count = self.partition[n_vars]
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
    def __init__(self, data_dir, batch_size=1):
        super(CnfDataSet, self).__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.cnf_generator = None

    def __iter__(self):
        yield
