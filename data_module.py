import argparse
import os
import pickle
import random
import uuid

import numpy as np
import pytorch_lightning as pl

import PyMiniSolvers.minisolvers as minisolvers
from utils import SatProblemDataSet
from utils.mk_problem import Problem


def generate_k_iclause(n, k):
    vs = np.random.choice(n, size=min(n, k), replace=False)
    return [v + 1 if random.random() < 0.5 else -(v + 1) for v in vs]


def write_dimacs_to(n_vars, iclauses, out_filename):
    with open(out_filename, 'w') as f:
        f.write("p cnf %d %d\n" % (n_vars, len(iclauses)))
        for c in iclauses:
            for x in c:
                f.write("%d " % x)
            f.write("0\n")


def parse_dimacs(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    i = 0
    while lines[i].strip().split(" ")[0] == "c":
        i += 1
    header = lines[i].strip().split(" ")
    assert (header[0] == "p")
    n_vars = int(header[2])
    iclauses = [[int(s) for s in line.strip().split(" ")[:-1]] for line in lines[i + 1:]]
    return n_vars, iclauses


def shift_ilit(x, offset):
    assert (x != 0)
    if x > 0:
        return x + offset
    else:
        return x - offset


def shift_iclauses(iclauses, offset):
    return [[shift_ilit(x, offset) for x in iclause] for iclause in iclauses]


def mk_batch_problem(problems):
    all_iclauses = []
    all_is_sat = []
    all_n_cells = []
    all_dimacs = []
    offset = 0

    prev_n_vars = None
    for dimacs, n_vars, iclauses, is_sat in problems:
        assert (prev_n_vars is None or n_vars == prev_n_vars)
        prev_n_vars = n_vars

        all_iclauses.extend(shift_iclauses(iclauses, offset))
        all_is_sat.append(is_sat)
        all_n_cells.append(sum([len(iclause) for iclause in iclauses]))
        all_dimacs.append(dimacs)
        offset += n_vars

    return Problem(offset, all_iclauses, all_is_sat, all_n_cells, all_dimacs)


def solve_sat(n_vars, iclauses):
    solver = minisolvers.MinisatSolver()
    for i in range(n_vars): solver.new_var(dvar=True)
    for iclause in iclauses: solver.add_clause(iclause)
    is_sat = solver.solve()
    stats = solver.get_stats()
    return is_sat, stats


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


def mk_dataset_filename(dimacs_dir, pickle_dir, max_nodes_per_batch, n_batches):
    dimacs_path = dimacs_dir.split("/")
    dimacs_dir = dimacs_path[-1] if dimacs_path[-1] != "" else dimacs_path[-2]
    return "%s/data_dir=%s_npb=%d_nb=%d.pkl" % (
        pickle_dir, dimacs_dir, max_nodes_per_batch, n_batches)


def make_pickle_from_dimacs(dimacs_dir, pickle_dir, max_nodes_per_batch, one):
    problems = []
    batches = []
    n_nodes_in_batch = 0
    prev_n_vars = None

    filenames = os.listdir(dimacs_dir)
    filenames = sorted(filenames)

    for filename in filenames:
        n_vars, iclauses = parse_dimacs("%s/%s" % (dimacs_dir, filename))
        n_clauses = len(iclauses)
        # n_cells = sum([len(iclause) for iclause in iclauses])

        n_nodes = 2 * n_vars + n_clauses
        if n_nodes > max_nodes_per_batch:
            continue

        batch_ready = False
        if (one and len(problems) > 0):
            batch_ready = True
        elif (prev_n_vars and n_vars != prev_n_vars):
            batch_ready = True
        elif (not one) and n_nodes_in_batch + n_nodes > max_nodes_per_batch:
            batch_ready = True

        if batch_ready:
            batches.append(mk_batch_problem(problems))
            del problems[:]
            n_nodes_in_batch = 0

        prev_n_vars = n_vars

        is_sat, stats = solve_sat(n_vars, iclauses)
        problems.append((filename, n_vars, iclauses, is_sat))
        n_nodes_in_batch += n_nodes

    if len(problems) > 0:
        batches.append(mk_batch_problem(problems))
        del problems[:]

    dataset_filename = mk_dataset_filename(
        dimacs_dir,
        pickle_dir,
        max_nodes_per_batch,
        len(batches)
    )
    with open(dataset_filename, 'wb') as f_dump:
        pickle.dump(batches, f_dump)


class SATDataModule(pl.LightningDataModule):
    __SAT_DIMACS_FILENAME_FORMAT = "{}/sr_n={:04d}_pk2={:.2f}_pg={:.2f}_t={}_sat=0.dimacs"
    __UNSAT_DIMACS_FILENAME_FORMAT = "{}/sr_n={:04d}_pk2={:.2f}_pg={:.2f}_t={}_sat=1.dimacs"
    __STAGES = ["train", "test"]

    def __init__(self, opts):
        super(SATDataModule, self).__init__()
        self.uuid = uuid.uuid4()
        if opts.run_dir is not None:
            opts.__setattr__("dimacs_dir", os.path.join(opts.run_dir, self.uuid.hex[:8], "data/dimacs"))
            opts.__setattr__("pickle_dir", os.path.join(opts.run_dir, self.uuid.hex[:8], "data/pickle"))

        self.opts = opts
        for s in SATDataModule.__STAGES:
            os.makedirs(os.path.join(self.opts.dimacs_dir, s))
            os.makedirs(os.path.join(self.opts.pickle_dir, s))

    def prepare_data(self):
        for stage in SATDataModule.__STAGES:
            for pair in range(self.opts.n_pairs):
                n_vars, iclauses, iclause_unsat, iclause_sat = gen_iclause_pair(
                    self.opts.min_n,
                    self.opts.max_n,
                    self.opts.p_k_2,
                    self.opts.p_geo
                )

                iclauses.append(iclause_unsat)
                write_dimacs_to(
                    n_vars,
                    iclauses,
                    SATDataModule.__UNSAT_DIMACS_FILENAME_FORMAT.format(
                        os.path.join(self.opts.dimacs_dir, stage),
                        n_vars,
                        self.opts.p_k_2,
                        self.opts.p_geo,
                        pair
                    )
                )

                iclauses[-1] = iclause_sat
                write_dimacs_to(
                    n_vars,
                    iclauses,
                    SATDataModule.__SAT_DIMACS_FILENAME_FORMAT.format(
                        os.path.join(self.opts.dimacs_dir, stage),
                        n_vars,
                        self.opts.p_k_2,
                        self.opts.p_geo,
                        pair
                    )
                )

    def setup(self, stage=None):
        for stage in SATDataModule.__STAGES:
            make_pickle_from_dimacs(
                os.path.join(self.opts.dimacs_dir, stage),
                os.path.join(self.opts.pickle_dir, stage),
                self.opts.max_nodes_per_batch,
                self.opts.one
            )

    def train_dataloader(self):
        return SatProblemDataSet(self.opts.pickle_dir + "train")

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        return SatProblemDataSet(self.opts.pickle_dir + "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('n_pairs', action='store', type=int)
    parser.add_argument('max_nodes_per_batch', action='store', type=int)

    parser.add_argument('--run_dir', action='store', type=str)
    parser.add_argument('--dimacs_dir', action='store', type=str)
    parser.add_argument('--pickle_dir', action='store', type=str)

    parser.add_argument('--min_n', action='store', dest='min_n', type=int, default=40)
    parser.add_argument('--max_n', action='store', dest='max_n', type=int, default=40)

    parser.add_argument('--p_k_2', action='store', dest='p_k_2', type=float, default=0.3)
    parser.add_argument('--p_geo', action='store', dest='p_geo', type=float, default=0.4)

    parser.add_argument('--py_seed', action='store', dest='py_seed', type=int, default=None)
    parser.add_argument('--np_seed', action='store', dest='np_seed', type=int, default=None)

    parser.add_argument('--print_interval', action='store', dest='print_interval', type=int, default=100)

    parser.add_argument('--one', action='store', dest='one', type=int, default=0)
    parser.add_argument('--max_dimacs', action='store', dest='max_dimacs', type=int, default=None)

    opts = parser.parse_args()

    data_module = SATDataModule(opts)
    data_module.prepare_data()
    data_module.setup()
