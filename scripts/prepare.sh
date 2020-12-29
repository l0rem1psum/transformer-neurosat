#!/bin/bash
#set -euxo pipefail
git clone https://github.com/liffiton/PyMiniSolvers.git
cd PyMiniSolvers
make
cd ..

mkdir data dimacs snapshots

