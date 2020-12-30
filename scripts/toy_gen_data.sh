#!/bin/bash

for ty in "train" "test"; do
  rm -rf data/pickle/$ty/sr10-40
  mkdir -p data/pickle/$ty/sr10-40
  for i in 1 2; do
    rm -rf data/dimacs/$ty/sr10-40/grp$i
    mkdir -p data/dimacs/$ty/sr10-40/grp$i
    python3 utils/gen_sr_dimacs.py data/dimacs/$ty/sr10-40/grp$i 10000 --min_n 10 --max_n 40
    python3 utils/dimacs_to_data.py data/dimacs/$ty/sr10-40/grp$i data/pickle/$ty/sr10-40 60000
  done
done
