#!/bin/bash

for ty in "train" "test"; do
  rm -rf data/pickle/$ty/sr5
  mkdir -p data/pickle/$ty/sr5
  for i in 1 2; do
    rm -rf data/dimacs/$ty/sr5/grp$i
    mkdir -p data/dimacs/$ty/sr5/grp$i
    python3 utils/gen_sr_dimacs.py data/dimacs/$ty/sr5/grp$i 50 --min_n 5 --max_n 10
    python3 utils/dimacs_to_data.py data/dimacs/$ty/sr5/grp$i data/pickle/$ty/sr5 60000
  done
done
