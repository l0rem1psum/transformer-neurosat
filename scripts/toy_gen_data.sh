#!/bin/bash

DIMACS_DIR=data/dimacs
PICKLE_DIR=data/pickle

rm -rf $DIMACS_DIR $PICKLE_DIR
mkdir -p $DIMACS_DIR $PICKLE_DIR

python3 data_module.py 100 60000 \
  --dimacs_dir $DIMACS_DIR \
  --pickle_dir $PICKLE_DIR \
  --min_n 5 \
  --max_n 10
