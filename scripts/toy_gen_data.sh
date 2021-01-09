#!/bin/bash

DATA_DIR=data

python3 utils/data_module.py 100 60000 \
  --data_dir $DATA_DIR/ \
  --min_n 5 \
  --max_n 10
