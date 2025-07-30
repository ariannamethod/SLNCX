#!/bin/bash
# Run daily maintenance tasks for Wulf1

set -e

python scripts/entropy_prune.py
python scripts/memory_vector.py
python wulf_train.py
