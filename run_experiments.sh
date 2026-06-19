#!/bin/bash
set -e
cd /Users/bytedance/ttadk-test/AGEA/agea
echo "=== Running AGEA Experiments ==="
python run_all_experiments.py 2>&1
