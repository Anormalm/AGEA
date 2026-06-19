#!/bin/bash
# Run FakeNews baselines and AGEA experiments
cd /Users/bytedance/ttadk-test/AGEA/agea

echo "============================================"
echo "Running FakeNews Baselines"
echo "============================================"
conda run -n ttadk-gnn python run_fakenews_baselines.py 2>&1

echo ""
echo "============================================"
echo "Running AGEA on PolitiFact"
echo "============================================"
conda run -n ttadk-gnn python run_agea.py --dataset fakenews_politifact 2>&1

echo ""
echo "============================================"
echo "Running AGEA on BuzzFeed"
echo "============================================"
conda run -n ttadk-gnn python run_agea.py --dataset fakenews_buzzfeed 2>&1

echo ""
echo "DONE"
