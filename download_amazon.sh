#!/bin/bash
set -e
cd /Users/bytedance/ttadk-test/AGEA/agea

echo "=== Downloading Amazon All_Beauty ==="
mkdir -p dataset/amazon
cd dataset/amazon
curl -L -o All_Beauty.jsonl.gz \
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz
ls -lh All_Beauty.jsonl.gz

echo "=== Done ==="
