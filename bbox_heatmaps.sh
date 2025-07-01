#!/bin/bash
#tar -xzf centroids_inputs.tar.gz

#python3 -m pip install --no-input pandas matplotlib

echo "Some debug info"
pwd
ls -lh

python3 centroids.py -f *.json

#rm centroids_inputs.tar.gz
