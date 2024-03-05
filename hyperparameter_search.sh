#!/bin/bash
for filename in configs/ar.yaml; do
    python3 src/tune.py $filename
done