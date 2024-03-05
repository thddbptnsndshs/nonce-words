#!/bin/bash
for filename in configs/*.yaml; do
    python3 src/train.py $filename
done