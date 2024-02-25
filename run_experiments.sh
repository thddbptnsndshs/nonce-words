#!/bin/bash
for filename in configs/*.yaml; do
    for ((i=0; i<=3; i++)); do
        python3 src/train.py $filename
    done
done