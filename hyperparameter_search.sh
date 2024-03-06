#!/bin/bash
for filename in configs/*; do
    python3 src/tune.py $filename
done