#!/bin/bash

for i in {0..9}; do
    echo "Processing script $i"
    python3 ./main_debug.py --i $i --fmap 'rf' --dataset 'synthetic' --activation 'relu_priv' > outputs/$i.txt
done
