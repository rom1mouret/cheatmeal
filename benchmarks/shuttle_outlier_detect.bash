#!/usr/bin/env bash

logs="shuttlelogs.txt"
input=$1
tmpset=/tmp/tmpset.csv

echo "" > $logs

for i in {0..20}; do
    shuf $input > $tmpset

    for f in 2 0; do
      python3 -u -m eval.outlier_detect_benchmarks $tmpset --forgiveness $f \
        --num-cols 0 1 2 3 4 5 6 7 8 9 --delimiter ' ' \
        --epochs 12 --split 40000 --merge harmonic --loss huber \
        --label-col 9 --normal-classes 1 --anomaly-classes 2 3 4 5 6 7 >> $logs 2>&1
    done
done
