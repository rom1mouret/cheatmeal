#!/usr/bin/env bash

logs="mammologs.txt"
input=$1
tmpset=/tmp/tmpset.csv

echo "" > $logs

for i in {0..20}; do
    shuf $input > $tmpset

    for f in 1 0; do
      python3 -u -m eval.outlier_detect_benchmarks $tmpset --forgiveness $f \
        --num-cols 0 1 2 3 4 5  --has-header \
        --epochs 20 --split 10000 --merge harmonic --loss huber \
        --label-col 6 --normal-classes "'-1'" --anomaly-classes "'1'"  >> $logs 2>&1
    done
done
