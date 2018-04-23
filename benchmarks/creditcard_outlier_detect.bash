#!/usr/bin/env bash

logs="creditcardlogs.txt"
input=$1
tmpset=/tmp/tmpset.csv

echo "" > $logs

for i in {0..20}; do
    shuf $input > $tmpset

    for f in 2 0; do
      python3 -u -m eval.outlier_detect_benchmarks $tmpset --forgiveness $f \
        --num-cols `for j in {0..29}; do echo $j; done`  \
        --has-header \
        --epochs 3 --split 120000 --merge harmonic --loss huber \
        --label-col 30 --normal-classes 0 --anomaly-classes 1  >> $logs 2>&1
    done
done
