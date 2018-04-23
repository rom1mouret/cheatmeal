#!/usr/bin/env bash

logs="covtype_logs.txt"
input=$1
tmpset=/tmp/tmpset.csv

echo "" > $logs

for i in {0..40}; do
    shuf $input > $tmpset

    for f in 1 0; do
      python3 -u -m eval.outlier_detect_benchmarks $tmpset --forgiveness $f \
        --binary-cols `for j in {10..53}; do echo $j; done` \
        --num-cols 1 2 3 4 5 6 7 8 9 \
        --epochs 8 --split 500000 \
        --merge harmonic \
        --ensemble 3 \
        --label-col 54 --normal-classes 2 --anomaly-classes 4 --gpu 0 0 0 >> $logs
    done
done
