#!/usr/bin/env bash

logs="creditcard_logs.txt"
input=$1
tmpset=/tmp/tmpset.csv

echo "" > $logs

for i in {0..99}; do
    shuf $input > $tmpset

    for control in 0 1; do
      python3 -u -m eval.dim_red_benchmarks $tmpset \
        --num-cols `for j in {0..29}; do echo $j; done`  \
        --has-header --control $control \
        --epochs 14 --split 120000 --loss abs \
        --label-col 30 --classes 0 1 >> $logs 2>&1
    done
done
