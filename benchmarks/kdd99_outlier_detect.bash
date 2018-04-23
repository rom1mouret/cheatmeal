#!/usr/bin/env bash

logs="kdd99logs.txt"
input=$1
tmpset=/tmp/tmpset.csv

echo "" > $logs

u2r="buffer_overflow. loadmodule. perl. rootkit."
r2l="ftp_write. guess_passwd. imap. multihop. phf. spy. warezclient. warezmaster."
probe="ipsweep. nmap. portsweep. satan."

for i in {0..50}; do
    shuf $input > $tmpset

    for f in 2 0; do
      python3 -u -m eval.outlier_detect_benchmarks $tmpset --forgiveness $f --gpu 0 0 0 \
        --ensemble 3 \
        --binary-cols 6 11 13 20 21 \
        --cat-cols 1 2 3 14 \
        --num-cols 0 4 5 7 8 9 10 12 15 16 17 18 19 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 \
        --epochs 1 --split 4000000 --merge harmonic \
        --label-col 41 --normal-classes "normal." --anomaly-classes $u2r $r2l $probe >> $logs 2>&1
    done
done
