#!/bin/sh
cp "/tmp/data/test.csv" test.csv
python predict.py
cp submission.csv "/var/log/result"