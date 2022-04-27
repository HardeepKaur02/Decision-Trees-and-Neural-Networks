#!/bin/bash

question=$1
train_data=$2
test_data=$3
part_num=$4

if [[ ${question} == "1" ]]; then
python3 Q1/q1.py $part_num
fi

if [[ ${question} == "2" ]]; then
python3 Q2/q2.py $part_num
fi