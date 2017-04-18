#!/bin/bash
for x in {0..2}
do
    for i in 2 3
    do
        for j in 100 30 10
        do
            python multimodel.py --num-experiments $i --sync $j
        done
    done
done
