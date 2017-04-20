#!/bin/bash
for x in {0..3}
do
    for i in 2 4 6
    do
        for j in 500 1000 2000
        do
            python multimodel.py --num-experiments $i --sync $j --runners 8 --aggr drop_half
            python multimodel.py --num-experiments $i --sync $j --runners 8 --aggr best
        done
    done
done
