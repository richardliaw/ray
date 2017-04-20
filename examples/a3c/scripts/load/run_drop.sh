#!/bin/bash
for x in {0..3}
do
    for i in 2 4
    do
        for j in 2000 1000 500
        do
            python multimodel.py --num-experiments $i --sync $j --runners 8 --aggr best --load ./progress/20170420_20_45_460483/policy.pkl
        done
    done
done
