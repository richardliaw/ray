#!/bin/bash
for x in {0..3}
do
    for j in 50 100 500
    do
        python multimodel.py --num-experiments 2 --sync $j --runners 8 --aggr average --load ./progress/20170420_20_45_460483/policy.pkl
    done
done
