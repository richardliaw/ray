#!/bin/bash
for x in {0..4}
do
    for i in 2 4 6
    do
        for j in 100 200 500 
        do
            python multimodel.py --num-experiments $i --sync $j --runners 8
        done
    done
done
