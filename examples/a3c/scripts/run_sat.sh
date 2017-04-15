#!/bin/bash
for x in {0..2}
do
    for i in 1 2 4
    do
        for j in 3 6
        do 
            python multimodel.py --num-experiments $i --runners $j 
        done
    done
done
