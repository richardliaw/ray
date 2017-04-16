#!/bin/bash
for x in {0..2}
do
    for i in 1 2 4
    do
        for j in 100 30 10  
        do 
            python multimodel.py --num-experiments $i --adam 0.001 --sync $j
            python multimodel.py --num-experiments $i --adam 0.0001 --sync $j
            python multimodel.py --num-experiments $i --adam 0.00001 --sync $j
            python multimodel.py --num-experiments $i --adam 0.000001 --sync $j
        done
    done
done
