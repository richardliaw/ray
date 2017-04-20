#!/bin/bash
for x in {0..4}
do
    python driver.py --runners 8 --load ./progress/20170420_20_45_460483/policy.pkl
done
