#!/bin/bash

set -x

# Cause the script to exit if a single command fails.
set -e

TP_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)

if [ ! -d $TP_DIR/arrow ]; then
  git clone https://github.com/apache/arrow/ "$TP_DIR/arrow"
fi
cd $TP_DIR/arrow
git pull origin master

git checkout 1514016a72385f28235fc36b19274b46e25c389a
