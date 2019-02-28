#!/bin/bash

while true; do
    PYTHONPATH=. hyperopt-mongo-worker --mongo=localhost:27017/sacred --workdir=$(pwd)
    if [[ $? -ne 0 ]]; then
        break
    fi
done
