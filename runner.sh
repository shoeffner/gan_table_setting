#!/bin/bash -e
PYTHONPATH=. hyperopt-mongo-worker --mongo=localhost:27017/sacred --poll-interval=5 --workdir=$(pwd)
