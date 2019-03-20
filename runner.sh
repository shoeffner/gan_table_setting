#!/bin/bash

while true; do
    PYTHONPATH=. hyperopt-mongo-worker --mongo="mongo://${MONGO_USER}:${MONGO_PASS}@${MONGO_HOST}/${MONGO_DB}/jobs?authSource=admin" --workdir="$(pwd)"
    if [[ $? -ne 0 ]]; then
        break
    fi
done
