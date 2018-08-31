#!/bin/bash
while true; do
    nvidia-smi | egrep -o [0-9]+\%;
    sleep 4;
done
