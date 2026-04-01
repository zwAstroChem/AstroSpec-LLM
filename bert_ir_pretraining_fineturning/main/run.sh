#!/bin/bash

python3 main/finetune.py &
pid=$!
while ps -p $pid > /dev/null; do
  sleep 10
done
shutdown -h now
