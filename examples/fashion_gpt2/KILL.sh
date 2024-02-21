#!/bin/bash
echo "Trying to kill processes matching keyword $1"
ps -ef | grep "$1" | grep -v grep | awk '{print $2}' | xargs kill -9
