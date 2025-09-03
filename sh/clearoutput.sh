#!/usr/bin/env bash

set -euo pipefail

nbs=$(find notebooks/ -type f -name "*.ipynb" -newer .last_cleanup) 
changed=0
for nb in $nbs; do
    jupyter nbconvert "$nb" --clear-output  
    changed=1
done

touch .last_cleanup

if [ $changed ]; then
    echo Some output was cleared!
    exit 1
else
    exit 0
fi