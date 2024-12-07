#!/bin/bash

# Find all Jupyter notebooks in the repository
notebooks=$(ls -R| grep '\.ipynb$')

# Loop over each notebook and clear its output
for notebook in $notebooks; do
    echo "Clearing output from $notebook"
    jupyter nbconvert --clear-output --inplace "$notebook"
done
