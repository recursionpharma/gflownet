#!/bin/bash

# Usage: ./generate_requirements.sh <ENV-NAME>  (e.g. ./generate_requirements.sh dev-3.10)

# set env variable
# to allow pip-compile-cross-platform to use pip with --find-links.
# not entirely sure this is needed.
export PIP_FIND_LINKS=https://data.pyg.org/whl/torch-2.1.2+cpu.html

# compile the dependencies from .in files
pip-compile-cross-platform requirements/$1.in --min-python-version 3.10 -o requirements/$1.txt

# remove the hashes from the .txt files
# this is slightly less safe in terms of reproducibility
# (e.g. if a package was re-uploaded to PyPI with the same version)
# but it is necessary to allow `pip install -r requirements` to use --find-links
# in our case, without --find-links, torch-cluster often cannot find the
# proper wheels and throws out an error `no torch module` when trying to build
sed -i '/--hash=/d' requirements/$1.txt
sed -i 's/\\//g' requirements/$1.txt

# removes the nvidia requirements
sed -i '/nvidia/d' requirements/$1.txt
