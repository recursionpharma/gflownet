#!/bin/bash

set -e
set -x

python3.9 -m pip install -r requirements.txt

################################################################################
#### Test Case: Package With CLI
rm -rf test-package || true
python3.9 test_generate.py y

pushd test-package

python3.9 -m roadie.cli.main yapf dump -o ./style.yapf
python3.9 -m roadie.cli.main lock -u ${PYPI_DOWNLOAD_USERNAME} -p ${PYPI_DOWNLOAD_PASSWORD} --all
python3.9 -m roadie.cli.main venv -u ${PYPI_DOWNLOAD_USERNAME} -p ${PYPI_DOWNLOAD_PASSWORD} --use-venv
source venv/bin/activate

flake8
yapf --style ./style.yapf --parallel --diff -r .
mypy
bandit -r . -x /tests/,/venv/
safety check $(ls requirements/*.txt | xargs -i echo -n ' -r {}')
CONFIGOME_ENV=test pytest
pip install --upgrade tox
tox --parallel

popd

################################################################################
#### Test Case: Package Without a CLI
rm -rf test-package || true
python3.9 test_generate.py n

pushd test-package

python3.9 -m roadie.cli.main yapf dump -o ./style.yapf
python3.9 -m roadie.cli.main lock -u ${PYPI_DOWNLOAD_USERNAME} -p ${PYPI_DOWNLOAD_PASSWORD} --all
python3.9 -m roadie.cli.main venv -u ${PYPI_DOWNLOAD_USERNAME} -p ${PYPI_DOWNLOAD_PASSWORD} --use-venv
source venv/bin/activate

flake8
yapf --style ./style.yapf --parallel --diff -r .
mypy
bandit -r . -x /tests/,/venv/
safety check $(ls requirements/*.txt | xargs -i echo -n ' -r {}')
CONFIGOME_ENV=test pytest
pip install --upgrade tox
tox --parallel

popd
