# Contributing

Contributions to the repository are welcome, and we encourage you to open issues and pull requests. In general, it is recommended to fork this repository and open a pull request from your fork to the `trunk` branch. PRs are encouraged to be short and focused, and to include tests and documentation where appropriate.

## Installation

To install the developers dependencies run:
```
pip install -e '.[dev]' --find-links https://data.pyg.org/whl/torch-2.1.2+cu121.html
```

## Dependencies

Dependencies are defined in `pyproject.toml`, and frozen versions that are known to work are provided in `requirements/`. 

To regenerate the frozen versions, run `./generate_requirements.sh <ENV-NAME>`. See comments within.

## Linting and testing

We use `tox` to run tests and linting, and `pre-commit` to run checks before committing.
To ensure that these checks pass, simply run `tox -e style` and `tox run` to run linters and tests, respectively.

`tox` itself runs many linters, but the most important ones are `black`, `ruff`, `isort`, and `mypy`. The full list
of linting tools is found in `.pre-commit-config.yaml`, while `tox.ini` defines the environments under which these 
linters (as well as tests) are run.

## Github Actions

We use Github Actions to run tests and linting on every push and pull request. The configuration for these actions is found in `.github/workflows/`.

## Style Guide

On top of `black`-as-a-style-guide, we generally adhere to the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). 
Our docstrings follow the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format, and we use type hints throughout the codebase.
