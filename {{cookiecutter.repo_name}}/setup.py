from setuptools import find_packages, setup

import roadie

(install_requires, extras_require) = roadie.get_requirements()

setup(
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
  {%- if cookiecutter.cli == 'y' %}
    entry_points={'console_scripts': ['{{cookiecutter.python_name}}={{cookiecutter.python_name}}.cli:cli']},
  {%- endif %}
)
