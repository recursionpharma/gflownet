from setuptools import find_packages, setup

import roadie

(install_requires, extras_require) = roadie.get_requirements()

setup(
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
)
