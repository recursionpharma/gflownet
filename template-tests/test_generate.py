from cookiecutter.main import cookiecutter
from roadie.constants import FLAKE8_STYLE, YAPF_STYLE
import sys


def main():

    yapf_configuration = ''
    for k, v in YAPF_STYLE.items():
        yapf_configuration += f'{k.lower()} = {str(v).lower()}\n'
    flake8_configuration = ''
    for k, v in FLAKE8_STYLE.items():
        flake8_configuration += f'{k.lower()} = {str(v).lower()}\n'

    cookiecutter_context = {
        'repo_name': "test-package",
        'description': "words go here",
        'type': 'package',
        'yapf': yapf_configuration,
        'flake8': flake8_configuration,
        'cli': sys.argv[1],
    }
    cookiecutter('..', no_input=True, extra_context=cookiecutter_context)


main()
