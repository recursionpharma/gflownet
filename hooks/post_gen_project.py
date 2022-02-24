import os
import shutil


def move_files(files_to_move):
    for f in files_to_move:
        shutil.move(f['src'], f['dest'])


def main():
    have_cli = {{cookiecutter.cli == 'y'}}
    python_name = '{{cookiecutter.python_name}}'


    files_to_delete = set()
    dirs_to_delete = set()

    if not have_cli:
        files_to_delete = files_to_delete.union({f'{python_name}/cli.py'})

    for f in files_to_delete:
        os.remove(f)

    for d in dirs_to_delete:
        shutil.rmtree(d)


main()
