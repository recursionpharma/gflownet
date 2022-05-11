import os.path
from setuptools import setup
from pathlib import Path
from typing import List, Tuple, Dict


# print("\n\n")
# import sys
# print(sys.path)
# print("\n\n")



# from src.gflownet import __version__

# print(__version__)

def get_version_from_package() -> str:
    """
    Read the package version from the source without importing it.
    """
    path = os.path.join(os.path.dirname(__file__), "src/gflownet/__init__.py")
    path = os.path.normpath(os.path.abspath(path))
    with open(path) as f:
        for line in f:
            if line.startswith("__version__"):
                token, version = line.split(" = ", 1)
                version = version.replace("\"", "").strip()
    return version


def _parse_requirement_file(req_file: Path) -> List[str]:
    lines = req_file.read_text().replace('\\\n', '').split('\n')
    lines = list(map(lambda x: x.split('#', 1)[0].strip(), lines))
    lines = [line for line in lines if len(line) and not line.startswith('-')]
    return lines


def get_requirements(path: str = 'requirements', ext: str = 'in') -> Tuple[List[str], Dict[str, List[str]]]:
    ext = ext[1:] if ext.startswith('.') else ext

    # Supports an arbitrary number of 'extra' packages
    install_requires = []
    extras_require = {}
    for req_file in Path(path).glob(f'*.{ext}'):
        lines = _parse_requirement_file(req_file)
        if req_file.stem == 'main':
            install_requires = lines
        else:
            extras_require[req_file.stem] = lines
    return install_requires, extras_require


install_requires, extras_require = get_requirements()

setup(
    install_requires=install_requires,
    extras_require=extras_require,
    version=get_version_from_package(),
    packages=["src/gflownet"],
)
