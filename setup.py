from pathlib import Path
from typing import Dict, List, Tuple

from setuptools import setup


def _parse_requirement_file(req_file: Path) -> List[str]:
    lines = req_file.read_text().replace("\\\n", "").split("\n")
    lines = list(map(lambda x: x.split("#", 1)[0].strip(), lines))
    lines = [line for line in lines if len(line) and (not line.startswith("-"))]
    return lines


def _get_requirements(path: str = "requirements", ext: str = "in") -> Tuple[List[str], Dict[str, List[str]]]:
    ext = ext[1:] if ext.startswith(".") else ext
    install_requires = []
    extras_require = {}
    for req_file in Path(path).glob(f"*.{ext}"):
        lines = _parse_requirement_file(req_file)
        if req_file.stem == "main":
            install_requires = lines
        else:
            extras_require[req_file.stem] = lines
    return (install_requires, extras_require)


install_requires, extras_require = _get_requirements()
setup(install_requires=install_requires, extras_require=extras_require)
