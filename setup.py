import os
from ast import literal_eval
from subprocess import check_output  # nosec - command is hard-coded, no possibility of injection

from setuptools import setup


def _get_next_version():
    if "SEMVER" in os.environ:
        return os.environ.get("SEMVER")

    # Note, this should only be used for development builds. Only robots can
    # create releases on PyPI from trunk, and the robots should know have the
    # `SEMVER` variable loaded at runtime.
    with open("VERSION", "r") as f:
        lines = f.read().splitlines()
    version_parts = {k: literal_eval(v) for k, v in map(lambda x: x.split("="), lines)}
    major = int(version_parts["MAJOR"])
    minor = int(version_parts["MINOR"])
    versions = check_output(["git", "tag", "--list"], encoding="utf-8").splitlines()  # nosec - command is hard-coded
    try:
        latest_patch = max(int(v.rsplit(".", 1)[1]) for v in versions if v.startswith(f"v{major}.{minor}."))
    except ValueError:  # no tags for this major.minor exist yet
        latest_patch = -1
    return f"{major}.{minor}.{latest_patch+1}"


setup(name="gflownet", version=_get_next_version())
