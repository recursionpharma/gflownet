import re

from gflownet.core import get_version


def test_get_version():
    versionPattern = r'\d+(=?\.(\d+(=?\.(\d+)*)*)*)*'
    assert re.match(versionPattern, get_version())
