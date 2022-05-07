from gflownet.core import get_version
import re


def test_get_version():
    versionPattern = r'\d+(=?\.(\d+(=?\.(\d+)*)*)*)*'
    assert re.match(versionPattern, get_version())

