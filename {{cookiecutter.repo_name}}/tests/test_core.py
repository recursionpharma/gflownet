from {{cookiecutter.python_name}}.core import get_version


def test_get_version():
    assert get_version() == 'set_version_placeholder'
