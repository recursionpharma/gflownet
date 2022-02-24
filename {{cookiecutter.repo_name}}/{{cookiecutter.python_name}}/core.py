def get_version() -> str:
    """Returns a string representation of the version of {{cookiecutter.python_name}} currently in use

    Returns
    -------
    str
        the version number installed of this package
    """
    try:
        from importlib.metadata import version  # type: ignore
        return version('{{cookiecutter.python_name}}')
    except ImportError:
        try:
            import pkg_resources
            return pkg_resources.get_distribution('{{cookiecutter.python_name}}').version
        except pkg_resources.DistributionNotFound:
            return 'set_version_placeholder'
    except ModuleNotFoundError:
        return 'set_version_placeholder'
