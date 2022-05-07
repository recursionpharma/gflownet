def get_version() -> str:
    """Returns a string representation of the version of gflownet currently in use

    Returns
    -------
    str
        the version number installed of this package
    """
    try:
        from importlib.metadata import version  # type: ignore
        return version('gflownet')
    except ImportError:
        try:
            import pkg_resources
            return pkg_resources.get_distribution('gflownet').version
        except pkg_resources.DistributionNotFound:
            return 'set_version_placeholder'
    except ModuleNotFoundError:
        return 'set_version_placeholder'
