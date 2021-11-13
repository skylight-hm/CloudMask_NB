from pathlib import Path

from setuptools import find_packages
from setuptools import setup


def version():
    """ Get the local package version. """
    namespace = {}
    path = Path("CloudMask_NB", "__version__.py")
    exec(path.read_text(), namespace)
    return namespace["__version__"]


def main() -> int:
    """ Execute the setup command.
    """
    _config = {
        "name": "CloudMask_NB",
        "author": "skylight-hm",
        "author_email": "mzlapqowjf321@qq.com",
        "packages": ['CloudMask_NB'],
        "console":["CloudMask_NB = CloudMask_NB.cli:main"]
    }
    _config.update({
        "version": version(),
    })
    setup(**_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
