import importlib_resources as resources
import os
import subprocess
import sys


def _get_executable():
    """Locate the compiled Helios executable."""
    return resources.files("_pyhelios") / "pyhelios" / "bin" / "helios++"


def helios_exec(args):
    #
    # Inject additional arguments to account for standard paths
    #

    # We always look for assets in the current working directory
    args = args + ["--assets", os.getcwd()]

    # We always look in the Python installation tree
    args = args + ["--assets", resources.files("pyhelios")]
    args = args + ["--assets", resources.files("pyhelios") / "data"]

    # Inject the legacy model switch. This is part of our transitioning strategy
    # to the new energy model.
    args = args + ["--legacyEnergyModel"]

    # Call the executable
    executable = _get_executable()
    return subprocess.call([executable] + args)


def helios_entrypoint():
    raise SystemExit(helios_exec(sys.argv[1:]))


if __name__ == "__main__":
    helios_entrypoint()
