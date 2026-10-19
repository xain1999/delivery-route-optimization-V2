"""What produced a result: library versions and the machine.

Store this with every saved plan or experiment (Chapter 12): a fixed seed
alone does not reproduce a search that stops on the clock, and a library
update can change results."""
from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version

PACKAGES = ("ortools", "pyvrp", "numpy", "scipy", "networkx")


def versions() -> dict:
    out = {"python": platform.python_version(), "machine": platform.machine(),
           "processor_count": __import__("os").cpu_count()}
    for pkg in PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out
