"""A small python package providing tools for statistics.
An accompanying package for visualization is `vplotly`.
"""
from importlib.metadata import version

from vstats._est_test_properties import \
    get_est_welchs_test_properties  # noqa: F401
from vstats._welchs_test import welchs_test  # noqa: F401

# read version from installed package
__version__ = version("vstats")

# To make autodoc document imported objects
# in package vstats and not in some submodule
# of it, we follow the approach described in
# https://stackoverflow.com/a/66996523 :

imported_objects = [
    get_est_welchs_test_properties,
    welchs_test
]

for o in imported_objects:
    o.__module__ = __name__

__all__ = [o.__name__ for o in imported_objects]
