"""
espaloma
Extensible Surrogate Potential of Ab initio Learned and Optimized by Message-passing Algorithm
"""

# Add imports here
from .espaloma import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
