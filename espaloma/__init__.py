"""
espaloma
Extensible Surrogate Potential of Ab initio Learned and Optimized by Message-passing Algorithm
"""

import espaloma.app
import espaloma.data
import espaloma.graphs
import espaloma.mm
import espaloma.nn
from espaloma.app.experiment import *
from espaloma.graphs.graph import Graph
from espaloma.metrics import GraphMetric
from espaloma.mm.geometry import *

# Add imports here
# import espaloma
from . import metrics, units

# Handle versioneer
from ._version import get_versions

#
# from openforcefield.utils.toolkits import ToolkitRegistry, OpenEyeToolkitWrapper, RDKitToolkitWrapper, AmberToolsToolkitWrapper
# toolkit_registry = ToolkitRegistry()
# toolkit_precedence = [ RDKitToolkitWrapper ] # , OpenEyeToolkitWrapper, AmberToolsToolkitWrapper]
# [ toolkit_registry.register_toolkit(toolkit) for toolkit in toolkit_precedence if toolkit.is_available() ]
#


versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
