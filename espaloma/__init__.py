"""
espaloma
Extensible Surrogate Potential of Ab initio Learned and Optimized by Message-passing Algorithm
"""

# Add imports here
# import espaloma
from . import units, metrics

#
# from openforcefield.utils.toolkits import ToolkitRegistry, OpenEyeToolkitWrapper, RDKitToolkitWrapper, AmberToolsToolkitWrapper
# toolkit_registry = ToolkitRegistry()
# toolkit_precedence = [ RDKitToolkitWrapper ] # , OpenEyeToolkitWrapper, AmberToolsToolkitWrapper]
# [ toolkit_registry.register_toolkit(toolkit) for toolkit in toolkit_precedence if toolkit.is_available() ]
#


from espaloma.graphs.graph import Graph
from espaloma.metrics import GraphMetric

import espaloma.data
import espaloma.nn
import espaloma.graphs
import espaloma.mm
import espaloma.app



from espaloma.mm.geometry import *

from espaloma.app.experiment import *

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
