"""
espaloma
Extensible Surrogate Potential of Ab initio Learned and Optimized by Message-passing Algorithm
"""

from . import metrics, units, data, app, graphs, mm, nn
from .app.experiment import *
from .graphs.graph import Graph
from .metrics import GraphMetric
from .mm.geometry import *
from .utils.model_fetch import get_model, get_model_path

# Add imports here
# import espaloma


# Handle versioneer
from ._version import get_versions

#
# from openff.toolkit.utils.toolkits import ToolkitRegistry, OpenEyeToolkitWrapper, RDKitToolkitWrapper, AmberToolsToolkitWrapper
# toolkit_registry = ToolkitRegistry()
# toolkit_precedence = [ RDKitToolkitWrapper ] # , OpenEyeToolkitWrapper, AmberToolsToolkitWrapper]
# [ toolkit_registry.register_toolkit(toolkit) for toolkit in toolkit_precedence if toolkit.is_available() ]
#


versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
