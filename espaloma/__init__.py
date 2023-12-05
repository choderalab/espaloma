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

#
# from openff.toolkit.utils.toolkits import ToolkitRegistry, OpenEyeToolkitWrapper, RDKitToolkitWrapper, AmberToolsToolkitWrapper
# toolkit_registry = ToolkitRegistry()
# toolkit_precedence = [ RDKitToolkitWrapper ] # , OpenEyeToolkitWrapper, AmberToolsToolkitWrapper]
# [ toolkit_registry.register_toolkit(toolkit) for toolkit in toolkit_precedence if toolkit.is_available() ]
#

from . import _version
__version__ = _version.get_versions()['version']
