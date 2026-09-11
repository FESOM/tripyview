# -*- coding: utf-8 -*-
#
# This file is part of tripyview
# Original code by Patrick Scholz, Dmitry Sidorenko, Nikolay Koldunov, Sergey Danilov
#
"""Top-level package for tripyview."""

__author__ = """Patrick Scholz"""
__email__ = "patrick.scholz@awi.de"
__version__ = "0.3.0"

import os


from .sub_mesh              import * 
from .sub_data              import * 
from .sub_plot              import * 
from .sub_climatology       import *
from .sub_index             import *
from .sub_transect          import *
from .sub_zmoc              import *
from .sub_dmoc              import *
from .sub_transp            import *
from .sub_utility           import *
from .sub_colormap          import *
from .sub_tripyrundriver    import *
from .sub_tripyrun          import *
from .sub_shortcut          import *
from .sub_notebookheader    import *
from .sub_warmup_numba      import *
#from .sub_3dsphere          import *
# Control VTK import with env var. sub_3dsphere pulls in pyvista+vtk (~0.5 s of
# every `import tripyview`, also in every papermill kernel and dask worker), so
# it is imported on first use instead: tripyview.create_3dsphere_ocean_mesh(...)
# still works, the import just happens at that moment (PEP 562 module __getattr__)
if os.environ.get("TRIPYVIEW_WITHOUT_VTK", "0") != "1":
    def __getattr__(name):
        # only called for names that are not already in this module's namespace
        if name.startswith('__'): raise AttributeError(name)
        import importlib
        try:
            # import_module and not `from . import sub_3dsphere`: the latter asks
            # the package for the attribute again and recurses into __getattr__
            sub_3dsphere = importlib.import_module('.sub_3dsphere', __name__)
        except ImportError as e:
            raise ImportError(f"tripyview.{name} needs sub_3dsphere (pyvista/vtk), which could not "
                              f"be imported: {e}. Set TRIPYVIEW_WITHOUT_VTK=1 to disable it.") from None
        try   : value = getattr(sub_3dsphere, name)
        except AttributeError: raise AttributeError(f"module 'tripyview' has no attribute '{name}'") from None
        globals()[name] = value # cache, so __getattr__ is not consulted again
        return value
else:
    print("VTK-related functionality (sub_3dsphere) disabled via TRIPYVIEW_WITHOUT_VTK=1")
