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
from .sub_warmup_numba      import warmup_numba
#from .sub_3dsphere          import *
# Control VTK import with env var
if os.environ.get("TRIPYVIEW_WITHOUT_VTK", "0") != "1":
    try:
        from .sub_3dsphere import *
    except ImportError as e:
        print("Warning: sub_3dsphere could not be imported:", e)
else:
    print("VTK-related functionality (sub_3dsphere) disabled via TRIPYVIEW_WITHOUT_VTK=1")
