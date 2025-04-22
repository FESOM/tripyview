# Create centralized notebook header. So if something needs to be changed in the 
# header of the notebook it can be done here, instead in doing it in every template 
# notebook separately
import os
import warnings
import time         as clock
import numpy        as np
import xarray       as xr
import shapefile    as shp
import tripyview    as tpv
import dask.array   as da
import dask
import gc
xr.set_options(keep_attrs=True)

client, use_existing_client = None, "tcp://0.0.0.0:0000"

def init_notebook(which_matplotlib="inline"):
    """Run Jupyter-specific configurations."""
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython:
        ipython.run_line_magic("matplotlib", which_matplotlib)
        ipython.run_line_magic("load_ext", "autoreload")
        ipython.run_line_magic("autoreload", "2")
        
    # Inject variables into the notebook's global namespace
    globals_ = ipython.user_global_ns if ipython else globals()
    globals_.update({
        "os": os,
        "warnings": warnings,
        "clock": clock,
        "tpv": tpv,
        "shp": shp,
        "xr": xr,
        "np": np,
        "da": da,
        "dask": dask,
        "gc": gc,
        "client": None,
        "use_existing_client": "tcp://0.0.0.0:0000"
    })
