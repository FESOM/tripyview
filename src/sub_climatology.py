# Patrick Scholz, 23.01.2018

import numpy as np
#import numpy.matlib
import time
import os
import xarray as xr

    
# ___LOAD CLIMATOLOGY DATA INTO XARRAY DATASET CLASS___________________________
#|                                                                             |
#|        *** LOAD CLIMATOLOGY DATA INTO --> XARRAY DATASET CLASS ***          |
#|                                                                             |
#|_____________________________________________________________________________|
def load_climatology(mesh, datapath, vname,depth=None, depidx=False,
                     **kwargs):
    #___________________________________________________________________________
    # load climatology data with xarray
    data = xr.open_dataset(datapath, **kwargs)
    
    # if there are multiple varaibles, than kick out varaible that is not needed
    if len(list(data.keys()))>1:
        vname_drop = list(data.keys())
        vname_drop.remove(vname)
        data = data.drop(labels=vname_drop)
    
    #___________________________________________________________________________
    # identify longitude dimension 
    lon_names_list = ['x','lon','longitude','long']
    lon_idx = [i for i, item in enumerate(list(data.dims)) if item.lower() in lon_names_list][0]
    lon_name = list(data.dims)[lon_idx]
    
    #___________________________________________________________________________
    # see if longitude dimension needs to be periodically rolled so it agrees with 
    # the fesom2 mesh focus 
    lon = data.coords[lon_name].values
    if any(lon>mesh.focus+180.0) or any(lon<mesh.focus-180.0):
        # identify rolling index 
        if   any(lon>mesh.focus+180.0):
            idx = np.where(lon>mesh.focus+180.0)[0]
            idx_roll = idx[0]
            lon[idx] = lon[idx]-360.0
        elif any(lon<mesh.focus-180.0): 
            idx = np.where(lon<mesh.focus+180.0)[0]
            idx_roll = -idx[-1]
            lon[idx] = lon[idx]+360.0
        
        # shift longitude coordinates    
        data.coords[lon_name].values = lon    
        
        # periodically roll data together with longitude dimension
        data = data.roll(dict({'lon':idx_roll}), roll_coords=True)
    
    
    
 
