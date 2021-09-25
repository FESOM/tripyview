# Patrick Scholz, 23.01.2018

import numpy as np
#import numpy.matlib
import time
import os
import xarray as xr
from sub_data import *
    
# ___LOAD CLIMATOLOGY DATA INTO XARRAY DATASET CLASS___________________________
#|                                                                             |
#|        *** LOAD CLIMATOLOGY DATA INTO --> XARRAY DATASET CLASS ***          |
#|                                                                             |
#|_____________________________________________________________________________|
def load_climatology(mesh, datapath, vname,depth=None, depidx=False,
                     do_zarithm='mean', do_hinterp='linear', 
                     do_compute=True, descript='clim', 
                     **kwargs):
    
    str_mdep = ''
    is_data = 'scalar'
    #___________________________________________________________________________
    # load climatology data with xarray
    data = xr.open_dataset(datapath, decode_times=False, **kwargs)
    
    # if there are multiple varaibles, than kick out varaible that is not needed
    if len(list(data.keys()))>1:
        vname_drop = list(data.keys())
        vname_drop.remove(vname)
        data = data.drop(labels=vname_drop)
    
    #___________________________________________________________________________
    # delete eventual time dimension from climatology data
    if 'time' in data.dims:
        data = data.squeeze(dim='time',drop=True )
    
    #___________________________________________________________________________
    # identify longitude dimension 
    lon_names_list = ['x','lon','longitude','long']
    lon_idx = [i for i, item in enumerate(list(data.dims)) if item.lower() in lon_names_list][0]
    lon_name = list(data.dims)[lon_idx]
    
    lat_names_list = ['y','lat','latitude']
    lat_idx = [i for i, item in enumerate(list(data.dims)) if item.lower() in lat_names_list][0]
    lat_name = list(data.dims)[lat_idx]
    
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
        #data.coords[lon_name].values = lon  
        data = data.assign_coords(dict({lon_name:lon}))
        
        # periodically roll data together with longitude dimension
        data = data.roll(dict({'lon':idx_roll}), roll_coords=True)
    
    #___________________________________________________________________________
    # do vertical interpolation
    if (depth) is not None:
        #_______________________________________________________________________
        zlev_names_list = ['z','depth','dep','level','lvl','zcoord','zlev','zlevel']
        zlev_idx = [i for i, item in enumerate(list(data.dims)) if item.lower() in zlev_names_list][0]
        zlev_name = list(data.dims)[zlev_idx]
        
        #_______________________________________________________________________
        # select depth level indices that are needed to interpolate the values 
        # in depth list,array
        zlev = data.coords[zlev_name].values
        ndimax = len(zlev)
        sel_levidx = do_comp_sel_levidx(zlev, depth, depidx, ndimax)
        
        #_______________________________________________________________________        
        # select vertical levels from data
        data = data.isel(dict({zlev_name:sel_levidx})) 
        
        #_______________________________________________________________________
        # do vertical interpolation and summation over interpolated levels 
        if depidx==False:
            str_mdep = ', '+str(do_zarithm)
            # do vertical interpolationof depth levels
            data = data.interp(dict({zlev_name:depth}), method="linear")
            
            # do z-arithmetic 
            if data[zlev_name].size>1: 
                data = do_depth_arithmetic(data, do_zarithm, zlev_name)
          
    #___________________________________________________________________________
    # do horizontal interplation to fesom grid 
    if (do_hinterp) is not None:
        if do_hinterp=='nearest':
            #add fesom2 mesh coordinatesro xarray dataset
            n_x = xr.DataArray(mesh.n_x, dims="nod2")
            n_y = xr.DataArray(mesh.n_y, dims="nod2")
            
            # interp data on nodes
            data = data.interp(lon=n_x, lat=n_y, method='nearest')
            del n_x, n_y
            
        elif do_hinterp=='linear':
            #add fesom2 mesh coordinatesro xarray dataset
            n_x = xr.DataArray(mesh.n_x, dims="nod2")
            n_y = xr.DataArray(mesh.n_y, dims="nod2")
            
            # interp data on nodes --> method linear
            data_lin = data.interp(lon=n_x, lat=n_y, method='linear')
            
            # fill up nan gaps as far as possible with nearest neighbours -->
            # gives better coastal edges
            if depth is not None:
                isnan = xr.ufuncs.isnan(data_lin[vname])
                data_lin[vname][isnan] = data[vname].interp(lon=n_x.sel(nod2=isnan), lat=n_y.sel(nod2=isnan), method='nearest')
                del isnan
            data = data_lin
            del data_lin, n_x, n_y
            
        elif do_hinterp=='regular': 
            ...
    
    #___________________________________________________________________________
    # write additional attribute info
    for vname in list(data.keys()):
        attr_dict=dict({'datapath':datapath, 'depth':depth, 'str_mdep':str_mdep, 
                        'depidx':depidx, 'do_zarithm':do_zarithm, 'do_hinterp':do_hinterp, 
                        'do_compute':do_compute, 'descript':descript})
        do_additional_attrs(data, vname, attr_dict)
    
    #___________________________________________________________________________
    if do_compute: data = data.compute()
    
    return(data)
