# Patrick Scholz, 23.01.2018
import numpy as np
import time
import os
import xarray as xr
import seawater as sw
from .sub_data import *

    
# ___LOAD CLIMATOLOGY DATA INTO XARRAY DATASET CLASS___________________________
#|                                                                             |
#|        *** LOAD CLIMATOLOGY DATA INTO --> XARRAY DATASET CLASS ***          |
#|                                                                             |
#|_____________________________________________________________________________|
def load_climatology(mesh, datapath, vname, depth=None, depidx=False,
                     do_zarithm='mean', do_hinterp='linear', do_zinterp=True, 
                     descript='clim', do_ptemp=True, pref =0.0, 
                     do_compute=False, do_load=True, do_persist=False, 
                     **kwargs):
    
    str_mdep = ''
    is_data = 'scalar'
    #___________________________________________________________________________
    # load climatology data with xarray
    data = xr.open_dataset(datapath, decode_times=False, **kwargs)
    
    #___________________________________________________________________________
    # delete eventual time dimension from climatology data
    if 'time' in data.dims:
        data = data.squeeze(dim='time',drop=True )
    
    #___________________________________________________________________________
    # identify dimension names
    list_lonstr  = ['x','lon','longitude','long', 'nx']
    list_latstr  = ['y','lat','latitude', 'ny']
    list_zlevstr = ['z','depth','dep','level','lvl','zcoord','zlev','zlevel', 'nz', 'Z']
    idx       = [i for i, item in enumerate(list(data.dims)) if item.lower() in list_lonstr][0]
    dim_lon   = list(data.dims)[idx]
    idx       = [i for i, item in enumerate(list(data.dims)) if item.lower() in list_latstr][0]
    dim_lat   = list(data.dims)[idx]
    idx       = [i for i, item in enumerate(list(data.dims)) if item.lower() in list_zlevstr][0]
    dim_zlev  = list(data.dims)[idx]
    
    # identify coordinate names
    idx       = [i for i, item in enumerate(list(data.coords)) if item.lower() in list_lonstr][0]
    coord_lon = list(data.coords)[idx]
    idx       = [i for i, item in enumerate(list(data.coords)) if item.lower() in list_latstr][0]
    coord_lat = list(data.coords)[idx]
    idx       = [i for i, item in enumerate(list(data.coords)) if item.lower() in list_zlevstr][0]
    coord_zlev= list(data.coords)[idx]
    
        
    #___________________________________________________________________________
    # compute potential temperature
    if vname == 'pdens' or 'sigma' in vname: do_ptemp=True
    if do_ptemp and (any( [a in vname for a in ['temp', 'T', 't00', 'temperature', 'pdens']]) or 'sigma' in vname) :
        for key in list(data.keys()):
            if any( [a in key for a in ['temp', 'T', 't00', 'temperature']]): vname_temp=key
            if any( [a in key for a in ['salt', 'S', 's00', 'salinity'   ]]): vname_salt=key
        #data_depth = data[coord_zlev]
        #data_depth = data_depth.expand_dims({dim_lat:data[coord_lat].data, 
                                             #dim_lon:data[coord_lon].data}
                                            #).transpose(dim_zlev,dim_lat,dim_lon)
        data_depth = data[coord_zlev].expand_dims(
                        dict({dim_lat:data[coord_lat].data, dim_lon:data[coord_lon].data})
                        ).transpose(dim_zlev,dim_lat,dim_lon)
        data[vname_temp].data = sw.ptmp(data[vname_salt].data, data[vname_temp].data, data_depth )
        
    #___________________________________________________________________________
    # if there are multiple variables, than kick out varaible that is not needed
    vname_drop = list(data.keys())            
    if vname == 'pdens' or 'sigma' in vname:
        if   vname == 'sigma'  : pref=0
        elif vname == 'sigma1' : pref=1000
        elif vname == 'sigma2' : pref=2000
        elif vname == 'sigma3' : pref=3000
        elif vname == 'sigma4' : pref=4000
        elif vname == 'sigma5' : pref=5000
        data = data.assign({vname: (list(data.dims), sw.pden(data[vname_salt].data, data[vname_temp].data, data_depth, pref)-1000.025)})
        #for labels in vname_drop:
        data = data.drop(labels=vname_drop)
        data[vname].attrs['units'] = 'kg/m^3'
    else:
        if len(list(data.keys()))>1:
            vname_drop.remove(vname)
            data = data.drop_vars(vname_drop)
    
    #___________________________________________________________________________
    # see if longitude dimension needs to be periodically rolled so it agrees with 
    # the fesom2 mesh focus 
    lon = data.coords[coord_lon].values
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
        #data.coords[dim_lon].values = lon  
        data = data.assign_coords(dict({dim_lon:lon}))
        
        # periodically roll data together with longitude dimension
        data = data.roll(dict({dim_lon:idx_roll}), roll_coords=True)
    
    #___________________________________________________________________________
    # do vertical interpolation
    if (depth) is not None:
        #_______________________________________________________________________
        # select depth level indices that are needed to interpolate the values 
        # in depth list,array
        zlev = data.coords[coord_zlev].values
        ndimax = len(zlev)
        sel_levidx = do_comp_sel_levidx(zlev, depth, depidx, ndimax)
        
        #_______________________________________________________________________        
        # select vertical levels from data
        data = data.isel(dict({dim_zlev:sel_levidx})) 
        
        #_______________________________________________________________________
        # do vertical interpolation and summation over interpolated levels 
        if depidx==False:
            str_mdep = ', '+str(do_zarithm)
            # do vertical interpolationof depth levels
            data = data.interp(dict({dim_zlev:depth}), method="linear")
            
            # do z-arithmetic 
            if data[coord_zlev].size>1: 
                data = do_depth_arithmetic(data, do_zarithm, dim_zlev)
          
    #___________________________________________________________________________
    # do horizontal interplation to fesom grid 
    if (do_hinterp) is not None:
        if do_hinterp=='nearest':
            #add fesom2 mesh coordinatesro xarray dataset
            n_x = xr.DataArray(mesh.n_x, dims="nod2")
            n_y = xr.DataArray(mesh.n_y, dims="nod2")
            
            # interp data on nodes
            # data = data.interp(lon=n_x, lat=n_y, method='nearest')
            data = data.interp(dict({dim_lon:n_x, dim_lat:n_y}), method='nearest')
            del n_x, n_y
            
        elif do_hinterp=='linear':
            #add fesom2 mesh coordinatesro xarray dataset
            n_x = xr.DataArray(mesh.n_x, dims="nod2")
            n_y = xr.DataArray(mesh.n_y, dims="nod2")
            
            # interp data on nodes --> method linear
            # data_lin = data.interp(dim_lon=n_x, dim_lat=n_y, method='linear')
            data_lin = data.interp(dict({dim_lon:n_x, dim_lat:n_y}), method='linear')
            
            # fill up nan gaps as far as possible with nearest neighbours -->
            # gives better coastal edges
            if depth is not None:
                #isnan = xr.ufuncs.isnan(data_lin[vname])
                isnan = np.isnan(data_lin[vname])
                #data_lin[vname][isnan] = data[vname].interp(dim_lon=n_x.sel(nod2=isnan), dim_lat=n_y.sel(nod2=isnan), method='nearest')
                data_lin[vname][isnan] = data[vname].interp(dict({dim_lon:n_x.sel(nod2=isnan), dim_lat:n_y.sel(nod2=isnan)}), method='nearest')
                del isnan
            data = data_lin
            del data_lin, n_x, n_y
            
        elif do_hinterp=='regular': 
            ...
        
        # re-chunk data along nod2
        data = data.chunk({'nod2':'auto'})    
        
    #___________________________________________________________________________
    # do vertical interplation to fesom grid
    if do_zinterp and (depth is None):
        #add fesom2 mesh coordinatesro xarray dataset
        zmid = xr.DataArray(np.abs(mesh.zmid), dims="nz1")
        
        # improvise extrapolation --> fesom depth levels reach usually deeper than 
        # the levels of the climatology --> therefor expand last layers of climatology 
        # so they cover the fesom depth range
        addlay = 3
        zlev   = data[coord_zlev].data
        dd_mat = np.ones((addlay,))*(zlev[-1]-zlev[-2])
        dd_mat = zlev[-1]+dd_mat.cumsum()
        zlev   = np.hstack([zlev, dd_mat])
        data   = data.pad({dim_zlev:(0, addlay)}, mode='edge')
        data   = data.assign_coords(dict({dim_zlev:zlev}))
        del(zlev, dd_mat)
        
        # interp data on nodes --> method linear
        data = data.interp(dict({dim_zlev:zmid}), method='linear')
        
        # re-chunk data along nz1
        data = data.chunk({'nz1':'auto'})  
        
    #___________________________________________________________________________
    # write additional attribute info
    for vname in list(data.keys()):
        attr_dict=dict({'datapath':datapath, 'depth':depth, 'str_mdep':str_mdep, 
                        'depidx':depidx, 'do_zarithm':do_zarithm, 'do_hinterp':do_hinterp, 
                        'do_compute':do_compute, 'descript':descript, 'runid':'fesom'})
        do_additional_attrs(data, vname, attr_dict)
    
    data = data.assign_coords(nz1=('nz1' ,-mesh.zmid))
    
    #if depth is None:
        #w_A = xr.DataArray(mesh.n_area[:-1,:].astype('float32'), dims=['nz1' , 'nod2']).chunk({'nod2':data.chunksizes['nod2'], 'nz1':data.chunksizes['nz1']})
        #w_A = w_A.where(~np.isnan(data[vname].data))
        #data = data.assign_coords(w_A=w_A)
    #else:
        #w_A = xr.DataArray(mesh.n_area[0,:].astype('float32'), dims=['nod2']).chunk({'nod2':data.chunksizes['nod2']})
        #data = data.assign_coords(w_A=w_A)
    #del(w_A)
    
    data, dim_vert, dim_horz = do_gridinfo_and_weights(mesh, data, do_zweight=False, do_hweight=True)
    
    
    #___________________________________________________________________________
    data = data.transpose()    
    data = data.astype('float32', copy=False)
    
    #___________________________________________________________________________
    warnings.filterwarnings("ignore", category=UserWarning, message="Sending large graph of size")
    warnings.filterwarnings("ignore", category=UserWarning, message="Large object of size 2.10 MiB detected in task graph")
    if do_compute: data = data.compute()
    if do_load   : data = data.load()
    if do_persist: data = data.persist()
    warnings.resetwarnings()
    
    #___________________________________________________________________________
    return(data)
