import numpy as np
import time as clock
import os
import xarray as xr
import matplotlib
matplotlib.rcParams['contour.negative_linestyle']= 'solid'
import matplotlib.pyplot as plt
#import matplotlib.patches as Polygon
#import matplotlib.path as mpltPath
#from matplotlib.tri import Triangulation
import shapefile as shp
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import interp1d
from numpy.matlib import repmat
from scipy import interpolate
import numpy.ma as ma

from .sub_colormap import *
from .sub_utility  import *
from .sub_plot     import *


#+___CALCULATE MERIDIONAL OVERTURNING IN DENSITY COORDINATES___________________+
#| Global MOC, Atlantik MOC, Indo-Pacific MOC, Indo MOC                        |
#|                                                                             |
#+_____________________________________________________________________________+
def load_dmoc_data(mesh, datapath, descript, year, which_transf, std_dens, #n_area=None, e_area=None, 
                   do_info=True, do_tarithm='mean', add_trend=False, do_wdiap=False, do_dflx=False, 
                   do_bolus=True, add_bolus=False, do_zcoord=True, do_useZinfo='std_dens_H', do_ndensz=False, 
                   do_compute=False, do_load=True, do_persist=False, do_parallel=False,
                   **kwargs):
    #___________________________________________________________________________
    # ensure that attributes are preserved  during operations with yarray 
    xr.set_options(keep_attrs=True)
    
    #___________________________________________________________________________
    # number of sigma2 density levels 
    dens         = xr.DataArray(std_dens, dims=["ndens"]).astype('float32')
    wd, w        = np.diff(std_dens), np.zeros(dens.size)
    w[0 ], w[-1] = wd[0   ]/2., wd[-1  ]/2.
    w[1:-1]      = (wd[0:-1]+wd[1:])/2. # drho @  std_dens level boundary
    weight_dens  = xr.DataArray(w, dims=["ndens"])
    
    #___________________________________________________________________________
    # Load netcdf data
    if do_info==True: print(' --> create xarray dataset and load std_* data')
    
    #___________________________________________________________________________
    # create xarray dataset to combine data for dmoc computation
    data_dMOC = xr.Dataset()
    
    #___________________________________________________________________________
    # add surface transformations 
    if ('srf' in which_transf or 'inner' in which_transf) or do_dflx: # add surface fluxes
        # compute combined density flux: heat_flux+freshwater_flux+restoring_flux
        if do_dflx: 
            data = load_data_fesom2(mesh, datapath, vname='std_heat_flux', year=year, 
                descript=descript , do_info=do_info, do_ie2n=False, 
                do_tarithm=do_tarithm, do_nan=False, 
                do_compute=do_compute, do_load=do_load, do_persist=do_persist, do_parallel=do_parallel)
            
            data['std_heat_flux'].data = data['std_heat_flux'].data +\
            load_data_fesom2(mesh, datapath, vname='std_frwt_flux', 
                year=year, descript=descript , do_info=do_info, do_ie2n=False, 
                do_tarithm=do_tarithm, do_nan=False, 
                do_compute=do_compute, do_load=do_load, do_persist=do_persist, 
                do_parallel=do_parallel)['std_frwt_flux'].data + \
            load_data_fesom2(mesh, datapath, vname='std_rest_flux', 
                year=year, descript=descript , do_info=do_info, do_ie2n=False, 
                do_tarithm=do_tarithm, do_nan=False, 
                do_compute=do_compute, do_load=do_load, do_persist=do_persist, 
                do_parallel=do_parallel)['std_rest_flux'].data
            data_attrs = data_dMOC.attrs # rescue attributes will get lost during multipolication
            data       = data.rename({'std_heat_flux':'dmoc_fd'}).assign_coords({'ndens' :("ndens",std_dens)})
            data_dMOC  = xr.merge([data_dMOC, data], combine_attrs="no_conflicts")
            del(data)
        
        # compute single flux from heat_flux & freshwater_flux &restoring_flux
        else:
            data_dMOC = xr.merge([data_dMOC, 
                load_data_fesom2(mesh, datapath, vname='std_heat_flux', 
                year=year, descript=descript , do_info=do_info, do_ie2n=False, 
                do_tarithm=do_tarithm, do_nan=False, 
                do_compute=do_compute, do_load=do_load, do_persist=do_persist, 
                do_parallel=do_parallel).rename({'std_heat_flux':'dmoc_fh'})], combine_attrs="no_conflicts") 
            
            data_dMOC = xr.merge([data_dMOC, 
                load_data_fesom2(mesh, datapath, vname='std_frwt_flux', 
                year=year, descript=descript , do_info=do_info, do_ie2n=False, 
                do_tarithm=do_tarithm, do_nan=False, 
                do_compute=do_compute, do_load=do_load, do_persist=do_persist, 
                do_parallel=do_parallel).rename({'std_frwt_flux':'dmoc_fw'})], combine_attrs="no_conflicts")   
            
            data_dMOC = xr.merge([data_dMOC, 
                load_data_fesom2(mesh, datapath, vname='std_rest_flux', 
                year=year, descript=descript , do_info=do_info, do_ie2n=False, 
                do_tarithm=do_tarithm, do_nan=False,
                do_compute=do_compute, do_load=do_load, do_persist=do_persist, 
                do_parallel=do_parallel).rename({'std_rest_flux':'dmoc_fr'})], combine_attrs="no_conflicts")   
        
        #_______________________________________________________________________
        # add density levels and density level weights
        # kick out dummy ndens varaible that is wrong in the files, replace it with 
        # proper dens variable
        data_dMOC = data_dMOC.drop_vars('ndens') 
        wd, w     = np.diff(std_dens), np.zeros(dens.size)
        w[0], w[1:-1], w[-1] = wd[0]/2., (wd[0:-1]+wd[1:])/2., wd[-1]/2.  # drho @  std_dens level boundary
        w_dens    = xr.DataArray(w, dims=["ndens"]).astype('float32')
        
        # check if input data have been chunked
        if any(data_dMOC.chunks.values()):
            w_dens = w_dens.chunk({'ndens':data_dMOC.chunksizes['ndens']})
            dens   = dens.chunk({  'ndens':data_dMOC.chunksizes['ndens']})
        data_dMOC = data_dMOC.assign_coords({ 'dens'  :dens  , \
                                              'w_dens':w_dens })
        del(w, wd, w_dens)
        
        #_______________________________________________________________________
        # rescue global attributes will get lost during multiplication
        gattrs= data_dMOC.attrs 
        
        # multiply with weights
        data_dMOC = data_dMOC / data_dMOC.w_dens * 1024
        if do_dflx: 
            data_dMOC = data_dMOC / data_dMOC.w_A#--> element area
        
        # put back global attributes
        data_dMOC = data_dMOC.assign_attrs(gattrs)
        del(gattrs)
        
    #___________________________________________________________________________
    # add volume trend  
    if add_trend:  
        data_dMOC = xr.merge([data_dMOC, 
                              load_data_fesom2(mesh, datapath, vname='std_dens_dVdT', 
                              year=year, descript=descript , do_info=do_info, do_ie2n=False, 
                              do_tarithm=do_tarithm, do_nan=False, do_compute=do_compute, 
                              do_parallel=do_parallel).rename({'std_heat_flux':'dmoc_dvdt'}).drop_vars('ndens')],
                              combine_attrs="no_conflicts")
        
    #___________________________________________________________________________
    # skip this when doing diapycnal vertical velocity
    if (not do_wdiap) and (not do_dflx) and do_zcoord:
        # the std_dens_Z variable from the DMOC diagnostic output of FESOM2, does
        # not really allow for a proper projection onto zcoord. the DMOC in zcoord
        # becomes way to shallow and unrealistic in that term. 
        # Best option do compute the projection is via the densitz class layer
        # thickness H, second best option is via is via the sigma2 density on vertices
        # and the interpolation of the densitz bins to estimate the vertical coordinate
        
        # check if input data have been chunked
        # if any(data_dMOC.chunks.values()) and any(dens.chunks.values())==False:
        if any(data_dMOC.chunks.values()) and dens.chunks is None:    
            dens = dens.chunk({  'ndens':data_dMOC.chunksizes['ndens']})
        #_______________________________________________________________________
        if do_useZinfo=='std_dens_H':
            # add vertical density class thickness
            data_h = load_data_fesom2(mesh, datapath, vname='std_dens_H', 
                        year=year, descript=descript , do_info=do_info, 
                        do_ie2n=False, do_tarithm=do_tarithm, do_nan=False, 
                        do_compute=do_compute, do_load=do_load, do_persist=do_persist, 
                        do_parallel=do_parallel).rename({'std_dens_H':'ndens_h'})
            data_h = data_h.assign_coords({'dens':dens})
            data_h = data_h.drop_vars(['ndens', 'elemi', 'lon']) #--> drop not needed variables
            data_dMOC = xr.merge([data_dMOC, data_h], combine_attrs="no_conflicts")
            if do_ndensz:
                # compute vertical density class z position from class thickness by 
                # cumulative summation 
                data_z = data_h.copy().rename({'ndens_h':'ndens_z'})
                data_z = data_z.cumsum(dim='ndens', skipna=True)
                #data_z = data_z.assign_coords({'ndens' :("ndens",std_dens)})
                data_z = data_z.where(data_h.ndens_h!=0.0,0.0)
                data_dMOC = xr.merge([data_dMOC, data_z], combine_attrs="no_conflicts")
                del(data_z)
            del(data_h)
            
        #_______________________________________________________________________
        elif do_useZinfo=='std_dens_Z':
            # add vertical density class position computed in FESOM2 -->
            # gives worst results for zcoordinate projection
            data_z = load_data_fesom2(mesh, datapath, vname='std_dens_Z'   , 
                        year=year, descript=descript , do_info=do_info, 
                        do_ie2n=False, do_tarithm=do_tarithm, do_nan=False, 
                        do_compute=do_compute, do_load=do_load, do_persist=do_persist, 
                        do_parallel=do_parallel).rename({'std_dens_Z':'ndens_z'})
            data_z = data_z.assign_coords({'dens':dens})
            data_z = data_z.drop_vars(['ndens', 'elemi', 'lon']) #--> drop not needed variables
            data_dMOC = xr.merge([data_dMOC, data_z], combine_attrs="no_conflicts")
            del(data_z)
            
        #_______________________________________________________________________
        elif do_useZinfo=='density_dMOC' or do_useZinfo=='hydrography':
            # load sigma2 density on nodes 
            if do_useZinfo=='density_dMOC':
                data_sigma2 = load_data_fesom2(mesh, datapath, vname='density_dMOC', 
                            year=year, descript=descript , do_info=do_info, 
                            do_ie2n=False, do_tarithm=do_tarithm, do_zarithm=None, do_nan=False, 
                            do_compute=do_compute, do_load=do_load, do_persist=do_persist, 
                            do_parallel=do_parallel).rename({'density_dMOC':'nz_rho'})
                
                # make land sea mask nan
                data_sigma2 = data_sigma2.where(data_sigma2!=0.0)
            
                # first put back here the land sea mask nan's than subtract ref 
                # denity --<> like that land remains Nan respective zero later
                data_sigma2 = data_sigma2 - 1000.00
            
            elif do_useZinfo=='hydrography':
                # load sigma2 density on nodes based on temperatur and salinity
                # hydrography
                data_sigma2 = load_data_fesom2(mesh, datapath, vname='sigma2', 
                            year=year, descript=descript , do_info=do_info, 
                            do_ie2n=False, do_tarithm=do_tarithm, do_zarithm=None, do_nan=False, 
                            do_compute=do_compute, do_load=do_load, do_persist=do_persist, 
                            do_parallel=do_parallel).rename({'sigma2':'nz_rho'})
                
                # make land sea mask nan --> here ref density is already substracted
                data_sigma2 = data_sigma2.where(data_sigma2!=0.0)
            
            data_sigma2  = data_sigma2.drop_vars(['ndens','lon','lat', 'nodi', 'nodiz', 'w_A']) 
            data_sigma2  = data_sigma2.assign_coords({'dens':dens})
            
            # have to do it via assign otherwise cant write [elem x ndens] into [nod2d x ndens] 
            # array an save the attributes in the same time
            if 'time' in list(data_sigma2.dims):
                data_sigma2  = data_sigma2.assign(nz_rho=data_sigma2[list(data_sigma2.keys())[0]][:, xr.DataArray(mesh.e_i, dims=["elem",'n3']),:].mean(dim="n3", keep_attrs=True) )
            else:
                data_sigma2  = data_sigma2.assign(nz_rho=data_sigma2[list(data_sigma2.keys())[0]][   xr.DataArray(mesh.e_i, dims=["elem",'n3']),:].mean(dim="n3", keep_attrs=True) )
            
            # Check if the dataset is empty
            if not data_dMOC.data_vars: data_sigma2 = data_sigma2.chunk({'elem':'auto', 'ndens':'auto'})
            else                      : data_sigma2 = data_sigma2.chunk({'elem':data_dMOC.chunksizes['elem'], 'ndens':data_dMOC.chunksizes['ndens']})
            
            data_sigma2 = data_sigma2.where(np.isnan(data_sigma2.nz_rho)==False,0.0)
            
            # add to Dataset
            data_dMOC = xr.merge([data_dMOC, data_sigma2], combine_attrs="no_conflicts")
            del(data_sigma2)
            
    #___________________________________________________________________________
    if (not do_dflx) and ( 'inner' in which_transf or 'dmoc' in which_transf ):
        # check if input data have been chunked
        # if any(data_dMOC.chunks.values()) and any(dens.chunks.values())==False:
        if any(data_dMOC.chunks.values()) and dens.chunks is None:
            dens = dens.chunk({  'ndens':data_dMOC.chunksizes['ndens']})
        
        # add divergence of density classes --> diapycnal velocity
        data_div  = load_data_fesom2(mesh, datapath, vname='std_dens_DIV' , 
                        year=year, descript=descript , do_info=do_info, 
                        do_ie2n=False, do_tarithm=do_tarithm, do_nan=False, 
                        do_compute=do_compute, do_load=do_load, do_persist=do_persist, 
                        do_parallel=do_parallel).rename({'std_dens_DIV':'dmoc'})
        data_div  = data_div.drop_vars(['ndens', 'nodi']) 
        data_div  = data_div.assign_coords({'dens':dens})
        
        # doing this step here so that the MOC amplitude is correct, setp 1 of 2
        # divide with verice area
        gattrs   = data_div.attrs
        data_div = data_div/data_div['w_A']    # --> vertice area
        
        # save global attributes
        data_div = data_div.assign_attrs(gattrs)
        
        # skip this when doing diapycnal vertical velocity
        if not do_wdiap:
            data_div  = data_div.drop_vars(['w_A', 'lon', 'lat']) # --> dont need from here on anymor
            
            # have to do it via assign otherwise cant write [elem x ndens] into [nod2d x ndens] 
            # array an save the attributes in the same time, also i need to unchunk
            # here the array with .load() otherwise i was not able to reindex it 
            # without causing problems 
            data_div['dmoc']     = data_div['dmoc'].load()
            if 'time' in list(data_div.dims):
                data_div = data_div.assign( dmoc=data_div['dmoc'][:, xr.DataArray(mesh.e_i, dims=["elem",'n3']),:].mean(dim="n3", keep_attrs=True) )
            else:
                data_div = data_div.assign( dmoc=data_div['dmoc'][   xr.DataArray(mesh.e_i, dims=["elem",'n3']),:].mean(dim="n3", keep_attrs=True) )
            
            if not data_dMOC.data_vars: data_div = data_div.chunk({'elem':'auto', 'ndens':'auto'})
            else                      : data_div = data_div.chunk({'elem':data_dMOC.chunksizes['elem'], 'ndens':data_dMOC.chunksizes['ndens']})
            
            # multiply with elemental area
            if 'w_A' not in list(data_dMOC.coords):
                data_dMOC = data_dMOC.assign_coords(lon=xr.DataArray(mesh.n_x[mesh.e_i].sum(axis=1)/3.0, dims=['elem']).chunk({'elem':data_div.chunksizes['elem']}))
                data_dMOC = data_dMOC.assign_coords(lat=xr.DataArray(mesh.n_y[mesh.e_i].sum(axis=1)/3.0, dims=['elem']).chunk({'elem':data_div.chunksizes['elem']}))
                data_dMOC = data_dMOC.assign_coords(w_A=xr.DataArray(mesh.e_area                       , dims=['elem']).chunk({'elem':data_div.chunksizes['elem']}))
            
            # doing this step here so that the MOC amplitude is correct, setp 2 of 2
            # multiply with elemental area
            gattrs   = data_div.attrs
            data_div = data_div*data_dMOC['w_A']# --> element area
            
            # save global attributes
            data_div = data_div.assign_attrs(gattrs)
            
        data_dMOC = xr.merge([data_dMOC, data_div], combine_attrs="no_conflicts")  
        del(data_div)
        
        # load density class divergence from bolus velolcity
        if (do_bolus): 
            # add divergence of density classes --> diapycnal velocity
            data_div_bolus  = load_data_fesom2(mesh, datapath, vname='std_dens_DIVbolus' , 
                                    year=year, descript=descript , do_info=do_info, 
                                    do_ie2n=False, do_tarithm=do_tarithm, do_nan=False, 
                                    do_compute=do_compute, do_load=do_load, do_persist=do_persist, 
                                    do_parallel=do_parallel).rename({'std_dens_DIVbolus':'dmoc_bolus'})
            data_div_bolus  = data_div_bolus.drop_vars(['ndens', 'nodi']) 
            data_div_bolus  = data_div_bolus.assign_coords({'dens':dens})
            
            # doing this step here so that the MOC amplitude is correct, setp 1 of 2
            # divide with verice area
            gattrs          = data_div_bolus.attrs
            data_div_bolus  = data_div_bolus/data_div_bolus['w_A'] # --> vertice area
            
            # save global attributes
            data_div_bolus  = data_div_bolus.assign_attrs(gattrs)
            
            # skip this when doing diapycnal vertical velocity
            if not do_wdiap:
                data_div_bolus  = data_div_bolus.drop_vars(['w_A', 'lon', 'lat']) # --> dont need from here on anymore
            
                # have to do it via assign otherwise cant write [elem x ndens] into [nod2d x ndens] 
                # array an save the attributes in the same time
                data_div_bolus['dmoc_bolus'] = data_div_bolus['dmoc_bolus'].load()
                if 'time' in list(data_div_bolus.dims):
                    data_div_bolus  = data_div_bolus.assign( dmoc_bolus=data_div_bolus['dmoc_bolus'][:, xr.DataArray(mesh.e_i, dims=["elem",'n3']), :].mean(dim="n3", keep_attrs=True) )
                else:
                    data_div_bolus  = data_div_bolus.assign( dmoc_bolus=data_div_bolus['dmoc_bolus'][   xr.DataArray(mesh.e_i, dims=["elem",'n3']),:].mean(dim="n3", keep_attrs=True) )
                
                # Check if the dataset is empty
                if not data_dMOC.data_vars: data_div_bolus = data_div_bolus.chunk({'elem':'auto', 'ndens':'auto'})
                else                      : data_div_bolus = data_div_bolus.chunk({'elem':data_dMOC.chunksizes['elem'], 'ndens':data_dMOC.chunksizes['ndens']})
                
                # doing this step here so that the MOC amplitude is correct, setp 1 of 2
                # multiply with elemental area
                data_div_bolus = data_div_bolus*data_dMOC['w_A']# --> element area
            
            # add bolus dmoc to existing dmoc --> thus only one array to process
            if add_bolus:
                data_dMOC['dmoc'].data = data_dMOC['dmoc'].data + data_div_bolus['dmoc_bolus'].data
            else:     
                data_dMOC = xr.merge([data_dMOC, data_div_bolus], combine_attrs="no_conflicts")  
            del(data_div_bolus)
    
    #___________________________________________________________________________
    # drop unnecessary coordinates
    if (not do_wdiap) and (not do_dflx):
        if 'lon' in list(data_dMOC.coords): data_dMOC = data_dMOC.drop_vars(['lon'])
    if 'elemi' in list(data_dMOC.coords): data_dMOC = data_dMOC.drop_vars(['elemi'])
    if 'nodi'  in list(data_dMOC.coords): data_dMOC = data_dMOC.drop_vars(['nodi'])
    if 'nzi'   in list(data_dMOC.coords): data_dMOC = data_dMOC.drop_vars(['nzi'])
    
    str_proj = 'dmoc'
    if do_zcoord: str_proj = str_proj + '+depth'
    else        : str_proj = str_proj + '+dens'
    data_dMOC.attrs['proj'] = str_proj
    
    #___________________________________________________________________________
    warnings.filterwarnings("ignore", category=UserWarning, message="Sending large graph of size")
    warnings.filterwarnings("ignore", category=UserWarning, message="Large object of size \\d+\\.\\d+ detected in task graph")
    if do_compute: data_dMOC = data_dMOC.compute()
    if do_load   : data_dMOC = data_dMOC.load()
    if do_persist: data_dMOC = data_dMOC.persist()
    warnings.resetwarnings()
    #___________________________________________________________________________
    # return combined xarray dataset object
    return(data_dMOC)
    

#+___CALCULATE MERIDIONAL OVERTURNING IN DENSITY COORDINATES___________________+
#| Global MOC, Atlantik MOC, Indo-Pacific MOC, Indo MOC                        |
#|                                                                             |
#+_____________________________________________________________________________+
def calc_dmoc(mesh, data_dMOC, dlat=1.0, which_moc='gmoc', which_transf=None, do_info=True, do_checkbasin=False,
              exclude_meditoce=True, do_bolus=True, do_parallel=False, n_workers=10, 
              do_compute=False, do_load=True, do_persist=False, do_dropvar=True, 
              **kwargs):
    
    # rescue global dataset attributes
    gattr = data_dMOC.attrs
    
    #___________________________________________________________________________
    # compute index for basin domain limitation
    ts1 = clock.time()
    idxin     = calc_basindomain_fast(mesh, which_moc=which_moc, do_onelem=True, exclude_meditoce=exclude_meditoce)

    # reduce to dMOC data to basin domain
    data_dMOC = data_dMOC.isel(elem=idxin)
    
    # check basin selection 
    if do_checkbasin:
        from matplotlib.tri import Triangulation
        tri = Triangulation(np.hstack((mesh.n_x,mesh.n_xa)), np.hstack((mesh.n_y,mesh.n_ya)), np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia)))
        plt.figure()
        plt.triplot(tri, color='k')
        plt.triplot(tri.x, tri.y, tri.triangles[ np.hstack((idxin[mesh.e_pbnd_0], idxin[mesh.e_pbnd_a])) ,:], color='r')
        plt.title('Basin selection')
        plt.show()
        
    #___________________________________________________________________________
    # prepare  weights for area weighted mean over the elements for all density 
    # classes and depth levels (in case nz_rho is loaded)
    edims  = dict()
    dimsrt = list()
    dtime='None'
    if 'time' in list(data_dMOC.dims): 
        edims['time'], dtime = data_dMOC['time'].data, 'time'
    edims['ndens'] = data_dMOC['ndens'].data
    delem, ddens = 'elem', 'ndens'
    
    if 'ndens_h' in list(data_dMOC.keys()) or 'ndens_z' in list(data_dMOC.keys()):
        # expand by ndens dimension --> need this here to get proper area weighting 
        # mean over the bottom topography!!!
        data_dMOC['ndens_w_A'] = data_dMOC['w_A'].expand_dims(edims).transpose(dtime, delem, ddens, missing_dims='ignore')
    
        # non-existing density classes (ndens_h==0) --> NaN
        data_dMOC['ndens_w_A'] = data_dMOC['ndens_w_A'].where(data_dMOC['ndens_h']!=0.0)
    
    if 'nz_rho' in list(data_dMOC.keys()):
        edims = dict()
        if 'time' in list(data_dMOC.dims): edims['time'] = data_dMOC['time'].data
        edims['nz1'] = data_dMOC['nz1'].data
        # expand by nz1 dimension
        data_dMOC['nz_w_A'   ] = data_dMOC['w_A'].expand_dims(edims).transpose(dtime, delem, 'nz1', missing_dims='ignore')
            
        # land sea mask --> NaN
        data_dMOC['nz_w_A'   ] = data_dMOC['nz_w_A'].where(data_dMOC['nz_rho']!=0.0)
    
    #___________________________________________________________________________
    # scale surface density fluxes are already area weighted for zonal 
    # integration
    if 'dmoc_fh'   in list(data_dMOC.keys()): data_dMOC['dmoc_fh'   ] = data_dMOC['dmoc_fh'   ] * 1.0e-6 
    if 'dmoc_fw'   in list(data_dMOC.keys()): data_dMOC['dmoc_fw'   ] = data_dMOC['dmoc_fw'   ] * 1.0e-6
    if 'dmoc_fr'   in list(data_dMOC.keys()): data_dMOC['dmoc_fr'   ] = data_dMOC['dmoc_fr'   ] * 1.0e-6 
    if 'dmoc_fd'   in list(data_dMOC.keys()): data_dMOC['dmoc_fd'   ] = data_dMOC['dmoc_fd'   ] * 1.0e-6     
    if 'dmoc_dvdt' in list(data_dMOC.keys()): data_dMOC['dmoc_dvdt' ] = data_dMOC['dmoc_dvdt' ] * 1.0e-6
    if 'dmoc'      in list(data_dMOC.keys()): data_dMOC['dmoc'      ] = data_dMOC['dmoc'      ] * 1.0e-6
    if 'dmoc_bolus'in list(data_dMOC.keys()): data_dMOC['dmoc_bolus'] = data_dMOC['dmoc_bolus'] * 1.0e-6

    # multiply with weights to prepare for area weighted zonal means 
    if 'ndens_h'   in list(data_dMOC.keys()): data_dMOC['ndens_h'   ] = data_dMOC['ndens_h'   ] * data_dMOC['ndens_w_A']
    if 'ndens_z'   in list(data_dMOC.keys()): data_dMOC['ndens_z'   ] = data_dMOC['ndens_z'   ] * data_dMOC['ndens_w_A']
    if 'nz_rho'    in list(data_dMOC.keys()): data_dMOC['nz_rho'    ] = data_dMOC['nz_rho'    ] * data_dMOC['nz_w_A'   ]
    data_dMOC = data_dMOC.load()
    
    #___________________________________________________________________________
    # create meridional bins --> this trick is from Nils Brückemann (ICON)
    lat_bin = xr.DataArray(data=np.round(data_dMOC['lat'].data/dlat)*dlat, dims='elem', name='lat')
    lat    = np.arange(lat_bin.min(), lat_bin.max()+dlat, dlat)
    
    #___________________________________________________________________________
    # define subroutine for binning over latitudes, allows for parallelisation
    def dmoc_over_lat(lat_i, lat_bin, data):
        #_______________________________________________________________________
        # compute which vertice is within the latitudinal bin
        # --> groupby is here a factor 5-6 slower than using isel+np.where
        data_latbin = data.isel(elem=np.where(lat_bin==lat_i)[0])
        data_latbin = data_latbin.sum(dim='elem', skipna=True)
        
        vlist = list(data_latbin.keys())
        # compute area weighted zonal mean only for the vertical zcoordinate 
        if 'ndens_h'   in vlist: data_latbin['ndens_h'] = data_latbin['ndens_h']/data_latbin['ndens_w_A']
        if 'ndens_z'   in vlist: data_latbin['ndens_z'] = data_latbin['ndens_z']/data_latbin['ndens_w_A']
        if 'nz_rho'    in vlist: data_latbin['nz_rho' ] = data_latbin['nz_rho' ]/data_latbin['nz_w_A'   ]
        # drop  now total area in bins over depth level and density class
        if 'ndens_w_A' in vlist: data_latbin            = data_latbin.drop_vars(['ndens_w_A'])
        if 'nz_w_A'    in vlist: data_latbin            = data_latbin.drop_vars(['nz_w_A'])
        return(data_latbin)
    
    #___________________________________________________________________________
    # do serial loop over latitudinal bins
    if not do_parallel:
        if do_info: print('\n ___loop over latitudinal bins___'+'_'*90, end='\n')
        data_lat = list()
        for iy, lat_i in enumerate(lat):
            # here compute sum over each lat-bin
            data_lat.append(dmoc_over_lat(lat_i, lat_bin, data_dMOC))
        dmoc = xr.concat(data_lat, dim='lat')    
    # do parallel loop over latitudinal bins
    else:
        if do_info: print('\n ___parallel loop over longitudinal bins___'+'_'*1, end='\n')
        from joblib import Parallel, delayed
        data_lat = Parallel(n_jobs=n_workers)(delayed(dmoc_over_lat)(lat_i, lat_bin, data_dMOC) for lat_i in lat)
        dmoc = xr.concat(data_lat, dim='lat')
    del(data_lat)  
        
    #___________________________________________________________________________
    # rearange dimension, add necessary coordinates, attributes ... 
    dmoc = dmoc.assign_coords({'lat':lat})
    for vari in list(dmoc.keys()):
        #_______________________________________________________________________
        # transpose dimension 
        dimv = 'ndens'
        if 'nz1'  in dmoc[vari].dims: dimv='nz1'
        if 'time' in dmoc[vari].dims: dmoc[vari] = dmoc[vari].transpose('time',dimv,'lat')
        else                        : dmoc[vari] = dmoc[vari].transpose(dimv,'lat')
        
        #_______________________________________________________________________
        # add more variable attributes
        strmoc = 'd'+which_moc.upper()
        vattr = data_dMOC[vari].attrs
        if   vari=='dmoc_fh'   : vattr.update({'long_name':'Transformation from heat flux'                 , 'short_name':strmoc+'_fh'   , 'units':'Sv'   })
        elif vari=='dmoc_fw'   : vattr.update({'long_name':'Transformation from freshwater flux'           , 'short_name':strmoc+'_fw'   , 'units':'Sv'   })
        elif vari=='dmoc_fr'   : vattr.update({'long_name':'Transformation from surface salinity restoring', 'short_name':strmoc+'_fr'   , 'units':'Sv'   })
        elif vari=='dmoc_fd'   : vattr.update({'long_name':'Transformation from total density flu'         , 'short_name':strmoc+'_fd'   , 'units':'Sv'   })
        elif vari=='dmoc_dvdt' : vattr.update({'long_name':'Transformation from volume change'             , 'short_name':strmoc+'_dv'   , 'units':'Sv'   })
        elif vari=='dmoc'      : vattr.update({'long_name':'Density MOC'                                   , 'short_name':strmoc         , 'units':'Sv'   })
        elif vari=='dmoc_bolus': vattr.update({'long_name':'Density MOC bolus vel.'                        , 'short_name':strmoc+'_bolus', 'units':'Sv'   })
        elif vari=='ndens_h'   : vattr.update({'long_name':'Density class thickness'                       , 'short_name':'ndens_h'      , 'units':'m'    })
        elif vari=='ndens_zfh' : vattr.update({'long_name':'Density class z position'                      , 'short_name':'ndens_zfh'    , 'units':'m'    })
        elif vari=='ndens_z'   : vattr.update({'long_name':'Density class z position'                      , 'short_name':'ndens_z'      , 'units':'m'    })
        elif vari=='nz_rho'    : vattr.update({'long_name':'sigma2 density in zcoord'                      , 'short_name':'nz_rho'       , 'units':'kg/m³'})
        dmoc[vari]=dmoc[vari].assign_attrs(vattr)
    del(data_dMOC)
    dmoc=dmoc.assign_attrs(gattr)
    
    #___________________________________________________________________________
    # exclude variables that should not be cumulatively integrated --> than do
    # cumulative sumation over lat
    var_list = list(dmoc.keys())
    if 'ndens_h'   in var_list: var_list.remove('ndens_h')
    if 'ndens_z'   in var_list: var_list.remove('ndens_z')
    if 'nz_rho'    in var_list: var_list.remove('nz_rho' )   
    
    # 1) flip dimension along latitude S->N --> N->S 
    # 2) do cumulative sumation from N->S
    # 3) flip dimension back to S->N
    if do_info==True: print(' --> do cumsum over latitudes')
    reverse = slice(None, None, -1)
    for var in var_list:
        dmoc[var] = -dmoc[var].isel(lat=reverse).cumsum(dim='lat', skipna=True).isel(lat=reverse)

    #___________________________________________________________________________
    # cumulative sum over density 
    if do_info==True: print(' --> do cumsum over density (bottom-->top)')
    if 'dmoc'      in list(dmoc.keys()):
        dmoc[ 'dmoc'       ] = dmoc[ 'dmoc' ].isel(ndens=reverse).cumsum(dim='ndens', skipna=True).isel(ndens=reverse)
    
    if 'dmoc_bolus'in list(dmoc.keys()):
        dmoc[ 'dmoc_bolus' ] = dmoc[ 'dmoc_bolus' ].isel(ndens=reverse).cumsum(dim='ndens', skipna=True).isel(ndens=reverse)
    
    #___________________________________________________________________________
    # compute z-position (z) from (f) density class thickness (h)
    if 'ndens_h'   in list(dmoc.keys()):
        dmoc[ 'ndens_zfh'  ] = dmoc[ 'ndens_h' ].cumsum(dim='ndens', skipna=True)
        dmoc[ 'ndens_zfh'  ] = dmoc[ 'ndens_zfh' ].roll({'ndens':1})
        dmoc[ 'ndens_zfh'  ].loc[dict(ndens=dmoc['ndens'][0])]=0.0
    
    #___________________________________________________________________________
    # Move grid related variables into coordinate section of the xarray dataset
    if 'ndens_h'   in list(dmoc.keys()): dmoc = dmoc.set_coords('ndens_h')
    if 'ndens_zfh' in list(dmoc.keys()): dmoc = dmoc.set_coords('ndens_zfh')
    if 'nz_rho'    in list(dmoc.keys()): dmoc = dmoc.set_coords('nz_rho')
    
    #___________________________________________________________________________
    # Add bolus MOC to normal MOC
    if 'dmoc' in list(dmoc.keys()) and 'dmoc_bolus' in list(dmoc.keys()) and do_bolus:  
        dmoc['dmoc'].data = dmoc['dmoc'].data + dmoc['dmoc_bolus'].data
        if do_dropvar: dmoc = dmoc.drop_vars('dmoc_bolus')
        
    if which_transf is not None:    
        if 'srf' in which_transf: 
            if 'dmoc_fh' in dmoc.data_vars and 'dmoc_fw' in dmoc.data_vars and 'dmoc_fr' in dmoc.data_vars:
                dmoc['dmoc_srf'] =  -(dmoc['dmoc_fh']+dmoc['dmoc_fw']+dmoc['dmoc_fr'])
                
                vattr = dmoc['dmoc_fh'].attrs
                vattr.update({'long_name':'Surface Transformation'                 , 'short_name':strmoc+'_srf'   , 'units':'Sv'   })
                dmoc['dmoc_srf'] = dmoc['dmoc_srf'].assign_attrs(vattr)
                if do_dropvar: dmoc  = dmoc.drop_vars(['dmoc_fh', 'dmoc_fw', 'dmoc_fr'])

        elif 'inner' in which_transf:
            if 'dmoc_fh' in dmoc.data_vars and 'dmoc_fw' in dmoc.data_vars and 'dmoc_fr' in dmoc.data_vars and 'dmoc' in dmoc.data_vars:
                dmoc['dmoc_inner'] =  dmoc['dmoc']+(dmoc['dmoc_fh']+dmoc['dmoc_fw']+dmoc['dmoc_fr'])
                
                vattr = dmoc['dmoc'].attrs
                vattr.update({'long_name':'Inner Transformation'                 , 'short_name':strmoc+'_inner'   , 'units':'Sv'   })
                dmoc['dmoc_inner'] = dmoc['dmoc_inner'].assign_attrs(vattr)
                if do_dropvar: dmoc  = dmoc.drop_vars(['dmoc_fh', 'dmoc_fw', 'dmoc_fr', 'dmoc'])
                
    #___________________________________________________________________________
    # compute depth of max and mean bottom topography
    elemz     = xr.DataArray(np.abs(mesh.zlev[mesh.e_iz]), dims=['elem'])
    elemz     = elemz.isel(elem=idxin)
    elemz_m   = elemz.groupby(lat_bin).max()
    dmoc      = dmoc.assign_coords(botmax =elemz_m.astype('float32'))
    
    elemz_m   = elemz.groupby(lat_bin).mean()
    dmoc      = dmoc.assign_coords(botmean=elemz_m.astype('float32'))
    del (elemz, elemz_m)
    
    #___________________________________________________________________________
    # compute index of max mean bottom topography
    elemiz    = xr.DataArray(mesh.e_iz, dims=['elem'])
    elemiz    = elemiz.isel(elem=idxin)
    elemiz    = elemiz.groupby(lat_bin).max()
    dmoc      = dmoc.assign_coords(botmaxi=elemiz.astype('uint8'))
    del(elemiz)
    
    #___________________________________________________________________________
    if do_compute: dmoc = dmoc.compute()
    if do_load   : dmoc = dmoc.load()
    if do_persist: dmoc = dmoc.persist()
    
    #___________________________________________________________________________
    return(dmoc)



#_______________________________________________________________________________     
# do creepy brute force play around to enforce more or less monotonicity in 
# dens_z, std_dens_Z --> not realy recommendet to do only as a last option
def do_ztransform(data):
    from scipy.interpolate import interp1d
    from numpy.matlib import repmat
    from scipy import interpolate
    import numpy.ma as ma
    
    # use depth information of depth of density classes at latitude
    data_y = data['ndens_z'].values[1:-1,:].copy()
#     data_y = data['dmoc_zpos'].values.copy()
#     data_y[1,:] = 0.0
#     data_y[-1,:] = -6250
    data_y[data_y>=-1.0]=np.nan
            
    # do dirty trick here !!! make sure that below the deepest depth at 
    # every latitude that there is no nan value or a value shallower than
    # the deepest depth left
    nlat  =  data['lat'].values.size
    ndens =  data_y.shape[0]
    for kk in range(nlat):
        min_datay = np.nanmin(data_y[:,kk])
        min_dep   = -6200.0
        for jj in list(range(0,ndens)[::-1]): # to bottom to top
            if data_y[jj,kk]==min_datay:break
            if np.isnan(data_y[jj,kk]) or data_y[jj,kk]>min_datay: data_y[jj,kk]=min_datay
#             if np.isnan(data_y[jj,kk]) or data_y[jj,kk]>min_datay: data_y[jj,kk]=min_dep
    del(min_datay)  
    # same but for the surface
    for kk in range(nlat):
        max_datay = np.nanmax(data_y[:,kk])
        max_dep   = 0.0
        for jj in list(range(0,ndens)): 
            if data_y[jj,kk]==max_datay:break
            if np.isnan(data_y[jj,kk]) or data_y[jj,kk]<max_datay: data_y[jj,kk]=max_datay        
#             if np.isnan(data_y[jj,kk]) or data_y[jj,kk]<max_datay: data_y[jj,kk]=max_dep        
    del(max_datay)        
    # do [ndens x nlat] matrix for latitudes
    data_x   = data['lat'].values.copy()
    data_x   = repmat(data_x,data['ndens'].values[1:-1].size,1)
#     data_x   = repmat(data_x,data['ndens'].values.size,1)
            
    ## do nearest neighbour interpolation of remaining nan values in depth_y
    xx, yy = np.meshgrid(data_x[0,:], data['ndens'].values[1:-1])
#     xx, yy = np.meshgrid(data_x[0,:], data['ndens'].values)
    data_y = np.ma.masked_invalid(data_y)
    data_y = interpolate.griddata((xx[~data_y.mask], yy[~data_y.mask]), data_y[~data_y.mask].ravel(),
                                  (xx, yy), method='nearest')
    
    
    for ni in range(1, data_y.shape[0]):
        idx =  np.where(data_y[ni-1,:]<data_y[ni,:])[0]
        if list(idx):
            data_y[ni,idx] = data_y[ni-1,idx]
            
#     data_y[-1,:] = -6000
    return(data_x, data_y)        

#
#
#_______________________________________________________________________________
def do_ztransform_martin(mesh, data):
    from scipy.interpolate import interp1d
    #___________________________________________________________________________
    lat, dens,  = data['lat'].values, data['ndens'].values
    dep         = np.abs(mesh.zmid)
    nlat, ndens = lat.size, dens.size 
    nz          = dep.size
    
    #___________________________________________________________________________
    sigma2    = data['nz_rho'].values
    dmoc      = data['dmoc'].values
    
    #___________________________________________________________________________
    data_dmocz=np.zeros(sigma2.shape)
    #f = interp1d(np.array(std_dens), dmoc, axis=0, bounds_error = False, fill_value=0)   
    f = interp1d(dens, dmoc, axis=0, bounds_error = False, fill_value=0)
    for li in range(nlat):
        data_dmocz[:,li] = f(sigma2[:,li])[:,li]
    
    return(lat, dep, data_dmocz, sigma2)

#
#
#_______________________________________________________________________________________
def do_ztransform_mom6(mesh, data):
    from scipy.interpolate import interp1d

    #___________________________________________________________________________________
    lat, dens, dep  = data['lat'].values, data['ndens'].values, np.abs(mesh.zmid)
    nlat, ndens, nz = lat.size, dens.size, dep.size
    sigma2 = data['nz_rho'].values
    sigma2[np.isnan(sigma2)]=40.0

    data_v = data['dmoc'].values[:,:]
    data_x = np.ones([ndens,nlat])*lat
    data_y = np.zeros([ndens,nlat])
    for li in range(nlat):
        f = interp1d(sigma2[:,li], dep, bounds_error=False)        
        #data_y[:,li] = f(std_dens[:])
        data_y[:,li] = f(dens[:])
    data_y[np.isnan(data_y)] = 0.0
    
    #print(data_x.shape)
    #print(data_y.shape)
    #print(data_v.shape)
    #densc=dict({'data_x':data_x, 'data_y':data_y, 'data_v':data_v})
    
    #___________________________________________________________________________
    data_dmocz     = np.zeros([nz, nlat])
    for li in range(nlat):
        f = interp1d(data_y[:,li], data_v[:,li], bounds_error=False)
        data_dmocz[:,li] = f(dep)
    

    return(lat, dep, data_dmocz)

#
#
#_______________________________________________________________________________
def do_ztransform_hydrography(mesh, data):
    from scipy.interpolate import interp1d
    #___________________________________________________________________________
    lat, dens,  = data['lat'].values, data['ndens'].values
    dep         = np.abs(mesh.zmid)
    nlat, ndens = lat.size, dens.size 
    nz          = dep.size
    
    #___________________________________________________________________________
    sigma2    = data['nz_rho'].values
    dmoc      = data['dmoc'].values
    
    #___________________________________________________________________________
    data_dmocz=np.zeros(sigma2.shape)
    f = interp1d(dens, dmoc, axis=0, bounds_error = False, fill_value=0)
    for li in range(nlat):
        #data_dmocz[:,li] = np.interp(sigma2[:,li], dens, dmoc[:,li])
        
        
        data_dmocz[:,li] = f(sigma2[:,li])[:,li]
    
    return(lat, dep, data_dmocz)
