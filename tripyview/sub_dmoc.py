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
                   do_compute=False, do_load=True, do_persist=False, 
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
                do_compute=do_compute, do_load=do_load, do_persist=do_persist)
            
            data['std_heat_flux'].data = data['std_heat_flux'].data +\
            load_data_fesom2(mesh, datapath, vname='std_frwt_flux', 
                year=year, descript=descript , do_info=do_info, do_ie2n=False, 
                do_tarithm=do_tarithm, do_nan=False, 
                do_compute=do_compute, do_load=do_load, do_persist=do_persist)['std_frwt_flux'].data + \
            load_data_fesom2(mesh, datapath, vname='std_rest_flux', 
                year=year, descript=descript , do_info=do_info, do_ie2n=False, 
                do_tarithm=do_tarithm, do_nan=False, 
                do_compute=do_compute, do_load=do_load, do_persist=do_persist)['std_rest_flux'].data
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
                do_compute=do_compute, do_load=do_load, do_persist=do_persist).rename({'std_heat_flux':'dmoc_fh'})], combine_attrs="no_conflicts") 
            
            data_dMOC = xr.merge([data_dMOC, 
                load_data_fesom2(mesh, datapath, vname='std_frwt_flux', 
                year=year, descript=descript , do_info=do_info, do_ie2n=False, 
                do_tarithm=do_tarithm, do_nan=False, 
                do_compute=do_compute, do_load=do_load, do_persist=do_persist).rename({'std_frwt_flux':'dmoc_fw'})], combine_attrs="no_conflicts")   
            
            data_dMOC = xr.merge([data_dMOC, 
                load_data_fesom2(mesh, datapath, vname='std_rest_flux', 
                year=year, descript=descript , do_info=do_info, do_ie2n=False, 
                do_tarithm=do_tarithm, do_nan=False,
                do_compute=do_compute, do_load=do_load, do_persist=do_persist).rename({'std_rest_flux':'dmoc_fr'})], combine_attrs="no_conflicts")   
        
        #_______________________________________________________________________
        # add density levels and density level weights
        # kick out dummy ndens varaible that is wrong in the files, replace it with 
        # proper dens variable
        data_dMOC = data_dMOC.drop_vars('ndens') 
        wd, w     = np.diff(std_dens), np.zeros(dens.size)
        w[0], w[1:-1], w[-1] = wd[0]/2., (wd[0:-1]+wd[1:])/2., wd[-1]/2.  # drho @  std_dens level boundary
        w_dens    = xr.DataArray(w, dims=["ndens"]).astype('float32')
        data_dMOC = data_dMOC.assign_coords({ 'dens'  :dens.chunk({  'ndens':data_dMOC.chunksizes['ndens']}), \
                                              'w_dens':w_dens.chunk({'ndens':data_dMOC.chunksizes['ndens']}) })
        del(w, wd, w_dens)
        
        #_______________________________________________________________________
        # rescue global attributes will get lost during multiplication
        data_attrs= data_dMOC.attrs 
        
        # multiply with weights
        data_dMOC = data_dMOC / data_dMOC.w_dens * 1024
        if do_dflx: 
            data_dMOC = data_dMOC / data_dMOC.w_A#--> element area
        
        # put back global attributes
        data_dMOC = data_dMOC.assign_attrs(data_attrs)
        del(data_attrs)
        
    #___________________________________________________________________________
    # add volume trend  
    if add_trend:  
        data_dMOC = xr.merge([data_dMOC, 
                              load_data_fesom2(mesh, datapath, vname='std_dens_dVdT', 
                              year=year, descript=descript , do_info=do_info, do_ie2n=False, 
                              do_tarithm=do_tarithm, do_nan=False, do_compute=do_compute).rename({'std_heat_flux':'dmoc_dvdt'}).drop_vars('ndens')],
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
        #_______________________________________________________________________
        if do_useZinfo=='std_dens_H':
            # add vertical density class thickness
            data_h = load_data_fesom2(mesh, datapath, vname='std_dens_H', 
                        year=year, descript=descript , do_info=do_info, 
                        do_ie2n=False, do_tarithm=do_tarithm, do_nan=False, 
                        do_compute=do_compute, do_load=do_load, do_persist=do_persist).rename({'std_dens_H':'ndens_h'})
            data_h = data_h.assign_coords({'dens':dens.chunk({'ndens':data_h.chunksizes['ndens']})})
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
                        do_compute=do_compute, do_load=do_load, do_persist=do_persist).rename({'std_dens_Z':'ndens_z'})
            data_z = data_z.assign_coords({'dens':dens.chunk({'ndens':data_z.chunksizes['ndens']})})
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
                            do_compute=do_compute, do_load=do_load, do_persist=do_persist).rename({'density_dMOC':'nz_rho'})
                
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
                            do_compute=do_compute, do_load=do_load, do_persist=do_persist).rename({'sigma2':'nz_rho'})
                
                # make land sea mask nan --> here ref density is already substracted
                data_sigma2 = data_sigma2.where(data_sigma2!=0.0)
            
            data_sigma2  = data_sigma2.drop_vars(['ndens','lon','lat', 'nodi', 'nodiz', 'w_A']) 
            data_sigma2  = data_sigma2.assign_coords({'dens':dens.chunk({'ndens':data_sigma2.chunksizes['ndens']})})
            
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
        # add divergence of density classes --> diapycnal velocity
        data_div  = load_data_fesom2(mesh, datapath, vname='std_dens_DIV' , 
                        year=year, descript=descript , do_info=do_info, 
                        do_ie2n=False, do_tarithm=do_tarithm, do_nan=False, 
                        do_compute=do_compute, do_load=do_load, do_persist=do_persist).rename({'std_dens_DIV':'dmoc'})
        data_div  = data_div.drop_vars(['ndens', 'nodi']) 
        data_div  = data_div.assign_coords({'dens':dens.chunk({'ndens':data_div.chunksizes['ndens']})})
        
        # doing this step here so that the MOC amplitude is correct, setp 1 of 2
        # divide with verice area
        data_div  = data_div/data_div['w_A']    # --> vertice area
        
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
            data_div  = data_div*data_dMOC['w_A']# --> element area
            
        data_dMOC = xr.merge([data_dMOC, data_div], combine_attrs="no_conflicts")  
        del(data_div)
        
        # load density class divergence from bolus velolcity
        if (do_bolus): 
            # add divergence of density classes --> diapycnal velocity
            data_div_bolus  = load_data_fesom2(mesh, datapath, vname='std_dens_DIVbolus' , 
                                    year=year, descript=descript , do_info=do_info, 
                                    do_ie2n=False, do_tarithm=do_tarithm, do_nan=False, 
                                    do_compute=do_compute, do_load=do_load, do_persist=do_persist).rename({'std_dens_DIVbolus':'dmoc_bolus'})
            data_div_bolus  = data_div_bolus.drop_vars(['ndens', 'nodi']) 
            data_div_bolus  = data_div_bolus.assign_coords({'dens':dens.chunk({'ndens':data_div_bolus.chunksizes['ndens']})})
            
            # doing this step here so that the MOC amplitude is correct, setp 1 of 2
            # divide with verice area
            data_div_bolus  = data_div_bolus/data_div_bolus['w_A'] # --> vertice area
            
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
def calc_dmoc(mesh, data_dMOC, dlat=1.0, which_moc='gmoc', do_info=True, do_checkbasin=False,
              exclude_meditoce=True, do_bolus=True, do_parallel=False, n_workers=10, 
              do_compute=False, do_load=True, do_persist=False, 
              **kwargs):
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
        vattr = data_dMOC[vari].attrs
        if   vari=='dmoc_fh'   : vattr.update({'long_name':'Transformation from heat flux'                 , 'units':'Sv'   })
        elif vari=='dmoc_fw'   : vattr.update({'long_name':'Transformation from freshwater flux'           , 'units':'Sv'   })
        elif vari=='dmoc_fr'   : vattr.update({'long_name':'Transformation from surface salinity restoring', 'units':'Sv'   })
        elif vari=='dmoc_fd'   : vattr.update({'long_name':'Transformation from total density flu'         , 'units':'Sv'   })
        elif vari=='dmoc_dvdt' : vattr.update({'long_name':'Transformation from volume change'             , 'units':'Sv'   })
        elif vari=='dmoc'      : vattr.update({'long_name':'Density MOC'                                   , 'units':'Sv'   })
        elif vari=='dmoc_bolus': vattr.update({'long_name':'Density MOC bolus vel.'                        , 'units':'Sv'   })
        elif vari=='ndens_h'   : vattr.update({'long_name':'Density class thickness'                       , 'units':'m'    })
        elif vari=='ndens_zfh' : vattr.update({'long_name':'Density class z position'                      , 'units':'m'    })
        elif vari=='ndens_z'   : vattr.update({'long_name':'Density class z position'                      , 'units':'m'    })
        elif vari=='nz_rho'    : vattr.update({'long_name':'sigma2 density in zcoord'                      , 'units':'kg/m³'})
        dmoc[vari]=dmoc[vari].assign_attrs(vattr)
    del(data_dMOC)
    
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
    # Add bolus MOC to normal MOC
    if 'dmoc_bolus'in list(dmoc.keys()) and do_bolus:  
        dmoc['dmoc'].data = dmoc['dmoc'].data + dmoc['dmoc_bolus'].data 
    
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



#+___PLOT MERIDIONAL OVERTRUNING CIRCULATION  _________________________________+
#|                                                                             |
#+_____________________________________________________________________________+
def plot_dmoc(mesh, data, which_moc='gmoc', which_transf='dmoc', figsize=[12, 6], 
              n_rc=[1, 1], do_grid=True, cinfo=None, do_rescale=None,
              do_reffig=False, ref_cinfo=None, ref_rescale=None,
              cbar_nl=8, cbar_orient='vertical', cbar_label=None, cbar_unit=None,
              do_bottom=True, color_bot=[0.6, 0.6, 0.6], 
              pos_fac=1.0, pos_gap=[0.01, 0.02], do_save=None, save_dpi=600, 
              do_contour=True, do_clabel=True, title='descript', 
              do_yrescale=True, do_zcoord=False, do_check=True, 
              pos_extend=[0.075, 0.075, 0.90, 0.95] ):
    #___________________________________________________________________________
    fontsize = 12
    
    #___________________________________________________________________________
    # make matrix with row colum index to know where to put labels
    rowlist = np.zeros((n_rc[0], n_rc[1]))
    collist = np.zeros((n_rc[0], n_rc[1]))       
    for ii in range(0,n_rc[0]): rowlist[ii,:]=ii
    for ii in range(0,n_rc[1]): collist[:,ii]=ii
    rowlist = rowlist.flatten()
    collist = collist.flatten()
    
    #___________________________________________________________________________    
    # create figure and axes
    fig, ax = plt.subplots( n_rc[0],n_rc[1],
                                figsize=figsize, 
                                gridspec_kw=dict(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.05,),
                                constrained_layout=False, sharex=True, sharey=True)
    
    #___________________________________________________________________________    
    # flatt axes if there are more than 1
    if isinstance(ax, np.ndarray): ax = ax.flatten()
    else:                          ax = [ax] 
    nax = len(ax)
     
    #___________________________________________________________________________
    # data must be list filled with xarray data
    if not isinstance(data  , list): data  = [data]
    ndata = len(data) 
    
    #___________________________________________________________________________
    # set up color info 
    if do_reffig:
        ref_cinfo = do_setupcinfo(ref_cinfo, [data[0]], ref_rescale, do_dmoc=which_transf)
        cinfo     = do_setupcinfo(cinfo    , data[1:] , do_rescale , do_dmoc=which_transf)
    else:
        cinfo     = do_setupcinfo(cinfo    , data     , do_rescale , do_dmoc=which_transf)
    
    #___________________________________________________________________________
    # compute remapping zcoord-->dens --> compute interpolants !!!
    if do_yrescale: 
        # compute remapping dens-->reg and reg-->dens  --> compute interpolants !!!
        #remap_d2r=np.array([ 0.00, 29.70, 30.50,
                            ## 0.00, 28.10, 28.90, 29.70, 30.50, 
                            #30.95, 31.50, 32.05, 32.60, 33.15, 
                            #33.70, 34.25, 34.75, 35.15, 35.50, 
                            #35.80, 36.04, 36.20, 36.38, 36.52, 
                            #36.62, 36.70, 36.77, 36.83, 36.89, 
                            #36.97, 36.98, 36.99, 37.00, 37.01, 
                            #37.02, 37.03, 37.04, 37.05, 37.06, 37.09, 37.11, 
                            #37.13, 37.15, 37.20, 37.30, 37.40, 40.])
        remap_d2r=np.hstack([0.00, 
                             np.arange(30.00, 35.99, 1.00),
                             np.arange(36.00, 36.64, 0.20),# 0.15
                             np.arange(36.65, 36.91, 0.05),
                             np.arange(36.92, 37.04, 0.02),
                             np.arange(37.05, 38.50, 0.25),
                             40.00])
        remap_d2r = np.sort(np.unique(remap_d2r))
        ramap_d2r_major = np.array([30.00, 36.00, 36.65, 36.92, 37.05])
        reg      = np.linspace(0, len(remap_d2r), len(remap_d2r))
        reg      = reg[::-1]
        dens2reg = interp1d(remap_d2r, reg      , kind='linear')
        reg2dens = interp1d(reg      , remap_d2r, kind='linear')

    #___________________________________________________________________________
    # loop over axes
    ndi, nli, nbi =0, 0, 0
    hpall=list()
    for ii in range(0,ndata):
        
        #_______________________________________________________________________
        # plot dmoc in z-coordinates
        if do_zcoord:
            # use depth information of depth of density classes at latitude
            if   'ndens_zfh'    in list(data[ii].keys()): 
                data_y      = -data[ii]['ndens_zfh'].values
                data_x      = np.ones(data_y.shape)*data[ii]['lat'].values
            
            elif 'nz_rho'    in list(data[ii].keys()): 
                #data_x, data_y, data_v, dum = do_ztransform_martin(mesh, data[ii])
                #data_x, data_y, data_v = do_ztransform_mom6(mesh, data[ii])
                data_x, data_y, data_v = do_ztransform_hydrography(mesh, data[ii])
                data_y = -data_y
            
            elif 'ndens_z'    in list(data[ii].keys()): 
                data_x, data_y = do_ztransform(data[ii])
                data_v = data[ii]['dmoc'].values.copy()
                data_v = data_v[1:-1,:]
            else:
                raise ValueError(' --> could not find any vertical position of the density class, no zlevel prokection possible!')
            
        #_______________________________________________________________________
        # plot dmoc in density-coordinates  
        else:
            data_x, data_y = data[ii]['lat'].values.copy(), data[ii]['dens'].values.copy()
            
        #_______________________________________________________________________
        # What  should be plotted: density MOC, Surface Transformation or Inner
        # Transformation
        if   which_transf == 'dmoc':
            if   'ndens_zfh'    in list(data[ii].keys()) and do_zcoord:
                data_plot = data[ii]['dmoc'].values.copy()
                do_check=True
            elif 'nz_rho' in list(data[ii].keys())       and do_zcoord: 
                data_plot = data_v
                do_check=False
            elif 'ndens_z'in list(data[ii].keys())       and do_zcoord: 
                data_plot = data_v
                do_check=False
            else:    
                data_plot = data[ii]['dmoc'].values.copy()
            
            #___________________________________________________________________
            # PLOT DMOC INFO: maximum/minimum dmoc sigma2 and depth
            if do_check:
                idxmax = data[ii]['dmoc'].argmax(dim=["ndens", "lat"])
                idxmin = data[ii]['dmoc'].argmin(dim=["ndens", "lat"])
                dmoc_max, dmoc_min = data[ii]['dmoc'].isel(idxmax).data, data[ii]['dmoc'].isel(idxmin).data
                if not do_zcoord:
                    s_max, l_max,  = data_y[idxmax['ndens'].data], data_x[idxmax['lat'].data], 
                    s_min, l_min,  = data_y[idxmin['ndens'].data], data_x[idxmin['lat'].data]
                    
                    d_max, d_min = np.nan, np.nan
                    if   'ndens_zfh'    in list(data[ii].keys()): 
                        d_max = data[ii]['ndens_zfh'].isel(idxmax).data
                        d_min = data[ii]['ndens_zfh'].isel(idxmin).data
                else:     
                    s_max = data[ii]['dens'].isel(ndens=idxmax['ndens']).data
                    l_max = data_x[idxmax['ndens'].data, idxmax['lat'].data]
                    d_max = data_y[idxmax['ndens'].data, idxmax['lat'].data]
                    s_min = data[ii]['dens'].isel(ndens=idxmin['ndens']).data
                    l_min = data_x[idxmin['ndens'].data, idxmin['lat'].data]
                    d_min = data_y[idxmin['ndens'].data, idxmin['lat'].data]
                print('DMOC_max={:5.1f} (sigma2={:5.2f}kg/m^3, depth={:5.0f}m, lat={:5.1f}°N)'.format(dmoc_max, s_max, d_max, l_max))
                print('DMOC_min={:5.1f} (sigma2={:5.2f}kg/m^3, depth={:5.0f}m, lat={:5.1f}°N)'.format(dmoc_min, s_min, d_min, l_min))                    
        
        elif which_transf == 'srf' : 
            data_plot = -(data[ii]['dmoc_fh'].values.copy()+ \
                          data[ii]['dmoc_fw'].values.copy()+ \
                          data[ii]['dmoc_fr'].values.copy())
            
        elif which_transf == 'inner':
            data_plot = data[ii]['dmoc'].values.copy()    + \
                        (data[ii]['dmoc_fh'].values.copy()+ \
                         data[ii]['dmoc_fw'].values.copy()+ \
                         data[ii]['dmoc_fr'].values.copy())
        
        #_______________________________________________________________________
        if do_reffig: 
            if ii==0: cinfo_plot = ref_cinfo
            else    : cinfo_plot = cinfo
        else: cinfo_plot = cinfo
        
        #_______________________________________________________________________
        data_plot[data_plot<cinfo_plot['clevel'][ 0]] = cinfo_plot['clevel'][ 0]+np.finfo(np.float32).eps
        data_plot[data_plot>cinfo_plot['clevel'][-1]] = cinfo_plot['clevel'][-1]-np.finfo(np.float32).eps
        
        #_______________________________________________________________________
        # if plot in density-coordinates first scale to regular y-axes, and flip  
        # the  y-axes of data_y and data_plot since ax.invert_yaxis() is not working 
        # with share plt.subplot(..., sharey=True)
        if not do_zcoord: 
            if do_yrescale: data_y = dens2reg(data_y)
            data_plot, data_y = data_plot[1:-1,:], data_y[1:-1]
            data_plot, data_y = data_plot[::-1,:], data_y[::-1]
        
        #if 'ndens_z'    in list(data[ii].keys()) and do_zcoord:
            #data_plot = data_plot[1:-1,:]
            
        #_______________________________________________________________________
        # plot DATA
        hp=ax[ii].contourf(data_x, data_y, data_plot, levels=cinfo_plot['clevel'], 
                           extend='both', cmap=cinfo_plot['cmap'])
        hpall.append(hp)
        
        if do_contour: 
            tickl    = cinfo_plot['clevel']
            ncbar_l  = len(tickl)
            idx_cref = np.where(cinfo_plot['clevel']==cinfo_plot['cref'])[0]
            #idx_cref = np.asscalar(idx_cref) --> asscalar replaced in numppy>1.16
            idx_cref = idx_cref.item()
            nstep    = ncbar_l/(cbar_nl)
            nstep    = np.max([np.int32(np.floor(nstep)),1])
            
            idx = np.arange(0, ncbar_l, 1)
            idxb = np.ones((ncbar_l,), dtype=bool)                
            idxb[idx_cref::nstep]  = False
            idxb[idx_cref::-nstep] = False
            idx_yes = idx[idxb==False]
            
            aux_clvl = cinfo_plot['clevel'][idx_yes]
            aux_clvl = aux_clvl[aux_clvl!=cinfo_plot['cref']]
            cont=ax[ii].contour(data_x, data_y, data_plot, 
                                levels=aux_clvl, colors='k', linewidths=[0.5]) #linewidths=[0.5,0.25])
            if do_clabel: 
                ax[ii].clabel(cont, cont.levels[np.where(cont.levels!=cinfo_plot['cref'])], 
                            inline=1, inline_spacing=1, fontsize=6, fmt='%1.1f Sv')
            ax[ii].contour(data_x, data_y, data_plot, 
                                levels=[0.0], colors='k', linewidths=[1.25]) #linewidths=[0.5,0.25])
        
        #_______________________________________________________________________
        # plot bottom representation in case of z-coordinates
        if do_zcoord: 
            #data_bot = np.nanmin(data_y, axis=0)
            data_bot = mesh.zlev[data[ii]['botmaxi'].max()+1]
            ax[ii].plot(data[ii]['lat'], -data[ii]['botmax'], color='k')
            ax[ii].fill_between(data[ii]['lat'], -data[ii]['botmax'], data_bot, color=color_bot, zorder=2)#,alpha=0.95)
            ax[ii].set_ylim([data_bot,0])
        
        #_______________________________________________________________________
        # set latitude limits
        xlim = list(ax[ii].get_xlim())  
        if   which_moc=='amoc' : xlim[1]=75
        elif which_moc=='ipmoc': xlim[1]=60
        elif which_moc=='pmoc' : xlim[1]=60
        ax[ii].set_xlim(xlim)
            
        #_______________________________________________________________________
        # in case y-axes should be rescaled (do_yrescale=true) and plot is density
        # coordinates give the now regular ticks new ticklabels with the proper
        # density values. Make the difference between major and minor ticks
        if do_yrescale and not do_zcoord: 
            xy, x_ind, y_ind = np.intersect1d(remap_d2r, ramap_d2r_major, return_indices=True)
            # --> this will become major tick marks (larger fontsize)
            ymajorticks = reg[x_ind] 
            ax[ii].set_yticks( ymajorticks, minor=False ) 
            
            ymajorlabel_list = np.around(reg2dens(ymajorticks), 3).tolist()
            ylabelmayjor_list_fmt=list()
            for num in ymajorlabel_list: ylabelmayjor_list_fmt.append('{:2.2f}'.format(num))
            ax[ii].set_yticklabels(ylabelmayjor_list_fmt, minor=False, size = 10)
            
            # --> this will become minor tick marks (smaller fontsize)
            yminorticks = np.setdiff1d(reg[1:-1], ymajorticks)
            ax[ii].set_yticks( yminorticks, minor=True )
            
            yminorlabel_list = np.around(reg2dens(yminorticks), 3).tolist()
            ylabelminor_list_fmt=list()
            for num in yminorlabel_list: ylabelminor_list_fmt.append('{:2.2f}'.format(num))
            ax[ii].set_yticklabels(ylabelminor_list_fmt, minor=True, size = 6)
            
        elif not do_yrescale:
            # only do invert axis for one single axis in the list --> if you do 
            # it for all the axis ax[ii].invert_yaxis() it wont't work !!!
            if ii==0: ax[0].invert_yaxis()
        
        #_______________________________________________________________________
        # fix color range
        for im in ax[ii].get_images(): im.set_clim(cinfo_plot['clevel'][ 0], cinfo_plot['clevel'][-1])
        
        #_______________________________________________________________________
        # plot grid lines 
        if do_grid: ax[ii].grid(color='k', linestyle='-', linewidth=0.25,alpha=1.0)
        
        #_______________________________________________________________________
        # set description string
        if title != None: 
            if not do_zcoord: 
                txtx = data_x[0]+(data_x[-1]-data_x[0])*0.025
                txty = data_y[0]+(data_y[-1]-data_y[0])*0.025    
            else:
                txtx = data_x.min()+(data_x.max()-data_x.min())*0.025
                #txty = data_y.min()+(data_y.max()-data_y.min())*0.025
                txty = data_bot+(0-data_bot)*0.025    
                data_bot
            if   isinstance(title,str) : 
                # if title string is 'descript' than use descript attribute from 
                # data to set plot title 
                dum_vname=list(data[ii].keys())[0]
                if title=='descript' and ('descript' in data[ii][dum_vname].attrs.keys() ):
                    txts = data[ii][dum_vname].attrs['descript']
                else:
                    txts = title
            # is title list of string        
            elif isinstance(title,list):   
                txts = title[ii]
            ax[ii].text(txtx, txty, txts, fontsize=12, fontweight='bold', horizontalalignment='left')
        
        #_______________________________________________________________________
        # set x and y labels for z-coordinates and density-coordinates
        if do_zcoord: 
            if collist[ii]==0: ax[ii].set_ylabel('Depth [m]',fontsize=12)
        else:
            if collist[ii]==0: ax[ii].set_ylabel('${\\sigma}_{2}$ potential density [kg/m${^3}$]',fontsize=12)
        if rowlist[ii]==n_rc[0]-1: ax[ii].set_xlabel('Latitude [deg]',fontsize=12)
        #_______________________________________________________________________    
        
    nax_fin = ii+1
    
    #___________________________________________________________________________
    # delete axes that are not needed
    #for jj in range(nax_fin, nax): fig.delaxes(ax[jj])
    for jj in range(ndata, nax): fig.delaxes(ax[jj])
    if nax != nax_fin-1: ax = ax[0:nax_fin]
    
    #___________________________________________________________________________
    # initialise colorbar
    if do_reffig==False:
        cbar = fig.colorbar(hp, orientation=cbar_orient, ax=ax, ticks=cinfo_plot['clevel'], 
                        extendrect=False, extendfrac=None,
                        drawedges=True, pad=0.025, shrink=1.0)
        
        # do formatting of colorbar 
        cbar = do_cbar_formatting(cbar, do_rescale, cbar_nl, fontsize, cinfo_plot['clevel'])
        
        # do labeling of colorbar
        #if n_rc[0]==1:
            #if   which_moc=='gmoc' : cbar_label = 'Global Meridional \n Overturning Circulation [Sv]'
            #elif which_moc=='amoc' : cbar_label = 'Atlantic Meridional \n Overturning Circulation [Sv]'
            #elif which_moc=='aamoc': cbar_label = 'Arctic-Atlantic Meridional \n Overturning Circulation [Sv]'
            #elif which_moc=='pmoc' : cbar_label = 'Pacific Meridional \n Overturning Circulation [Sv]'
            #elif which_moc=='ipmoc': cbar_label = 'Indo-Pacific Meridional \n Overturning Circulation [Sv]'
            #elif which_moc=='imoc' : cbar_label = 'Indo Meridional \n Overturning Circulation [Sv]'
        #else:
            #if   which_moc=='gmoc' : cbar_label = 'Global Meridional Overturning Circulation [Sv]'
            #elif which_moc=='amoc' : cbar_label = 'Atlantic Meridional Overturning Circulation [Sv]'
            #elif which_moc=='aamoc': cbar_label = 'Arctic-Atlantic Meridional Overturning Circulation [Sv]'
            #elif which_moc=='pmoc' : cbar_label = 'Pacific Meridional Overturning Circulation [Sv]'
            #elif which_moc=='ipmoc': cbar_label = 'Indo-Pacific Meridional Overturning Circulation [Sv]'
            #elif which_moc=='imoc' : cbar_label = 'Indo Meridional Overturning Circulation [Sv]'
        if   which_moc=='gmoc' : cbar_label = 'Global MOC [Sv]'
        elif which_moc=='amoc' : cbar_label = 'Atlantic MOC [Sv]'
        elif which_moc=='aamoc': cbar_label = 'Arctic-Atlantic MOC [Sv]'
        elif which_moc=='pmoc' : cbar_label = 'Pacific MOC [Sv]'
        elif which_moc=='ipmoc': cbar_label = 'Indo-Pacific MOC [Sv]'
        elif which_moc=='imoc' : cbar_label = 'Indo MOC [Sv]' 
        
        dum_vname=list(data[ii].keys())[0]
        if 'str_ltim' in data[0][dum_vname].attrs.keys():
            cbar_label = cbar_label+'\n'+data[0][dum_vname].attrs['str_ltim']
            
        if   which_transf=='dmoc' : cbar.set_label('Density - '       +cbar_label, size=fontsize+2)
        elif which_transf=='srf'  : cbar.set_label('Srf. Transf. - '  +cbar_label, size=fontsize+2)
        elif which_transf=='inner': cbar.set_label('Inner. Transf. - '+cbar_label, size=fontsize+2)
    else:
        cbar=list()
        for ii, aux_ax in enumerate(ax): 
            cbar_label = ''
            if ii==0:
                aux_cbar = fig.colorbar(hpall[ii], orientation=cbar_orient, ax=aux_ax, ticks=ref_cinfo['clevel'], 
                                        extendrect=False, extendfrac=None, drawedges=True, pad=0.025, shrink=1.0)
                aux_cbar = do_cbar_formatting(aux_cbar, ref_rescale, cbar_nl, fontsize, ref_cinfo['clevel'])
            else:
                aux_cbar = fig.colorbar(hpall[ii], orientation=cbar_orient, ax=aux_ax, ticks=cinfo['clevel'], 
                                        extendrect=False, extendfrac=None, drawedges=True, pad=0.025, shrink=1.0)
                aux_cbar = do_cbar_formatting(aux_cbar, do_rescale, cbar_nl, fontsize, cinfo['clevel'])
                cbar_label = 'anom. '
            if   which_moc=='gmoc' : cbar_label = cbar_label+'Global MOC [Sv]'
            elif which_moc=='amoc' : cbar_label = cbar_label+'Atlantic MOC [Sv]'
            elif which_moc=='aamoc': cbar_label = cbar_label+'Arctic-Atlantic MOC [Sv]'
            elif which_moc=='pmoc' : cbar_label = cbar_label+'Pacific MOC [Sv]'
            elif which_moc=='ipmoc': cbar_label = cbar_label+'Indo-Pacific MOC [Sv]'
            elif which_moc=='imoc' : cbar_label = cbar_label+'Indo MOC [Sv]'    
            
            if 'str_ltim' in data[0]['dmoc'].attrs.keys():
                cbar_label = cbar_label+'\n'+data[0]['dmoc'].attrs['str_ltim']
            if   which_transf=='dmoc' : aux_cbar.set_label('Density - '+cbar_label, size=fontsize+2)
            elif which_transf=='srf'  : aux_cbar.set_label('Srf. Transf. - '+cbar_label, size=fontsize+2)
            elif which_transf=='inner': aux_cbar.set_label('Inner. Transf. - '+cbar_label, size=fontsize+2) 
            cbar.append(aux_cbar)
            
    #___________________________________________________________________________
    # repositioning of axes and colorbar
    if do_reffig==False:
        ax, cbar = do_reposition_ax_cbar(ax, cbar, rowlist, collist, pos_fac, pos_gap, 
                                        title=None, extend=pos_extend)
    fig.canvas.draw()
    
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, fig, dpi=save_dpi)
    plt.show()
    
    #___________________________________________________________________________
    return(fig, ax, cbar)



#+___PLOT MERIDIONAL OVERTRUNING CIRCULATION TIME-SERIES_______________________+
#|                                                                             |
#+_____________________________________________________________________________+
def plot_dmoc_tseries(moct_list, input_names, which_cycl=None, which_lat=['max'], 
                       which_moc='amoc', do_allcycl=False, do_concat=False, ymaxstep=1, xmaxstep=5,
                       str_descript='', str_time='', figsize=[], do_rapid=None, 
                       do_save=None, save_dpi=600, do_pltmean=True, do_pltstd=False ):    
    
    import matplotlib.patheffects as path_effects
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator

    if len(figsize)==0: figsize=[13,6.5]
    if do_concat: figsize[0] = figsize[0]*2
    
    #___________________________________________________________________________
    # loop over which_lat list, either with single latitude entry 45.0 or 
    # string 'max'
    for lat in which_lat: 
        
        #_______________________________________________________________________
        # loop over vars dmoc_nadw or dmoc_aabw
        for var in list(moct_list[0].keys()):
            fig,ax= plt.figure(figsize=figsize),plt.gca()
        
            #___________________________________________________________________
            # setup colormap
            if do_allcycl: 
                if which_cycl != None:
                    cmap = categorical_cmap(np.int32(len(moct_list)/which_cycl), which_cycl, cmap="tab10")
                else: cmap = categorical_cmap(len(moct_list), 1, cmap="tab10")
            else: cmap = categorical_cmap(len(moct_list), 1, cmap="tab10")
            
            #___________________________________________________________________
            ii, ii_cycle = 0, 1
            if which_cycl is None: aux_which_cycl = 1
            else                 : aux_which_cycl = which_cycl
            
            #___________________________________________________________________
            # loop over time series in moct_list
            for ii_ts, (tseries, tname) in enumerate(zip(moct_list, input_names)):
                data = tseries[var]
                #_______________________________________________________________
                # select moc values from single latitude or latidude range
                if lat=='max':
                    if var=='dmoc_aabw': data = data.isel(lat=(data.lat>40) & (data.lat<60)).min(dim='lat') 
                    if var=='dmoc_nadw': data = data.isel(lat=(data.lat>40) & (data.lat<60)).max(dim='lat') 
                    str_label= f'{40}°N<lat<{60}°N'
                elif isinstance(lat, list):    
                    if var=='dmoc_aabw': data = data.isel(lat=(data.lat>lat[0]) & (data.lat<lat[1])).min(dim='lat') 
                    if var=='dmoc_nadw': data = data.isel(lat=(data.lat>lat[0]) & (data.lat<lat[1])).max(dim='lat') 
                    str_label= f'{lat[0]}°N<lat<{lat[1]}°N'
                else:     
                    #data = data.sel(lat=lat)
                    data = data.isel(lat=np.argmin(np.abs(data['lat'].data-lat)))
                    if lat>=0: str_label= f'{lat}°N'
                    else     : str_label= f'{lat}°S'   
                    
                #_______________________________________________________________
                # set time axes in units of years
                time = data['time']
                year = np.unique(data['time.year'])
                totdayperyear = np.where(time.dt.is_leap_year, 366, 365)
                time = auxtime = time.dt.year + (time.dt.dayofyear-time.dt.day[0])/totdayperyear            
                tlim, tdel = [time[0], time[-1]], time[-1]-time[0]
                if do_concat: auxtime = auxtime + (tdel+1)*(ii_cycle-1)
                
                #_______________________________________________________________
                hp=ax.plot(auxtime, data, linewidth=1.5, label=tname, color=cmap.colors[ii_ts,:], marker='o', markerfacecolor='w', markersize=5, zorder=2)
                if np.mod(ii_ts+1,aux_which_cycl)==0 or do_allcycl==False:
                    dict_plt = {'markeredgecolor':'k', 'markeredgewidth':0.5, 'color':hp[0].get_color(), 'clip_box':False, 'clip_on':False, 'zorder':3}
                    if do_pltmean: 
                        plt.plot(time[0]-(tdel)*0.0120, data.mean(), marker='<', **dict_plt)
                    if do_pltstd:
                        plt.plot(time[0]-(tdel)*0.015, data.mean()+data.std(), marker='^', markersize=6, **dict_plt)
                        plt.plot(time[0]-(tdel)*0.015, data.mean()-data.std(), marker='v', markersize=6, **dict_plt)
                #_______________________________________________________________
                ii_cycle=ii_cycle+1
                if ii_cycle>aux_which_cycl: ii_cycle=1
                
            #___________________________________________________________________
            # add Rapid moc data @26.5°
            if do_rapid != None and var == 'dmoc_nadw': 
                path = do_rapid
                rapid26 = xr.open_dataset(path)['moc_mar_hc10']
                rapid26_ym = rapid26.groupby('time.year').mean('time', skipna=True)
                time_rapid = rapid26_ym.year
                if do_allcycl: 
                    time_rapid = time_rapid + (aux_which_cycl-1)*(time[-1]-time[0]+1)
                    
                hpr=plt.plot(time_rapid,rapid26_ym.data,
                        linewidth=2, label='Rapid @ 26.5°N', color='k', marker='o', markerfacecolor='w', 
                        markersize=5, zorder=2)
                
                dict_plt = {'markeredgecolor':'k', 'markeredgewidth':0.5, 'color':'k', 'clip_box':False, 'clip_on':False, 'zorder':3}
                if do_pltmean: 
                    plt.plot(time[0]-(tdel)*0.0120, rapid26_ym.data.mean(), marker='<', markersize=8, **dict_plt)
                if do_pltstd:
                    plt.plot(time[0]-(tdel)*0.015, rapid26_ym.data.mean()+rapid26_ym.data.std(), marker='^', markersize=6, **dict_plt)                        
                    plt.plot(time[0]-(tdel)*0.015, rapid26_ym.data.mean()-rapid26_ym.data.std(), marker='v', markersize=6, **dict_plt)    
                del(rapid26)
            
            #___________________________________________________________________
            ax.legend(shadow=True, fancybox=True, frameon=True, #mode='None', 
                    bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
                    #bbox_to_anchor=(1.04, 1.0), ncol=1) #loc='lower right', 
            ax.set_xlabel('Time [years]',fontsize=12)
            ax.set_ylabel('Density {:s} in [Sv]'.format(which_moc.upper()),fontsize=12)
            
            if   var=='dmoc_nadw': str_cell, str_cells = 'upper cell strength', 'nadw'
            elif var=='dmoc_aabw': str_cell, str_cells = 'lower cell strength', 'aabw'
            ax.set_title(f'{str_cell} @ {str_label}', fontsize=12, fontweight='bold')
            
            #___________________________________________________________________
            xmajor_locator = MultipleLocator(base=xmaxstep) # this locator puts ticks at regular intervals
            ymajor_locator = MultipleLocator(base=ymaxstep) # this locator puts ticks at regular intervals
            ax.xaxis.set_major_locator(xmajor_locator)
            ax.yaxis.set_major_locator(ymajor_locator)
            
            xminor_locator = AutoMinorLocator(5)
            yminor_locator = AutoMinorLocator(4)
            ax.yaxis.set_minor_locator(yminor_locator)
            ax.xaxis.set_minor_locator(xminor_locator)
            
            plt.grid(which='major')
            if not do_concat:
                plt.xlim(time[0]-(time[-1]-time[0])*0.015,time[-1]+(time[-1]-time[0])*0.015)    
            else:    
                plt.xlim(time[0]-(time[-1]-time[0])*0.015,time[-1]+(time[-1]-time[0]+1)*(aux_which_cycl-1)+(time[-1]-time[0])*0.015)    
                
            #___________________________________________________________________
            plt.show()
            fig.canvas.draw()
            
            #___________________________________________________________________
            # save figure based on do_save contains either None or pathname
            aux_do_save = do_save
            if do_save != None:
                aux_do_save = '{:s}_{:s}_{:s}{:s}'.format(do_save[:-4], str_cells, str_label.replace('°','').replace(' ','_'), do_save[-4:])
            do_savefigure(aux_do_save, fig, dpi=save_dpi)
    
    #___________________________________________________________________________
    return(fig,ax)



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
