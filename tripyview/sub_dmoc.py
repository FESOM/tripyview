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
import gc
from .sub_colormap import *
from .sub_utility  import *
from .sub_plot     import *


# ___CALCULATE MERIDIONAL OVERTURNING IN DENSITY COORDINATES___________________
#| Global MOC, Atlantik MOC, Indo-Pacific MOC, Indo MOC                        |
#|                                                                             |
#|                                                                             |
#|_____________________________________________________________________________|
def load_dmoc_data(mesh                         , 
                   datapath                     , 
                   std_dens                     ,
                   year         = None          , 
                   which_transf = 'dmoc'        , 
                   do_tarithm   = 'mean'        , 
                   do_bolus     = True          , 
                   add_bolus    = False         , 
                   add_trend    = False         , 
                   do_wdiap     = False         , 
                   do_dflx      = False         , 
                   do_zcoord    = True          , 
                   do_useZinfo  = 'std_dens_H'  ,
                   do_ndensz    = False         , 
                   descript     = ''            , 
                   do_compute   = False         , 
                   do_load      = False         ,  
                   do_persist   = False         , 
                   do_parallel  = False         , 
                   do_info      = True          ,
                   chunks       = { 'time' :'auto', 'elem':'auto', 'nod2':'auto', \
                                    'edg_n':'auto', 'nz'  :'auto', 'nz1' :'auto', \
                                    'ndens':'auto'},
                   **kwargs):
    """
    --> load data that are neccessary for density moc computation
    
    Parameters:
    
        :mesh:          fesom2 tripyview mesh object,  with all mesh information 
        
        :datapath:      str, path that leads to the FESOM2 data
        
        :std_dens:      np.array with sigma2 density bins that were used in FESOM2
                        for the DMOC diagnostic
        
        :year:          int, list, np.array, range, default: None, single year or 
                        list/array of years whos file should be opened
        
        :which_transf:  str (default='dmoc') which transformation should be computed
                        options area
                        
                        - 'dmoc'    compute dmoc density transformation
                        - 'srf'     compute density transform. from surface forcing 
                        - 'inner'   compute density transform. from interior mixing (dmoc-srf)
        
        :do_tarithm:    str (default='mean') do time arithmetic on time selection
                        option are: None, 'None', 'mean', 'median', 'std', 'var', 'max'
                        'min', 'sum'
            
        :do_bolus:      bool (default=False) load density class divergence from bolus velolcity
                        and add them to the total density class divergence
        
        :add_bolus:     bool (default=False) include density class divergence from bolus velolcity
                        as separate varaible in xarray dataset object
        
        :add_trend:     bool (default=False) include density class volume trend  
                        as separate varaible in xarray dataset object
        
        :do_wdiap:      bool (default=False) load data to be used to only look at
                        diapycnal vertical velocity
        
        :do_dflx:       bool (default=False) load data to be used for the computation 
                        surface buoyancy forced transformations vertical velocities
        
        :do_zcoord:     bool (default=True)  do density MOC remapping back into zcoord
                        space
        
        :do_useZinfo:   str (default='std_dens_H') which data should be used for the 
                        zcoord remapping. Options are:
                        
                        - 'std_dens_H'   use mean layerthickness of density classes (best option)
                        - 'hydrography'  use mean sigma2 hydrography to estime z position of density classes (OK), 
                        - 'density_dMOC' use density_dMOC variable to estime z position of density classes (Bad), 
                        - 'std_dens_Z'   use mean depth of density classes (very bad)
        
        :do_ndensz:     bool (default=False) alreadz compute here the density class z position
                        from the density class layer thickness by cumulatic sumation
        
        :descript:      str (default=''), string to describe dataset is written into 
                        variable attributes                
        
        :do_compute:    bool (default=False), do xarray dataset compute() at the end
                        data = data.compute(), creates a new dataobject the original
                        data object seems to persist
        
        :do_load:       bool (default=True), do xarray dataset load() at the end
                        data = data.load(), applies all operations to the original
                        dataset
        
        :do_persist:    bool (default=False), do xarray dataset persist() at the end
                        data = data.persist(), keeps the dataset as dask array, keeps
                        the chunking   
        
        :chunks:        dict(), (default=dict({'time':'auto', ...})) dictionary 
                        with chunksize of specific dimensions. By default setted 
                        to auto but can also be setted to specific value. In my 
                        observation it revealed that the loading of data was a factor 2-3
                        faster with auto-chunking but this might has its limitation
                        for very large meshes 
        
        :do_info:       bool (defalt=True), print variable info at the end 
        
    Returns:
    
        :data:          object, returns xarray dataset object with density class informations
        
    ____________________________________________________________________________
    """
    
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
    input_dict = dict({ 'year':year, 'descript':descript , 'do_info':do_info, 'do_ie2n':False, 
                        'do_tarithm':do_tarithm, 'do_nan':False, 'do_ie2n':False,
                        'do_parallel':do_parallel, 'chunks':chunks, 
                        'do_compute':do_compute, 'do_load':do_load, 'do_persist':do_persist})
    
    #___________________________________________________________________________
    # add surface transformations 
    if ('srf' in which_transf or 'inner' in which_transf) or do_dflx: # add surface fluxes
        # compute combined density flux: heat_flux+freshwater_flux+restoring_flux
        if do_dflx: 
            data = load_data_fesom2(mesh, datapath, vname='std_heat_flux', **input_dict).persist()
            
            # Add subsequent variables one by one, freeing memory as we go
            for var in ['std_frwt_flux', 'std_rest_flux']:
                data['std_heat_flux'] += load_data_fesom2(mesh, datapath, vname=var, **input_dict)[var].persist()
                gc.collect()
            
            data_attrs = data_dMOC.attrs # rescue attributes will get lost during multipolication
            data       = data.rename({'std_heat_flux':'dmoc_fd'}).assign_coords({'ndens' :("ndens",std_dens)})
            data_dMOC  = xr.merge([data_dMOC, data], combine_attrs="no_conflicts")
            #gc.collect()
            del(data)
            
        # compute single flux from heat_flux & freshwater_flux &restoring_flux
        else:
            data = load_data_fesom2(mesh, datapath, vname='std_heat_flux', **input_dict).rename({'std_heat_flux':'dmoc_fh'}).persist()
            data_dMOC = xr.merge([data_dMOC, data], combine_attrs="no_conflicts") 
            gc.collect()
            del(data)
            data = load_data_fesom2(mesh, datapath, vname='std_frwt_flux', **input_dict).rename({'std_frwt_flux':'dmoc_fw'}).persist()
            data_dMOC = xr.merge([data_dMOC, data], combine_attrs="no_conflicts")   
            gc.collect()
            del (data)
            data = load_data_fesom2(mesh, datapath, vname='std_rest_flux', **input_dict).rename({'std_rest_flux':'dmoc_fr'}).persist()
            data_dMOC = xr.merge([data_dMOC, data], combine_attrs="no_conflicts")   
            gc.collect()
            del(data)
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
        data = load_data_fesom2(mesh, datapath, vname='std_dens_dVdT', **input_dict).rename({'std_heat_flux':'dmoc_dvdt'}).drop_vars('ndens').persit()
        data_dMOC = xr.merge([data_dMOC, data], combine_attrs="no_conflicts")
        gc.collect()
        del(data)
        
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
            data_h = load_data_fesom2(mesh, datapath, vname='std_dens_H', **input_dict).rename({'std_dens_H':'ndens_h'}).persist()
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
            data_z = load_data_fesom2(mesh, datapath, vname='std_dens_Z', **input_dict).rename({'std_dens_Z':'ndens_z'}).persist()
            data_z = data_z.assign_coords({'dens':dens})
            data_z = data_z.drop_vars(['ndens', 'elemi', 'lon']) #--> drop not needed variables
            data_dMOC = xr.merge([data_dMOC, data_z], combine_attrs="no_conflicts")
            del(data_z)
            
        #_______________________________________________________________________
        elif do_useZinfo=='density_dMOC' or do_useZinfo=='hydrography':
            # load sigma2 density on nodes 
            if do_useZinfo=='density_dMOC':
                data_sigma2 = load_data_fesom2(mesh, datapath, vname='density_dMOC', **input_dict).rename({'density_dMOC':'nz_rho'}).persist()
                
                # make land sea mask nan
                data_sigma2 = data_sigma2.where(data_sigma2!=0.0)
            
                # first put back here the land sea mask nan's than subtract ref 
                # denity --<> like that land remains Nan respective zero later
                data_sigma2 = data_sigma2 - 1000.00
            
            elif do_useZinfo=='hydrography':
                # load sigma2 density on nodes based on temperatur and salinity
                # hydrography
                data_sigma2 = load_data_fesom2(mesh, datapath, vname='sigma2', **input_dict).rename({'sigma2':'nz_rho'}).persist()
                
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
        data_div  = load_data_fesom2(mesh, datapath, vname='std_dens_DIV', 
                        **{**input_dict, 'chunks': {'nod2': -1, 'ndens': 1, 'time': 1}}).rename({'std_dens_DIV':'dmoc'}).persist()
        data_div  = data_div.drop_vars(['ndens', 'nodi', 'ispbnd']) 
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
            # array an save the attributes in the same time, I do this here in a little bit 
            # weird way to be efficient in terms of dask operation and not to run in issues with
            # dask task graph, 
            e_i = xr.DataArray(mesh.e_i[:,0], dims=['elem'])
            aux_dmoc_div =                data_div['dmoc'].isel(nod2=e_i)
            e_i = xr.DataArray(mesh.e_i[:,1], dims=['elem'])
            aux_dmoc_div = aux_dmoc_div + data_div['dmoc'].isel(nod2=e_i)
            e_i = xr.DataArray(mesh.e_i[:,2], dims=['elem'])
            aux_dmoc_div = aux_dmoc_div + data_div['dmoc'].isel(nod2=e_i)
            del(e_i)
            aux_dmoc_div = aux_dmoc_div/3.0
            warnings.filterwarnings("ignore", category=UserWarning, message="Sending large graph of size")
            warnings.filterwarnings("ignore", category=UserWarning, message="Large object of size \\d+\\.\\d+ detected in task graph")
            aux_dmoc_div = aux_dmoc_div.assign_attrs(data_div['dmoc'].attrs).persist()
            data_div = data_div.assign(dmoc=aux_dmoc_div).chunk({'elem':chunks['elem'] , 'ndens':-1})
            del(aux_dmoc_div)
            
            # data_div['dmoc']     = data_div['dmoc'].load()
            # if 'time' in list(data_div.dims):
            #     data_div = data_div.assign( dmoc=data_div['dmoc'][:, xr.DataArray(mesh.e_i, dims=["elem",'n3']),:].mean(dim="n3", keep_attrs=True) )
            # else:
            #     data_div = data_div.assign( dmoc=data_div['dmoc'][   xr.DataArray(mesh.e_i, dims=["elem",'n3']),:].mean(dim="n3", keep_attrs=True) )
            
            # if not data_dMOC.data_vars: data_div = data_div.chunk({'elem':'auto', 'ndens':'auto'})
            # else                      : data_div = data_div.chunk({'elem':data_dMOC.chunksizes['elem'], 'ndens':data_dMOC.chunksizes['ndens']})
            
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
            data_div_bolus  = load_data_fesom2(mesh, datapath, vname='std_dens_DIVbolus', 
                                    **{**input_dict, 'chunks': {'nod2': -1, 'ndens': 1, 'time': 1}}).rename({'std_dens_DIVbolus':'dmoc_bolus'}).persist()
            data_div_bolus  = data_div_bolus.drop_vars(['ndens', 'nodi', 'ispbnd']) 
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
                e_i = xr.DataArray(mesh.e_i[:,0], dims=['elem'])
                aux_dmoc_div =                data_div['dmoc_bolus'].isel(nod2=e_i)
                e_i = xr.DataArray(mesh.e_i[:,1], dims=['elem'])
                aux_dmoc_div = aux_dmoc_div + data_div['dmoc_bolus'].isel(nod2=e_i)
                e_i = xr.DataArray(mesh.e_i[:,2], dims=['elem'])
                aux_dmoc_div = aux_dmoc_div + data_div['dmoc_bolus'].isel(nod2=e_i)
                del(e_i)
                aux_dmoc_div = aux_dmoc_div/3.0
                warnings.filterwarnings("ignore", category=UserWarning, message="Sending large graph of size")
                warnings.filterwarnings("ignore", category=UserWarning, message="Large object of size \\d+\\.\\d+ detected in task graph")
                aux_dmoc_div = aux_dmoc_div.assign_attrs(data_div['dmoc_bolus'].attrs).persist()
                data_div_bolus = data_div_bolus.assign(dmoc_bolus=aux_dmoc_div).chunk({'elem':chunks['elem'] , 'ndens':-1})
                del(aux_dmoc_div)
                
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
    


# ___CALCULATE MERIDIONAL OVERTURNING IN DENSITY COORDINATES___________________
#| Global MOC, Atlantik MOC, Indo-Pacific MOC, Indo MOC                        |
#|                                                                             |
#|                                                                             |
#|_____________________________________________________________________________|
def calc_dmoc(mesh, 
              data_dMOC, 
              dlat              = 1.0       , 
              which_moc         = 'gmoc'    , 
              which_transf      = None      , 
              do_checkbasin     = False     ,
              exclude_meditoce  = False     , 
              do_bolus          = True      , 
              do_parallel       = False     , 
              n_workers         = 10        , 
              do_compute        = False     , 
              do_load           = True      , 
              do_persist        = False     , 
              do_info           = True      , 
              do_dropvar        = True      , 
              **kwargs):
    """
    --> calculate meridional overturning circulation from vertical velocities 
        (Pseudostreamfunction) either on vertices or elements
    
    Parameters:
    
        :mesh:          fesom2 tripyview mesh object,  with all mesh information 
        
        :data_dMOC:     xarray dataset object with 3d density class data
        
        :dlat:          float (default=1.0), latitudinal binning resolution
        
        :which_moc:     str, shp.Reader() (default='gmox') which global or regional 
                        MOC should be computed based on present day shapefiles. 
                        ·Options are:
                        
                        - 'gmoc'  ... compute global MOC
                        - 'amoc'  ... compute MOC for Atlantic Basin
                        - 'aamoc' ... compute MOC for Atlantic+Arctic Basin
                        - 'pmoc'  ... compute MOC for Pacific Basin
                        - 'ipmoc' ... compute MOC for Indo-Pacific Basin (PMOC how it should be)
                        - 'imoc'  ... compute MOC for Indian-Ocean Basin
                        - shp.Reader('path') ... compute MOC based on custom shapefile
                        
                        Important:
                        Between 'amoc' and 'aamoc' there is not much difference 
                        in variability, but upto 1.5Sv in amplitude. Where 'aamoc'
                        is stronger than 'amoc'. There is no clear rule which one 
                        is better, just be sure you are consistent       
                        
        :which_transf:  str (default='dmoc') which transformation should be computed
                        options area
                        
                        - 'dmoc'    compute dmoc density transformation
                        - 'srf'     compute density transform. from surface forcing 
                        - 'inner'   compute density transform. from interior mixing (dmoc-srf)
        
        
        :do_checkbasin: bool (default=False) provide plot with regional basin selection
        
        :exclude_meditoce: bool (default=False) exclude mediteranian sea from basin selection
        
        :do_bolus:      bool (default=False) load density class divergence from bolus velolcity
                        and add them to the total density class divergence
                        
        :do_dropvar:    bool (default=true) drop all variables from dataset that are not                
                        absolutely needed
                        
        :do_parallel:   bool (default=False) do computation of binning based MOC 
                        in parallel
        
        :n_workers:     int (default=10) how many worker (CPUs) are used for the 
                        parallelized MOC computation
                        
                        
        :do_compute:    bool (default=False), do xarray dataset compute() at the end
                        data = data.compute(), creates a new dataobject the original
                        data object seems to persist
        
        :do_load:       bool (default=True), do xarray dataset load() at the end
                        data = data.load(), applies all operations to the original
                        dataset
                        
        :do_persist:    bool (default=False), do xarray dataset persist() at the end
                        data = data.persist(), keeps the dataset as dask array, keeps
                        the chunking   
                        
        :do_info:       bool (defalt=True), print variable info at the end 
    
    Returns:
    
        :dmoc:          object, returns xarray dataset object with DMOC
        
    ::
    
        data_list = list()
        
        data = tpv.load_dmoc_data(mesh, datapath, std_dens, year=year, which_transf='dmoc', descript=descript,
                      do_zcoord=True, do_bolus=True, do_load=False, do_persist=True)
    
    
        dmoc     = tpv.calc_dmoc(mesh, data, dlat=1.0, which_moc=vname, which_transf='dmoc')
        
        data_list.append( dmoc )
    
    ____________________________________________________________________________
    """
    
    # rescue global dataset attributes
    gattr = data_dMOC.attrs
    
    #___________________________________________________________________________
    t1=clock.time()
    # In case MOC is defined via string
    if isinstance(which_moc, str):
        which_moc_name = which_moc
        if do_info==True: print('_____calc. '+which_moc.upper()+' from vertical velocities via meridional bins_____')
    
    # In case MOC is defined via  custom shapefile e.g for deep paleo time slices
    elif isinstance(which_moc, shp.Reader):
        # Extract the 'Name' attribute for each shape
        field_names = [field[0] for field in which_moc.fields[1:]]
        which_moc_name, which_moc_region = 'moc', ''
        # search for "Name" attribute in shapefile
        if "Name" in field_names:
            index = [field[0] for field in which_moc.fields[1:] ].index("Name")
            which_moc_name = [record[index] for record in which_moc.records()][0]
        # search for "Region" attribute in shapefile    
        if "Region" in field_names:
            index = [field[0] for field in which_moc.fields[1:] ].index("Region")
            which_moc_region = [record[index] for record in which_moc.records()][0]
            
        if do_info==True: print('_____calc. '+which_moc_name.upper()+' from vertical velocities via meridional bins_____')
        
    #___________________________________________________________________________
    # compute/use index for basin domain limitation
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
    lat     = np.arange(lat_bin.data.min(), lat_bin.data.max()+dlat, dlat)
    
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
        strmoc = 'd'+which_moc_name.upper()
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



# ___CALCULATE MERIDIONAL OVERTURNING IN DENSITY COORDINATES___________________
#| Global MOC, Atlantik MOC, Indo-Pacific MOC, Indo MOC                        |
#|                                                                             |
#|                                                                             |
#|_____________________________________________________________________________|
def calc_dmoc_dask( mesh                          , 
                    data                          , 
                    do_parallel                   , 
                    parallel_nprc                 ,
                    dlat              = 1.0       , 
                    which_moc         = 'gmoc'    , 
                    which_transf      = None      , 
                    do_checkbasin     = False     ,
                    do_exclude        = False     , 
                    exclude_list      = list(['ocean_basins/Mediterranean_Basin.shp', [26,42,39.5,47]])   ,
                    do_bolus          = True      , 
                    do_info           = True      , 
                    do_dropvar        = True      , 
                    do_botmax_z       = True      ,
                    do_botmax_dens    = True      ,
                    **kwargs):
    """
    --> calculate meridional overturning circulation from vertical velocities 
        (Pseudostreamfunction) either on vertices or elements
    
    Parameters:
    
        :mesh:          fesom2 tripyview mesh object,  with all mesh information 
        
        :data:     xarray dataset object with 3d density class data
        
        :do_parallel:   bool, (default=False) is a dask client running
        
        :parallel_nprc: int, (default=64), number of parallel processes
        
        :dlat:          float (default=1.0), latitudinal binning resolution
        
        :which_moc:     str, shp.Reader() (default='gmox') which global or regional 
                        MOC should be computed based on present day shapefiles. 
                        ·Options are:
                        
                        - 'gmoc'  ... compute global MOC
                        - 'amoc'  ... compute MOC for Atlantic Basin
                        - 'aamoc' ... compute MOC for Atlantic+Arctic Basin
                        - 'pmoc'  ... compute MOC for Pacific Basin
                        - 'ipmoc' ... compute MOC for Indo-Pacific Basin (PMOC how it should be)
                        - 'imoc'  ... compute MOC for Indian-Ocean Basin
                        - shp.Reader('path') ... compute MOC based on custom shapefile
                        
                        Important:
                        Between 'amoc' and 'aamoc' there is not much difference 
                        in variability, but upto 1.5Sv in amplitude. Where 'aamoc'
                        is stronger than 'amoc'. There is no clear rule which one 
                        is better, just be sure you are consistent       
                        
        :which_transf:  str (default='dmoc') which transformation should be computed
                        options area
                        
                        - 'dmoc'    compute dmoc density transformation
                        - 'srf'     compute density transform. from surface forcing 
                        - 'inner'   compute density transform. from interior mixing (dmoc-srf)
        
        
        :do_checkbasin: bool (default=False) provide plot with regional basin selection
        
        :exclude_meditoce: bool (default=False) exclude mediteranian sea from basin selection
        
        :do_bolus:      bool (default=False) load density class divergence from bolus velolcity
                        and add them to the total density class divergence
                        
        :do_dropvar:    bool (default=true) drop all variables from dataset that are not                
                        absolutely needed
                        
        :do_info:       bool (defalt=True), print variable info at the end 
    
    Returns:
    
        :dmoc:          object, returns xarray dataset object with DMOC
        
    ::
    
        data_list = list()
        
        data = tpv.load_dmoc_data(mesh, datapath, std_dens, year=year, which_transf='dmoc', descript=descript,
                      do_zcoord=True, do_bolus=True, do_load=False, do_persist=True)
    
    
        dmoc     = tpv.calc_dmoc(mesh, data, dlat=1.0, which_moc=vname, which_transf='dmoc')
        
        data_list.append( dmoc )
    
    ____________________________________________________________________________
    """
    
    # rescue global dataset attributes
    gattr = data.attrs
    vname_list = list(data.data_vars)
    dimn_h, dimn_v, dimn_t = 'None', 'None', 'None'
    if   ('nod2'  in data.dims): dimn_h = 'nod2'
    elif ('elem'  in data.dims): dimn_h = 'elem'
    if   ('nz'    in data.dims): dimn_v = 'nz'    
    elif ('nz1'   in data.dims): dimn_v = 'nz1'
    elif ('nz_1'  in data.dims): dimn_v = 'nz_1'
    elif ('ndens' in data.dims): dimn_v = 'ndens'
    elif ('time'  in data.dims): dimn_t = 'time'
    
    #___________________________________________________________________________
    t1=clock.time()
    # In case MOC is defined via string
    if isinstance(which_moc, str):
        which_moc_name = which_moc
        if do_info==True: print('_____calc. '+which_moc.upper()+' from vertical velocities via meridional bins_____')
    
    # In case MOC is defined via  custom shapefile e.g for deep paleo time slices
    elif isinstance(which_moc, shp.Reader):
        # Extract the 'Name' attribute for each shape
        field_names = [field[0] for field in which_moc.fields[1:]]
        which_moc_name, which_moc_region = 'moc', ''
        # search for "Name" attribute in shapefile
        if "Name" in field_names:
            index = [field[0] for field in which_moc.fields[1:] ].index("Name")
            which_moc_name = [record[index] for record in which_moc.records()][0]
        # search for "Region" attribute in shapefile    
        if "Region" in field_names:
            index = [field[0] for field in which_moc.fields[1:] ].index("Region")
            which_moc_region = [record[index] for record in which_moc.records()][0]
            
        if do_info==True: print('_____calc. '+which_moc_name.upper()+' from vertical velocities via meridional bins_____')
        
    #___________________________________________________________________________
    # compute/use index for basin domain limitation
    idxin = calc_basindomain_fast(mesh, 
                                  which_moc    = which_moc, 
                                  do_onelem    = True, 
                                  do_exclude   = do_exclude,
                                  exclude_list = exclude_list)

    # reduce to dMOC data to basin domain
    data  = data.isel({dimn_h:idxin})

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
    # determine/adapt actual chunksize
    nchunk = 1
    if do_parallel and isinstance(data[vname_list[0]].data, da.Array)==True :
            
        nchunk = len(data.chunks[dimn_h])
            
        # after all the time and depth operation after the loading there will 
        # be worker who have no chunk piece to work on  --> therfore we need
        # to rechunk make sure the workload is distributed between all 
        # availabel worker equally         
        if nchunk<parallel_nprc*0.75:
            print(f' --> rechunk: {nchunk}', end='')
            if 'time' in data.dims:
                data = data.chunk({dimn_h: np.ceil(data.dims[dimn_h]/(parallel_nprc)).astype('int'), dimn_v:-1, 'time':-1})
            else:
                data = data.chunk({dimn_h: np.ceil(data.dims[dimn_h]/(parallel_nprc)).astype('int'), dimn_v:-1})
            nchunk = len(data.chunks[dimn_h])
            print(f' -> {nchunk}', end='')    
            if 'time' not in data.dims: print('')
            
    #___________________________________________________________________________
    # create meridional bins
    lat_min    = np.floor(data['lat'].min().compute())
    lat_max    = np.ceil( data['lat'].max().compute())
    lat_bins   = np.arange(lat_min, lat_max+dlat*0.5, dlat)
    lat        = (lat_bins[1:]+lat_bins[:-1])*0.5
    nlat, nlev = len(lat_bins)-1, data.dims['ndens']
    
    #___________________________________________________________________________
    # prepare  weights for area weighted mean over the elements for all density 
    # classes and depth levels (in case nz_rho is loaded)
    if 'ndens_h' in vname_list or 'ndens_z' in vname_list:
        # expand by ndens dimension --> need this here to get proper area weighting 
        # mean over the bottom topography!!!
        data['ndens_w_A'] = data['w_A'].broadcast_like(data['ndens_h'])
        
        # non-existing density classes (ndens_h==0) --> NaN
        data['ndens_w_A'] = data['ndens_w_A'].where(data['ndens_h'] > 0, 0.0)
        data = data.drop_vars(['w_A'])
    
    ##___________________________________________________________________________
    #if 'elem_pbnd' not in data.coords: 
            #data = data.assign_coords(elem_pbnd=xr.DataArray(np.zeros(data['lat'].shape, dtype=bool), dims=data['lat'].dims))
            #if isinstance(data['lat'].data, da.Array)==True: 
                #data['elem_pbnd'] = data['elem_pbnd'].chunk(data['lat'].chunks)
                
    #___________________________________________________________________________
    lat_chnksize  = data['lat'].chunksizes
    
    # prepare chunked input to routine that should act on chunks
    if 'time' in data.dims:
        drop_axis, ntime = [0,1], data.dims['time']
        chnk_lat     = data['lat'].data[None, None, :]
        chnk_ispbnd  = data['ispbnd'].data[None, None, :]
        #chnk_pbnd = data['elem_pbnd'].data[None, :, None]
    else:
        drop_axis, ntime = [0], 1
        chnk_lat     = data['lat'].data[None, :]
        chnk_ispbnd  = data['ispbnd'].data[None, :]
        #chnk_pbnd = data['elem_pbnd'].data[      :, None]
        
    chnk_wA, chnk_h = None, None    
    chnk_dmoc, chnk_dmoc_bolus, chnk_fh, chnk_fw, chnk_fr, chnk_fd, chnk_dvdt= None, None, None, None, None, None, None
    nvar=0
    if 'dmoc'       in vname_list: chnk_dmoc      , nvar = data['dmoc'      ].data, nvar+1
    if 'dmoc_bolus' in vname_list: chnk_dmoc_bolus, nvar = data['dmoc_bolus'].data, nvar+1
    if 'dmoc_fh'    in vname_list: chnk_fh        , nvar = data['dmoc_fh'   ].data, nvar+1
    if 'dmoc_fw'    in vname_list: chnk_fw        , nvar = data['dmoc_fw'   ].data, nvar+1
    if 'dmoc_fr'    in vname_list: chnk_fr        , nvar = data['dmoc_fr'   ].data, nvar+1
    if 'dmoc_fd'    in vname_list: chnk_fd        , nvar = data['dmoc_fd'   ].data, nvar+1
    if 'dmoc_dvdt'  in vname_list: chnk_dvdt      , nvar = data['dmoc_dvdt' ].data, nvar+1
    
    if 'ndens_h'    in vname_list: chnk_h         , nvar = data['ndens_h'   ].data, nvar+1
    if 'ndens_h'    in vname_list: chnk_wA        , nvar = data['ndens_w_A' ].data, nvar+1
    
    dmoc = da.map_blocks(calc_dmoc_chnk             , # function that act chunks
                         lat_bins                   , # mean bin definitions
                         chnk_lat                   , # lat nod2 coordinates
                         chnk_wA                    , # area weight
                         chnk_ispbnd                , # area weight
                         nvar                       , # number of input/output variables
                         chnk_h                     , # density class thickness
                         chnk_dmoc                  , # density class divergence
                         chnk_dmoc_bolus            , # density class divergence bolus
                         chnk_fh                    , # transf. by heatflux
                         chnk_fw                    , # transf. by freahwaterflx
                         chnk_fr                    , # transf. by radiationflx
                         chnk_fd                    , # transf. by total flx
                         chnk_dvdt                  ,   
                         dtype     = np.float32     , # Tuple dtype
                         drop_axis = drop_axis      , # drop dim nz1
                         chunks    = (nvar*ntime*nlev*nlat,) # Output shape
                        )
    
    
    #___________________________________________________________________________
    # reshape axis over chunks 
    if 'time' in data.dims: dmoc = dmoc.reshape((nchunk, nvar, ntime, nlev, nlat))
    else                  : dmoc = dmoc.reshape((nchunk, nvar,        nlev, nlat))
    
    #___________________________________________________________________________
    # do dask axis reduction across chunks dimension
    dmoc = da.reduction(dmoc,                   
                        chunk     = lambda x, axis=None, keepdims=None: x,  # this is a do nothing routine that acts within the chunks
                        aggregate = np.sum, # that the sum that goes over the chunk
                        dtype     = np.float32,
                        axis      = 0,
                       ).compute()
    del(chnk_wA, chnk_h, chnk_dmoc, chnk_dmoc_bolus, chnk_fh, chnk_fw, chnk_fr, chnk_fd, chnk_dvdt)
    
    #___________________________________________________________________________
    # do area weighted mean density class thickness
    if 'ndens_h'    in vname_list: 
        with np.errstate(divide='ignore', invalid='ignore'):
            dmoc[-2] = np.where(dmoc[-1]>0, dmoc[-2]/dmoc[-1], np.nan)  

    #___________________________________________________________________________
    # write variables into dataset
    inSv = 1e-6
    data_vars, coords, dim_name, dim_size,  = dict(), dict(), list(), list()
    if 'time' in list(data.dims): dim_name.append('time')
    if 'ndens'in list(data.dims): dim_name.append('ndens')
    dim_name.append('lat')
    
    for dim_ni in dim_name:
        if   dim_ni=='time' : dim_size.append(data.sizes['time'] ); coords['time' ]=(['time' ], data['time' ].data ) 
        elif dim_ni=='lat'  : dim_size.append(lat.size           ); coords['lat'  ]=(['lat'  ], lat.astype('float32')          ) 
        elif dim_ni=='ndens' : dim_size.append(data.sizes['ndens']); coords['dens' ]=(['ndens' ], data['dens'].values.astype('float32') )
        
    nvar=0
    if 'dmoc'       in vname_list: data_vars['dmoc'      ], nvar = (dim_name, dmoc[nvar]*inSv) , nvar+1
    if 'dmoc_bolus' in vname_list: data_vars['dmoc_bolus'], nvar = (dim_name, dmoc[nvar]*inSv) , nvar+1
    if 'dmoc_fh'    in vname_list: data_vars['dmoc_fh'   ], nvar = (dim_name, dmoc[nvar]*inSv) , nvar+1
    if 'dmoc_fw'    in vname_list: data_vars['dmoc_fw'   ], nvar = (dim_name, dmoc[nvar]*inSv) , nvar+1
    if 'dmoc_fr'    in vname_list: data_vars['dmoc_fr'   ], nvar = (dim_name, dmoc[nvar]*inSv) , nvar+1
    if 'dmoc_fd'    in vname_list: data_vars['dmoc_fd'   ], nvar = (dim_name, dmoc[nvar]*inSv) , nvar+1
    if 'dmoc_dvdt'  in vname_list: data_vars['dmoc_dvdt' ], nvar = (dim_name, dmoc[nvar]*inSv) , nvar+1
    if 'ndens_h'    in vname_list: data_vars['ndens_h'   ], nvar = (dim_name, dmoc[nvar]     ) , nvar+1
    dmoc = xr.Dataset(data_vars=data_vars, coords=coords, attrs=data.attrs)
   
    #___________________________________________________________________________
    # write attributes to dataset
    for vari in vname_list:
        #_______________________________________________________________________
        # add more variable attributes
        strmoc = 'd'+which_moc_name.upper()
        vattr = data[vari].attrs
        if   vari=='dmoc_fh'   : vattr.update({'long_name':'Transformation from heat flux'                 , 'short_name':strmoc+'_fh'   , 'units':'Sv'   })
        elif vari=='dmoc_fw'   : vattr.update({'long_name':'Transformation from freshwater flux'           , 'short_name':strmoc+'_fw'   , 'units':'Sv'   })
        elif vari=='dmoc_fr'   : vattr.update({'long_name':'Transformation from surface salinity restoring', 'short_name':strmoc+'_fr'   , 'units':'Sv'   })
        elif vari=='dmoc_fd'   : vattr.update({'long_name':'Transformation from total density flu'         , 'short_name':strmoc+'_fd'   , 'units':'Sv'   })
        elif vari=='dmoc_dvdt' : vattr.update({'long_name':'Transformation from volume change'             , 'short_name':strmoc+'_dv'   , 'units':'Sv'   })
        elif vari=='dmoc'      : vattr.update({'long_name':'Density MOC'                                   , 'short_name':strmoc         , 'units':'Sv'   })
        elif vari=='dmoc_bolus': vattr.update({'long_name':'Density MOC bolus vel.'                        , 'short_name':strmoc+'_bolus', 'units':'Sv'   })
        elif vari=='ndens_h'   : vattr.update({'long_name':'Density class thickness'                       , 'short_name':'ndens_h'      , 'units':'m'    })
        dmoc[vari]=dmoc[vari].assign_attrs(vattr)
    
    chnk_lat = data['lat'] # --> need this later for the bottom max patch computation
    del(data)
    
    #___________________________________________________________________________
    # exclude variables that should not be cumulatively integrated --> than do
    # cumulative sumation over lat
    var_list = list(dmoc.keys())
    if 'ndens_h'   in var_list: var_list.remove('ndens_h')
    
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
    if 'dmoc'      in list(dmoc.data_vars):
        dmoc[ 'dmoc'       ] = dmoc[ 'dmoc' ].isel(ndens=reverse).cumsum(dim='ndens', skipna=True).isel(ndens=reverse)
    
    if 'dmoc_bolus'in list(dmoc.data_vars):
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
    if 'dmoc' in list(dmoc.data_vars) and 'dmoc_bolus' in list(dmoc.data_vars) and do_bolus:  
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
    # compute depth of max topography based on zcoord
    if do_botmax_z:
        botmax = xr.DataArray(np.abs(mesh.zlev[mesh.e_iz-1]), dims='elem').isel({'elem':idxin}).chunk({'elem':lat_chnksize['elem']})
        
        # collect chunk pieces
        #botmax = da.map_blocks(bottommax_latbin_chnk, lat_bins, data['lat'].data, botmax.data, 
        botmax = da.map_blocks(bottommax_latbin_chnk, lat_bins, chnk_lat.data, botmax.data, 
                            dtype=np.float32, chunks=(nlat,) ).reshape((nchunk, nlat))
        
        # do dask axis reduction across chunks dimension
        botmax = da.reduction(botmax,                   
                            chunk     = lambda x, axis=None, keepdims=None: x,  # this is a do nothing routine that acts within the chunks
                            aggregate = np.max, # that the sum that goes over the chunk
                            dtype     = np.float32,
                            axis      = 0,
                            ).compute()
        
        # slightly smooth bottom batch
        filt   = np.array([1,3,1])
        botmax = np.concatenate( (np.ones((filt.size,))*botmax[0], botmax, np.ones((filt.size,))*botmax[-1] ) )
        botmax = np.convolve(botmax, filt/np.sum(filt), mode='same')[filt.size:-filt.size]
        dmoc   = dmoc.assign_coords(botmax=xr.DataArray(botmax, dims='nlat').astype('float32'))
    
    # compute depth of max topography based on density classes
    if do_botmax_dens and 'ndens_zfh' in dmoc.coords:
        filt   = np.array([1,3,1])
        botmax_dens=dmoc['ndens_zfh'].max('ndens').data
        botmax_dens = np.concatenate( (np.ones((filt.size,))*botmax_dens[0], botmax_dens, np.ones((filt.size,))*botmax_dens[-1] ) )
        botmax_dens = np.convolve(botmax_dens, filt/np.sum(filt), mode='same')[filt.size:-filt.size]
        dmoc   = dmoc.assign_coords(botmax_dens=botmax_dens.astype('float32'))
    
    #___________________________________________________________________________
    return(dmoc)



#
#
#_______________________________________________________________________________  
def calc_dmoc_chnk(lat_bins, chnk_lat, chnk_wA, chnk_ispbnd, 
                   nvar            , # number of input/output variables
                   chnk_h          , # density class thickness
                   chnk_d          , # density class divergence
                   chnk_d_bolus    , # density class divergence bolus
                   chnk_fh         , # transf. by heatflux
                   chnk_fw         , # transf. by freahwaterflx
                   chnk_fr         , # transf. by radiationflx
                   chnk_fd         , # transf. by total flx
                   chnk_dvdt
                   ):
    """

    """
    #n11 = chnk_lat.shape
    #n21 = chnk_wA.shape
    #n31 = chnk_d.shape
    
    #___________________________________________________________________________
    # create chunked variable do this avoud if condtion within the for loop
    chnk_var_list = list()
    if chnk_d       is not None: chnk_var_list.append(chnk_d)
    if chnk_d_bolus is not None: chnk_var_list.append(chnk_d_bolus)
    if chnk_fh      is not None: chnk_var_list.append(chnk_fh)
    if chnk_fw      is not None: chnk_var_list.append(chnk_fw)
    if chnk_fr      is not None: chnk_var_list.append(chnk_fr)
    if chnk_fd      is not None: chnk_var_list.append(chnk_fd)
    if chnk_dvdt    is not None: chnk_var_list.append(chnk_dvdt)
    
    # only need the additional dimension at the point where the function is initialised
    if   np.ndim(chnk_var_list[0]) == 2: 
        chnk_lat    = chnk_lat[0, :] # 2D --> now is 1D again
        chnk_ispbnd = chnk_ispbnd[0, :] # 2D --> now is 1D again
    elif np.ndim(chnk_var_list[0]) == 3:
        chnk_lat    = chnk_lat[0, 0, :] # 3D --> now is 1D again
        chnk_ispbnd = chnk_ispbnd[0, 0, :] # 3D --> now is 1D again
        if chnk_h is not None: chnk_wA  = chnk_wA[ 0, :, :] # 3D --> now is 2D again
    
    # Use np.digitize to find bin indices for longitudes and latitudes
    idx_lat = np.digitize(chnk_lat, lat_bins)-1  # Adjust to get 0-based index
    nlat    = len(lat_bins)-1
    
    # Initialize binned data storage 
    if   np.ndim(chnk_var_list[0]) == 3: 
        # Replace NaNs with 0 value to summation issues
        ntime, nlev, nnod = chnk_var_list[0].shape
        binned_d    = np.zeros((nvar, ntime, nlev, nlat), dtype=np.float32)
        # binned_d[ 0,...] - data
        # binned_d[-2,...] - area weight sum
        # binned_d[-1,...] - area weight sum
        
    elif np.ndim(chnk_var_list[0]) == 2: 
        # Replace NaNs with 0 value to summation issues
        nlev, nnod  = chnk_var_list[0].shape
        binned_d    = np.zeros((nvar, nlev, nlat), dtype=np.float32)  
        # binned_d[ 0,...] - data
        # binned_d[-2,...] - class thickness * area weight
        # binned_d[-1,...] - area weight sum
        
    # Precompute mask outside the loop
    idx_valid = (idx_lat >= 0) & (idx_lat < nlat) & ~chnk_ispbnd
    del(chnk_ispbnd)

    # Apply mask before looping
    idx_lat   = idx_lat[idx_valid   ]
    if chnk_h is not None:  chnk_wA   = chnk_wA[:, idx_valid]
    nnod      = len(idx_lat)
    
    # Sum data based on binned indices with time dimension: [2, ntime, nlat, nlev]
    if   np.ndim(chnk_var_list[0]) == 3: 
        if chnk_h is not None: chnk_h = chnk_h[:, :, idx_valid]
        for ii, chnk_var in enumerate(chnk_var_list): chnk_var_list[ii] = chnk_var[:, :, idx_valid]
        for nod_i in range(0,nnod):
            jj = idx_lat[nod_i]
            for ii, chnk_var in enumerate(chnk_var_list):
                binned_d[ii, :, :, jj] = binned_d[ii, :, :, jj] + chnk_var[:, :, nod_i]
            
            if chnk_h is not None:    
                binned_d[-2, :, :, jj] = binned_d[-2, :, :, jj] + chnk_h[  :, :, nod_i]*chnk_wA[:, nod_i]    
                binned_d[-1, :, :, jj] = binned_d[-1, :, :, jj] + chnk_wA[    :, nod_i]
    
    # Sum data based on binned indices withou time dimension: [2, nlat, nlev]
    elif np.ndim(chnk_var_list[0]) == 2:  
        if chnk_h is not None: chnk_h = chnk_h[:, idx_valid]
        for ii, chnk_var in enumerate(chnk_var_list): chnk_var_list[ii] = chnk_var[:, idx_valid]
        for nod_i in range(0,nnod):
            jj = idx_lat[nod_i]
            for ii, chnk_var in enumerate(chnk_var_list):
                binned_d[ii, :, jj] = binned_d[ii, :, jj] + chnk_var[:, nod_i]
                
            if chnk_h is not None:        
                binned_d[-2, :, jj] = binned_d[-2, :, jj] + chnk_h[ :, nod_i]*chnk_wA[:, nod_i]
                binned_d[-1, :, jj] = binned_d[-1, :, jj] + chnk_wA[:, nod_i]
    
    #___________________________________________________________________________
    return binned_d.flatten()



#
#
#_______________________________________________________________________________
# compute maximum bottom topography within each bin
def bottommax_latbin_chnk(lat_bins, chnk_lat, chnk_d):
    #if np.ndim(chnk_lat)==3: chnk_lat = chnk_lat[0,:,0]
    #if np.ndim(chnk_lat)==2: chnk_lat = chnk_lat[:,0]
    idx_lat   = np.digitize(chnk_lat, lat_bins)-1  # Adjust to get 0-based index
    nlat      = len(lat_bins)-1
    binned_d  = np.zeros((nlat, ), dtype=np.float32)
    idx_valid = (idx_lat >= 0) & (idx_lat < nlat)
    idx_lat   = idx_lat[idx_valid]
    chnk_d    = chnk_d[idx_valid]
    for nod_i in range(0,len(idx_lat)):
        jj = idx_lat[nod_i]
        binned_d[jj] = np.maximum(binned_d[jj], chnk_d[nod_i])
    #___________________________________________________________________________
    return(binned_d.flatten())    
    
    

#_______________________________________________________________________________     
# do creepy brute force play around to enforce more or less monotonicity in 
# dens_z, std_dens_Z --> not realy recommendet to do only as a last option
def do_ztransform(data):
    """
    --> 
    ____________________________________________________________________________
    """
    
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
    """
    --> 
    ____________________________________________________________________________
    """
    
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
#_______________________________________________________________________________
def do_ztransform_mom6(mesh, data):
    """
    --> 
    ____________________________________________________________________________
    """
    
    from scipy.interpolate import interp1d

    #___________________________________________________________________________
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
    """
    --> 
    ____________________________________________________________________________
    """
    
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
