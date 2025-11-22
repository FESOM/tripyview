# Patrick Scholz, 23.01.2018
import numpy as np
import time  as clock
import os
import warnings
import xarray as xr
import netCDF4 as nc
#import seawater as sw
import gsw as gsw
from .sub_mesh import *
import warnings
import dask.array as da
from dask.array import broadcast_arrays
import gc


#xr.set_options(enable_cftimeindex=False)
# ___LOAD FESOM2 DATA INTO XARRAY DATASET CLASS________________________________
#|                                                                             |
#|           *** LOAD FESOM2 DATA INTO --> XARRAY DATASET CLASS ***            |
#|                                                                             |
#|_____________________________________________________________________________|
def load_data_fesom2(mesh, 
                     datapath, 
                     vname          = None      ,
                     year           = None      ,
                     mon            = None      ,
                     day            = None      ,
                     record         = None      ,
                     depth          = None      ,
                     depidx         = False     ,
                     do_tarithm     = 'mean'    ,
                     do_zarithm     = 'mean'    ,
                     do_zweight     = False     ,
                     do_hweight     = True      ,
                     do_nan         = True      ,
                     do_ie2n        = True      ,
                     do_rot         = True      ,
                     do_filename    = False     ,
                     do_file        = 'run'     ,
                     descript       = ''        ,
                     runid          = 'fesom'   ,
                     do_prec        = 'float32' ,
                     do_f14cmip6    = False     ,
                     do_multiio     = False     ,
                     do_compute     = False     ,
                     do_load        = True      ,
                     do_persist     = False     ,
                     do_parallel    = False     ,
                     chunks         = { 'time' :'auto', 'elem':'auto', 'nod2'  :'auto', \
                                        'edg_n':'auto', 'nz'  :'auto', 'nz1'   :'auto', \
                                        'ndens':'auto', 'x'   :'auto', 'ncells':'auto', \
                                        'node' :'auto'},
                     do_showtime    = False     ,
                     do_info        = True      ,
                     client         = None      , 
                     engine         = 'netcdf4' , # 'h5netcdf'
                     diagpath       = None      ,
                     **kwargs):
    """
    --> general loading of fesom2 and fesom14cmip6 data
    
    Parameters:
    
        :mesh:          fesom2 tripyview mesh object,  with all mesh information 
        
        :datapath:      str, path that leads to the FESOM2 data
        
        :vname:         str, (default: None), variable name that should be loaded
        
        :year:          int, list, np.array, range, default: None, single year or 
                        list/array of years whos file should be opened
        
        :mon:           list, (default=None), specific month that should be selected 
                        from dataset. If mon selection leads to no data selection, 
                        because data maybe annual, selection is set to mon=None
        
        :day:           list, (default=None), same as mon but for day
        
        :record:        int,list, (default=None), load specific record number from 
                        dataset. Overwrites effect of mon and sel_day
        
        :depth:         int, list, np.array, range (default=None). Select single 
                        depth level that will be interpolated or select list of depth 
                        levels that will be interpolated and vertically averaged. If 
                        None all vertical layers in data are loaded
        
        :depidx:        bool, (default:False) if depth is int and depidx=True, depth
                        index is selected that is closest to depth. No interpolation 
                        will be done
        
        :do_tarithm:    str (default='mean') do time arithmetic on time selection
                        option are: None, 'None', 'mean', 'median', 'std', 'var', 'max'
                        'min', 'sum'
        
        :do_zarithm:    str (default='mean') do arithmetic on selected vertical layers
                        options are: None, 'None', 'mean', 'max', 'min', 'sum'
        
        :do_zweight:    bool, (defaull=False) store weights for vertical weigthed 
                        averages within the dataset
        
        :do_hweight:    bool, (default=True), store weightsd for weighted horizontal
                        averages
        
        :do_nan:        bool (default=True), do replace bottom fill values with nan                
        
        :do_ie2n:       bool (default=True), if data are on elements automatically 
                        interpolates them to vertices --> easier to plot 
        
        :do_rot:        bool (default=True), if vector data are loaded e.g. 
                        vname='vec+u+v' rotates the from rotated frame (in which 
                        they are stored) to geo coordinates
        
        :do_filename:   str, (default=None) load this specific filname string instead
                        of path selection via datapath and year
        
        :do_file:       str, (default='run'), which data should be loaded options are
        
                        - 'run' ... fesom2 simulation files should be load, 
                        - 'restart_oce' ... fesom2 ocean restart file should be loaded, 
                        - 'restart_ice' ... fesom2 ice restart file should be loaded 
                        - 'blowup'      ... fesom2 ocean blowup file will be loaded
        
        :descript:      str (default=''), string to describe dataset is written into 
                        variable attributes
        
        :runid:         str (default='fesom'), runid of loaded data        
        
        :do_prec:       str, (default='float32') which precision is used for the 
                        loading of the data
                        
        :do_f14cmip6:   bool, (default=False), Set to true when loading cmorized 
                        FESOM1.4 CMIP6 data when computing AMOC
                        
        :do_multiio:    bool, (default=False), Set to true when loading FESOM2 
                        data processed with MULTIIO when computing AMOC
        
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
                        
        :do_showtime:   bool, (default=False) show time information stored in dataset
        
        :do_info:       bool (defalt=True), print variable info at the end 
        
        :client:        dask client object (default=None)
        
        :diagpath:      str (default=None) if str give custom path to specific fesom2
                        fesom.mesh.diag.nc file, if None routine looks automatically in    
                        meshfolder and original datapath folder (stored as attribute in)
                        xarray dataset object 
        
    Returns:
    
        :data:          object, returns xarray dataset object
        
    ____________________________________________________________________________
    """
    #___________________________________________________________________________
    # default values 
    is_data = 'scalar'
    is_ie2n = False
    do_vec  = False
    do_sclrv= None # do scalar velocity compononent
    do_norm = False
    do_gradx= False
    do_grady= False
    do_pdens= False
    str_adep, str_atim = '', '' # string for arithmetic
    str_ldep, str_ltim = '', '' # string for labels
    str_lsave = ''    
    xr.set_options(keep_attrs=True)
    
    #___________________________________________________________________________
    # Related to bug especially on albedo netcdf-c not being threat save since netcdf1.6.1: 
    # https://github.com/pydata/xarray/issues/7079
    # https://github.com/xCDAT/xcdat/issues/561
    # https://forum.access-hive.org.au/t/netcdf-not-a-valid-id-errors/389/24
    import dask
    import distributed
    if distributed.worker_client() is None:
        dask.config.set(scheduler="single-threaded")

    #___________________________________________________________________________
    # Create xarray dataset object with all grid information 
    if vname in ['topography','zcoord', 
                 'narea', 'n_area', 'clusterarea', 'scalararea', 
                 'earea', 'e_area', 'triarea',
                 'nresol', 'n_resol', 'resolution', 'resol', 
                 'eresol','e_resol','triresolution','triresol',
                 'edepth','etopo','e_depth','e_topo',
                 'ndepth', 'ntopo', 'n_depth', 'n_topo', ]:
        data = xr.Dataset()                        
        #___________________________________________________________________________
        # store topography in data
        if   any(x in vname for x in ['ndepth', 'ntopo', 'n_depth', 'n_topo', 'topography', 'zcoord']):
            data['ntopo'] = ("nod2", np.abs(mesh.n_z))
            data['ntopo'].attrs["description"]='Depth'
            data['ntopo'].attrs["descript"   ]='Depth'
            data['ntopo'].attrs["long_name"  ]='Depth'
            data['ntopo'].attrs["units"      ]='m'
            data['ntopo'].attrs["is_data"    ]=is_data
            data = data.chunk(chunks['nod2'])
        
        elif any(x in vname for x in ['edepth', 'etopo', 'e_depth', 'e_topo' ]):
            data['etopo'] = ("elem", np.abs(mesh.zlev[mesh.e_iz]))
            data['etopo'].attrs["description"]='Depth'
            data['etopo'].attrs["descript"   ]='Depth'
            data['etopo'].attrs["long_name"  ]='Depth'
            data['etopo'].attrs["units"      ]='m'
            data['etopo'].attrs["is_data"    ]=is_data
            data = data.chunk(chunks['elem'])
            data, dim_vert, dim_horz = do_gridinfo_and_weights(mesh,data,do_zweight=do_zarithm)
        
        # store vertice cluster area in data    
        elif any(x in vname for x in ['narea', 'n_area', 'clusterarea', 'scalararea']):
            if len(mesh.n_area)==0: mesh=mesh.compute_n_area()
            data['narea'] = ("nod2", mesh.n_area[0,:])
            data['narea'].attrs["description"]='Vertice area'
            data['narea'].attrs["descript"   ]='Vertice area'
            data['narea'].attrs["long_name"  ]='Vertice area'
            data['narea'].attrs["units"      ]='m^2'
            data['narea'].attrs["is_data"    ]=is_data
            data = data.chunk(chunks['nod2'])
            data, dim_vert, dim_horz = do_gridinfo_and_weights(mesh,data,do_zweight=do_zarithm)
        
        # store vertice resolution in data               
        elif any(x in vname for x in ['nresol', 'n_resol', 'resolution', 'resol']):
            if len(mesh.n_resol)==0: mesh=mesh.compute_n_resol()
            data['nresol'] = ("nod2", mesh.n_resol/1000)
            data['nresol'].attrs["description"]='Resolution'
            data['nresol'].attrs["descript"   ]='Resolution'
            data['nresol'].attrs["long_name"  ]='Resolution'
            data['nresol'].attrs["units"      ]='km'
            data['nresol'].attrs["is_data"    ]=is_data
            data = data.chunk(chunks['nod2'])
            data, dim_vert, dim_horz = do_gridinfo_and_weights(mesh,data,do_zweight=do_zarithm)
        
        # store element area in data    
        elif any(x in vname for x in ['earea', 'e_area', 'triarea']):
            if len(mesh.e_area)==0: mesh=mesh.compute_e_area()
            data['earea'] = ("elem", mesh.e_area)
            data['earea'].attrs["description"]='Element area'
            data['earea'].attrs["descript"   ]='Element area'
            data['earea'].attrs["long_name"  ]='Element area'
            data['earea'].attrs["units"      ]='m^2'
            data['earea'].attrs["is_data"    ]=is_data
            data = data.chunk(chunks['elem'])
           
        # store element resolution in data               
        elif any(x in vname for x in ['eresol', 'e_resol', 'triresolution', 'triresol']):
            if len(mesh.e_resol)==0: mesh=mesh.compute_e_resol()
            data['eresol'] = ("elem", mesh.e_resol/1000)
            data['eresol'].attrs["description"]='Element resolution'
            data['eresol'].attrs["descript"   ]='Element resolution'
            data['eresol'].attrs["long_name"  ]='Element resolution'
            data['eresol'].attrs["units"      ]='km'
            data['eresol'].attrs["is_data"    ]=is_data
            data = data.chunk(chunks['elem'])
           
        #_______________________________________________________________________
        data, dim_vert, dim_horz = do_gridinfo_and_weights(mesh,data,do_zweight=do_zarithm)
        if do_compute: data = data.compute()
        if do_load   : data = data.load()
        if do_persist: data = data.persist()
        return(data)
    
    #___________________________________________________________________________
    #  ||   ||   ||   ||   ||   ||   ||   ||   ||   ||   ||   ||   ||   ||   ||  
    # _||_ _||_ _||_ _||_ _||_ _||_ _||_ _||_ _||_ _||_ _||_ _||_ _||_ _||_ _||_ 
    # \  / \  / \  / \  / \  / \  / \  / \  / \  / \  / \  / \  / \  / \  / \  / 
    #  \/   \/   \/   \/   \/   \/   \/   \/   \/   \/   \/   \/   \/   \/   \/  
    #___________________________________________________________________________    
    # analyse vname input if vector data should be load  "vec+vnameu+vnamev"
    vname2, vname_tmp = None, None
    if ('vec' in vname) or ('norm' in vname) or ('grad' in vname):
        if ('vec'    in vname): do_vec   = True
        if ('norm'   in vname): do_norm  = True
        if ('gradx'  in vname): do_gradx = True
        if ('grady'  in vname): do_grady = True
        if ('gradxy' in vname): do_gradx, do_grady = True, True
        
        # in case you want to plot a single scalar velocity component, the velocities
        # might still need to be rotated depending what are the settings in the model
        # for the rotation you still need both components. After rotation the unnecessary 
        # component can be kicked out. The component that needs to be keept is 
        # defined by ":" vname = 'vec+u+v:v', the variable that is kept
        # is written into do_sclrv
        if ':' in vname: vname, do_sclrv = vname.split(':')    
        
        # determine the varaibles for the two vector component separated by "+"
        aux = vname.split('+')
        
        if   do_vec   : aux.remove('vec'   )
        if   do_norm  : aux.remove('norm'  )
        if   do_gradx and do_grady: aux.remove('gradxy') 
        elif do_gradx : aux.remove('gradx' ) 
        elif do_grady : aux.remove('grady' ) 
        
        if ((do_vec) or (do_norm)) and (not do_gradx and not do_grady):
            if len(aux)==1: raise ValueError(" to load vector or norm of data two variables need to be defined: vec+u+v")
            vname, vname2 = aux[0], aux[1]
        elif do_gradx or do_grady:
            vname = aux[0]
        del aux
        
    elif ('sigma' in vname) or ('pdens' in vname):
        do_pdens=True 
        vname_tmp = vname
        vname, vname2 = 'temp', 'salt'
        
    #___________________________________________________________________________
    # create path name list that needs to be loaded
    if isinstance(datapath, str):
        if '~/' in datapath: datapath = os.path.abspath(os.path.expanduser(datapath))
    pathlist, str_ltim = do_pathlist(year, datapath, do_filename, do_file, vname, runid)
    
    # if pathlist is empty jump out of the routine and return none 
    if len(pathlist)==0: 
        data = None
        return data
    
    #___________________________________________________________________________
    # set specfic type when loading --> #convert to specific precision
    from functools import partial
    def _preprocess(x, do_prec, transpose):
        #if transpose is not None:
           #for var in list(x.data_vars):
                 #x[var] = x[var].transpose(*transpose)  # Correct variable reference
        return x.astype(do_prec, copy=False)
    #partial_func = partial(_preprocess, do_prec=do_prec, transpose=['nod2','nz1','time'])
    partial_func = partial(_preprocess, do_prec=do_prec, transpose=None)
    
    #___________________________________________________________________________
    # load multiple files
    
    # Avoid Warning Message:
    # SerializationWarning: Unable to decode time axis into full numpy.datetime64 
    # objects, continuing using cftime.datetime objects instead, reason: dates out 
    # of range dtype = _decode_cf_datetime_dtype(data, u
    use_cftime = False
    if year[0]>2262 or year[1]>2262: use_cftime=True
    
    if   engine == 'netcdf4' : 
        #engine_dict = dict({})
        engine_dict = dict({
                            'engine'        :"netcdf4" , 
                            #'combine'       :'nested', 
                            #'compat'        :'override', !!! ATTENTION DO NOT USE THAT OPTION it overrides concated years with NaNs!!!
                            'decode_coords' :False     , 
                            'decode_times'  :True      , 
                            'use_cftime'    :use_cftime, 
                            'autoclose'     :False     , 
                            'lock'          :False     , 
                            'backend_kwargs':{'format': 'NETCDF4', 'mode':'r'}
                            })# load normal FESOM2 run file
    elif engine == 'h5netcdf': 
        engine_dict = dict({'engine':"h5netcdf",'backend_kwargs':{'phony_dims': 'sort', 'decode_vlen_strings':False, 'invalid_netcdf':'ignore'}})# load normal FESOM2 run file
    
    if do_file=='run':
        data = xr.open_mfdataset(pathlist, parallel=do_parallel, chunks=chunks, 
                                 preprocess=partial_func, 
                                 **engine_dict, 
                                 **kwargs)
        
        # !!! --> this is not a good idea, to do chunking after loading requires 
        # !!! --> massivly more RAM than giving the chunk operation directly into 
        # !!! --> loading routine                          
        #data = xr.open_mfdataset(pathlist, parallel=do_parallel, 
        #                         autoclose=True, preprocess=partial_func, 
        #                         **kwargs)
        #data = data.chunk({key: chunks[key] for key in data.dims})
        
        
        if do_showtime: 
            print(data.time.data)
            print(data['time.year'])
        
        # in case of vector load also meridional data and merge into 
        # dataset structure
        if (do_vec or do_norm or do_pdens) and vname2 is not None:
            pathlist, dum = do_pathlist(year, datapath, do_filename, do_file, vname2, runid)
            data     = xr.merge([data, xr.open_mfdataset(pathlist,  parallel=do_parallel, chunks=chunks, 
                                                         preprocess=partial_func, 
                                                         **engine_dict, 
                                                         **kwargs)])
            # !!! --> this is not a good idea, to do chunking after loading requires 
            # !!! --> massivly more RAM than giving the chunk operation directly into 
            # !!! --> loading routine                          
            #data     = xr.merge([data, xr.open_mfdataset(pathlist,  parallel=do_parallel, chunks=chunks, 
            #                                             autoclose=True, preprocess=partial_func, 
            #                                             **kwargs).chunk({key: chunks[key] for key in data.dims})])
            if do_vec: is_data='vector'
        
        ## rechunking leads to extended memory demand at runtime of xarray with
        ## dask client!!! --> this here is not a good idea!!!
        #data = data.chunk({'time': data.sizes['time']})
        
    # load restart or blowup files
    else:
        print(pathlist)
        data = xr.open_mfdataset(pathlist, parallel=do_parallel, chunks=chunks, 
                                 autoclose=False, preprocess=partial_func, **kwargs)
        if (do_vec or do_norm or do_pdens) and vname2 is not None:
            # which variables should be dropped 
            vname_drop = list(data.keys())
            print(' > var in file:', vname_drop)
            vname_drop.remove(vname)
            vname_drop.remove(vname2)
            
        else:    
            # which variables should be dropped 
            vname_drop = list(data.keys())
            print(' > var in file:', vname_drop)
            vname_drop.remove(vname)
            
        # remove variables that are not needed
        #data = data.drop(labels=vname_drop)
        data = data.drop_vars(vname_drop)
    
    #___________________________________________________________________________    
    if do_parallel and do_info: display(data)
    
    #___________________________________________________________________________    
    # This is for icepack data over thickness classes make class selection always 
    # based in indices
    if ('ncat' in data.dims): depidx = True
    
    #___________________________________________________________________________    
    # rename all dimension naming that do not agree with actual fesom2 standard
    # 'node' --> 'nod2'
    # 'nz_1' --> 'nz1'
    if ('node'      in data.dims     ): data = data.rename_dims({'node':'nod2'})
    if ('nz_1'      in data.dims     ): data = data.rename_dims({'nz_1':'nz1'})
    
    # Solution for Dmitry Stupak --> rechange dimension naming after various cdo
    # operations 
    if ('x'         in      data.dims):
        if   data.sizes['x']==mesh.n2dn: data = data.rename_dims({'x':'nod2'})
        elif data.sizes['x']==mesh.n2de: data = data.rename_dims({'x':'elem'})
    
    # convert ncells dimension if found to nod2 (happens for fesom14cmip6 and 
    # MULTIIO)
    if ('ncells'    in data.dims     ): data = data.rename_dims({'ncells':'nod2'})
    
    # kick out *_bnds variables if found we dont need them in moment in tripyview
    # and it makes the dataset smaller
    if ('lon_bnds'  in data.data_vars): data = data.drop_vars(['lon_bnds' ])
    if ('lat_bnds'  in data.data_vars): data = data.drop_vars(['lat_bnds' ])
    if ('time_bnds' in data.data_vars): data = data.drop_vars(['time_bnds'])
    
    # change depth dimension naming in case of fesom14cmip6 and MULTIIO data to 
    # fesom2 convention
    if ('depth' in data.coords) or ('lev' in data.coords): 
        
        if   ('depth' in data.coords): dimn_vold='depth' 
        elif ('lev'   in data.coords): dimn_vold='lev'
        
        if   (data.sizes[dimn_vold]==mesh.nlev):
            data = data.rename({dimn_vold :'nz'  })
            if 'nz' not in data.indexes: data = data.set_index(nz='nz')
            
            # rename dimension: depth --> nz
            if (dimn_vold in data.dims  ): data = data.swap_dims({dimn_vold: 'nz'})
            
        elif (data.sizes[dimn_vold]==mesh.nlev-1):
            data = data.rename({dimn_vold :'nz1'  })
            if 'nz1' not in data.indexes: data = data.set_index(nz1='nz1')
        
            # rename dimension: depth --> nz
            if (dimn_vold  in data.dims  ): data = data.swap_dims({dimn_vold: 'nz1'})
        
        del dimn_vold
    
    #___________________________________________________________________________
    # check if mesh and data fit together
    if   'nod2' in data.dims: 
        if data.sizes['nod2'] != mesh.n2dn  : raise ValueError(' --> vertice length of mesh and data does not fit togeather')
    elif 'elem' in data.dims: 
        if data.sizes['elem'] != mesh.n2de  : raise ValueError(' --> element length of mesh and data does not fit togeather')
    if   'nz1' in data.dims: 
        if data.sizes['nz1' ] != mesh.nlev-1: raise ValueError(' --> zmid length of mesh and data does not fit togeather')
    elif 'nz'  in data.dims: 
        if data.sizes['nz' ]  != mesh.nlev  : raise ValueError(' --> zlev length of mesh and data does not fit togeather')
    
    #___________________________________________________________________________
    # ensure proper dimnesion permutation for data it must be [time, nod2, nz]
    dimn_h, dimn_v = 'dum', 'dum'
    if   ('nod2' in data.dims): dimn_h = 'nod2'
    elif ('elem' in data.dims): dimn_h = 'elem'
    elif ('edg_n'in data.dims): dimn_h = 'edg_n'
    if   ('nz'   in data.dims): dimn_v = 'nz'
    elif ('nz1'  in data.dims): dimn_v = 'nz1'
    elif ('ndens'in data.dims): dimn_v = 'ndens'
    
    # check dimension ordering
    if 'time' in data.dims:
        if   ( len(data.dims)==3 and list(data.dims) != ['time', dimn_v, dimn_h]): data = data.transpose('time', dimn_v, dimn_h)
    else:    
        if ( len(data.dims)==2 and list(data.dims) != [dimn_v, dimn_h])        : data = data.transpose(dimn_v, dimn_h)
    del dimn_h, dimn_v
    
    #___________________________________________________________________________
    # add depth axes since its not included in restart and blowup files
    # also add weights
    if do_zarithm in ['wmean','wint']: do_zweight=True
    data = data.unify_chunks()
    data, dim_vert, dim_horz = do_gridinfo_and_weights(mesh, data, do_zweight=do_zweight, do_hweight=do_hweight)
    
    #___________________________________________________________________________
    # years are selected by the files that are open, need to select mon or day 
    # or record 
    data, mon, day, str_ltim = do_select_time(data, mon, day, record, str_ltim)
    
    # do time arithmetic on data
    if 'time' in data.dims: data, str_atim = do_time_arithmetic(data, do_tarithm)
    
    #___________________________________________________________________________
    # set bottom to nan --> in moment the bottom fill value is zero would be 
    # better to make here a different fill value in the netcdf files !!!
    data = do_setbottomnan(mesh, data, do_nan)
    
    #___________________________________________________________________________
    # select depth levels also for vertical interpolation 
    # found 3d data based mid-depth levels (temp, salt, pressure, ....)
    # if ( ('nz1' in data[vname].dims) or ('nz'  in data[vname].dims) ) and (depth is not None):
    if ( bool(set(['nz1','nz', 'ncat']).intersection(data.dims)) ) and (depth is not None):
        #print('~~ >-))))o> o0O ~~~ A')
        #_______________________________________________________________________
        data, str_ldep = do_select_levidx(data, mesh, depth, depidx, dim_vert)
        
        #_______________________________________________________________________
        if do_pdens: 
            data, vname = do_potential_density(data, do_pdens, vname, vname2, vname_tmp)
            
        #_______________________________________________________________________
        # do vertical interpolation and summation over interpolated levels 
        if depidx==False:
            str_adep = ', '+str(do_zarithm)
            
            auxdepth = depth
            if isinstance(depth,list) and len(depth)==1: auxdepth = depth[0]
                
            if   ('nz1' in data.dims):
                # this seems to be in moment slightly faster  than using here the 
                # map_blocks option!!!
                data = data.interp(nz1=auxdepth, method="linear")
                data['nzi'] = data['nzi'].astype("uint8")
                
                #template = data.isel(nz1=slice(0, 2)).interp(nz1=auxdepth, method="linear")
                #def interp_block(data_block, new_depth=None):
                    #return data_block.interp(nz1=new_depth, method="linear")
                #data = data.chunk({dim_vert: -1}).map_blocks(interp_block, 
                                                          #kwargs={"new_depth": auxdepth}, 
                                                          #template=template)
                
                if data['nz1'].size>1: 
                    data = do_depth_arithmetic(data, do_zarithm, "nz1")
                 
            elif ('nz'  in data.dims):    
                data = data.interp(nz=auxdepth, method="linear")
                if data['nz'].size>1:   
                    data = do_depth_arithmetic(data, do_zarithm, "nz") 
            
    #___________________________________________________________________________
    # select all depth levels but do vertical summation over it --> done for 
    # merid heatflux
    elif ( (bool(set(['nz1', 'nz', 'ncat', 'ndens']).intersection(data.dims))) and 
           (depth is None) and 
           (do_zarithm in ['sum','mean','wmean','wint', 'max', 'min']) ): 
        if   ('nz1'  in data.dims): data = do_depth_arithmetic(data, do_zarithm, "nz1" )
        elif ('nz'   in data.dims): data = do_depth_arithmetic(data, do_zarithm, "nz"  )
        elif ('ncat' in data.dims): data = do_depth_arithmetic(data, do_zarithm, "ncat")     
    
    # only 2D data found            
    else: depth=None
    
    #___________________________________________________________________________
    # rotate the vectors if do_rot=True and do_vec=True
    data = do_vector_rotation(data, mesh, do_vec, do_rot, do_sclrv)
    
    #___________________________________________________________________________
    # compute potential density if do_pdens=True    
    if do_pdens and depth is None: 
        data, vname = do_potential_density(data, do_pdens, vname, vname2, vname_tmp)
    
    #___________________________________________________________________________
    # compute gradient,  do_gradx=True, do_grady=True
    data = do_gradient_xy(data, mesh, datapath, do_gradx, do_grady, do_rot=True, 
                          diagpath=diagpath, runid=runid, chunks=chunks, 
                          do_info=True) 
    
    #___________________________________________________________________________
    # compute norm of the vectors if do_norm=True    
    data = do_vector_norm(data, do_norm)
    
    #___________________________________________________________________________
    # interpolate from elements to node
    if ('elem' in list(data.dims)) and do_ie2n: is_ie2n=True
    data = do_interp_e2n(data, mesh, do_ie2n, client=client)
    
    #___________________________________________________________________________
    # write additional attribute info
    str_lsave = str_ltim+str_ldep
    str_lsave = str_lsave.replace(' ','_').replace(',','').replace(':','')
    
    for vname in list(data.keys()):
        attr_dict=dict({'datapath':datapath, 'runid':runid, 'do_file':do_file, 'do_filename':do_filename, 
                        'year':year, 'mon':mon, 'day':day, 'record':record, 'depth':depth, 
                        'depidx':depidx, 'do_tarithm':str_atim,
                        'do_zarithm':str_adep, 'str_ltim':str_ltim,'str_ldep':str_ldep,'str_lsave':str_lsave,
                        'is_data':is_data, 'is_ie2n':is_ie2n, 'do_compute':do_compute, 
                        'descript':descript})
        
        # in case of icepack data write thickness class as attribute
        if ('ncat' in data.dims) and (depth is not None): attr_dict.update({'ncat':depth})
            
        data = do_additional_attrs(data, vname, attr_dict)
    
    #___________________________________________________________________________
    warnings.filterwarnings("ignore", category=UserWarning, message="Sending large graph of size")
    warnings.filterwarnings("ignore", category=UserWarning, message="Large object of size \\d+\\.\\d+ detected in task graph")
    if   do_compute: data = data.compute()
    elif do_load   : data = data.load()
    elif do_persist: data = data.persist()
    if any([do_compute, do_load, do_persist]): data.close()
    gc.collect()  # Trigger garbage collection
    if client is not None: client.rebalance()
    warnings.resetwarnings()
    #___________________________________________________________________________
    if do_info: 
        info_txt ="""___FESOM2 DATA INFO________________________
 > Dimensions : {}
 > {}
 > {}
 ___________________________________________""".format(
        str(data.dims),str(data.coords), str(data.data_vars))
        print(info_txt)    
        #print(list(data.dims))
        #print(data.coords)
        #print(data.data_vars)

    #___________________________________________________________________________
    return(data)

    #___________________________________________________________________________
    #  /\   /\   /\   /\   /\   /\   /\   /\   /\   /\   /\   /\   /\   /\   /\ 
    # /  \ /  \ /  \ /  \ /  \ /  \ /  \ /  \ /  \ /  \ /  \ /  \ /  \ /  \ /  \
    # -||- -||- -||- -||- -||- -||- -||- -||- -||- -||- -||- -||- -||- -||- -||-
    #  ||   ||   ||   ||   ||   ||   ||   ||   ||   ||   ||   ||   ||   ||   ||
    #___________________________________________________________________________



#
#    
# ___CREATE FILENAME MASK FOR: RUN, RESTART AND BLOWUP FILES___________________
def do_fnamemask(do_file,vname,runid,year):
    """
    --> contains filename mask to distinguish between run, restart and blowup file
    that can be loaded    
    
    Parameters:
    
        :do_file:   str, which kind of file should be loaded, 'run','restart_oce', 'restart_ice', 'blowup'
        
        :vname:     str, name of variable 
        
        :runid:     str, runid of simulation usually 'fesom'
        
        :year:      int, year number that should be loaded
    
    Returns:
    
        :fname:   str, filename
    
    ____________________________________________________________________________
    """
    if   do_file=='run'            : fname = '{}.{}.{}.nc'.format(   vname,runid,year)
    elif do_file=='restart_oce'    : fname = '{}.{}.oce.restart/{}.nc'.format(runid,year,vname)
    elif do_file=='restart_ice'    : fname = '{}.{}.ice.restart/{}.nc'.format(runid,year,vname)
    elif do_file=='restart_icepack': fname = '{}.{}.icepack.restart/{}.nc'.format(runid,year,vname)
    elif do_file=='blowup'         : fname = '{}.{}.oce.blowup.nc'.format( runid,year)
    #elif do_file=='restart_oce': fname = '{}.{}.oce.restart.nc'.format(runid,year)
    #elif do_file=='restart_ice': fname = '{}.{}.ice.restart.nc'.format(runid,year)
    
    #___________________________________________________________________________
    return(fname)



#
#
# ___CREATE PATHLIST TO DATAFILES______________________________________________
def do_pathlist(year, datapath, do_filename, do_file, vname, runid):
    """
    --> create path/file list of data that should be loaded 
    
    Parameters:
    
        :year:          int, list, np.array, range of years that should be loaded
        
        :datapath:      str, path that leads to the FESOM2 data
        
        :do_filename:   bool, (default=None) load this specific filname string instead
        
        :fo_file:       str, which kind of file should be loaded, 'run', 
                        'restart_oce', 'restart_ice', 'blowup'   
        
        :vname:         str, name of variable 
        
        :runid:         str, runid of simulation usually 'fesom' 
    
    Returns:
    
        :pathlist:      str, list
    
    ____________________________________________________________________________ 
    """

    pathlist=[]
    if datapath is None: return(pathlist,'')

    # specific filename and path is given to load 
    if  do_filename: 
        pathlist = datapath
        if isinstance(datapath, list):
            if isinstance(year, list) and len(year)==2:
                str_mtim = 'y:{}-{}'.format(str(year[0]), str(year[1]))
                
            elif isinstance(year, int):
                str_mtim = 'y:{}'.format(year)    
                
        else:
            str_mtim = os.path.basename(datapath)
        
    # list, np.array or range of years is given to load files
    elif isinstance(year, (list, np.ndarray, range)):
        # year = [yr_start, yr_end]
        if isinstance(year, list) and len(year)==2: 
            year_in = range(year[0],year[1]+1)
            str_mtim = 'y:{}-{}'.format(str(year[0]), str(year[1]))
        # year = [year1,year2,year3....]            
        else:           
            year_in = year
            str_mtim = 'y:{}-{}'.format(str(year[0]), str(year[-1]))
        # loop over year to create filename list 
        for yr in year_in:
            fname = do_fnamemask(do_file,vname,runid,yr)
            path  = os.path.join(datapath,fname)
            if os.path.isfile(path):
                pathlist.append(path)  
            else:
                print(f'--> No file: {path}')
    
    # a single year is given to load
    elif isinstance(year, int):
        fname = do_fnamemask(do_file,vname,runid,year)
        path  = os.path.join(datapath,fname)
        if os.path.isfile(path):
            pathlist.append(path)  
        else:
            print(f'--> No file: {path}')
        str_mtim = 'y:{}'.format(year)
    else:
        raise ValueError( " year can be integer, list, np.array or range(start,end)")
    
    #___________________________________________________________________________
    return(pathlist,str_mtim)



#
#
# ___CREATE GRIDINFO AND AREA &N DEPTH WEIGHTS__________________________________
def do_gridinfo_and_weights(mesh, data, do_hweight=True, do_zweight=False):
    """
    --> apply all coordinate information to the dataset. Apply vertice/ element centroids
    lon/lat positions, horizontal area weights for the scalar and vector cell and 
    vertical depth weight
    
    Parameters:
    
        :mesh:          fesom2 tripyview mesh object,  with all mesh information 
        
        :data:          xarray dataset object
        
        :do_hweight:    bool, (default=True) store weightsd for weighted horizontal
                        averages
        
        :do_zweight:    bool, (default=False) store weights for vertical weigthed 
                        averages within the dataset
                        
    Returns:
        
        :data:          xarray dataset object, with coordinate + weight information
        
    """
    
    # Suppress the specific warning about sending large graphs
    #warnings.filterwarnings("ignore", category=UserWarning, message="Sending large graph of size")

    #___________________________________________________________________________
    # setup vertical and horizontal dimension names 
    dimn_v, dimn_h = None, None
    if   'nz1'   in data.dims: dimn_v = 'nz1'        
    elif 'nz'    in data.dims: dimn_v = 'nz'
    elif 'ndens' in data.dims: dimn_v = 'ndens'
    elif 'ncat'  in data.dims: dimn_v = 'ncat'
    if   'nod2'  in data.dims: dimn_h = 'nod2'
    elif 'elem'  in data.dims: dimn_h = 'elem'
    elif 'edg_n' in data.dims: dimn_h = 'edg_n'
    set_chnk_h, set_chnk_v = dict(), dict()
    if dimn_h in data.chunksizes.keys(): set_chnk_h = {dimn_h: data.chunksizes[dimn_h]}
    if dimn_v in data.chunksizes.keys(): set_chnk_v = {dimn_v: data.chunksizes[dimn_v]}
    set_chnk_hv = {**set_chnk_h, **set_chnk_v}
    
    #___________________________________________________________________________
    # set vertical coordinates and grid info
    grid_info = dict()
    if   ('nz1'  in data.dims): 
        if data.sizes['nz1'] == len(mesh.zmid):
            if   ('nz1'  in data.coords): data = data.drop_vars('nz1' ) 
            grid_info['nz1']    = xr.DataArray(np.abs(mesh.zmid).astype('float32')                , dims=dimn_v).chunk(set_chnk_v)
            grid_info['nzi']    = xr.DataArray(np.arange(0,mesh.zmid.size, dtype='uint8')         , dims=dimn_v).chunk(set_chnk_v)
        
    elif ('nz'   in data.dims):
        if data.sizes['nz'] == len(mesh.zlev):
            if ('nz'  in data.coords): data = data.drop_vars('nz' ) 
            grid_info['nz' ]    = xr.DataArray(np.abs(mesh.zlev).astype('float32')                , dims=dimn_v).chunk(set_chnk_v) 
            grid_info['nzi']    = xr.DataArray(np.arange(0,mesh.zlev.size, dtype='uint8')         , dims=dimn_v).chunk(set_chnk_v) 
    
    #___________________________________________________________________________
    # set horizontal coordinates and gridinfo 
    # set vertice coordinates
    if   ('nod2' in data.dims):
        grid_info['lon'   ] = xr.DataArray(mesh.n_x.astype('float32')                         , dims=dimn_h).chunk(set_chnk_h)
        grid_info['lat'   ] = xr.DataArray(mesh.n_y.astype('float32')                         , dims=dimn_h).chunk(set_chnk_h)
        grid_info['nodi'  ] = xr.DataArray(np.arange(0,mesh.n2dn, dtype='int32')              , dims=dimn_h).chunk(set_chnk_h)
        grid_info['ispbnd'] = xr.DataArray(np.zeros(mesh.n2dn, dtype='bool')                  , dims=dimn_h).chunk(set_chnk_h)
        if   'nz1' == dimn_v: grid_info['nodiz'] = xr.DataArray(mesh.n_iz.astype('uint8'  )-1 , dims=dimn_h).chunk(set_chnk_h)
        elif 'nz'  == dimn_v: grid_info['nodiz'] = xr.DataArray(mesh.n_iz.astype('uint8'  )   , dims=dimn_h).chunk(set_chnk_h)
        
        #_______________________________________________________________________
        # do horizontal weighting for weighted mean computation on vertices
        if do_hweight:
            # need area weight for 3d data on mid depth levels
            if   'nz1' == dimn_v:
                if data.sizes['nz1'] == len(mesh.zmid):
                    grid_info['w_A'] = xr.DataArray(mesh.n_area[:-1,:].astype('float32')      , dims=[dimn_v, dimn_h]).chunk(set_chnk_hv)
                else:
                    # do this to add grid weights on data that have been already 
                    # vertically selcected
                    nzidx = data['nzi'].values.astype('uint8')
                    grid_info['w_A'] = xr.DataArray(mesh.n_area[nzidx, :].astype('float32')      , dims=[dimn_v, dimn_h]).chunk(set_chnk_hv)
                    
            # need area weight for 3d data on full depth levels
            elif 'nz'  == dimn_v:
                if mesh.n_area.ndim==1: # in case fesom14cmip6 n_area is not depth dependent, therefor ndims=1
                    grid_info['w_A'] = xr.DataArray(mesh.n_area.astype('float32')             , dims=[        dimn_h]).chunk(set_chnk_h)
                else:    
                    if data.sizes['nz'] == len(mesh.zmid):
                        grid_info['w_A'] = xr.DataArray(mesh.n_area.astype('float32')             , dims=[dimn_v, dimn_h]).chunk(set_chnk_hv)
                    else:
                        # do this to add grid weights on data that have been already 
                        # vertically selcected
                        nzidx = data['nz'].values.astype('uint8')
                        grid_info['w_A'] = xr.DataArray(mesh.n_area[nzidx, :].astype('float32')             , dims=[dimn_v, dimn_h]).chunk(set_chnk_hv)
            
            # only need area weights for 2d data
            else:   
                if mesh.n_area.ndim==1: # in case fesom14cmip6 n_area is not depth dependent, therefor ndims=1
                    grid_info['w_A'] = xr.DataArray(mesh.n_area.astype('float32')             , dims=[        dimn_h]).chunk(set_chnk_h)
                else:    
                    grid_info['w_A'] = xr.DataArray(mesh.n_area[0, :].astype('float32')       , dims=[        dimn_h]).chunk(set_chnk_h)
        
        #_______________________________________________________________________
        # do vertical weighting/volumen weight 
        if do_zweight and dimn_v is not None:  
            if   'nz1' == dimn_v: dz = mesh.zlev[:-1]-mesh.zlev[1:]
            elif 'nz'  == dimn_v: dz = np.hstack(((mesh.zlev[0]-mesh.zlev[1])/2.0, mesh.zmid[:-1]-mesh.zmid[1:], (mesh.zlev[-2]-mesh.zlev[-1])/2.0))
            grid_info['w_z']  = xr.DataArray(np.abs(dz).astype('float16')               , dims=[dimn_v        ]).chunk(set_chnk_v)
            del(dz)    
    
    #___________________________________________________________________________
    # set element coordinates
    elif ('elem' in data.dims):                          
        grid_info['lon'  ]  = xr.DataArray((mesh.n_x[mesh.e_i].sum(axis=1)/3.0).astype('float32'), dims=dimn_h).chunk(set_chnk_h)
        grid_info['lat'  ]  = xr.DataArray((mesh.n_y[mesh.e_i].sum(axis=1)/3.0).astype('float32'), dims=dimn_h).chunk(set_chnk_h)
        grid_info['elemi']  = xr.DataArray(np.arange(0,mesh.n2de, dtype='int32')                 , dims=dimn_h).chunk(set_chnk_h)
        grid_info['ispbnd'] = xr.DataArray(np.isin(np.arange(0, mesh.n2de, dtype='int32'), mesh.e_pbnd_1), dims=dimn_h).chunk(set_chnk_h)
        if   'nz1' == dimn_v: grid_info['elemiz'] = xr.DataArray(mesh.e_iz.astype('uint8')-1     , dims=dimn_h).chunk(set_chnk_h)
        elif 'nz'  == dimn_v: grid_info['elemiz'] = xr.DataArray(mesh.e_iz.astype('uint8')       , dims=dimn_h).chunk(set_chnk_h)
        
        #_______________________________________________________________________
        # do weighting for weighted mean computation on elements
        if do_hweight:
            grid_info['w_A'] = xr.DataArray(mesh.e_area.astype( 'float32')                       , dims=dimn_h).chunk(set_chnk_h)
        
        if do_zweight and dimn_v is not None:    
            if   'nz1' == dimn_v: dz = mesh.zlev[:-1]-mesh.zlev[1:]
            elif 'nz'  == dimn_v: dz = np.hstack(((mesh.zlev[0]-mesh.zlev[1])/2.0, mesh.zmid[:-1]-mesh.zmid[1:], (mesh.zlev[-2]-mesh.zlev[-1])/2.0))    
            grid_info['w_z']  = xr.DataArray(np.abs(dz).astype('float16')               , dims=[dimn_v        ]).chunk(set_chnk_v)
            del(dz)    
            
            #dz = xr.DataArray(np.abs(dz).astype('float16'), dims=dimn_v).chunk(set_chnk_v)
            
            #mat_nhor_iz    = data['elemiz'].expand_dims(dim=dimn_v)              # --> Shape (1, n)
            #mat_nzi_hor    = data['nzi'   ].expand_dims(dim=dimn_h, axis=-1)     # --> Shape (m, 1)
            ## Broadcast the arrays together (Dask-aware operation)
            #mat_nhor_iz, mat_nzi_hor = xr.broadcast(mat_nhor_iz, mat_nzi_hor) # --> Shape (m, n)

            #grid_info['w_z']= xr.where(mat_nzi_hor <= mat_nhor_iz, 1, 0 )*dz
            #del(dz, mat_nhor_iz, mat_nzi_hor)
            
    #___________________________________________________________________________
    # now return assigned grid_info to the dataset
    data = data.assign_coords(grid_info)
    gc.collect()
    return(data, dimn_v, dimn_h)

    

#
#
# ___SET 3D BOTTOM VALUES TO NAN_______________________________________________
def do_setbottomnan(mesh, data, do_nan):
    """
    --> replace bottom fill values with nan (default value is zero)
    
    Parameters:
    
        :mesh:      fesom2 tripyview mesh object,  with all mesh information
        
        :data:      xarray dataset object     
        
        :do_nan:    bool,  do replace bottom fill values with nan
        
    Returns:
    
        :data:      xarray dataset object     
        
    ____________________________________________________________________________
    """
    # set bottom to nan --> in moment the bottom fill value is zero would be 
    # better to make here a different fill value in the netcdf files !!!
    if do_nan and any(x in data.dims for x in ['nz1','nz']): 
        
        dimn_v = 'nz1' if 'nz1'  in data.dims else 'nz'
        dimn_h = 'nod2'if 'nod2' in data.dims else 'elem'
        if   ('nod2' in data.dims):
            # from Shape (n,)  --> Shape (1, n)
            mat_nhor_iz = data['nodiz'].expand_dims(dim=dimn_v) 
        elif('elem' in data.dims):
            # from Shape (n,)  --> Shape (1, n)
            mat_nhor_iz = data['elemiz'].expand_dims(dim=dimn_v) 
            
        # from Shape (m,) --> Shape (m, 1)    
        mat_nzi_hor = data['nzi'  ].expand_dims(dim=dimn_h, axis=-1)

        # Broadcast the arrays together (this will align them properly across chunks)
        # create here both arrays to have the size (m, n)
        mat_nhor_iz, mat_nzi_hor = broadcast_arrays(mat_nhor_iz, mat_nzi_hor)
        
        # set nan values where mat_nzi_hor <= mat_nhor_iz
        data_nan = data.where(mat_nzi_hor <= mat_nhor_iz)
        
        #if len(data.data_vars)==2:
            #vname, vname2 = list(data.data_vars)
            #data[vname ] = (data[vname].dims, da.where(mat_nzi_hor <= mat_nhor_iz, data[vname ], np.nan))
            #data[vname2] = (data[vname].dims, da.where(mat_nzi_hor <= mat_nhor_iz, data[vname2], np.nan))
        #else:
            #vname = list(data.data_vars)[0]
            #data[vname] = (data[vname].dims, da.where(mat_nzi_hor <= mat_nhor_iz, data[vname], np.nan))
            
        del mat_nhor_iz, mat_nzi_hor
        #___________________________________________________________________________
        del(data)
        gc.collect()
        return(data_nan)
    else:
        return(data)



#
#
# ___DO TIME SELECTION_________________________________________________________
def do_select_time(data, mon, day, record, str_mtim):
    """
    --> select specific month, dayy or record number                             
    
    Paramters:
    
        :data:      xarray dataset object                                   
        
        :mon:       list, (default=None), specific month that should be     
                    selected from dataset. If mon selection leads to no data
                    selection, because data maybe annual, selection is set to
                    mon=None                                                
        
        :day:       list, (default=None), same as mon but for day           
        
        :record:    int, list, (default=None), load specific record number from
                    dataset. Overwrites effect of mon and sel_day    
                    
        :str_mtim:  str, time label string input here contains already selection
                    from years 'y:1958-2019', now add info from selction of month or 
                    days
    
    Returns:
    
        :data:      returns xarray dataset object      
        
        :mon:       list, with mon that got selected otherwise None
        
        :day:       list, with day  that got selected otherwise None
        
        :str_ltim:  str, with selected time information 
    
    ____________________________________________________________________________
    """
    
    #___________________________________________________________________________
    # select no time use entire yearly file
    if (mon is None) and (day is None) and (record is None):
        return(data, mon, day, str_mtim)
    
    #___________________________________________________________________________
    # select time based on record index --> overwrites mon and day selection        
    elif (record is not None):
        data = data.isel(time=record)
        # do time information string 
        str_mtim = '{}, rec: {}'.format(str_mtim, record)
    
    #___________________________________________________________________________
    # select time based on mon and or day selection 
    elif (mon is not None) or (day is not None):
        
        if isinstance(mon, int): mon = [mon]
        if isinstance(day, int): day = [day]
        
        # by default select everything
        sel_mon = np.full((data['time'].size, ), True, dtype=bool)
        sel_day = np.full((data['time'].size, ), True, dtype=bool)
        
        # than check if mon or day is defined and overwrite selction mon day
        # selction array
        if   (mon is not None): sel_mon = np.isin( data['time.month'], mon)
        if   (day is not None): sel_day = np.inin( data['time.day']  , day)
        
        # check if selection would discard all time slices 
        if np.all(sel_mon==False): 
            sel_mon = np.full((data['time'].size, ), True, dtype=bool)
            mon     = None
            print(" > your mon selection was discarded, no time slice would have been selected!")
            print("   The loaded data might be only annual mean")
        if np.all(sel_day==False): 
            sel_mday = np.full((data['time'].size, ), True, dtype=bool)
            day      = None
            print(" > your day selection was discarded, no time slice would have been selected!")
            print("   The loaded data might be only annual or monthly mean")
            
        # select matching time slices
        data = data.isel(time=np.logical_and(sel_mon,sel_day))
        
        # do time information string for month
        if (mon is not None) and len(mon)!=12:
            mon_list_short='JFMAMJJASOND'
            mon_list_lon=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            if len(mon)==1: 
                str_mtim = '{}, m:{}.'.format(str_mtim, mon_list_lon[mon[0]-1])
            else:
                aux_mon = ''
                aux_mon = ['{}{}'.format(aux_mon,mon_list_short[i-1]) for i in mon]
                aux_mon = ''.join(aux_mon)
                str_mtim = '{}, m:{}'.format(str_mtim, str(aux_mon) )
        
        # do time information string for day
        if (day is not None):
            if len(mon)==1: 
                str_mtim = '{}, d:{}.'.format(str_mtim, str(day))
            else:
                str_mtim = '{}, d:{}-{}'.format(str_mtim, str(day[0]), str(day[-1]) )
        
    #___________________________________________________________________________
    gc.collect()
    return(data, mon, day, str_mtim)    



#
#
# ___DO VERTICAL LEVEL SELECTION_______________________________________________
def do_select_levidx(data, mesh, depth, depidx, dimn_v):
    """
    --> select vertical levels based on depth list
    
    Parameters:
    
        :data:      xarray dataset object
        
        :mesh:      fesom2 mesh object
        
        :depth:     int, list, np.array, range (default=None). Select single
                    depth level that will be interpolated or select list of
                    depth levels that will be interpolated and vertically
                    averaged. If None all vertical layers in data are loaded
        
        :depidx:    bool, (default:False) if depth is int and depidx=True,
                    depth index is selected that is closest to depth. No
                    interpolation will be done
    
    Returns:
    
        :data:      xarray dataset object
        
    ____________________________________________________________________________
    """
    #___________________________________________________________________________
    # no depth selecetion at all
    if   depth is None: 
        str_ldep = ''
        return(data, str_ldep)
    else:
        ndimax  = data.sizes[dimn_v]
        #_______________________________________________________________________
        # found 3d data based on mid-depth levels (w, Kv,...) --> compute 
        # selection index
        if   dimn_v == 'nz1':
            sel_levidx = do_comp_sel_levidx(data[dimn_v], depth, depidx, ndimax)
            aux_strdep = 'depidx'
        
        #_______________________________________________________________________
        # found 3d data based on full-depth levels (w, Kv,...) --> compute 
        # selection index
        elif dimn_v == 'nz':    
            sel_levidx = do_comp_sel_levidx(data[dimn_v], depth, depidx, ndimax)
            aux_strdep = 'depidx'
        
        #_______________________________________________________________________
        # found 3d data based based on icepack thickness classes -->
        # selection index
        elif dimn_v == 'ncat':
            sel_levidx = do_comp_sel_levidx(np.arange(1,ndimax+1,1), depth, depidx, ndimax)
            aux_strdep = 'ncat'
    
    #___________________________________________________________________________
    # select depth index
    data = data.isel({dimn_v:sel_levidx})#.chunk({dimn_v:-1})    
        
    #___________________________________________________________________________
    # do depth information string
    if (depth is not None) and not depidx:
        if   isinstance(depth,(int, float)):
            str_ldep = ', dep:{}m'.format(str(depth))
        elif isinstance(depth,(list, np.ndarray, range)):   
            if len(depth)>0:
                str_ldep = ', dep:{}-{}m'.format(str(depth[0]), str(depth[-1]))
            else:    
                str_ldep = ', dep:{}-{}m'.format(str(mesh.zlev[0]), str(mesh.zlev[-1]))
                
    elif (depth is not None) and depidx:            
        str_ldep = ', {}:{}'.format(aux_strdep, str(depth))
    #___________________________________________________________________________
    gc.collect()
    return(data, str_ldep)



#
#
# ___COMPUTE VERTICAL LEVEL SELECTION INDEX____________________________________
def do_comp_sel_levidx(zlev, depth, depidx, ndimax):
    """
    --> compute level indices that are needed to interpolate the depth levels
    
    Parameters:
    
        :zlev:          list, depth vector of the datas 
        
        :depth:         list, with depth levels that should be interpolated
        
        :depidx:        bool, (default:False) if depth is int and depidx=True,
                        depth index is selected that is closest to depth. No  
                        interpolation will be done 
        
        :ndimax:        int, maximum number of levels  mesh.n_iz.max()-1 for mid
                        depth datas, mesh.n_iz.max() for full level data
    
    Returns:
    
        :sel_levidx:    list, with level indices that should be extracted
        
    ____________________________________________________________________________
    """
    
    #___________________________________________________________________________
    # select indices for vertical interpolation for single depth layer
    if   isinstance(depth,(int, float)):
        # select index closest to depth
        if depidx:
            sel_levidx = np.argmin(abs(zlev-depth))
        # select index for interpoaltion 
        else:
            auxidx  = np.searchsorted(zlev,depth)
            if   auxidx>ndimax : sel_levidx = [ndimax-1,ndimax]       
            elif auxidx>=1     : sel_levidx = [auxidx-1,auxidx]
            else               : sel_levidx = [auxidx,auxidx+1]   
        
    #___________________________________________________________________________
    # select indices for vertical interpolation for multiple defined 
    # depth layer
    elif isinstance(depth,(list, np.ndarray, range)):   
        sel_levidx=[]
        for depi in depth:
            auxidx     = np.searchsorted(zlev, depi)
            if auxidx>ndimax and ndimax not in sel_levidx: sel_levidx.append(ndimax)    
            if auxidx>=1 and auxidx-1 not in sel_levidx: sel_levidx.append(auxidx-1)
            if (auxidx not in sel_levidx): sel_levidx.append(auxidx)
            if (auxidx==0 and 1 not in sel_levidx): sel_levidx.append(auxidx+1)
    
    #___________________________________________________________________________
    return(sel_levidx)
    
    

#
#    
# ___COMPUTE TIME ARITHMETICS ON DATA__________________________________________
def do_time_arithmetic(data, do_tarithm):
    """
    --> do arithmetic over time dimension
    
    Parameters:
    
        :data:          xarray dataset object
        
        :do_tarithm:    str (default='mean') do time arithmetic on time selection
                        option are
                        
                        - None, 'None'
                        - 'mean' mean over entire time dimension
                        - 'median' 
                        - 'std'
                        - 'var'
                        - 'max', 
                        - 'min', 
                        - 'sum'
                        - 'ymean','annual' mean over year dimension
                        - 'mmean','monthly' mean over month dimension
                        - 'dmean','daily' mean over day dimension

    Returns:
    
        :data:          xarray dataset object
    
        :str_atim:      str, which time arithmetic was applied 
        
    ____________________________________________________________________________
    """
    str_atim = None
    if do_tarithm is not None:
        
        str_atim = str(do_tarithm)
        
        #_______________________________________________________________________
        if   do_tarithm=='mean':
            data_tmean = data.mean(  dim="time", keep_attrs=True).persist()
        
        elif do_tarithm=='median':
            data_tmean = data.median(dim="time", keep_attrs=True).persist()
        
        elif do_tarithm=='std':
            data_tmean = data.std(   dim="time", keep_attrs=True).persist()
        
        elif do_tarithm=='var':
            data_tmean = data.var(   dim="time", keep_attrs=True).persist()       
        
        elif do_tarithm=='max':
            data_tmean = data.max(   dim="time", keep_attrs=True).persist()
        
        elif do_tarithm=='min':
            data_tmean = data.min(   dim="time", keep_attrs=True).persist()  
        
        elif do_tarithm=='sum':
            data_tmean = data.sum(   dim="time", keep_attrs=True).persist()    
        
        #_______________________________________________________________________
        # yearly means 
        elif do_tarithm in ['ymean','annual']:
            import datetime
            data_tmean     = data.groupby('time.year').mean('time')
            # recreate time axes based on year
            data_tmean     = data_tmean.rename_dims({'year':'time'})
            
            warnings.filterwarnings("ignore", category=UserWarning, message="Sending large graph of size")
            warnings.filterwarnings("ignore", category=UserWarning, message="Large object of size \\d+\\.\\d+ detected in task graph")
            
            aux_time = xr.cftime_range(start='{:d}-01-01'.format(data_tmean.year[1]), periods=len(data['time']), freq='YS')
            data_tmean     = data_tmean.drop_vars('year')
            data_tmean     = data_tmean.assign_coords(time=aux_time)
            del(aux_time)
            warnings.resetwarnings()
        
        #_______________________________________________________________________
        # monthly means --> seasonal cycle 
        elif do_tarithm in ['mmean','monthly']:
            import datetime
            data_tmean     = data.groupby('time.month').mean('time')
            # recreate time axes based on year
            data_tmean     = data_tmean.rename_dims({'month':'time'})
            
            warnings.filterwarnings("ignore", category=UserWarning, message="Sending large graph of size")
            warnings.filterwarnings("ignore", category=UserWarning, message="Large object of size \\d+\\.\\d+ detected in task graph")
            
            aux_time = xr.cftime_range(start='0001-01-01', periods=len(data_tmean['time']), freq='MS')
            data_tmean     = data_tmean.drop_vars('month')
            data_tmean     = data_tmean.assign_coords(time=aux_time)
            del(aux_time)
            warnings.resetwarnings()
        
        #_______________________________________________________________________
        # daily means --> 1...365
        elif do_tarithm in ['dmean','daily']:
            import datetime
            data_tmean     = data.groupby('time.day').mean('time')
            # recreate time axes based on year
            data_tmean     = data_tmean.rename_dims({'day':'time'})
            
            warnings.filterwarnings("ignore", category=UserWarning, message="Sending large graph of size")
            warnings.filterwarnings("ignore", category=UserWarning, message="Large object of size \\d+\\.\\d+ detected in task graph")
            
            aux_time = xr.cftime_range(start='0001-01-01', periods=len(data_tmean['time']), freq='DS')
            data_tmean     = data_tmean.drop_vars('day')
            data_tmean     = data_tmean.assign_coords(time=aux_time).drop_vars('day')
            del(aux_time)
            warnings.resetwarnings()
        
        elif do_tarithm=='None':
            return(data, str_atim)
        
        else:
            raise ValueError(' the time arithmetic of do_tarithm={} is not supported'.format(str(do_tarithm))) 
        
        #_______________________________________________________________________
        del(data)
        gc.collect()  # Trigger garbage collection
        return(data_tmean, str_atim)
    else:
        return(data, str_atim)



#
#
# ___COMPUTE HORIZONTAL ARITHMETICS ON DATA____________________________________
def do_horiz_arithmetic(data, do_harithm, dim_name):
    """
    --> do arithmetic on depth dimension
    
    Parameters:

        :data:          xarray dataset object

        :do_harithm:    str (default='mean') do arithmetic on selected vertical
                        layers options are
                        
                        - None, 'None'
                        - 'mean'  ... arithmetic mean 
                        - 'median'
                        - 'std'
                        - 'var'
                        - 'max' 
                        - 'min'
                        - 'sum'   ... arithmetic sum 
                        - 'wint'  ... weighted horizontal integral int( DATA*dxdy)
                        - 'wmean' ... weighted horizontal mean

        :dim_name:      str, name of depth dimension, is different for full-level
                        and mid-level data

    Returns:
    
        :data:          xarray dataset object

    """
    if do_harithm is not None:
        
        if   do_harithm=='mean':
            data_hmean = data.mean(  dim=dim_name, keep_attrs=True, skipna=True)
        
        elif do_harithm=='median':
            data_hmean = data.median(dim=dim_name, keep_attrs=True, skipna=True)
        
        elif do_harithm=='std':
            data_hmean = data.std(   dim=dim_name, keep_attrs=True, skipna=True) 
        
        elif do_harithm=='var':
            data_hmean = data.var(   dim=dim_name, keep_attrs=True, skipna=True)       
        
        elif do_harithm=='max':
            data_hmean = data.max(   dim=dim_name, keep_attrs=True, skipna=True)
        
        elif do_harithm=='min':
            data_hmean = data.min(   dim=dim_name, keep_attrs=True, skipna=True)  
        
        elif do_harithm=='sum':
            data_hmean = data.sum(   dim=dim_name, keep_attrs=True, skipna=True)            
        
        elif do_harithm=='wint':
            data_hmean    = data.weighted(data['w_A']).sum(dim=dim_name, keep_attrs=True, skipna=True)
            
        elif do_harithm=='wmean':
            # this solution needs way less RAM and scales better with dask
            data_hmean    = data.weighted(data['w_A']).mean(dim=dim_name, keep_attrs=True, skipna=True)
            
        elif do_harithm=='None' or do_harithm is None:
            return(data)
        
        else:
            raise ValueError(' the time arithmetic of do_tarithm={} is not supported'.format(str(do_tarithm))) 
        
        #_______________________________________________________________________
        del(data)
        gc.collect()  # Trigger garbage collection
        return(data_hmean)
    else:
        return(data)


#
#
#
# ___COMPUTE DEPTH ARITHMETICS ON DATA_________________________________________
def do_depth_arithmetic(data, do_zarithm, dim_name):
    """
    --> do arithmetic on depth dimension
    
    Parameters:
        :data:          xarray dataset object 

        :do_zarithm:    str (default='mean') do arithmetic on selected vertical
                        layers options are
                        
                        - None, 'None'
                        - 'mean' 
                        - 'max' 
                        - 'min'
                        - 'sum'
                        - 'wint'
                        - 'wmean'
 
        :dim_name:      str, name of depth dimension, is different for full-level
                        and mid-level data
 
    Returns:
    
        :data:   xarray dataset object
        
    ____________________________________________________________________________
    """
    if do_zarithm is not None:
        
        if   do_zarithm=='mean':
            data_zmean    = data.mean(dim=dim_name, keep_attrs=True, skipna=True).persist()
        
        elif do_zarithm=='max':
            data_zmean    = data.max( dim=dim_name, keep_attrs=True).persist()
        
        elif do_zarithm=='min':
            data_zmean    = data.min( dim=dim_name, keep_attrs=True).persist()  
        
        elif do_zarithm=='sum':
            data_zmean    = data.sum( dim=dim_name, keep_attrs=True, skipna=True).persist()
        
        elif do_zarithm=='wint':
            #print(data['w_z'].values)
            #data_zmean    = data.weighted(data['w_z']).sum(dim=dim_name, keep_attrs=True, skipna=True).persist()
            
            
            data_zmean    = data*data['w_z']
            data_zmean    = data_zmean.sum( dim=dim_name, keep_attrs=True, skipna=True).persist()
            
        elif do_zarithm=='wmean':
            data_zmean    = data.weighted(data['w_z']).mean(dim=dim_name, keep_attrs=True, skipna=True).persist()
        
        elif do_zarithm=='None' or do_zarithm is None:
            return(data)
        
        else:
            raise ValueError(' the depth arithmetic of do_zarithm={} is not supported'.format(str(do_zarithm))) 
        #_______________________________________________________________________
        del(data)
        gc.collect()  # Trigger garbage collection
        return(data_zmean)
    else:
        return(data)



#
#
# ___COMPUTE GRID ROTATION OF VECTOR DATA______________________________________
def do_vector_rotation(data, mesh, do_vec, do_rot, do_sclrv):
    """
    --> compute roration of vector: vname='vec+u+v'
    
    Parameters:
    
        :data:          xarray dataset object
        
        :mesh:          fesom2 mesh object
        
        :do_vec:        bool, should data be considered as vectors
        
        :do_rot:     bool, should rotation be applied
    
    Returns:
    
        :data:          xarray dataset object
        
    ____________________________________________________________________________
    """
    if do_vec and do_rot:
        # which varaibles are in data, must be two to compute vector rotation
        vname = list(data.keys())
        
        # vector data are on vertices 
        #if ('nod2' in data[vname[0]].dims) or ('node' in data[vname[0]].dims):
        print(' > do vector rotation')
        data[vname[0]].data, data[vname[1]].data = vec_r2g_dask(mesh.abg      , 
                                                    data['lon'].data     , 
                                                    data['lat'].data     , 
                                                    data[vname[0]].data  ,  
                                                    data[vname[1]].data  , 
                                                    gridis='geo' )
        
        # in case only a scalar vector component is needed, rotation might still 
        # need to be done. After rotation the other vector component can be dropped
        if do_sclrv is not None:
            vname_drop = list(data.data_vars)
            vname_drop.remove(do_sclrv)
            print(' > keep vector component: ', do_sclrv)
            print(' > drop vector component: ', vname_drop[-1])
            data = data.drop_vars(vname_drop)
        gc.collect()    
    #___________________________________________________________________________
    return(data)



#
#
# ___COMPUTE NORM OF VECTOR DATA_______________________________________________
def do_vector_norm(data, do_norm):
    """
    --> compute vector norm: vname='vec+u+v'
    
    Parameters:
    
        :data:      xarray dataset object
        
        :do_norm:   bool, should vector norm be computed
    
    Returns:
    
        :data:      xarray dataset object
    
    ____________________________________________________________________________
    """
    if do_norm:
        print(' > compute norm')

        # Extract variable names (assuming exactly two variables exist)
        vname = list(data.data_vars)  # Use `.data_vars` instead of `keys()` to ensure only variables are considered
        
        # Define new variable name
        new_vname = 'norm+{}+{}'.format(vname[0],vname[1])
        
        # Compute norm efficiently using `apply_ufunc`
        # --> this option is minimal faster than np.sqrt( np.square(data[vname[0]]) + 
        #     np.square(data[vname[1]]) ) but produces a much cleaner task graph 
        #     in dask
        data[new_vname] = xr.apply_ufunc(np.hypot,  # Efficient sqrt(x^2 + y^2) function
                                        data[vname[0]],
                                        data[vname[1]],
                                        dask="parallelized",  # Enables Dask parallelization
                                        output_dtypes=[data[vname[0]].dtype])
        #data[new_vname] = np.sqrt( np.square(data[vname[0]]) + np.square(data[vname[1]]) )
        
        # rescue attributes
        vattrs = data[vname[0]].attrs
        if 'long_name' in vattrs:
            if 'zonal'      in vattrs['long_name'  ]: vattrs['long_name'  ] = vattrs['long_name'  ].replace('zonal'     , "norm")
            if 'meridional' in vattrs['long_name'  ]: vattrs['long_name'  ] = vattrs['long_name'  ].replace('meridional', "norm")
        
        if 'desciptiion' in vattrs:
            if 'zonal'      in vattrs['description']: vattrs['description'] = vattrs['description'].replace('zonal'     , "norm")
            if 'meridional' in vattrs['description']: vattrs['description'] = vattrs['description'].replace('meridional', "norm")
        data[new_vname] = data[new_vname].assign_attrs(vattrs)
        
        # delet variable vname2 from Dataset
        data = data.drop_vars(vname)
        gc.collect()
    #___________________________________________________________________________    
    return(data)  



#
#
# ___COMPUTE GRADIENT OF SCALAR DATA____________________________________________
def do_gradient_xy(data, mesh, datapath, do_gradx, do_grady, 
                diagpath = None    , 
                runid    = 'fesom' , 
                chunks   = dict()  ,
                do_rot   = True    ,
                do_info  = True):
                #check_clockwise   = False   ,
                
    """
    --> compute gradients
    
    Parameters:
    
        :data:      xarray dataset object
        
        :mesh:      fesom2 tripyview mesh object,  with all mesh information
        
        :datapath:  str, path that leads to the FESOM2 data
        
        :do_gradx:  bool, compute gradient in zonal directions
        
        :do_grady:  bool, compute gradient in meridional directions
        
        :diagpath:  None, provide path to fesom.mesh.diag.nc file 
        
        :runid:     str, runid of loaded data and mesh.diag file 
        
        :chunks:    dict(), impose chunks 
        
        :do_rot:    bool, do rotation from rot2geo
        
        :do_info:   bool, print information
    
    Returns:
    
        :data:      xarray dataset object
    
    ____________________________________________________________________________
    """
    if do_gradx or do_grady:
        
        # Extract variable names (assuming exactly two variables exist)
        vname = list(data.data_vars)[0]  # Use `.data_vars` instead of `keys()` to ensure only variables are considered
        gattrs, vattrs = data.attrs, data[vname].attrs
        
        # Define new variable name
        if do_gradx: vname_grdx = 'gradx_{}'.format(vname)
        if do_grady: vname_grdy = 'grady_{}'.format(vname)
        
        #_______________________________________________________________________
        # scan for diagnostic files in meshpath, datapath and datapath/1/ or use 
        # diagpath directly 
        if diagpath is None:
            fname = runid+'.mesh.diag.nc'
            
            if   os.path.isfile( os.path.join(datapath, fname) ): 
                dname = datapath
            elif os.path.isfile( os.path.join( os.path.join(os.path.dirname(os.path.normpath(datapath)),'1/'), fname) ): 
                dname = os.path.join(os.path.dirname(os.path.normpath(datapath)),'1/')
            elif os.path.isfile( os.path.join(mesh.path,fname) ): 
                dname = mesh.path
            else:
                raise ValueError('could not find directory with...mesh.diag.nc file')
            
            diagpath = os.path.join(dname,fname)
            if do_info: print(' > found diag in directory: {:s}'.format(diagpath))
        
        #_______________________________________________________________________
        # decide over elemental chunking of the gradients
        if  'elem' in data.chunksizes:
            set_chnk = {'elem': data.chunksizes['elem']}
        elif 'elem' in chunks: 
            set_chnk = {'elem': chunks['elem']}
        else:
            set_chnk = {'elem': 'auto'}
        
        #_______________________________________________________________________
        # load gradient weights  from diagnostic file for scalar gradients
        if   'nod2' in data.dims:
            if do_info: print(' > compute gradient on vertices')
            list_dropvar = list(data.coords)
            data   = data.drop_vars(list_dropvar) 
            data   = data.chunk({'nod2':-1}).persist()
            data   = data.persist()
            
            # load only weights from diag file to compute gradients, drop everything
            # else!
            w_grad_name = list(['gradient_sca_x', 'gradient_sca_y', 'face_nodes', 'lon', 'lat'])
            #w_grad = xr.open_mfdataset(diagpath, parallel=True, data_vars=w_grad_name, chunks=set_chnk).persist()
            w_grad = xr.open_mfdataset(diagpath, parallel=True, chunks=set_chnk)
            list_dropvar = list(w_grad.data_vars)
            for dropvar in w_grad_name: list_dropvar.remove(dropvar)
            w_grad = w_grad.drop_vars(list_dropvar).persist()
            
            # uncomment this to test if the gradient computation works
            # data[vname].values = mesh.n_y
            # data[vname] = data[vname].persist()
            
            ## in fesom2 we rely on that the vertices indices in the elem array are 
            ## clockwise sorted. This is checked in the model and if not the case
            ## it is imposed in the model. But this can lead to the fact the elem
            ## array in the model can be different from the elem array in elem2d.out
            ## Here we try to check on this and impose the same correction 
            # permute = [0, 1, 2]
            ## check based on first triangle in the elem array if vertices orientation
            ## is clockwise (iscw=True) or counter-clockwise (iscw=False)
            # if check_clockwise:
            #     x1, y1 = mesh.n_x[mesh.e_i[0,0]], mesh.n_y[mesh.e_i[0,0]]
            #     x2, y2 = mesh.n_x[mesh.e_i[0,1]], mesh.n_y[mesh.e_i[0,1]]
            #     x3, y3 = mesh.n_x[mesh.e_i[0,2]], mesh.n_y[mesh.e_i[0,2]]
            #     # Signed area (2x)
            #     area2 = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1)
            #     iscw = np.sign(area2)==-1
            #     if not iscw: 
            #         print(' > found counter-clockwise orientation, do permute:', permute)
            #         permute=[0, 2, 1]
            
            # !!! ATTENTION !!!
            # to avoid here problem of clockwise or counterclockwise oriented 
            # elem array  we will use here always the elem array of the meshdiag
            # file (w_grad['face_nodes']) to compute the gradients !!!
            
            # I do this here in a little bit weird way to be efficient in terms of 
            # reindexing and dask operation and to avoid exeedingly high memory demand 
            # for very large grids
            grad_x, grad_y = 0, 0
            for ii in range(3):
                e_i     =  w_grad['face_nodes'].isel(n3=ii)-1
                data_e  = data[vname].isel(nod2=e_i)
                grad_x += data_e * w_grad['gradient_sca_x'].isel(n3=ii)
                grad_y += data_e * w_grad['gradient_sca_y'].isel(n3=ii)
            del e_i, data_e
            
        # load gradient weights  from diagnostic file for vector gradients, they only 
        # got added in a late version of fesom>2.6.8
        elif 'elem' in data.dims:
            if do_info: print(' > compute gradient on elem')
            list_dropvar = list(data.coords)
            data= data.drop_vars(list_dropvar) 
            data = data.chunk({'elem':-1}).persist()
            
            # load only weights from diag file to compute gradients
            w_grad_name = list(['gradient_vec_x', 'gradient_vec_y', 'face_links', 'face_nodes', 'lon', 'lat'])
            w_grad = xr.open_mfdataset(diagpath, parallel=True, data_vars=w_grad_name, chunks=set_chnk)
            list_dropvar = list(w_grad.data_vars)
            for dropvar in w_grad_name: list_dropvar.remove(dropvar)
            w_grad = w_grad.drop_vars(list_dropvar).persist()
            
            # I do this here in a little bit weird way to be efficient in terms of 
            # reindexing and dask operation and to avoid exeedingly high memory demand 
            # for very large grids
            grad_x, grad_y = 0, 0
            for ii in range(3):
                e_i    = w_grad['face_links'].isel(n3=ii)
                data_e = data[vname].isel(elem=e_i)
                grad_x += data_e * w_grad['gradient_vec_x'].isel(n3=ii)
                grad_y += data_e * w_grad['gradient_vec_y'].isel(n3=ii)
            del e_i, data_e
            
        #_______________________________________________________________________
        # since gradient_sca_x/y is in the rotated coordinates of the model the  
        # final gradients needs to be rotated back into geo coordinates
        if do_rot:
            if do_info: print(' > do gradient rotation')
            e_i = w_grad['face_nodes'].load()-1
            lon = w_grad['lon'].isel(nod2=e_i).mean(dim='n3').chunk(set_chnk)
            lat = w_grad['lat'].isel(nod2=e_i).mean(dim='n3').chunk(set_chnk)
            grad_x.data, grad_y.data = vec_r2g_dask(mesh.abg, 
                                                    lon.data, lat.data, 
                                                    grad_x.data, grad_y.data, 
                                                    gridis='geo', do_info=False)
        del w_grad
        gc.collect
        
        #_______________________________________________________________________
        # add attributes
        if do_gradx: 
            vattrs['description'] = 'zonal ' + data[vname].attrs['description'] + ' gradient'
            vattrs['long_name']   = 'zonal ' + data[vname].attrs['long_name'  ] + ' gradient'
            grad_x = grad_x.assign_attrs(vattrs)
                
        if do_grady: 
            vattrs['description'] = 'meridional ' + data[vname].attrs['description'] + ' gradient'
            vattrs['long_name']   = 'meridional ' + data[vname].attrs['long_name'  ] + ' gradient'
            grad_y = grad_y.assign_attrs(vattrs)  
        del(data)
        gc.collect()
        
        #_______________________________________________________________________
        # create new gradient dataset
        data_vars  = dict()
        if do_gradx: data_vars[vname_grdx] = grad_x.persist()
        if do_grady: data_vars[vname_grdy] = grad_y.persist()
        data       = xr.Dataset(data_vars=data_vars, attrs=gattrs)
        data, _, _ = do_gridinfo_and_weights(mesh, data, do_zweight=False, do_hweight=True)
        
    #___________________________________________________________________________    
    return(data)  


#
#
# ___COMPUTE NORM OF VECTOR DATA_______________________________________________
def do_potential_density(data, do_pdens, vname, vname2, vname_tmp):
    """
    --> compute potential density based on temp and salt
    
    Parameters:
    
        :data:      xarray dataset object, containing temperature and salinity data
        
        :do_pdens:  bool, should potential densitz be compute_boundary_edges
        
        :vname:     str, name of temperature variable in dataset
        
        :vname2:    str, name of salinity variable in dataset
        
        :vname_tmp: str, which potential density should be computed
                    - 'sigma0'  ... pref=0
                    - 'sigma1'  ... pref=1000
                    - 'sigma2'  ... pref=2000
                    - 'sigma3'  ... pref=3000
                    - 'sigma4'  ... pref=4000
                    - 'sigma5'  ... pref=5000
                    
    Returns:
    
        :data:      xarray dataset object, containing potential density
        
        :vname:     str, string with variable name of potential density
        
    ____________________________________________________________________________
    """
    if do_pdens:
        pref=0
        if   vname_tmp == 'sigma' or vname_tmp == 'sigma0'  : pref=0
        elif vname_tmp == 'sigma1' : pref=1000
        elif vname_tmp == 'sigma2' : pref=2000
        elif vname_tmp == 'sigma3' : pref=3000
        elif vname_tmp == 'sigma4' : pref=4000
        elif vname_tmp == 'sigma5' : pref=5000

        data_depth = data['nz1'].expand_dims({'nod2':data.sizes['nod2']})
        data_lat   = data['lat'].expand_dims({'nz1' :data.sizes['nz1' ]})        
        data_p     =  xr.apply_ufunc(gsw.p_from_z, -data_depth, data_lat, 
                            dask='parallelized',
                            output_dtypes=[float])
        # Expand p to match T,S dims, gsw.p_from_z want depth to be downward negative
        if 'time' in data.dims: 
            data_p =  data_p.expand_dims(time=data.sizes['time'])
            data_p = data_p.transpose('time', 'nz1', 'nod2')
            dims   = ['time', 'nz1', 'nod2']
            
        else:     
            data_p = data_p.transpose('nz1', 'nod2')
            dims   = ['nz1', 'nod2']
        del(data_depth, data_lat)
        
        # convert Practical Salinity --> Absolute Salinity unit
        SA = xr.apply_ufunc(gsw.SA_from_SP,
                            data[vname2].data, data_p, data['lon'].data, data['lat'].data,
                            dask='parallelized',
                            output_dtypes=[float])
        del(data_p)
        
        # convert Potential Temperature --> Conservative Temperature
        CT = xr.apply_ufunc(gsw.CT_from_pt,
                            SA, data[vname].data,
                            dask='parallelized',
                            output_dtypes=[float])
        
        # compute density at reference pressure pref
        rho = xr.apply_ufunc(gsw.rho,
                            SA, CT, pref,
                            dask='parallelized',
                            output_dtypes=[float])
        
        sigma = (rho - 1000.)#.persist()
        
        #data = data.assign({vname_tmp: (list(data[vname].dims), sigma.data)})
        data = data.assign({vname_tmp: (dims, sigma.data)})
        #data[vname_tmp] = data[vname_tmp].where(data[vname2]!=0,drop=0.0)
        data = data.persist()
        del(SA, CT, rho, sigma)
        
        data = data.drop_vars([vname, vname2])
        data[vname_tmp].attrs['units'] = 'kg/m^3'
        vname = vname_tmp
        
    #___________________________________________________________________________    
    return(data, vname)  



#
#
# ___INTERPOLATE ELEMENTAL DATA TO VERTICES____________________________________
def do_interp_e2n(data, mesh, do_ie2n, client=None):
    """
    --> interpolate data on elements to vertices
    
    Parameters:
    
        :data:      xarray dataset object
        
        :mesh:      fesom2 mesh object   
        
        :do_ie2n:   bool, True/False if interpolation should be applied
    
    Returns:
        
        :data:      xarray dataset object
        
    ____________________________________________________________________________
    """
    # which variables are stored in dataset
    vname_list = list(data.keys())
    if ('elem' in data[vname_list[0]].dims) and do_ie2n:
        print(' > do interpolation e2n')
        #_______________________________________________________________________
        if len(vname_list)==2: 
            aux, aux2  = grid_interp_e2n(mesh,data[vname_list[0]].values, data_e2=data[vname_list[1]].values, client=client)
            vname_new  = 'n_'+vname_list[0]
            vname_new2 = 'n_'+vname_list[1]
                    
            dim_list = list()
            if   'time' in data.dims: dim_list.append('time')
            if   'nz'   in data.dims: dim_list.append('nz')
            elif 'nz1'  in data.dims: dim_list.append('nz1')    
            dim_list.append('nod2')    
                
            data = xr.merge([ data, xr.Dataset({vname_new: (dim_list, aux), vname_new2: (dim_list, aux2)})], combine_attrs="no_conflicts")
            data = data.unify_chunks()
                
            # copy attributes from elem to vertice variable 
            data[vname_new].attrs  = data[vname_list[0]].attrs
            data[vname_new2].attrs = data[vname_list[1]].attrs
                
            # delete elem variable from dataset
            data = data.drop_vars(vname_list).rename({vname_new:vname_list[0], vname_new2:vname_list[1]})        
            del(aux, aux2)
            
        else:    
            for vname in vname_list:
                # interpolate elem to vertices
                #aux = grid_interp_e2n(mesh,data[vname].data)
                #with np.errstate(divide='ignore',invalid='ignore'):
                aux = grid_interp_e2n(mesh,data[vname].values, client=client)
                
                # new variable name 
                vname_new = 'n_'+vname
                
                # add vertice interpolated variable to dataset
                #print(data)
                dim_list = list()
                if   'time' in data.dims: dim_list.append('time')
                if   'nz'   in data.dims: dim_list.append('nz')
                elif 'nz1'  in data.dims: dim_list.append('nz1')    
                dim_list.append('nod2')    
                
                data = xr.merge([ data, xr.Dataset({vname_new: (dim_list, aux)})], combine_attrs="no_conflicts")
                data = data.unify_chunks()
                
                # copy attributes from elem to vertice variable 
                data[vname_new].attrs = data[vname].attrs
                
                # delete elem variable from dataset
                data = data.drop_vars(vname).rename({vname_new:vname})
                del(aux)
            
        #_______________________________________________________________________
        # kick out element related coordinates 
        for coordi in list(data.coords):
            if 'elem' in data[coordi].dims: data = data.drop_vars(coordi)
        
        #_______________________________________________________________________
        # enter area weights for nodes
        data, _ , _ = do_gridinfo_and_weights(mesh, data, do_hweight=True, do_zweight=False)
    #___________________________________________________________________________
    return(data)



#
#
# ___PUT ADDITIONAL VARIABLE INFORMATION INTO ATTRIBUTES_______________________
def do_additional_attrs(data, vname, attr_dict):
    """
    --> write additional information to variable attributes
    
    Parameters:
    
        :data:      xarray dataset object
        
        :vname:     str, (default: None), variable name that should be loaded
        
        :attr_dict: dict with different infos that are written to dataset 
                    variable attributes
        
    Returns:    

        :data:      xarray dataset object

    ____________________________________________________________________________
    """
    #___________________________________________________________________________
    for key in attr_dict:
        data[vname].attrs[key] = str(attr_dict[key])
        
    if 'description' not in data[vname].attrs.keys():
        data[vname].attrs['description'] = str(vname)
    if 'long_name'   not in data[vname].attrs.keys():
        data[vname].attrs['long_name']   = str(data[vname].attrs['description'])
    
    #___________________________________________________________________________
    return(data)



#
#
# ___DO ANOMALY________________________________________________________________
def do_anomaly(data1,data2):
    """
    --> compute anomaly between two xarray Datasets
    
    Parameters:
    
        :data1:   xarray dataset object

        :data2:   xarray dataset object

    Returns:
    
        :anom:   xarray dataset object, data1-data2

    ____________________________________________________________________________
    """
    # copy datasets object 
    anom = data1.copy()
    
    data1_vname = list(data1.keys())
    data2_vname = list(data2.keys())
    #for vname in list(anom.keys()):
    for vname, vname2 in zip(data1_vname, data2_vname):
        # do anomalous data 
        if vname=='dmoc_zpos':
            anom[vname].data = data1[vname].data
        else:
            anom[vname].data = data1[vname].data - data2[vname2].data
            
        # do anomalous attributes 
        attrs_data1 = data1[vname].attrs
        attrs_data2 = data2[vname2].attrs
        
        for key in attrs_data1.keys():
            if (key in attrs_data1.keys()) and (key in attrs_data2.keys()):
                if key in ['long_name']:
                   # anom[vname].attrs[key] = 'anomalous '+anom[vname].attrs[key]
                   anom[vname].attrs[key] = 'anom. '+anom[vname].attrs[key].capitalize()
                   
                elif key in ['short_name']:
                   # anom[vname].attrs[key] = 'anomalous '+anom[vname].attrs[key]
                   anom[vname].attrs[key] = 'anom. '+anom[vname].attrs[key]    
                   
                elif key in ['units']: 
                    continue
                
                elif key in ['descript']: 
                    # print(len(data1[vname].attrs[key])+len(data2[vname2].attrs[key]))
                    if len(data1[vname].attrs[key])+len(data2[vname2].attrs[key])>75:
                        anom[vname].attrs[key]  = data1[vname].attrs[key]+'\n - '+data2[vname2].attrs[key]
                    else:     
                        anom[vname].attrs[key]  = data1[vname].attrs[key]+' - '+data2[vname2].attrs[key]
                        
                elif key in ['do_rescale']: 
                    anom[vname].attrs[key]  = data1[vname].attrs[key]    
                    
                elif data1[vname].attrs[key] != data2[vname2].attrs[key]:
                    anom[vname].attrs[key]  = data1[vname].attrs[key]+' - '+data2[vname2].attrs[key]
    
    #___________________________________________________________________________
    return(anom)


#
#
#_______________________________________________________________________________
def coarsegrain_h_dask(data, do_parallel, parallel_nprc, dlon=1.0, dlat=1.0, client=None ):
    import dask.array as da
    
    #___________________________________________________________________________
    if len(list(data.data_vars))==2:
        vname, vname2 = list(data.data_vars)
    else:     
        vname = list(data.data_vars)[0]
    
    #___________________________________________________________________________
    # determine actual chunksize
    nchunk = 1
    if do_parallel and ('elem' in data.chunks or 'nod2' in data.chunks):
        if   'elem' in data.dims: nchunk = len(data.chunks['elem'])
        elif 'nod2' in data.dims: nchunk = len(data.chunks['nod2'])
        print(' --> nchunk=', nchunk)  
        
        #___________________________________________________________________________
        # after all the time and depth operation after the loading there will be worker who have no chunk
        # piece to work on  --> therfore we needtro rechunk 
        # make sure the workload is distributed between all availbel worker equally         
        if nchunk<parallel_nprc*0.75:
            print(' --> rechunk array size', end='')
            if   'elem' in data.dims: 
                data = data.chunk({'elem': np.ceil(data.sizes['elem']/parallel_nprc).astype('int')})
                nchunk = len(data.chunks['elem'])
            elif 'nod2' in data.dims: 
                data = data.chunk({'nod2': np.ceil(data.sizes['nod2']/parallel_nprc).astype('int')})
                nchunk = len(data.chunks['nod2'])
            # print(data.chunks)        
            print(' --> nchunk_new=', nchunk)        
    
    #___________________________________________________________________________
    # The centroid position of the periodic boundary trinagle causes problems when determining in which 
    # bin they should be --> therefor we kick them out 
    if 'ispbnd' not in data.coords: 
        data = data.assign_coords(ispbnd=xr.DataArray(np.zeros(data[vname].shape, dtype=bool), dims=data[vname].dims).chunk((data[vname].chunks) ))
    
    #___________________________________________________________________________
    # create lon lat bins 
    rad     , Rearth   = np.pi/180, 6371e3
    lon_min , lon_max  = float(np.floor(data['lon'].min().compute())), float(np.ceil( data['lon'].max().compute()))
    lat_min , lat_max  = float(np.floor(data['lat'].min().compute())), float(np.ceil( data['lat'].max().compute()))
    lon_bins, lat_bins = np.arange(lon_min, lon_max+dlon/2, dlon), np.arange(lat_min, lat_max+dlat/2, dlat)
    nlon    , nlat     = len(lon_bins)-1, len(lat_bins)-1
    lon     , lat      = (lon_bins[1:]+lon_bins[:-1])*0.5, (lat_bins[1:]+lat_bins[:-1])*0.5
    dx      , dy       = Rearth*dlon*rad*np.cos((lat)/2.0*rad), Rearth*dlat*rad, 
    dA                 = np.tile(dx*dy, (nlon,1)).T
    del(dx, dy, lon_min, lon_max, lat_min, lat_max)
    
    #___________________________________________________________________________
    # Apply coarse-graining over chunks for both u and v velocities
    if len(list(data.data_vars))==2:
        binned_d = da.map_blocks(coarsegrain_h_chnk    ,
                                lon_bins               , 
                                lat_bins               ,
                                data['lon'      ].data ,  # lon mesh coordinates of chunk piece
                                data['lat'      ].data ,  # lat mesh coordinates of chunk piece
                                data['w_A'      ].data ,  # area weight
                                data['ispbnd'   ].data ,  # index if triangle is boundary triangle, can be None for nodes
                                data[vname      ].data ,  # zonal vel. of chunk piece
                                data[vname2     ].data ,  # meridional vel. Chunked data for u and v
                                dtype  = np.float32    ,  # Tuple dtype
                                chunks = (3*nlon*nlat,)  # Output shape
                                )
        # reshape axis over chunks 
        binned_d = binned_d.reshape((nchunk, 3, nlat, nlon ))
    
    # Apply coarse-graining over chunks for single data
    else:   
        # Apply coarse-graining over chunks for both u and v velocities
        binned_d = da.map_blocks(coarsegrain_h_chnk    ,
                                lon_bins               , 
                                lat_bins               ,
                                data['lon'      ].data ,  # lon mesh coordinates of chunk piece
                                data['lat'      ].data ,  # lat mesh coordinates of chunk piece
                                data['w_A'      ].data ,  # area weight
                                data['ispbnd'   ].data ,  # index if triangle is boundary triangle, can be None for nodes
                                data[vname      ].data ,  # single data chunk piece
                                None                   ,
                                dtype  = np.float32    ,  # Tuple dtype
                                chunks = (2*nlon*nlat,)  # Output shape
                                )
        # reshape axis over chunks 
        binned_d = binned_d.reshape((nchunk, 2, nlat, nlon ))
        
    #___________________________________________________________________________
    # do dask axis reduction across chunks dimension
    binned_d = da.reduction(binned_d,                   
                            chunk     = lambda x, axis=None, keepdims=None: x,  # this is a do nothing function definition
                            aggregate = np.sum, 
                            dtype     = np.float32,  # Tuple dtype
                            axis      = 0,
                            ).compute()
    if client is not None: client.rebalance()
    
    #___________________________________________________________________________
    # deal with u,v data
    if len(list(data.data_vars))==2:
        # compute mean velocities ber bin for u/v--> avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            binned_d[0] = np.where(binned_d[2] > 0, binned_d[0] / binned_d[2], np.nan)
            binned_d[1] = np.where(binned_d[2] > 0, binned_d[1] / binned_d[2], np.nan)
        
        # build data_vars dictionary 
        data_vars =  dict({vname    : (('lat','lon'), binned_d[0], data[vname].attrs), 
                           vname2   : (('lat','lon'), binned_d[1], data[vname2].attrs)})
    
    # deal with single data
    else:
        # compute mean velocities ber bin for single data--> avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            binned_d[0] = np.where(binned_d[1] > 0, binned_d[0] / binned_d[1], np.nan)
        
        # build data_vars dictionary 
        data_vars =  dict({vname    : (('lat','lon'), binned_d[0], data[vname].attrs)})
    
    #___________________________________________________________________________
    # write xarray dataset
    data_reg = xr.Dataset(data_vars = data_vars,
                           coords    = {'lon'    : (('lon'      ), lon.astype(np.float32)), 
                                        'lat'    : (('lat'      ), lat.astype(np.float32)), 
                                        'lon_bnd': (('lon_bnd'  ), lon_bins.astype(np.float32)) , 
                                        'lat_bnd': (('lat_bnd'  ), lat_bins.astype(np.float32)) , 
                                        'w_A'    : (('lat','lon'), dA.astype(np.float32))},
                           attrs     = data.attrs)
    #___________________________________________________________________________
    #data_reg = data_reg.load()
    return(data_reg)



#
#
#_______________________________________________________________________________
def coarsegrain_h_chnk(lon_bins, lat_bins, chnk_lon, chnk_lat, chnk_wA, chnk_pbnd, chnk_d, chnk_d2):
    """
    Coarse-grain unstructured chunked data into longitude-latitude bins.
    """
    # Replace NaNs with 0 value to summation issues
    chnk_wA     = np.where(np.isnan(chnk_d ), 0, chnk_wA)
    chnk_d      = np.where(np.isnan(chnk_d ), 0, chnk_d )
    if chnk_d2 is not None:
        chnk_d2 = np.where(np.isnan(chnk_d2), 0, chnk_d2)
        
    # Use np.digitize to find bin indices for longitudes and latitudes
    lon_indices = np.digitize(chnk_lon, lon_bins) - 1  # Adjust to get 0-based index
    lat_indices = np.digitize(chnk_lat, lat_bins) - 1  # Adjust to get 0-based index
    nlon, nlat  = len(lon_bins)-1, len(lat_bins)-1
    
    # Initialize binned data storage for both u and v velocities
    if chnk_d2 is None:
        binned_d= np.zeros((2, len(lat_bins) - 1, len(lon_bins) - 1))
        # binned_d[0, nlat, nlon] - sum area weight data
        # binned_d[1, nlat, nlon] - area weight sum
        
    # Initialize binned data storage for both single data
    else:
        binned_d= np.zeros((3, len(lat_bins) - 1, len(lon_bins) - 1))
        # binned_d[0, nlat, nlon] - sum area weight zonal data
        # binned_d[1, nlat, nlon] - sum area weight merid data
        # binned_d[2, nlat, nlon] - area weight sum
    
    # Precompute mask outside the loop
    idx_valid   = ((lon_indices >= 0) & (lon_indices < nlon) & 
                   (lat_indices >= 0) & (lat_indices < nlat) &    
                   (~chnk_pbnd))
    del(chnk_pbnd)
    
    # Apply mask before looping
    lat_indices = lat_indices[idx_valid]
    lon_indices = lon_indices[idx_valid]
    chnk_wA     = chnk_wA[    idx_valid]
    chnk_d      = chnk_d [    idx_valid]
    if chnk_d2 is not None:
        chnk_d2 = chnk_d2[    idx_valid]
    nnod        = len(chnk_d)
    
    # do binning for single data
    if chnk_d2 is None:
        for nod_i in range(nnod):
            ii, jj = lon_indices[nod_i], lat_indices[nod_i]
            binned_d[0, jj, ii] = binned_d[0, jj, ii] + chnk_d[ nod_i] * chnk_wA[nod_i]
            binned_d[1, jj, ii] = binned_d[1, jj, ii] + chnk_wA[nod_i] # area weight counter
    
    # do binning for zonal/merid data
    else:
        for nod_i in range(nnod):
            ii, jj = lon_indices[nod_i], lat_indices[nod_i]
            binned_d[0, jj, ii] = binned_d[0, jj, ii] + chnk_d[ nod_i] * chnk_wA[nod_i]
            binned_d[1, jj, ii] = binned_d[1, jj, ii] + chnk_d2[nod_i] * chnk_wA[nod_i]
            binned_d[2, jj, ii] = binned_d[2, jj, ii] + chnk_wA[nod_i] # area weight counter

    return binned_d.flatten()


def vec_r2g_dask(abg, lon, lat, urot, vrot, gridis='geo', do_info=False):
    """
    Rotate vector data from rotated coordinates into geographical coordinates.
    This version uses Dask for parallelization.
    """
    #___________________________________________________________________________
    # create grid coorinates for geo and rotated frame
    if any(x in gridis for x in ['geo', 'g', 'geographical']):
        rlon, rlat = grid_g2r(abg, lon, lat)
    elif any(x in gridis for x in ['rot', 'r', 'rotated']):
        rlon, rlat = lon, lat
        lon , lat  = grid_r2g(abg, rlon, rlat)
    else:
        raise ValueError(f"Unsupported gridis={gridis}, expected 'geo' or 'rot'.")

    #___________________________________________________________________________
    # compute rotation matrix
    rmat = grid_rotmat_dask(abg)

    # Use map_blocks to calculate the pseudo-inverse in parallel
    rmat = np.linalg.pinv(rmat.compute())
    
    #___________________________________________________________________________
    # degree --> radian 
    rad        = np.pi / 180  # Degree to radian conversion
    lon , lat  = lon  * rad, lat  * rad
    rlon, rlat = rlon * rad, rlat * rad
    if   vrot.ndim == 2 or urot.ndim == 2: 
        lon , lat  =  lon[None, :],  lat[None, :]
        rlon, rlat = rlon[None, :], rlat[None, :]
    elif vrot.ndim == 3 or urot.ndim == 3: 
        lon ,  lat =  lon[None, None, :] ,  lat[None, None, :]
        rlon, rlat = rlon[None, None, :] , rlat[None, None, :]
        
    #___________________________________________________________________________
    # Precompute trigonometric functions for efficiency (avoid recalculating in 
    # loop)
    sin_lat , cos_lat  = da.sin(lat ), da.cos(lat )
    sin_lon , cos_lon  = da.sin(lon ), da.cos(lon )
    sin_rlat, cos_rlat = da.sin(rlat), da.cos(rlat)
    sin_rlon, cos_rlon = da.sin(rlon), da.cos(rlon)

    #___________________________________________________________________________
    # rotation of one dimensional vector data
    # Handle 1D/2d/3d data (vectorized with Dask)
    
    # Compute vector in rotated cartesian coordinates
    vxr = -vrot*sin_rlat*cos_rlon - urot*sin_rlon
    vyr = -vrot*sin_rlat*sin_rlon + urot*cos_rlon
    vzr =  vrot*cos_rlat

    # Compute vector in geo cartesian coordinates
    vxg = rmat[0, 0]*vxr + rmat[0, 1]*vyr + rmat[0, 2]*vzr
    vyg = rmat[1, 0]*vxr + rmat[1, 1]*vyr + rmat[1, 2]*vzr
    vzg = rmat[2, 0]*vxr + rmat[2, 1]*vyr + rmat[2, 2]*vzr

    # Compute vector in geo coordinates
    vgeo = vxg*-sin_lat*cos_lon - vyg*sin_lat*sin_lon + vzg*cos_lat
    ugeo = vxg*-sin_lon         + vyg*cos_lon

    return ugeo.astype(np.float32), vgeo.astype(np.float32)



#_______________________________________________________________________________
def isotherm_depth_dask(data, which_isotherm, client=None,):
    import dask.array as da
    vname = list(data.keys())[0]  # Get temperature variable name
    if   'nod2'  in data.dims: dimn_h = 'nod2'
    elif 'elem'  in data.dims: dimn_h = 'elem'
    elif 'edg_n' in data.dims: dimn_h = 'edg_n'
    #___________________________________________________________________________
    # Apply function to chunks over dask client 
    isothermz = da.map_blocks(isotherm_depth_chnk            , # input function isotherm_depth_chnk 
                              data[vname].data               , # temp_chunk
                              data.coords['nz1'].data        , # depth
                              which_isotherm                 , # which isotherm value 
                              dtype=np.float32, 
                              drop_axis=0,
                             )
    isothermz = isothermz.compute()
    if client is not None: client.rebalance()
        
    #___________________________________________________________________________
    # build xarray dataset
    isotdep = xr.Dataset(data_vars = {'isotdep': ('nod2', isothermz, data[vname].attrs)},
                             coords    = {'lon'    : data.coords['lon'], 
                                          'lat'    : data.coords['lat'], 
                                          'w_A'    : data.coords['w_A'].isel(nz1=0)},
                             attrs     = data.attrs)
    isotdep['isotdep'].attrs['long_name'  ] = 'depth of {}°C isotherm'.format(which_isotherm)
    isotdep['isotdep'].attrs['description'] = 'depth of {}°C isotherm'.format(which_isotherm)
    isotdep['isotdep'].attrs['units'      ] = 'm'
    isotdep = isotdep.load()
    #___________________________________________________________________________
    return(isotdep) 



#_______________________________________________________________________________
# compute isotherm depth for each chunk block, hereby its important that the vertical dimension
# nz1 is NOT chunked and consecutive
def isotherm_depth_chnk(temp_chnk, depth_vals, which_isotherm):
    import dask.array as da
    """Efficiently compute the isotherm depth for each node."""
    nz1, nod2    = temp_chnk.shape
    
    # Replace NaNs with a large negative value to avoid issues
    temp_chnk    = da.where(da.isnan(temp_chnk), da.inf, temp_chnk)

    # Find below indices where temp crosses isotherm
    idx_below    = da.argmax(temp_chnk < which_isotherm, axis=0)
    # it can haben that some chunk contain no valid situation for 
    # temp_chnk < which_isotherm in this da.argmax would return an empty array
    if len(idx_below)==0: idx_below = da.zeros(nod2)

    # Find below indices where temp crosses isotherm
    idx_above    = idx_below-1

    # Find depth layers above below where isotherm crosses
    depth_below  = depth_vals[idx_below]
    depth_above  = depth_vals[idx_above]

    # if there is no valid isotherm crossing set depth layers to NaN
    depth_above  = da.where(idx_above<=-1, da.nan, depth_above)
    depth_below  = da.where(idx_above<=-1, da.nan, depth_below)
    
    # Create a tuple of indices for (nod2, idx_below)
    # Convert the tuple of indices into a linear index for the flattened temp_block
    # Use the linear index to get values from the flattened temp_chnk[nod2, nz1]
    #flat_indices = da.arange(nod2) * nz1 + idx_below # --> for temp_chnk shape (nod2, nlev)
    flat_indices = da.arange(nod2) + idx_below*nod2   # --> for temp_chnk shape (nlev, nod2) !!!
    temp_below   = temp_chnk.flatten()[flat_indices]

    #flat_indices = da.arange(nod2) * nz1 + idx_above # --> for temp_chnk shape (nod2, nlev)
    flat_indices = da.arange(nod2) + idx_above*nod2   # --> for temp_chnk shape (nlev, nod2) !!!
    temp_above   = temp_chnk.flatten()[flat_indices]
    del(flat_indices)

    # avoid division by zero
    denom_temp   = temp_below - temp_above
    denom_temp   = da.where(denom_temp == 0, da.nan, denom_temp)
    
    # linearly interpolate isotherm depth
    isothermz = depth_above + ((which_isotherm - temp_above) * (depth_below - depth_above) / denom_temp)
    
    return (isothermz)
    #return np.stack([nz1, nod2])
