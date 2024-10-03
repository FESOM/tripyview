# Patrick Scholz, 23.01.2018
import numpy as np
import time  as clock
import os
import warnings
import xarray as xr
import netCDF4 as nc
import seawater as sw
#import gsw as gsw
from .sub_mesh import *
import warnings


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
                     do_vecrot      = True      , 
                     do_filename    = False     , 
                     do_file        = 'run'     , 
                     descript       = ''        , 
                     runid          = 'fesom'   , 
                     do_prec        = 'float32' , 
                     do_f14cmip6    = False     , 
                     do_compute     = False     , 
                     do_load        = True      , 
                     do_persist     = False     , 
                     do_parallel    = False     ,
                     chunks         = { 'time' :'auto', 'elem':'auto', 'nod2':'auto', \
                                        'edg_n':'auto', 'nz'  :'auto', 'nz1' :'auto', \
                                        'ndens':'auto'},
                     do_showtime    = False     , 
                     do_info        = True      , 
                     **kwargs):
    """
    --> general loading of 
    
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
        
        :do_vecrot:     bool (default=True), if vector data are loaded e.g. 
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
        
    Returns:
    
        :data:          object, returns xarray dataset object
        
    ____________________________________________________________________________
    """
    #___________________________________________________________________________
    # default values 
    is_data = 'scalar'
    is_ie2n = False
    do_vec  = False
    do_norm = False
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
    if ('vec' in vname) or ('norm' in vname):
        if ('vec'  in vname): do_vec =True
        if ('norm' in vname): do_norm=True
        aux = vname.split('+')
        if len(aux)==2 or aux[-1]=='': 
            raise ValueError(" to load vector or norm of data two variables need to be defined: vec+u+v")
        vname, vname2 = aux[1], aux[2]
        del aux
    elif ('sigma' in vname) or ('pdens' in vname):
        do_pdens=True 
        vname_tmp = vname
        vname, vname2 = 'temp', 'salt'
        
    #___________________________________________________________________________
    # create path name list that needs to be loaded
    if '~/' in datapath: datapath = os.path.abspath(os.path.expanduser(datapath))
    pathlist, str_ltim = do_pathlist(year, datapath, do_filename, do_file, vname, runid)
    if len(pathlist)==0: 
        data = None
        return data
    
    #___________________________________________________________________________
    # set specfic type when loading --> #convert to specific precision
    from functools import partial
    def _preprocess(x, do_prec):
        return x.astype(do_prec, copy=False)
    
    #def _preprocess(x, do_prec):
        #for var in list(x.coords):
            #if var == 'time_bnds': x = x.drop_vars(var)
        #return x.astype(do_prec, copy=False)

    partial_func = partial(_preprocess, do_prec=do_prec)
    
    #___________________________________________________________________________
    # load multiple files
    # load normal FESOM2 run file
    if do_file=='run':
        data = xr.open_mfdataset(pathlist, parallel=do_parallel, chunks=chunks, 
                                 autoclose=False, preprocess=partial_func, **kwargs)
        if do_showtime: 
            print(data.time.data)
            print(data['time.year'])
        
        # in case of vector load also meridional data and merge into 
        # dataset structure
        if do_vec or do_norm or do_pdens:
            pathlist, dum = do_pathlist(year, datapath, do_filename, do_file, vname2, runid)
            data     = xr.merge([data, xr.open_mfdataset(pathlist,  parallel=do_parallel, chunks=chunks, 
                                                         autoclose=False, preprocess=partial_func, **kwargs)])
            if do_vec: is_data='vector'
        
        ## rechunking leads to extended memory demand at runtime of xarray with
        ## dask client!!! --> this here is not a good idea!!!
        #data = data.chunk({'time': data.sizes['time']})
        
    # load restart or blowup files
    else:
        print(pathlist)
        data = xr.open_mfdataset(pathlist, parallel=do_parallel, chunks=chunks, 
                                 autoclose=False, preprocess=partial_func, **kwargs)
        if do_vec or do_norm or do_pdens:
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
    # rename old vertices dimension name to new
    # 'node' --> 'nod2'
    # 'nz_1' --> 'nz1'
    if ('node' in data.dims): data = data.rename_dims({'node':'nod2'})
    if ('nz_1' in data.dims): data = data.rename_dims({'nz_1':'nz1'})
    
    # convert dimensions name from fesom14cmip6 --> fesom2
    if do_f14cmip6: 
        data = data.drop_vars(['lon_bnds','lat_bnds', 'time_bnds'])
        
        if ('ncells' in data.dims  ): data = data.rename_dims({'ncells':'nod2'})
        
        # rename coordinate: depth --> nz, do this first than rename dimension otherwise
        # it triggers a warning message
        if ('depth'  in data.coords): 
            data = data.rename({'depth' :'nz'  })
            if 'nz' not in data.indexes:
                data = data.set_index(nz='nz')
        # rename dimension: depth --> nz
        if ('depth'  in data.dims  ): data = data.swap_dims({'depth': 'nz'})
        
        if ('time' in data.dims) and \
           ('nod2' in data.dims) and \
           ('nz'   in data.dims): data = data.transpose('time', 'nod2', 'nz')
        data = data.unify_chunks()
        
    #___________________________________________________________________________    
    # add depth axes since its not included in restart and blowup files
    # also add weights
    if do_zarithm == 'wmean': do_zweight=True
    data, dim_vert, dim_horz = do_gridinfo_and_weights(mesh, data, do_zweight=do_zweight, do_hweight=do_hweight)
    
    #___________________________________________________________________________
    # years are selected by the files that are open, need to select mon or day 
    # or record 
    data, mon, day, str_ltim = do_select_time(data, mon, day, record, str_ltim)

    # do time arithmetic on data
    if 'time' in data.dims:
        data, str_atim = do_time_arithmetic(data, do_tarithm)
    
    #___________________________________________________________________________
    # make sure datas are alligned in [time, elem, nz] and not [time, nz, elem]
    if 'time' in data.dims:
        if dim_vert is not None: data = data.transpose('time', dim_horz, dim_vert)
    else: 
        if dim_vert is not None: data = data.transpose(dim_horz, dim_vert)

    #___________________________________________________________________________
    # set bottom to nan --> in moment the bottom fill value is zero would be 
    # better to make here a different fill value in the netcdf files !!!
    data = do_setbottomnan(mesh, data, do_nan)
    
    #___________________________________________________________________________
    # select depth levels also for vertical interpolation 
    # found 3d data based mid-depth levels (temp, salt, pressure, ....)
    # if ( ('nz1' in data[vname].dims) or ('nz'  in data[vname].dims) ) and (depth is not None):
    if ( bool(set(['nz1','nz']).intersection(data.dims)) ) and (depth is not None):
        #print('~~ >-))))o> o0O ~~~ A')
        #_______________________________________________________________________
        data, str_ldep = do_select_levidx(data, mesh, depth, depidx)
        
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
                data = data.interp(nz1=auxdepth, method="linear")
                if data['nz1'].size>1: 
                    data = do_depth_arithmetic(data, do_zarithm, "nz1")
                    
            elif ('nz'  in data.dims):    
                data = data.interp(nz=auxdepth, method="linear")
                if data['nz'].size>1:   
                    data = do_depth_arithmetic(data, do_zarithm, "nz") 
    
    #___________________________________________________________________________
    # select all depth levels but do vertical summation over it --> done for 
    # merid heatflux
    elif ( bool(set(['nz1', 'nz']).intersection(data.dims)) ) and (depth is None) and (do_zarithm in ['sum','mean','wmean']): 
        #print('~~ >-))))o> o0O ~~~ B')
        if   ('nz1'  in data.dims): data = do_depth_arithmetic(data, do_zarithm, "nz1" )
        elif ('nz'   in data.dims): data = do_depth_arithmetic(data, do_zarithm, "nz"  )     
    # only 2D data found            
    else:
        #print('~~ >-))))o> o0O ~~~ C')
        depth=None
    
    #___________________________________________________________________________
    # rotate the vectors if do_vecrot=True and do_vec=True
    data = do_vector_rotation(data, mesh, do_vec, do_vecrot)
    
    #___________________________________________________________________________
    # compute norm of the vectors if do_norm=True    
    data = do_vector_norm(data, do_norm)
    
    ##___________________________________________________________________________
    ## compute norm of the vectors if do_norm=True    
    #data = do_rescaling(data, do_rescale)

    #___________________________________________________________________________
    # compute potential density if do_pdens=True    
    if do_pdens and depth is None: 
        data, vname = do_potential_density(data, do_pdens, vname, vname2, vname_tmp)
    
    #___________________________________________________________________________
    # interpolate from elements to node
    if ('elem' in list(data.dims)) and do_ie2n: is_ie2n=True
    data = do_interp_e2n(data, mesh, do_ie2n)
    
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
        data = do_additional_attrs(data, vname, attr_dict)
    
    #___________________________________________________________________________
    warnings.filterwarnings("ignore", category=UserWarning, message="Sending large graph of size")
    warnings.filterwarnings("ignore", category=UserWarning, message="Large object of size \\d+\\.\\d+ detected in task graph")
    if do_compute: data = data.compute()
    if do_load   : data = data.load()
    if do_persist: data = data.persist()
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
    if   do_file=='run'        : fname = '{}.{}.{}.nc'.format(   vname,runid,year)
    elif do_file=='restart_oce': fname = '{}.{}.oce.restart.nc'.format(runid,year)
    elif do_file=='restart_ice': fname = '{}.{}.ice.restart.nc'.format(runid,year)
    elif do_file=='blowup'     : fname = '{}.{}.oce.blowup.nc'.format( runid,year)
    
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
    # specific filename and path is given to load 
    if  do_filename: 
        pathlist = datapath
        if isinstance(datapath, list):
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
                print(f'--> No file: {path}\n')
    
    # a single year is given to load
    elif isinstance(year, int):
        fname = do_fnamemask(do_file,vname,runid,year)
        path  = os.path.join(datapath,fname)
        if os.path.isfile(path):
            pathlist.append(path)  
        else:
            print(f'--> No file: {path}\n')
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
    warnings.filterwarnings("ignore", category=UserWarning, message="Sending large graph of size")

    dimv = None
    if   ('nz1'  in data.dims): 
        dimv = 'nz1'
        if   ('nz1'  in data.coords): data = data.drop_vars('nz1' ) 
        elif ('nz_1' in data.coords): data = data.drop_vars('nz_1') 
        set_chunk = dict({dimv:data.chunksizes[dimv]}) 
        data = data.assign_coords({'nz1': xr.DataArray(-mesh.zmid,                  dims=dimv).astype('float32').chunk(set_chunk) })
        data = data.assign_coords({'nzi': xr.DataArray(np.arange(0,mesh.zmid.size), dims=dimv).astype('uint8'  ).chunk(set_chunk) })
        
    elif ('nz'   in data.dims): 
        dimv = 'nz'
        if ('nz'  in data.coords): data = data.drop_vars('nz' ) 
        set_chunk = dict({dimv:data.chunksizes[dimv]}) 
        data = data.assign_coords(nz   = xr.DataArray(-mesh.zlev,                  dims=dimv).astype('float32').chunk(set_chunk) )
        data = data.assign_coords(nzi  = xr.DataArray(np.arange(0,mesh.zlev.size), dims=dimv).astype('uint8').chunk(set_chunk) )
        
    elif ('ndens'   in data.dims): 
        dimv = 'ndens'
        
    dimh = None
    if   ('nod2' in data.dims):
        dimh = 'nod2'
        if dimh in data.chunksizes: set_chunk = dict({dimh: data.chunksizes[dimh]})
        else                      : set_chunk = dict({})
        
        #_______________________________________________________________________
        # set coordinates
        data = data.assign_coords(lon  = xr.DataArray(mesh.n_x              , dims=dimh).chunk(set_chunk))
        data = data.assign_coords(lat  = xr.DataArray(mesh.n_y              , dims=dimh).chunk(set_chunk))
        data = data.assign_coords(nodi = xr.DataArray(np.arange(0,mesh.n2dn), dims=dimh).astype('int32').chunk(set_chunk))
        if   dimv in ['nz1']: 
            data = data.assign_coords(nodiz = xr.DataArray(mesh.n_iz-1      , dims=dimh).astype('uint8').chunk(set_chunk))
        elif dimv in ['nz']: 
            data = data.assign_coords(nodiz = xr.DataArray(mesh.n_iz        , dims=dimh).astype('uint8').chunk(set_chunk))
        
        #_______________________________________________________________________
        # do horiz weighting for weighted mean computation on nodes
        if do_hweight:
            if   'nz1'  == dimv:
                set_chunk = dict({dimh: data.chunksizes[dimh], dimv:data.chunksizes[dimv]}) 
                w_A = xr.DataArray(mesh.n_area[:-1,:].astype('float32'), dims=[dimv, dimh]).chunk(set_chunk)
            elif 'nz'   == dimv:
                if mesh.n_area.ndim==1: # in case fesom14cmip6 n_area is not depth dependent, therefor ndims=1
                    set_chunk = dict({dimh: data.chunksizes[dimh]}) 
                    w_A = xr.DataArray(mesh.n_area.astype( 'float32'), dims=[        dimh]).chunk(set_chunk)
                else:    
                    set_chunk = dict({dimh: data.chunksizes[dimh], dimv:data.chunksizes[dimv]}) 
                    w_A = xr.DataArray(mesh.n_area.astype('float32'), dims=[dimv, dimh]).chunk(set_chunk)
            else:   
                set_chunk = dict({dimh: data.chunksizes[dimh]}) 
                if mesh.n_area.ndim==1: # in case fesom14cmip6 n_area is not depth dependent, therefor ndims=1
                    w_A = xr.DataArray(mesh.n_area.astype( 'float32'), dims=[        dimh]).chunk(set_chunk)
                else:    
                    w_A = xr.DataArray(mesh.n_area[0, :].astype( 'float32'), dims=[        dimh]).chunk(set_chunk)
            data = data.assign_coords(w_A=w_A)
            del(w_A)
        
        # do vertical weighting/volumen weight
        if do_zweight and dimv is not None:  
            if   'nz1' == dimv:
                set_chunk = dict({dimv:data.chunksizes[dimv]}) 
                #w_An = xr.DataArray(mesh.n_area[:-1,:].astype('float32'), dims=['nz1', 'nod2']).chunk(data.chunksizes['nod2'])
                w_z  = xr.DataArray(mesh.zlev[:-1]-mesh.zlev[1:], dims=dimv).chunk(set_chunk)
                #data = data.assign_coords(w_z=w_z*w_An)
            elif 'nz' == dimv:
                set_chunk = dict({dimv:data.chunksizes[dimv]}) 
                #w_An = xr.DataArray(mesh.n_area.astype('float32'), dims=['nz', 'nod2']).chunk(data.chunksizes['nod2'])
                w_z  = xr.DataArray(np.hstack(((mesh.zlev[0]-mesh.zlev[1])/2.0, mesh.zmid[:-1]-mesh.zmid[1:], (mesh.zlev[-2]-mesh.zlev[-1])/2.0)), dims=dimv).chunk(set_chunk)
            #data = data.drop('w_A')    
            #data = data.assign_coords(w_z=w_An*w_z)
            data = data.assign_coords(w_z=w_z)
            del(w_z)
        
    elif ('elem' in data.dims):                          
        dimh = 'elem'
        if dimh in data.chunksizes: set_chunk = dict({dimh: data.chunksizes[dimh]})
        else                      : set_chunk = dict({})
        
        #_______________________________________________________________________
        # set coordinates
        data = data.assign_coords(lon  = xr.DataArray(mesh.n_x[mesh.e_i].sum(axis=1)/3.0, dims=dimh).chunk(set_chunk))
        data = data.assign_coords(lat  = xr.DataArray(mesh.n_y[mesh.e_i].sum(axis=1)/3.0, dims=dimh).chunk(set_chunk))
        data = data.assign_coords(elemi= xr.DataArray(np.arange(0,mesh.n2de)            , dims=dimh).astype('int32').chunk(set_chunk))
        if   dimv in ['nz1']: 
            data = data.assign_coords(elemiz= xr.DataArray(mesh.e_iz-1                  , dims=dimh).astype('uint8').chunk(set_chunk))
        elif dimv in ['nz']: 
            data = data.assign_coords(elemiz= xr.DataArray(mesh.e_iz                    , dims=dimh).astype('uint8').chunk(set_chunk))
        
        #_______________________________________________________________________
        # do weighting for weighted mean computation on elements
        if do_hweight:
            data = data.assign_coords(w_A  = xr.DataArray(mesh.e_area                   , dims=dimh).chunk(set_chunk))
        
        if do_zweight and dimv is not None:    
            if   'nz1' == dimv:
                set_chunk = dict({dimh: data.chunksizes[dimh], dimv:data.chunksizes[dimv]})
                w_A  = np.zeros((mesh.nlev-1, mesh.n2de))
                for ei in range(0,mesh.n2de): w_A[mesh.e_iz[ei]+1-1:,ei]=np.nan
                w_Ae = xr.DataArray(w_A, dims=[dimv, dimh]).chunk(set_chunk) 
                                               
                set_chunk = dict({dimv:data.chunksizes[dimv]})
                w_z  = xr.DataArray(mesh.zlev[:-1]-mesh.zlev[1:], dims=dimv).chunk(set_chunk)
                
            elif 'nz' == dimv:
                set_chunk = dict({dimh: data.chunksizes[dimh], dimv:data.chunksizes[dimv]})
                w_A  = np.zeros((mesh.nlev, mesh.n2de))
                for ei in range(0,mesh.n2de): w_A[mesh.e_iz[ei]+1:,ei]=np.nan
                w_Ae = xr.DataArray(w_A, dims=[dimv, dimh]).chunk(set_chunk)
                
                set_chunk = dict({dimv:data.chunksizes[dimv]})
                w_z  = xr.DataArray(mesh.zlev[:-1]-mesh.zlev[1:], dims=dimv).chunk(set_chunk)
                
            data = data.assign_coords(w_z=w_z*w_Ae)
            del(w_z, w_Ae, w_A)
    
    #___________________________________________________________________________
    warnings.resetwarnings()
    return(data, dimv, dimh)

    

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
        if   ('nod2' in data.dims):
            #if vname in ['Kv', 'Av']: 
            if   ('nz1'  in data.dims): mat_nodiz= data['nodiz'].expand_dims({'nz1': data['nz1']}).transpose()
            elif ('nz'   in data.dims): mat_nodiz= data['nodiz'].expand_dims({'nz': data['nz']}).transpose()
            mat_nzinod= data['nzi'].expand_dims({'nod2': data['nod2']}).drop_vars('nod2')
            
            # kickout all cooordinates from mat_nodiz and mat_nzinod
            mat_nodiz = mat_nodiz.drop_vars(list(mat_nodiz.coords))
            mat_nzinod= mat_nzinod.drop_vars(list(mat_nzinod.coords))
            
            data = data.where(mat_nzinod<=mat_nodiz)
            del mat_nodiz, mat_nzinod
                
        elif('elem' in data.dims):
            if   ('nz1'  in data.dims): mat_elemiz= data['elemiz'].expand_dims({'nz1': data['nz1']}).transpose()
            elif ('nz'   in data.dims): mat_elemiz= data['elemiz'].expand_dims({'nz': data['nz']}).transpose()
            mat_nzielem= data['nzi'].expand_dims({'elem': data['elemi']}).drop_vars('elem')
            
            # kickout all cooordinates from mat_nzielem
            mat_elemiz = mat_elemiz.drop_vars(list(mat_elemiz.coords))
            mat_nzielem= mat_nzielem.drop_vars(list(mat_nzielem.coords))
            
            data = data.where(mat_nzielem<=mat_elemiz)
            del mat_elemiz, mat_nzielem
        
    #___________________________________________________________________________
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
        if   (mon is not None): sel_mon = np.in1d( data['time.month'], mon)
        if   (day is not None): sel_day = np.in1d( data['time.day']  , day)
        
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
    return(data, mon, day, str_mtim)    



#
#
# ___DO VERTICAL LEVEL SELECTION_______________________________________________
def do_select_levidx(data, mesh, depth, depidx):
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
    if   (depth is None): 
        str_ldep = ''
        return(data, str_ldep)
    
    #___________________________________________________________________________
    # found 3d data based on mid-depth levels (w, Kv,...) --> compute 
    # selection index
    elif ('nz1' in data.dims) and (depth is not None):
        #_______________________________________________________________________
        # compute selection index either single layer of for interpolation 
        ndimax = mesh.n_iz.max()-1
        sel_levidx = do_comp_sel_levidx(-mesh.zmid, depth, depidx, ndimax)
        
        #_______________________________________________________________________        
        # select vertical levels from data
        data = data.isel(nz1=sel_levidx)        
        
    #___________________________________________________________________________
    # found 3d data based on full-depth levels (w, Kv,...) --> compute 
    # selection index
    elif ('nz' in data.dims) and (depth is not None):  
        #_______________________________________________________________________
        # compute selection index either single layer of for interpolation 
        ndimax  = mesh.n_iz.max()
        sel_levidx = do_comp_sel_levidx(-mesh.zlev, depth, depidx, ndimax)
        
        #_______________________________________________________________________        
        # select vertical levels from data
        data = data.isel(nz=sel_levidx)
        
    #___________________________________________________________________________
    # do depth information string
    if (depth is not None):
        if   isinstance(depth,(int, float)):
            str_ldep = ', dep:{}m'.format(str(depth))
        elif isinstance(depth,(list, np.ndarray, range)):   
            if len(depth)>0:
                str_ldep = ', dep:{}-{}m'.format(str(depth[0]), str(depth[-1]))
            else:    
                str_ldep = ', dep:{}-{}m'.format(str(mesh.zlev[0]), str(mesh.zlev[-1]))
    #___________________________________________________________________________
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
            data = data.mean(  dim="time", keep_attrs=True)
        
        elif do_tarithm=='median':
            data = data.median(dim="time", keep_attrs=True)
        
        elif do_tarithm=='std':
            data = data.std(   dim="time", keep_attrs=True) 
        
        elif do_tarithm=='var':
            data = data.var(   dim="time", keep_attrs=True)       
        
        elif do_tarithm=='max':
            data = data.max(   dim="time", keep_attrs=True)
        
        elif do_tarithm=='min':
            data = data.min(   dim="time", keep_attrs=True)  
        
        elif do_tarithm=='sum':
            data = data.sum(   dim="time", keep_attrs=True)    
        
        #_______________________________________________________________________
        # yearly means 
        elif do_tarithm in ['ymean','annual']:
            import datetime
            data     = data.groupby('time.year').mean('time')
            # recreate time axes based on year
            data     = data.rename_dims({'year':'time'})
            
            warnings.filterwarnings("ignore", category=UserWarning, message="Sending large graph of size")
            warnings.filterwarnings("ignore", category=UserWarning, message="Large object of size \\d+\\.\\d+ detected in task graph")
            
            aux_time = xr.cftime_range(start='{:d}-01-01'.format(data.year[1]), periods=len(data['time']), freq='YS')
            data     = data.drop_vars('year')
            data     = data.assign_coords(time=aux_time)
            del(aux_time)
            warnings.resetwarnings()
        
        #_______________________________________________________________________
        # monthly means --> seasonal cycle 
        elif do_tarithm in ['mmean','monthly']:
            import datetime
            data     = data.groupby('time.month').mean('time')
            # recreate time axes based on year
            data     = data.rename_dims({'month':'time'})
            
            warnings.filterwarnings("ignore", category=UserWarning, message="Sending large graph of size")
            warnings.filterwarnings("ignore", category=UserWarning, message="Large object of size \\d+\\.\\d+ detected in task graph")
            
            aux_time = xr.cftime_range(start='0001-01-01', periods=len(data['time']), freq='MS')
            data     = data.drop_vars('month')
            data     = data.assign_coords(time=aux_time)
            del(aux_time)
            warnings.resetwarnings()
        
        #_______________________________________________________________________
        # daily means --> 1...365
        elif do_tarithm in ['dmean','daily']:
            import datetime
            data     = data.groupby('time.day').mean('time')
            # recreate time axes based on year
            data     = data.rename_dims({'day':'time'})
            
            warnings.filterwarnings("ignore", category=UserWarning, message="Sending large graph of size")
            warnings.filterwarnings("ignore", category=UserWarning, message="Large object of size \\d+\\.\\d+ detected in task graph")
            
            aux_time = xr.cftime_range(start='0001-01-01', periods=len(data['time']), freq='DS')
            data     = data.drop_vars('day')
            data     = data.assign_coords(time=aux_time).drop_vars('day')
            del(aux_time)
            warnings.resetwarnings()
        
        elif do_tarithm=='None':
            ...
        
        else:
            raise ValueError(' the time arithmetic of do_tarithm={} is not supported'.format(str(do_tarithm))) 
        
    #___________________________________________________________________________
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
            data = data.mean(  dim=dim_name, keep_attrs=True, skipna=True)
        
        elif do_harithm=='median':
            data = data.median(dim=dim_name, keep_attrs=True, skipna=True)
        
        elif do_harithm=='std':
            data = data.std(   dim=dim_name, keep_attrs=True, skipna=True) 
        
        elif do_harithm=='var':
            data = data.var(   dim=dim_name, keep_attrs=True, skipna=True)       
        
        elif do_harithm=='max':
            data = data.max(   dim=dim_name, keep_attrs=True, skipna=True)
        
        elif do_harithm=='min':
            data = data.min(   dim=dim_name, keep_attrs=True, skipna=True)  
        
        elif do_harithm=='sum':
            data = data.sum(   dim=dim_name, keep_attrs=True, skipna=True)            
        
        elif do_harithm=='wint':
            data    = data*data['w_A']
            data    = data.sum(   dim=dim_name, keep_attrs=True, skipna=True)      
        
        elif do_harithm=='wmean':
            weights = data['w_A']
            data    = data.drop_vars('w_A')
            weights = weights.where(np.isnan(data)==False)
            weights = weights/weights.sum(dim=dim_name, skipna=True)
            data    = data*weights
            del weights
            data    = data.sum(   dim=dim_name, keep_attrs=True, skipna=True)  
            data    = data.where(data!=0)
        
        elif do_harithm=='None' or do_zarithm is None:
            ...
        
        else:
            raise ValueError(' the time arithmetic of do_tarithm={} is not supported'.format(str(do_tarithm))) 
    
    #___________________________________________________________________________
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
            data    = data.mean(dim=dim_name, keep_attrs=True, skipna=True)
        
        elif do_zarithm=='max':
            data    = data.max( dim=dim_name, keep_attrs=True)
        
        elif do_zarithm=='min':
            data    = data.min( dim=dim_name, keep_attrs=True)  
        
        elif do_zarithm=='sum':
            data    = data.sum( dim=dim_name, keep_attrs=True, skipna=True)
        
        elif do_zarithm=='wint':
            data    = data*data['w_z']
            data    = data.sum(   dim=dim_name, keep_attrs=True, skipna=True)          
        
        elif do_zarithm=='wmean':
            if   ('nod2' in data.dims):
                weights = data['w_A']*data['w_z']
            else:    
                weights = data['w_z']
                
            weights = weights/weights.sum(dim=dim_name, skipna=True)
            data    = data*weights
            data    = data.sum( dim=dim_name, keep_attrs=True, skipna=True) 
            del weights
        
        elif do_zarithm=='None' or do_zarithm is None:
            ...
        
        else:
            raise ValueError(' the depth arithmetic of do_zarithm={} is not supported'.format(str(do_zarithm))) 
    #___________________________________________________________________________
    return(data)



#
#
# ___COMPUTE GRID ROTATION OF VECTOR DATA______________________________________
def do_vector_rotation(data, mesh, do_vec, do_vecrot):
    """
    --> compute roration of vector: vname='vec+u+v'
    
    Parameters:
    
        :data:          xarray dataset object
        
        :mesh:          fesom2 mesh object
        
        :do_vec:        bool, should data be considered as vectors
        
        :do_vecrot:     bool, should rotation be applied
    
    Returns:
    
        :data:          xarray dataset object
        
    ____________________________________________________________________________
    """
    if do_vec and do_vecrot:
        # which varaibles are in data, must be two to compute vector rotation
        vname = list(data.keys())
        
        # vector data are on vertices 
        if ('nod2' in data[vname[0]].dims) or ('node' in data[vname[0]].dims):
            print(' > do nod2 vector rotation')
            data[vname[0] ].data,\
            data[vname[1]].data = vec_r2g(mesh.abg, mesh.n_x, mesh.n_y, 
                                        data[vname[0]].data, data[vname[1]].data, 
                                        gridis='geo' )
        
        # vector data are on elements
        if ('elem' in data[vname[0]].dims):
            print(' > do elem vector rotation')
            data[vname[0] ].data,\
            data[vname[1]].data = vec_r2g(mesh.abg, 
                                        mesh.n_x[mesh.e_i].sum(axis=1)/3, 
                                        mesh.n_y[mesh.e_i].sum(axis=1)/3, 
                                        data[vname[0]].data, data[vname[1]].data, 
                                        gridis='geo' )  
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
        print(' > do compute norm')
        # which varaibles are in data, must be two to compute norm
        vname = list(data.keys())
        
        # rename variable vname
        new_vname = 'norm+{}+{}'.format(vname[0],vname[1])
        
        ## estimate chunksize
        #if   'nod2' in data.dims: dim_horz   = 'nod2'            
        #elif 'elem' in data.dims: dim_horz   = 'elem'
        #chunkssize = data.chunksizes[dim_horz]
        
        # compute norm in variable  vname
        #data[vname[0] ].data = np.sqrt(data[vname[0]].data**2 + data[vname[1]].data**2)
        #data[vname[0] ] = np.sqrt(np.square(data[vname[0]]) + np.square(data[vname[1]]))
        #data[new_vname] = np.sqrt(data[vname[0]].data**2 + data[vname[1]].data**2)
        #data[new_vname] = xr.DataArray(np.sqrt(np.square(data[vname[0]].data) + np.square(data[vname[1]].data)), dims=[dim_horz]).chunk(chunkssize)
        
        data[vname[0]] = np.sqrt( np.square(data[vname[0]]) + np.square(data[vname[1]]) )
        
        # rename variable vname[0]
        data = data.rename({vname[0]:new_vname})
        
        # delet variable vname2 from Dataset
        data = data.drop_vars(vname[1])
 
    #___________________________________________________________________________    
    return(data)  


##
##
##
## ___COMPUTE LOGARYTHMIC RESCALING_____________________________________________
#def do_rescaling(data, do_rescale):
    #"""
    #compute vector norm: vname='vec+u+v'
    
    #Parameters: 
    
        #:data:          xarray dataset object
        
        #:do_rescale:    string 'log10'
        
    #Returns:

        #:data:          xarray dataset object
    #____________________________________________________________________________
    #"""
    #if do_rescale=='log10':
        #print(' > do compute log10 rescaling')
        ## which varaibles are in data, must be two to compute norm
        #vname = list(data.keys())[0]
        
        ## compute log10
        ##data[vname] = xr.ufuncs.log10(data[vname])
        #attr_glob = data.attrs        # rescue global attributes --> get lost with xr.where
        #attr_loc  = data[vname].attrs # rescue local  attributes --> get lost with xr.where 
        #data = xr.where(data!=0, xr.ufuncs.log10(data), 0.0)
        #data.attrs        = attr_glob # put back global attributes
        #data[vname].attrs = attr_loc  # put back local  attributes
        
        ## set attribute for rescaling
        #data[vname].attrs['do_rescale'] = 'log10()'
        
    ##___________________________________________________________________________    
    #return(data)  



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
        
        if 'time' in data.dims:
            data_depth = data['nz1'].expand_dims(dict({'time':data.dims['time'], 'nod2':data.dims['nod2']}))
        else:
            data_depth = data['nz1'].expand_dims(dict({'nod2':data.dims['nod2']}))
            
        # data = data.assign({vname_tmp: (list(data[vname].dims), sw.pden(data[vname2].data, data[vname].data, data_depth, pref)-1000.00)})
        data = data.assign({vname_tmp: (list(data[vname].dims), sw.dens(data[vname2].data, data[vname].data, pref)-1000.00)})
        
        del(data_depth)
        
        data[vname_tmp] = data[vname_tmp].where(data[vname2]!=0,drop=0.0)
        
        #data = data.drop(labels=[vname, vname2])
        data = data.drop_vars([vname, vname2])
        data[vname_tmp].attrs['units'] = 'kg/m^3'
        vname = vname_tmp
        
    #___________________________________________________________________________    
    return(data, vname)  



#
#
# ___INTERPOLATE ELEMENTAL DATA TO VERTICES____________________________________
def do_interp_e2n(data, mesh, do_ie2n):
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
        for vname in vname_list:
            # interpolate elem to vertices
            #aux = grid_interp_e2n(mesh,data[vname].data)
            #with np.errstate(divide='ignore',invalid='ignore'):
            aux = grid_interp_e2n(mesh,data[vname].values)
            
            # new variable name 
            vname_new = 'n_'+vname
            
            # add vertice interpolated variable to dataset
            #print(data)
            if   'nz' in data.dims:
                data = xr.merge([ data, xr.Dataset({vname_new: ( ['nod2','nz'],aux)})], combine_attrs="no_conflicts")
            elif 'nz1' in data.dims:
                data = xr.merge([ data, xr.Dataset({vname_new: ( ['nod2','nz1'],aux)})], combine_attrs="no_conflicts")
            else:
                data = xr.merge([ data, xr.Dataset({vname_new: ( 'nod2',aux)})], combine_attrs="no_conflicts")
            # copy attributes from elem to vertice variable 
            data[vname_new].attrs = data[vname].attrs
            
            # delete elem variable from dataset
            data = data.drop_vars(vname)

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
