# Patrick Scholz, 23.01.2018
import numpy as np
import time
import os
import xarray as xr
import seawater as sw
from .sub_mesh import *

# ___LOAD FESOM2 DATA INTO XARRAY DATASET CLASS________________________________
#|                                                                             |
#|           *** LOAD FESOM2 DATA INTO --> XARRAY DATASET CLASS ***            |
#|                                                                             |
#|_____________________________________________________________________________|
def load_data_fesom2(mesh, datapath, vname=None, year=None, mon=None, day=None, 
                     record=None, depth=None, depidx=False, do_nan=True, 
                     do_tarithm='mean', do_zarithm='mean', do_ie2n=True,
                     do_vecrot=True, do_filename=None, do_file='run', do_info=True, 
                     do_compute=True, descript='',  runid='fesom',
                     **kwargs):
    """
    ---> load FESOM2 data:
    ___INPUT:___________________________________________________________________
    mesh        :   fesom2 mesh object,  with all mesh information 
    datapath    :   str, path that leads to the FESOM2 data
    vname       :   str, (default: None), variable name that should be loaded
    year        :   int, list, np.array, range, default: None, single year or 
                    list/array of years whos file should be opened
    mon         :   list, (default=None), specific month that should be selected 
                    from dataset. If mon selection leads to no data selection, 
                    because data maybe annual, selection is set to mon=None
    day         :   list, (default=None), same as mon but for day
    record      :   int,list, (default=None), load specific record number from 
                    dataset. Overwrites effect of mon and sel_day
    depth       :   int, list, np.array, range (default=None). Select single 
                    depth level that will be interpolated or select list of depth 
                    levels that will be interpolated and vertically averaged. If 
                    None all vertical layers in data are loaded
    depidx      :   bool, (default:False) if depth is int and depidx=True, depth
                    index is selected that is closest to depth. No interpolation 
                    will be done
    do_nan      :   bool (default=True), do replace bottom fill values with nan                
    do_tarithm  :   str (default='mean') do time arithmetic on time selection
                    option are: None, 'None', 'mean', 'median', 'std', 'var', 'max'
                    'min', 'sum'
    do_zarithm  :   str (default='mean') do arithmetic on selected vertical layers
                    options are: None, 'None', 'mean', 'max', 'min', 'sum'
    do_ie2n     :   bool (default=True), if data are on elements automatically 
                    interpolates them to vertices --> easier to plot 
    do_vecrot   :   bool (default=True), if vector data are loaded e.g. 
                    vname='vec+u+v' rotates the from rotated frame (in which 
                    they are stored) to geo coordinates
    do_filename :   str, (default=None) load this specific filname string instead
                    of path selection via datapath and year
    do_file     :   str, (default='run'), which data should be loaded options are:
                    'run'-fesom2 simulation files should be load, 'restart_oce'-
                    fesom2 ocean restart file should be loaded, 'restart_ice'-
                    fesom2 ice restart file should be loaded and 'blowup'- fesom2
                    ocean blowup file will be loaded
    do_info     :   bool (defalt=True), print variable info at the end 
    do_compute  :   bool (default=True), do xarray dataset compute() at the end
                    data = data.compute()
    descript    :   str (default=''), string to describe dataset is written into 
                    variable attributes
    runid       :   str (default='fesom'), runid of loaded data                
    ___RETURNS:_________________________________________________________________
    data        :   object, returns xarray dataset object
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
    #___________________________________________________________________________
    # Create xarray dataset object with all grid information 
    if vname in ['depth', 'topo', 'topography','zcoord', 'narea', 'n_area', 'clusterarea', 'scalararea'
                 'nresol', 'n_resol', 'resolution', 'resol', 'earea', 'e_area', 'triarea',
                 'eresol','e_resol','triresolution','triresol']:
        data = xr.Dataset(coords={"lon"  :( "nod2"         ,mesh.n_x), 
                                  "lat"  :( "nod2"         ,mesh.n_y), 
                                  "faces":(["elem","three"],mesh.e_i),
                                  "zlev" :( "nz"           ,mesh.zlev),
                                  "zmid" :( "nz1"          ,mesh.zmid)} )
                                
    #___________________________________________________________________________
    # store topography in data
    if   any(x in vname for x in ['depth','topo','topography','zcoord']):
        data['topo'] = ("nod2", -mesh.n_z)
        data['topo'].attrs["description"]='Depth'
        data['topo'].attrs["long_name"  ]='Depth'
        data['topo'].attrs["units"      ]='m'
        data['topo'].attrs["is_data"    ]=is_data
        return(data)
    # store vertice cluster area in data    
    elif any(x in vname for x in ['narea','n_area','clusterarea','scalararea']):
        if len(mesh.n_area)==0: mesh=mesh.compute_n_area()
        data['narea'] = ("nod2", mesh.n_area)
        data['narea'].attrs["description"]='Vertice area'
        data['narea'].attrs["long_name"  ]='Vertice area'
        data['narea'].attrs["units"      ]='m^2'
        data['narea'].attrs["is_data"    ]=is_data
        return(data)
    # store vertice resolution in data               
    elif any(x in vname for x in ['nresol','n_resol','resolution','resol']):
        if len(mesh.n_resol)==0: mesh=mesh.compute_n_resol()
        data['nresol'] = ("nod2", mesh.n_resol)
        data['nresol'].attrs["description"]='Resolution'
        data['nresol'].attrs["long_name"  ]='Resolution'
        data['nresol'].attrs["units"      ]='m'
        data['nresol'].attrs["is_data"    ]=is_data
        return(data)
    # store element area in data    
    elif any(x in vname for x in ['earea','e_area','triarea']):
        if len(mesh.e_area)==0: mesh=mesh.compute_e_area()
        data['earea'] = ("elem", mesh.e_area)
        data['earea'].attrs["description"]='Element area'
        data['earea'].attrs["long_name"  ]='Element area'
        data['earea'].attrs["units"      ]='m^2'
        data['earea'].attrs["is_data"    ]=is_data
        return(data)
    # store element resolution in data               
    elif any(x in vname for x in ['eresol','e_resol','triresolution','triresol']):
        if len(mesh.e_resol)==0: mesh=mesh.compute_e_resol()
        data['eresol'] = ("elem", mesh.e_resol)
        data['eresol'].attrs["description"]='Element resolution'
        data['eresol'].attrs["long_name"  ]='Element resolution'
        data['eresol'].attrs["units"      ]='m'
        data['eresol'].attrs["is_data"    ]=is_data
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
    pathlist, str_ltim = do_pathlist(year, datapath, do_filename, do_file, vname, runid)
    
    #___________________________________________________________________________
    # load multiple files
    # load normal FESOM2 run file
    if do_file=='run':
        data = xr.open_mfdataset(pathlist, parallel=True, **kwargs) 
        
        # in case of vector load also meridional data and merge into 
        # dataset structure
        if do_vec or do_norm or do_pdens:
            pathlist, dum = do_pathlist(year, datapath, do_filename, do_file, vname2, runid)
            data     = xr.merge([data, xr.open_mfdataset(pathlist,**kwargs)])
            if do_vec: is_data='vector'
            
    # load restart or blowup files
    else:
        print(pathlist)
        data = xr.open_mfdataset(pathlist, parallel=True, **kwargs)
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
        data = data.drop(labels=vname_drop)
        record=0
        
        # add depth axes since its not included in restart and blowup files
        if   ('nz_1' in data.dims): data = data.assign_coords(nz_1=("nz_1",-mesh.zmid))
        if   ('nz1'  in data.dims): data = data.assign_coords(nz1 =("nz1" ,-mesh.zmid))
        if   ('nz'   in data.dims): data = data.assign_coords(nz  =("nz"  ,-mesh.zlev))
    
    #___________________________________________________________________________
    # years are selected by the files that are open, need to select mon or day 
    # or record 
    data, mon, day, str_ltim = do_select_time(data, mon, day, record, str_ltim)
    
    #___________________________________________________________________________
    # set bottom to nan --> in moment the bottom fill value is zero would be 
    # better to make here a different fill value in the netcdf files !!!
    if do_nan and any(x in data.dims for x in ['nz_1','nz1','nz']): 
        data = data.where(data[vname]!=0)
    
    #___________________________________________________________________________
    # select depth levels also for vertical interpolation 
    # found 3d data based mid-depth levels (temp, salt, pressure, ....)
    # if ( ('nz1' in data[vname].dims) or ('nz'  in data[vname].dims) ) and (depth is not None):
    if ( bool(set(['nz1','nz_1','nz']).intersection(data.dims)) ) and (depth is not None):
        #_______________________________________________________________________
        # in old fesom2 output the vertical depth levels are not included --> 
        # add depth axes by hand
        if   ('nz_1' in data.dims) and ('nz_1' not in list(data.coords)): 
            data = data.assign_coords(nz_1=("nz_1",-mesh.zmid))
        elif ('nz1'  in data.dims) and ('nz1'  not in list(data.coords)): 
            data = data.assign_coords(nz1 =("nz1" ,-mesh.zmid))
        elif ('nz'   in data.dims) and ('nz'   not in list(data.coords)): 
            data = data.assign_coords(nz  =("nz"  ,-mesh.zlev))
        
        #_______________________________________________________________________
        data, str_ldep = do_select_levidx(data, mesh, depth, depidx)
        
        #_______________________________________________________________________
        if do_pdens: 
            data, vname = do_potential_desnsity(data, do_pdens, vname, vname2, vname_tmp)
            
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
                    
            if   ('nz_1' in data.dims):
                data = data.interp(nz_1=auxdepth, method="linear")
                if data['nz_1'].size>1: 
                    data = do_depth_arithmetic(data, do_zarithm, "nz_1")
                    
            elif ('nz'  in data.dims):    
                data = data.interp(nz=auxdepth, method="linear")
                if data['nz'].size>1:   
                    data = do_depth_arithmetic(data, do_zarithm, "nz") 
    
    # only 2D data found            
    else:
        depth=None
        
    #___________________________________________________________________________
    # do arithmetic on data
    data, str_atim = do_time_arithmetic(data, do_tarithm)
    
    #___________________________________________________________________________
    # rotate the vectors if do_vecrot=True and do_vec=True
    data = do_vector_rotation(data, mesh, do_vec, do_vecrot)

    #___________________________________________________________________________
    # compute norm of the vectors if do_norm=True    
    data = do_vector_norm(data, do_norm)
    
    #___________________________________________________________________________
    # compute potential density if do_pdens=True    
    if do_pdens and depth is None: 
        data, vname = do_potential_desnsity(data, do_pdens, vname, vname2, vname_tmp)
    
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
    if do_compute: data = data.compute()
    
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


    
# ___CREATE FILENAME MASK FOR: RUN, RESTART AND BLOWUP FILES___________________
#| contains filename mask to distinguish between run, restart and blowup file  |
#| that can be loaded                                                          |
#| ___INPUT_________________________________________________________________   |
#| do_file      :   str, which kind of file should be loaded, 'run',           |
#|                  'restart_oce', 'restart_ice', 'blowup'                     |
#| vname        :   str, name of variable                                      |
#| runid        :   str, runid of simulation usually 'fesom'                   |
#| year         :   int, year number that should be loaded                     |
#| ___RETURNS_______________________________________________________________   |
#| fname        :   str, filename                                              |
#|_____________________________________________________________________________| 
def do_fnamemask(do_file,vname,runid,year):
    #___________________________________________________________________________
    if   do_file=='run'        : fname = '{}.{}.{}.nc'.format(vname,runid,year)
    elif do_file=='restart_oce': fname = '{}.{}.oce.restart.nc'.format(runid,year)
    elif do_file=='restart_ice': fname = '{}.{}.ice.restart.nc'.format(runid,year)
    elif do_file=='blowup'     : fname = '{}.{}.oce.blowup.nc'.format(runid,year)
    
    #___________________________________________________________________________
    return(fname)



# ___CREATE PATHLIST TO DATAFILES______________________________________________
#| create path/file list of data that should be loaded                         |
#| ___INPUT_________________________________________________________________   |
#| year         :   int, list, np.array, range of years that should be loaded  |
#| datapath     :   str, path that leads to the FESOM2 data                    |
#| do_filename  :   str, (default=None) load this specific filname string      |
#|                  instead                                                    |
#| fo_file      :   str, which kind of file should be loaded, 'run',           |
#|                  'restart_oce', 'restart_ice', 'blowup'                     |
#| vname        :   str, name of variable                                      |
#| runid        :   str, runid of simulation usually 'fesom'                   |
#| ___RETURNS_______________________________________________________________   |
#| pathlist     :   str, list                                                  |
#|_____________________________________________________________________________| 
def do_pathlist(year, datapath, do_filename, do_file, vname, runid):
    pathlist=[]
    
    # specific filename and path is given to load 
    if do_filename: 
        pathlist = do_filename
        str_mtim = os.path.basename(do_filename)
        
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
            path = os.path.join(datapath,fname)
            pathlist.append(path)  
    
    # a single year is given to load
    elif isinstance(year, int):
        fname = do_fnamemask(do_file,vname,runid,year)
        pathlist.append(os.path.join(datapath,fname))
        str_mtim = 'y:{}'.format(year)
    else:
        raise ValueError( " year can be integer, list, np.array or range(start,end)")
    
    #___________________________________________________________________________
    return(pathlist,str_mtim)



# ___DO TIME SELECTION_________________________________________________________
#| select specific month, dayy or record number                                |
#| ___INPUT_________________________________________________________________   |
#| data         :   xarray dataset object                                      |
#| mon          :   list, (default=None), specific month that should be        |
#|                  selected from dataset. If mon selection leads to no data   |
#|                  selection,because data maybe annual, selection is set to   |
#|                  mon=None                                                   |
#| day          :   list, (default=None), same as mon but for day              |
#| record       :   int,list, (default=None), load specific record number from |
#|                  dataset. Overwrites effect of mon and sel_day              |
#| ___RETURNS_______________________________________________________________   |
#| data         :   returns xarray dataset object                              |
#|_____________________________________________________________________________|
def do_select_time(data, mon, day, record, str_mtim):
    
    #___________________________________________________________________________
    # select no time use entire yearly file
    if (mon is None) and (day is None) and (record is None):
        return(data, mon, day, str_mtim)
    
    #___________________________________________________________________________
    # select time based on record index --> overwrites mon and day selection        
    elif (record is not None):
        data.isel(time=record)
        
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



# ___DO VERTICAL LEVEL SELECTION_______________________________________________
#| selct vertical levels based on depth list                                   |
#| ___INPUT_________________________________________________________________   |
#| data         :   xarray dataset object                                      |
#| mesh         :   fesom2 mesh object                                         |
#| depth        :   int, list, np.array, range (default=None). Select single   |
#|                  depth level that will be interpolated or select list of    |
#|                  depth levels that will be interpolated and vertically      | 
#|                  averaged. If None all vertical layers in data are loaded   |
#| depidx       :   bool, (default:False) if depth is int and depidx=True,     |
#|                  depth index is selected that is closest to depth. No       |
#|                  interpolation will be done                                 |
#| ___RETURNS_______________________________________________________________   |
#| data         :   xarray dataset object                                      |
#|_____________________________________________________________________________|
def do_select_levidx(data, mesh, depth, depidx):
    
    #___________________________________________________________________________
    # no depth selecetion at all
    if   (depth is None): 
        str_ldep = ''
        return(data, str_ldep)
    
    #___________________________________________________________________________
    # found 3d data based on mid-depth levels (w, Kv,...) --> compute 
    # selection index
    elif (('nz1' in data.dims) or ('nz_1' in data.dims)) and (depth is not None):
        #_______________________________________________________________________
        # compute selection index either single layer of for interpolation 
        ndimax = mesh.n_iz.max()-1
        sel_levidx = do_comp_sel_levidx(-mesh.zmid, depth, depidx, ndimax)
        
        #_______________________________________________________________________        
        # select vertical levels from data
        if ('nz1'  in data.dims): data = data.isel(nz1=sel_levidx)        
        if ('nz_1' in data.dims): data = data.isel(nz_1=sel_levidx)        
        
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
            str_ldep = ', dep:{}-{}m'.format(str(depth[0]), str(depth[-1]))
            
    #___________________________________________________________________________
    return(data, str_ldep)



# ___COMPUTE VERTICAL LEVEL SELECTION INDEX____________________________________
#| compute level indices that are needed to interpolate the depth levels       |                                                                             |
#| ___INPUT_________________________________________________________________   |
#| zlev         :   list, depth vector of the datas                            |
#| depth        :   list, with depth levels that should be interpolated        |
#| depidx       :   bool, (default:False) if depth is int and depidx=True,     |
#|                  depth index is selected that is closest to depth. No       |
#|                  interpolation will be done                                 |
#| ndimax       :   int, maximum number of levels  mesh.n_iz.max()-1 for mid   | 
#|                  depth datas, mesh.n_iz.max() for full level data           |              
#| ___RETURNS_______________________________________________________________   |
#| sel_levidx   : list, with level indices that should be extracted            |
#|_____________________________________________________________________________|
def do_comp_sel_levidx(zlev, depth, depidx, ndimax):
    
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
    
    
    
# ___COMPUTE TIME ARITHMETICS ON DATA__________________________________________
#| do arithmetic on time dimension                                             |                                                                            |
#| ___INPUT_________________________________________________________________   |
#| data         :   xarray dataset object                                      |
#| do_tarithm   :   str (default='mean') do time arithmetic on time selection  |
#|                  option are: None, 'None', 'mean', 'median', 'std', 'var',  |
#|                  'max', 'min', 'sum'                                        |
#| ___RETURNS_______________________________________________________________   |
#| data         :   xarray dataset object                                      |
#|_____________________________________________________________________________|    
def do_time_arithmetic(data, do_tarithm):
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
        elif do_tarithm=='None':
            ...
        else:
            raise ValueError(' the time arithmetic of do_tarithm={} is not supported'.format(str(do_tarithm))) 
        
    #___________________________________________________________________________
    return(data, str_atim)



# ___COMPUTE DEPTH ARITHMETICS ON DATA_________________________________________
#| do arithmetic on depth dimension                                            |
#| ___INPUT_________________________________________________________________   |
#| data         :   xarray dataset object                                      |
#| do_zarithm   :   str (default='mean') do arithmetic on selected vertical    |
#|                  layers options are: None, 'None', 'mean', 'max', 'min',    |
#|                  'sum'                                                      |
#| dim_name     :   str, name of depth dimension, is different for full-level  |
#|                  and mid-level data                                         |
#| ___RETURNS_______________________________________________________________   |
#| data         :   xarray dataset object                                      |
#|_____________________________________________________________________________|    
def do_depth_arithmetic(data, do_zarithm, dim_name):
    if do_zarithm is not None:
        
        #_______________________________________________________________________
        if   do_zarithm=='mean':
            data = data.mean(  dim=dim_name, keep_attrs=True, skipna=True)
        elif do_zarithm=='max':
            data = data.max(   dim=dim_name, keep_attrs=True)
        elif do_zarithm=='min':
            data = data.min(   dim=dim_name, keep_attrs=True)  
        elif do_zarithm=='sum':
            data = data.sum(   dim=dim_name, keep_attrs=True, skipna=True)      
        elif do_zarithm=='None':
            ...
        else:
            raise ValueError(' the depth arithmetic of do_zarithm={} is not supported'.format(str(do_zarithm))) 
    #___________________________________________________________________________
    return(data)



# ___COMPUTE GRID ROTATION OF VECTOR DATA______________________________________
#| compute roration of vector: vname='vec+u+v'                                 |
#| ___INPUT_________________________________________________________________   |
#| data         :   xarray dataset object                                      |
#| mesh         :   fesom2 mesh object                                         |
#| do_vec       :   bool, should data be considered as vectors                 |
#| do_vecrot    :   bool, should rotation be applied                           |
#| ___RETURNS_______________________________________________________________   |
#| data         :   xarray dataset object                                      |
#|_____________________________________________________________________________|  
def do_vector_rotation(data, mesh, do_vec, do_vecrot):
    if do_vec and do_vecrot:
        # which varaibles are in data, must be two to compute vector rotation
        vname = list(data.keys())
        
        # vector data are on vertices 
        if ('nod2' in data[vname[0]].dims) or ('node' in data[vname[0]].dims):
            print(' > do vector rotation')
            data[vname[0] ].data,\
            data[vname[1]].data = vec_r2g(mesh.abg, mesh.n_x, mesh.n_y, 
                                        data[vname[0]].data, data[vname[1]].data, 
                                        gridis='geo' )
        
        # vector data are on elements
        if ('elem' in data[vname[0]].dims):
            print(' > do vector rotation')
            data[vname[0] ].data,\
            data[vname[1]].data = vec_r2g(mesh.abg, 
                                        mesh.n_x[mesh.e_i].sum(axis=1)/3, 
                                        mesh.n_y[mesh.e_i].sum(axis=1)/3, 
                                        data[vname[0]].data, data[vname[1]].data, 
                                        gridis='geo' )  
    #___________________________________________________________________________
    return(data)



# ___COMPUTE NORM OF VECTOR DATA_______________________________________________
#| compute vector norm: vname='vec+u+v'                                        |
#| ___INPUT_________________________________________________________________   |
#| data         :   xarray dataset object                                      |
#| do_norm      :   bool, should vector norm be computed                       |
#| ___RETURNS_______________________________________________________________   |
#| data         :   xarray dataset object                                      |
#|_____________________________________________________________________________|  
def do_vector_norm(data, do_norm):
    if do_norm:
        print(' > do compute norm')
        # which varaibles are in data, must be two to compute norm
        vname = list(data.keys())
        
        # compute norm in variable  vname
        data[vname[0] ].data = np.sqrt(data[vname[0]].data**2 + data[vname[1]].data**2)
        
        # delte variable vname2 from Dataset
        data      = data.drop(labels=vname[1])
         
        # rename variable vname
        new_vname = 'norm+{}+{}'.format(vname[0],vname[1])
        #new_vname = 'sqrt({}²+{}²)'.format(vname,vname2)
        data      = data.rename({vname[0]:new_vname})
        
    #___________________________________________________________________________    
    return(data)  


# ___COMPUTE NORM OF VECTOR DATA_______________________________________________
#| compute vector norm: vname='vec+u+v'                                        |
#| ___INPUT_________________________________________________________________   |
#| data         :   xarray dataset object                                      |
#| do_norm      :   bool, should vector norm be computed                       |
#| ___RETURNS_______________________________________________________________   |
#| data         :   xarray dataset object                                      |
#|_____________________________________________________________________________|  
def do_potential_desnsity(data, do_pdens, vname, vname2, vname_tmp):
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
        data = data.assign({vname_tmp: (list(data.dims), sw.pden(data[vname2].data, data[vname].data, data_depth, pref)-1000.025)})
        del(data_depth)
        data = data.drop(labels=[vname, vname2])
        data[vname_tmp].attrs['units'] = 'kg/m^3'
        vname = vname_tmp
        
    #___________________________________________________________________________    
    return(data, vname)  




# ___INTERPOLATE ELEMENTAL DATA TO VERTICES____________________________________
#| interpolate data on elements to vertices                                    |
#| ___INPUT_________________________________________________________________   |
#| data         :   xarray dataset object                                      |
#| mesh         :   fesom2 mesh object                                         |
#  do_ie2n      :   bool, True/False if interpolation should be applied        |
#| ___RETURNS_______________________________________________________________   |
#| data         :   xarray dataset object                                      |
#|_____________________________________________________________________________|  
def do_interp_e2n(data, mesh, do_ie2n):
    
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
            data = xr.merge([ data, xr.Dataset({vname_new: ( 'nod2',aux)}) ])
            
            # copy attributes from elem to vertice variable 
            data[vname_new].attrs = data[vname].attrs
            
            # delete elem variable from dataset
            data = data.drop(labels=vname)
    
    #___________________________________________________________________________
    return(data)



# ___PUT ADDITIONAL VARIABLE INFORMATION INTO ATTRIBUTES_______________________
#| write additional information to variable attributes                         |
#| ___INPUT_________________________________________________________________   |
#| data         :   xarray dataset object                                      |
#| vname        :   str, (default: None), variable name that should be loaded  |
#| attr_dict    :   dict with different infos that are written to attributes   |
#| ___RETURNS_______________________________________________________________   |
#| data         :   xarray dataset object                                      |
#|_____________________________________________________________________________|  
#def do_additional_attrs(data, vname, datapath, do_file, do_filename, 
                        #year, mon, day, record, depth, str_mdep, depidx,  
                        #do_tarithm, is_data, is_ie2n, do_compute, descript):
def do_additional_attrs(data, vname, attr_dict):

    #___________________________________________________________________________
    for key in attr_dict:
        data[vname].attrs[key] = str(attr_dict[key])
        
    if 'description' not in data[vname].attrs.keys():
        data[vname].attrs['description'] = str(vname)
    if 'long_name'   not in data[vname].attrs.keys():
        data[vname].attrs['long_name']   = str(data[vname].attrs['description'])
    
    #___________________________________________________________________________
    return(data)



# ___DO ANOMALY________________________________________________________________
#| compute anomaly between two xarray Datasets                                 |
#| ___INPUT_________________________________________________________________   |
#| data1        :   xarray dataset object                                      |
#| data2        :   xarray dataset object                                      |
#| ___RETURNS_______________________________________________________________   |
#| anom         :   xarray dataset object, data1-data2                         |
#|_____________________________________________________________________________|
def do_anomaly(data1,data2):
    
    # copy datasets object 
    anom = data1.copy()
    
    data1_vname = list(data1.keys())
    data2_vname = list(data2.keys())
    #for vname in list(anom.keys()):
    for vname, vname2 in zip(data1_vname, data2_vname):
        # do anomalous data 
        anom[vname].data = data1[vname].data - data2[vname2].data
        
        # do anomalous attributes 
        attrs_data1 = data1[vname].attrs
        attrs_data2 = data2[vname2].attrs
        
        for key in attrs_data1.keys():
            if (key in attrs_data1.keys()) and (key in attrs_data2.keys()):
                if key in ['long_name']:
                   anom[vname].attrs[key] = 'anomalous '+anom[vname].attrs[key] 
                elif key in ['units',]: 
                    continue
                elif data1[vname].attrs[key] != data2[vname2].attrs[key]:
                    anom[vname].attrs[key]  = data1[vname].attrs[key]+' - '+data2[vname2].attrs[key]
    
    #___________________________________________________________________________
    return(anom)
