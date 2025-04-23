# Patrick, Scholz 02.09.2018
import numpy as np
import time as clock
import os
from netCDF4 import Dataset
import xarray as xr
import matplotlib
matplotlib.rcParams['contour.negative_linestyle']= 'solid'
import matplotlib.pyplot as plt
#import matplotlib.patches as Polygon
#import matplotlib.path as mpltPath
#from matplotlib.tri import Triangulation
from numba import jit, njit, prange
import shapefile as shp
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import interp1d
from numpy.matlib import repmat
from scipy import interpolate
import numpy.ma as ma

from .sub_colormap import *
from .sub_utility  import *
from .sub_plot     import *
import warnings

# ___CALCULATE MERIDIONAL OVERTURNING FROM VERTICAL VELOCITIES_________________
#|                                                                             |
#|                                                                             |
#|_____________________________________________________________________________|
def calc_zmoc(mesh, 
              data, 
              dlat          = 1.0       , 
              which_moc     = 'gmoc'    , 
              do_onelem     = False     , 
              diagpath      = None      , 
              do_checkbasin = False     , 
              do_parallel   = False     , 
              n_workers     = 10        , 
              do_compute    = False     , 
              do_load       = True      , 
              do_persist    = False     , 
              do_info       = True      , 
              **kwargs                  , 
             ):
    """
    --> calculate meridional overturning circulation from vertical velocities 
        (Pseudostreamfunction) either on vertices or elements
    
    Parameters:
    
        :mesh:          fesom2 tripyview mesh object,  with all mesh information 
        
        :data:          xarray dataset object with 3d vertical velocities
        
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
                        
        :do_onelem:     bool (default=False) ... should computation be done on 
                        elements or vertices
        
        :diagpath:      str (default=None) if str give custom path to specific fesom2
                        fesom.mesh.diag.nc file, if None routine looks automatically in    
                        meshfolder and original datapath folder (stored as attribute in)
                        xarray dataset object 
        
        :do_checkbasin: bool (default=False) provide plot with regional basin selection
        
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
    
        :zmoc:          object, returns xarray dataset object with MOC
        
    ::
    
        data_list = list()
        
        data = tpv.load_data_fesom2(mesh, datapath, vname='w', year=year, descript=descript, 
                                    do_zarithm=None, do_nan=False, do_load=False, do_persist=True)
        
        zmoc = tpv.calc_zmoc(mesh, data, dlat=1.0, which_moc='amoc')
        
        data_list.append( zmoc )
    
    ____________________________________________________________________________
    """
    
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
    if do_onelem:
        idxin = xr.DataArray(calc_basindomain_fast(mesh, which_moc=which_moc, do_onelem=do_onelem), dims='elem')
    else:
        idxin = xr.DataArray(calc_basindomain_fast(mesh, which_moc=which_moc, do_onelem=do_onelem), dims='nod2')
    
    #___________________________________________________________________________
    if do_checkbasin:
        from matplotlib.tri import Triangulation
        tri = Triangulation(np.hstack((mesh.n_x,mesh.n_xa)), np.hstack((mesh.n_y,mesh.n_ya)), np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia)))
        plt.figure()
        plt.triplot(tri, color='k')
        if do_onelem:
            plt.triplot(tri.x, tri.y, tri.triangles[ np.hstack((idxin[mesh.e_pbnd_0], idxin[mesh.e_pbnd_a])) ,:], color='r')
        else:
            plt.plot(mesh.n_x[idxin], mesh.n_y[idxin], 'or', linestyle='None', markersize=1)
        plt.title('Basin selection')
        plt.show()
    
    #___________________________________________________________________________
    # rescue global and variable attributes
    vname = list(data.keys())[0]
    gattr = data.attrs
    vattr = data[vname].attrs
    
    #___________________________________________________________________________
    # do moc calculation either on nodes or on elements        
    # keep in mind that node area info is changing over depth--> therefor load from file 
    if diagpath is None:
        fname = data['w'].attrs['runid']+'.mesh.diag.nc'
        
        if   os.path.isfile( os.path.join(data['w'].attrs['datapath'], fname) ): 
            dname = data['w'].attrs['datapath']
        elif os.path.isfile( os.path.join( os.path.join(os.path.dirname(os.path.normpath(data['w'].attrs['datapath'])),'1/'), fname) ): 
            dname = os.path.join(os.path.dirname(os.path.normpath(data['w'].attrs['datapath'])),'1/')
        elif os.path.isfile( os.path.join(mesh.path,fname) ): 
            dname = mesh.path
        else:
            raise ValueError('could not find directory with...mesh.diag.nc file')
        
        diagpath = os.path.join(dname,fname)
        if do_info: print(' --> found diag in directory:{}', diagpath)
        
    #___________________________________________________________________________
    # compute area weighted vertical velocities on elements
    if do_onelem:
        #_______________________________________________________________________
        edims = dict()
        dtime, delem, dnz = 'None', 'elem', 'nz'
        if 'time' in list(data.dims): dtime = 'time'
        
        #_______________________________________________________________________
        # load elem area from diag file
        if ( os.path.isfile(diagpath)):
            nz_w_A = xr.open_mfdataset(diagpath, parallel=True, **kwargs)['elem_area']#.chunk({'elem':1e4})
            if 'elem_n' in list(nz_w_A.dims): nz_w_A = nz_w_A.rename({'elem_n':'elem'})
            if 'nl'     in list(nz_w_A.dims): nz_w_A = nz_w_A.rename({'nl'    :'nz'  })
            if 'nl1'    in list(nz_w_A.dims): nz_w_A = nz_w_A.rename({'nl1'   :'nz1' }) 
            
        else: 
            if len(mesh.e_area)>0:
                nz_w_A = xr.DataArray(mesh.e_area, dims=['elem'])
            else:    
                raise ValueError('could not find ...mesh.diag.nc file or the variable mesh.e_area')
            
        nz_w_A = nz_w_A.isel( elem=xr.DataArray(idxin, dims=['elem']) )
        
        #_______________________________________________________________________    
        # average from vertices towards elements
        e_i = xr.DataArray(mesh.e_i, dims=["elem",'n3'])
        if 'time' in list(data.dims): 
            data = data.assign(w=data['w'][:, e_i, :].sum(dim="n3", keep_attrs=True)/3.0 )
        else:  
            data = data.assign(w=data['w'][   e_i, :].sum(dim="n3", keep_attrs=True)/3.0 )
        
        # drop un-necessary variables 
        for vdrop in ['lon', 'lat', 'nodi', 'nodiz', 'w_A']:
            if vdrop in list(data.coords): data = data.drop(vdrop)
        #data = data.drop(['lon', 'lat', 'nodi', 'nodiz', 'w_A'])    
        data = data.assign_coords(elemiz= xr.DataArray(mesh.e_iz, dims=['elem']))
        data = data.assign_coords(elemi = xr.DataArray(np.arange(0,mesh.n2de), dims=['elem']))
        
        #_______________________________________________________________________    
        # select MOC basin 
        data = data.isel(elem=idxin)
        
        #_______________________________________________________________________    
        # enforce bottom topography --> !!! important otherwise results will look 
        # weired
        mat_elemiz = data['elemiz'].expand_dims({'nz': data['nzi']}).transpose()
        mat_nzielem= data['nzi'].expand_dims({'elem': data['elemi']})
        data = data.where(mat_nzielem.data<mat_elemiz.data)
        del(mat_elemiz, mat_nzielem)
        
        #_______________________________________________________________________
        # calculate area weighted mean
        data = data.transpose(dtime, dnz, delem, missing_dims='ignore') * nz_w_A * 1e-6
        data = data.transpose(dtime, delem, dnz, missing_dims='ignore')
        data = data.fillna(0.0)
        del(nz_w_A)
        
        #_______________________________________________________________________
        # create meridional bins --> this trick is from Nils Brückemann (ICON)
        lat     = mesh.n_y[mesh.e_i].sum(axis=1)/3.0
        lat_bin = xr.DataArray(data=np.round(lat[idxin]/dlat)*dlat, dims='elem', name='lat')  
        
    #___________________________________________________________________________
    # compute area weighted vertical velocities on vertices
    else:    
        #_______________________________________________________________________
        # load vertice cluster area from diag file
        if 'w_A' not in list(data.coords):
            if ( os.path.isfile(diagpath)):
                nz_w_A = xr.open_mfdataset(diagpath, parallel=True, **kwargs)['nod_area']#.chunk({'nod2':1e4})
                if 'nod_n' in list(nz_w_A.dims): nz_w_A = nz_w_A.rename({'nod_n':'nod2'})
                if 'nl'    in list(nz_w_A.dims): nz_w_A = nz_w_A.rename({'nl'   :'nz'  })
                if 'nl1'   in list(nz_w_A.dims): nz_w_A = nz_w_A.rename({'nl1'  :'nz1' })
                # you need to drop here the coordinates for nz since they go from 
                # 0...-6000 the coordinates of nz in the data go from 0...6000 that 
                # causes otherwise troubles
                nz_w_A = nz_w_A.drop_vars(['nz'])
            else: 
                if len(mesh.n_area)>0:
                    if mesh.n_area.ndim == 1:
                        # this here is only for the special case when fesom14-CMIP6
                        # data are load to compute the MOC
                        nz_w_A = xr.DataArray(mesh.n_area, dims=['nod2'])
                    else: 
                        # normal case when fesom2 data are used!!!
                        nz_w_A = xr.DataArray(mesh.n_area, dims=['nz','nod2'])
                else: 
                    raise ValueError('could not find ...mesh.diag.nc file')
            
            if any(data.chunks.values()): nz_w_A=nz_w_A.chunk(data.chunksizes)
            data = data.assign_coords(w_A=nz_w_A)
            del(nz_w_A)
        
        #_______________________________________________________________________    
        # select MOC basin
        data = data.isel(nod2=idxin)
        
        #_______________________________________________________________________
        # calculate area weighted mean
        warnings.filterwarnings("ignore", category=UserWarning, message="Sending large graph of size")
        warnings.filterwarnings("ignore", category=UserWarning, message="Large object of size")
        data = data[vname]*data['w_A']*1e-6
        data = data.fillna(0.0)
        data = data.load()
        #t3 = clock.time()
        #print('  --> comp. area weighted mean', t3-t2)
        #_______________________________________________________________________
        # create meridional bins --> this trick is from Nils Brückemann (ICON)
        lat_bin = xr.DataArray(data=np.round(data.lat/dlat)*dlat, dims='nod2', name='lat')
        lat     = np.arange(lat_bin.data.min(), lat_bin.data.max()+dlat, dlat)
        warnings.resetwarnings()
        #t4 = clock.time()
        #print('  --> comp. lat_bin', t4-t3)
        
    #___________________________________________________________________________
    # create ZMOC xarray Dataset
    # define variable attributes    
    if   which_moc_name=='gmoc' : str_region='Global '
    elif which_moc_name=='amoc' : str_region='Atlantic '
    elif which_moc_name=='aamoc': str_region='Atlantic-Arctic '
    elif which_moc_name=='pmoc' : str_region='Pacific '
    elif which_moc_name=='ipmoc': str_region='Indo-Pacific '
    elif which_moc_name=='imoc' : str_region='Indo '
    elif isinstance(which_moc,shp.Reader): str_region=which_moc_region
    
    # for the choice of vertical plotting mode
    gattr['proj'         ]= 'zmoc'
    
    vattr['long_name'    ]= which_moc_name.upper()
    vattr['short_name'   ]= which_moc_name.upper()
    vattr['standard_name']= str_region+'Meridional Overturning Circulation'
    vattr['description'  ]= str_region+'Meridional Overturning Circulation Streamfunction, positive: clockwise, negative: counter-clockwise circulation', 
    vattr['units'        ]= 'Sv'
    # define data_vars dict, coordinate dict, as well as list of dimension name 
    # and size 
    data_vars, coords, dim_n, dim_s,  = dict(), dict(), list(), list()
    if 'time' in list(data.dims): dim_n.append('time')
    if 'nz1'  in list(data.dims): dim_n.append('nz1')
    if 'nz'   in list(data.dims): dim_n.append('nz')
    dim_n.append('lat')
    for dim_ni in dim_n:
        if   dim_ni=='time': dim_s.append(data.sizes['time']); coords['time' ]=(['time'], data['time'].data ) 
        elif dim_ni=='lat' : dim_s.append(lat.size          ); coords['lat'  ]=(['lat' ], lat          ) 
        elif dim_ni=='nz1'  : dim_s.append(data.sizes['nz1'] ); coords['depth']=(['nz1'  ], data['nz1' ].data )
        elif dim_ni=='nz'   : dim_s.append(data.sizes['nz' ] ); coords['depth']=(['nz'   ], data['nz'  ].data ) 
    data_vars['zmoc'] = (dim_n, np.zeros(dim_s, dtype='float32'), vattr) 
    # create dataset
    zmoc = xr.Dataset(data_vars=data_vars, coords=coords, attrs=gattr)
    
    #___________________________________________________________________________
    # define subroutine for binning over latitudes, allows for parallelisation
    def moc_over_lat(lat_i, lat_bin, data):
        #_______________________________________________________________________
        # compute which vertice is within the latitudinal bin
        # --> groupby is here a factor 5-6 slower than using isel+np.where
        data_latbin = data.isel(nod2=np.where(lat_bin==lat_i)[0])
        data_latbin = data_latbin.sum(dim='nod2', skipna=True)
        return(data_latbin)
    
    #___________________________________________________________________________
    # do serial loop over latitudinal bins
    if not do_parallel:
        if do_info: print('\n ___loop over latitudinal bins___'+'_'*90, end='\n')
        for iy, lat_i in enumerate(lat):
            if 'time' in data.dims: zmoc['zmoc'][:,:,iy] = moc_over_lat(lat_i, lat_bin, data)
            else                  : zmoc['zmoc'][  :,iy] = moc_over_lat(lat_i, lat_bin, data)
    
    # do parallel loop over latitudinal bins
    else:
        if do_info: print('\n ___parallel loop over longitudinal bins___'+'_'*1, end='\n')
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_workers)(delayed(moc_over_lat)(lat_i, lat_bin, data) for lat_i in zmoc.lat)
        if 'time' in data.dims: zmoc['zmoc'][:,:,:] = xr.concat(results, dim='nlat').transpose('time','nz','lat')
        else                  : zmoc['zmoc'][  :,:] = xr.concat(results, dim='nlat').transpose('nz','lat')
    
    del(data)
    
    ##___________________________________________________________________________
    ## group data by bins --> this trick is from Nils Brückemann (ICON)
    #if do_info==True: print(' --> do binning of latitudes')
    #data    = data.rename_vars({'w':'zmoc', 'nz':'depth'})
    #data    = data.groupby(lat_bin)
    #t5 = clock.time()
    #print('  --> comp. bins', t5-t4)

    ## zonal sumation/integration over bins
    #if do_info==True: print(' --> do sumation/integration over bins')
    #data    = data.sum().load()
    #t6 = clock.time()
    #print('  --> comp. zonal int', t6-t5)
    
    ## transpose data from [lat x nz] --> [nz x lat]
    #dtime, dhz, dnz = 'None', 'lat', 'nz'
    #if 'time' in list(data.dims): dtime = 'time'
    #data = data.transpose(dtime, dnz, dhz, missing_dims='ignore')
    
    #___________________________________________________________________________
    # cumulative sum over latitudes
    if do_info==True: print(' --> do cumsum over latitudes')
    zmoc['zmoc'] = -zmoc['zmoc'].reindex(lat=zmoc['lat'][::-1]).cumsum(dim='lat', skipna=True).reindex(lat=zmoc['lat'])
    
    #___________________________________________________________________________
    # compute depth of max and nice bottom topography
    if do_onelem: zmoc = calc_bottom_patch(zmoc, lat_bin, xr.DataArray(mesh.e_iz, dims=['elem']), idxin)        
    else        : zmoc = calc_bottom_patch(zmoc, lat_bin, xr.DataArray(mesh.n_iz, dims=['nod2']), idxin)
    
    #___________________________________________________________________________
    if do_compute: zmoc = zmoc.compute()
    if do_load   : zmoc = zmoc.load()
    if do_persist: zmoc = zmoc.persist()
        
    #___________________________________________________________________________
    # write some infos 
    if do_info==True: 
        print(' --> total time:{:.3f} s'.format(clock.time()-t1))
        if 'time' not in list(zmoc.dims):
            if which_moc_name in ['amoc', 'aamoc', 'gmoc']:
                maxv = zmoc.isel(nz=zmoc['depth']>= 700 , lat=zmoc['lat']>0.0)['zmoc'].max().values
                minv = zmoc.isel(nz=zmoc['depth']>= 2500, lat=zmoc['lat']>-50.0)['zmoc'].min().values
                print(' max. NADW_{:s} = {:.2f} Sv'.format(zmoc['zmoc'].attrs['descript'],maxv))
                print(' max. AABW_{:s} = {:.2f} Sv'.format(zmoc['zmoc'].attrs['descript'],minv))
            elif which_moc_name in ['pmoc', 'ipmoc']:
                minv = zmoc['zmoc'].isel(nz=zmoc['depth']>= 2000, lat=zmoc['lat']>-50.0)['moc'].min().values
                print(' max. AABW_{:s} = {:.2f} Sv'.format(zmoc['zmoc'].attrs['descript'],minv))
    
    #___________________________________________________________________________
    return(zmoc)


# ___CALCULATE MERIDIONAL OVERTURNING FROM VERTICAL VELOCITIES_________________
#|                                                                             |
#|                                                                             |
#|_____________________________________________________________________________|
def calc_zmoc_dask( mesh                      , 
                    data                      , 
                    do_parallel               , 
                    parallel_nprc             ,
                    dlat          = 1.0       ,
                    which_moc     = 'gmoc'    , 
                    diagpath      = None      , 
                    do_checkbasin = False     , 
                    do_exclude    = False     , 
                    exclude_list  = list(['ocean_basins/Mediterranean_Basin.shp', [26,42,39.5,47]])   ,
                    do_info       = True      , 
                    do_persist    = True      ,
                    **kwargs                  , 
                    ):
    """
    --> calculate meridional overturning circulation from vertical velocities 
        (Pseudostreamfunction) either on vertices or elements
    
    Parameters:
    
        :mesh:          fesom2 tripyview mesh object,  with all mesh information 
        
        :data:          xarray dataset object with 3d vertical velocities
        
        :do_parallel:   bool, (default=False) is a dask client running
        
        :parallel_nprc: int, (default=64), number of parallel processes
        
        :data:          xarray dataset object with 3d vertical velocities
        
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
                        
        :diagpath       str (default=None) if str give custom path to specific fesom2
                        fesom.mesh.diag.nc file, if None routine looks automatically in    
                        meshfolder and original datapath folder (stored as attribute in)
                        xarray dataset object 
        
        :do_checkbasin: bool (default=False) provide plot with regional basin selection
        
        :do_parallel:   bool (default=False) do computation of binning based MOC 
                        in parallel
        
        :n_workers:     int (default=10) how many worker (CPUs) are used for the 
                        parallelized MOC computation
                        
                                              
        :do_info:       bool (defalt=True), print variable info at the end 
    
    Returns:
    
        :zmoc:          object, returns xarray dataset object with MOC
        
    ::
    
        data_list = list()
        
        data = tpv.load_data_fesom2(mesh, datapath, vname='w', year=year, descript=descript, 
                                    do_zarithm=None, do_nan=False, do_load=False, do_persist=True)
        
        zmoc = tpv.calc_zmoc(mesh, data, dlat=1.0, which_moc='amoc')
        
        data_list.append( zmoc )
    
    ____________________________________________________________________________
    """
    
    #___________________________________________________________________________
    #t1=clock.time()
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
    idxin = xr.DataArray(calc_basindomain_fast(mesh, 
                                               which_moc    = which_moc, 
                                               do_onelem    = False, 
                                               do_exclude   = do_exclude, 
                                               exclude_list = exclude_list), dims='nod2')
    
    #___________________________________________________________________________
    if do_checkbasin:
        from matplotlib.tri import Triangulation
        tri = Triangulation(np.hstack((mesh.n_x,mesh.n_xa)), np.hstack((mesh.n_y,mesh.n_ya)), np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia)))
        plt.figure()
        plt.triplot(tri, color='k')
        plt.plot(mesh.n_x[idxin], mesh.n_y[idxin], 'or', linestyle='None', markersize=1)
        plt.title('Basin selection')
        plt.show()
    
    #___________________________________________________________________________
    # rescue global and variable attributes
    vname = list(data.keys())[0]
    gattr = data.attrs
    vattr = data[vname].attrs
    dimn_h, dimn_v = 'dum', 'dum'
    if   ('nod2' in data.dims): dimn_h = 'nod2'
    elif ('elem' in data.dims): dimn_h = 'elem'
    if   'nz'  in list(data[vname].dims): 
        dimn_v  = 'nz'    
    elif 'nz1' in list(data[vname].dims) or 'nz_1' in list(data[vname].dims): 
        dimn_v  = 'nz1'
        
    #___________________________________________________________________________
    # do moc calculation either on nodes or on elements        
    # keep in mind that node area info is changing over depth--> therefor load from file 
    if diagpath is None:
        fname = data['w'].attrs['runid']+'.mesh.diag.nc'
        
        if   os.path.isfile( os.path.join(data['w'].attrs['datapath'], fname) ): 
            dname = data['w'].attrs['datapath']
        elif os.path.isfile( os.path.join( os.path.join(os.path.dirname(os.path.normpath(data['w'].attrs['datapath'])),'1/'), fname) ): 
            dname = os.path.join(os.path.dirname(os.path.normpath(data['w'].attrs['datapath'])),'1/')
        elif os.path.isfile( os.path.join(mesh.path,fname) ): 
            dname = mesh.path
        else:
            raise ValueError('could not find directory with...mesh.diag.nc file')
        
        diagpath = os.path.join(dname,fname)
        if do_info: print(' --> found diag in directory:{}', diagpath)
        
    
    #___________________________________________________________________________
    # load vertice cluster area from diag file
    if 'w_A' not in list(data.coords):
        if ( os.path.isfile(diagpath)):
            nz_w_A = xr.open_mfdataset(diagpath, parallel=True, **kwargs)['nod_area']#.chunk({'nod2':1e4})
            if 'nod_n' in list(nz_w_A.dims): nz_w_A = nz_w_A.rename({'nod_n':'nod2'})
            if 'nl'    in list(nz_w_A.dims): nz_w_A = nz_w_A.rename({'nl'   :'nz'  })
            if 'nl1'   in list(nz_w_A.dims): nz_w_A = nz_w_A.rename({'nl1'  :'nz1' })
            # you need to drop here the coordinates for nz since they go from 
            # 0...-6000 the coordinates of nz in the data go from 0...6000 that 
            # causes otherwise troubles
            nz_w_A = nz_w_A.drop_vars(['nz'])
        else: 
            if len(mesh.n_area)>0:
                if mesh.n_area.ndim == 1:
                    # this here is only for the special case when fesom14-CMIP6
                    # data are load to compute the MOC
                    nz_w_A = xr.DataArray(mesh.n_area, dims=['nod2'])
                else: 
                    # normal case when fesom2 data are used!!!
                    nz_w_A = xr.DataArray(mesh.n_area, dims=['nz','nod2'])
            else: 
                raise ValueError('could not find ...mesh.diag.nc file')
        
        if any(data.chunks.values()): nz_w_A=nz_w_A.chunk(data.chunksizes)
        data = data.assign_coords(w_A=nz_w_A)
        del(nz_w_A)
        
    #___________________________________________________________________________    
    # select MOC basin
    data = data.isel(nod2=idxin)
      
    #___________________________________________________________________________
    # determine/adapt actual chunksize
    nchunk = 1
    if do_parallel and isinstance(data[vname].data, da.Array)==True :
            
        nchunk = len(data.chunks['nod2'])
            
        # after all the time and depth operation after the loading there will 
        # be worker who have no chunk piece to work on  --> therfore we need
        # to rechunk make sure the workload is distributed between all 
        # availabel worker equally         
        if nchunk<parallel_nprc*0.75:
            print(f' --> rechunk: {nchunk}', end='')
            if 'time' in data.dims:
                data = data.chunk({'nod2': np.ceil(data.dims['nod2']/(parallel_nprc)).astype('int'), dimn_v:-1, 'time':-1})
            else:
                data = data.chunk({'nod2': np.ceil(data.dims['nod2']/(parallel_nprc)).astype('int'), dimn_v:-1})
            nchunk = len(data.chunks['nod2'])
            print(f' -> {nchunk}', end='')    
            if 'time' not in data.dims: print('')
        
    #___________________________________________________________________________
    # create meridional bins
    lat_min    = float(np.floor(data[ 'lat' ].min().compute()))
    lat_max    = float(np.ceil( data[ 'lat' ].max().compute()))
    lat_bins   = np.arange(lat_min, lat_max+dlat*0.5, dlat)
    lat        = (lat_bins[1:]+lat_bins[:-1])*0.5
    nlat, nlev = len(lat_bins)-1, data.dims[dimn_v]
    
    #_______________________________________________________________________
    if do_persist: data = data.persist()
    #display(data_zm)
        
    #___________________________________________________________________________
    # Apply zonal integration over chunk
    # its important to use: 
    # data_zm[do_lonlat  ].data[:,None]
    # data_zm['elem_pbnd'].data[:,None]
    # all the input to da.map_blocks(calc_transect_zm_mean_chnk... must have the
    # same dimensionality otherwise it wont work. I also hat to use drop_axis = [1] 
    # which drops the chunking along the second dimension only leaving  the 
    # chunking of the first dimension. I also only managed to return the results
    # as a flattened array the attempt to return as a more dimensional matrix
    # failed. THats why i need to use shape afterwards 
    if 'time' in data.dims:
        drop_axis   = [0,1]
        ntime       = data.dims['time']
        chnk_lat    = data['lat'].data[None, None, : ] 
        chnk_wA     = data['w_A'].data[None, : , :   ]
    else:
        drop_axis   = [0]
        ntime       = 1
        chnk_lat    = data['lat'].data[None, :] 
        chnk_wA     = data['w_A'].data
    
    zmoc = da.map_blocks(calc_zmoc_chnk             ,
                         lat_bins                   ,   # mean bin definitions
                         chnk_lat                   ,   # lat nod2 coordinates
                         chnk_wA                    ,   # area weight
                         data[vname].data           ,   # data chunk piece
                         dtype     = np.float32     ,   # Tuple dtype
                         drop_axis = drop_axis      ,   # drop dim nz1
                         chunks    = (2*ntime*nlat*nlev,) # Output shape
                         )
    
       
    #___________________________________________________________________________
    # reshape axis over chunks 
    if 'time' in data.dims: zmoc = zmoc.reshape((nchunk, 2, ntime, nlev, nlat))
    else                  : zmoc = zmoc.reshape((nchunk, 2,        nlev, nlat))
        
    #___________________________________________________________________________
    # do dask axis reduction across chunks dimension
    zmoc = da.reduction(zmoc,                   
                        chunk     = lambda x, axis=None, keepdims=None: x,  # this is a do nothing routine that acts within the chunks
                        aggregate = np.sum, # that the sum that goes over the chunk
                        dtype     = np.float32,
                        axis      = 0,
                       ).compute()
    
    #___________________________________________________________________________
    # cumulative sum over latitudes from N-->S
    inSv = 1e-6
    if 'time' in data.dims:
        zmoc[0] = -np.flip(np.flip(zmoc[0], axis=2).cumsum(axis=2), axis=2) * inSv
    else:
        zmoc[0] = -np.flip(np.flip(zmoc[0], axis=1).cumsum(axis=1), axis=1) * inSv
    
    #___________________________________________________________________________
    # create land seamask --> zmoc[0] contains the bin summated area weight where 
    # its zero there bottom topography
    with np.errstate(divide='ignore', invalid='ignore'):
        zmoc = np.where(zmoc[1]>0, zmoc[0], np.nan)  

    #___________________________________________________________________________
    # transfer from [nlat, nlev] ---> [nlev, nlat]
    #if 'time' in data.dims : zmoc = zmoc.transpose([0,2,1])
    #else                   : zmoc = zmoc.transpose([1,0])
    
    #___________________________________________________________________________
    # create ZMOC xarray Dataset
    # define variable attributes    
    if   which_moc_name=='gmoc' : str_region='Global '
    elif which_moc_name=='amoc' : str_region='Atlantic '
    elif which_moc_name=='aamoc': str_region='Atlantic-Arctic '
    elif which_moc_name=='pmoc' : str_region='Pacific '
    elif which_moc_name=='ipmoc': str_region='Indo-Pacific '
    elif which_moc_name=='imoc' : str_region='Indo '
    elif isinstance(which_moc,shp.Reader): str_region=which_moc_region
    
    # for the choice of vertical plotting mode
    gattr['proj'         ]= 'zmoc'
    
    # variable attributes
    vattr['long_name'    ]= which_moc_name.upper()
    vattr['short_name'   ]= which_moc_name.upper()
    vattr['standard_name']= str_region+'Meridional Overturning Circulation'
    vattr['description'  ]= str_region+'Meridional Overturning Circulation Streamfunction, positive: clockwise, negative: counter-clockwise circulation', 
    vattr['units'        ]= 'Sv'
    
    # define data_vars dict, coordinate dict, as well as list of dimension name 
    # and size 
    data_vars, coords, dim_n, dim_s,  = dict(), dict(), list(), list()
    if 'time' in list(data.dims): dim_n.append('time')
    if 'nz1'  in list(data.dims): dim_n.append('nz1')
    if 'nz'   in list(data.dims): dim_n.append('nz')
    dim_n.append('lat')
    for dim_ni in dim_n:
        if   dim_ni=='time': dim_s.append(data.sizes['time']); coords['time' ]=(['time'], data['time'].data ) 
        elif dim_ni=='lat' : dim_s.append(lat.size          ); coords['lat'  ]=(['lat' ], lat          ) 
        elif dim_ni=='nz1'  : dim_s.append(data.sizes['nz1'] ); coords['depth']=(['nz1'  ], data['nz1' ].data )
        elif dim_ni=='nz'   : dim_s.append(data.sizes['nz' ] ); coords['depth']=(['nz'   ], data['nz'  ].data ) 
    data_vars['zmoc'] = (dim_n, zmoc, vattr) 
    
    # create dataset
    zmoc = xr.Dataset(data_vars=data_vars, coords=coords, attrs=gattr)
   
    #___________________________________________________________________________
    # compute depth of max and nice bottom topography
    #lat_bin_nod2 = xr.DataArray(data=np.round(data.lat/dlat)*dlat, dims='nod2', name='lat')
    #zmoc = calc_bottom_patch(zmoc, lat_bin_nod2, xr.DataArray(mesh.n_iz, dims=['nod2']), idxin)
    if 'time' in data.dims: idx_bot = np.argmax(np.isnan(zmoc['zmoc'].data[0,:,:]), axis=0)
    else                  : idx_bot = np.argmax(np.isnan(zmoc['zmoc'].data), axis=0)
    filt    = np.array([1,3,1])
    botmax  = zmoc['depth'].data[idx_bot-1]
    botmax  = np.concatenate( (np.ones((filt.size,))*botmax[0], botmax, np.ones((filt.size,))*botmax[-1] ) )
    botmax  = np.convolve(botmax, filt/np.sum(filt), mode='same')[filt.size:-filt.size]
    zmoc    = zmoc.assign_coords(botmax = botmax)
    del(idx_bot, botmax, filt)
    
    #___________________________________________________________________________
    # write some infos 
    if do_info==True: 
        print(' --> total time:{:.3f} s'.format(clock.time()-t1))
        if 'time' not in list(zmoc.dims):
            if which_moc_name in ['amoc', 'aamoc', 'gmoc']:
                maxv = zmoc.isel(nz=zmoc['depth']>= 700 , lat=zmoc['lat']>0.0)['zmoc'].max().values
                minv = zmoc.isel(nz=zmoc['depth']>= 2500, lat=zmoc['lat']>-50.0)['zmoc'].min().values
                print(' max. NADW_{:s} = {:.2f} Sv'.format(zmoc['zmoc'].attrs['descript'],maxv))
                print(' max. AABW_{:s} = {:.2f} Sv'.format(zmoc['zmoc'].attrs['descript'],minv))
            elif which_moc_name in ['pmoc', 'ipmoc']:
                minv = zmoc['zmoc'].isel(nz=zmoc['depth']>= 2000, lat=zmoc['lat']>-50.0)['moc'].min().values
                print(' max. AABW_{:s} = {:.2f} Sv'.format(zmoc['zmoc'].attrs['descript'],minv))
    
    #___________________________________________________________________________
    return(zmoc)


#
#
#_______________________________________________________________________________  
def calc_zmoc_chnk(lat_bins, chnk_lat, chnk_wA, chnk_d):
    """

    """
    #n11 = chnk_lat.shape
    #n21 = chnk_wA.shape
    #n31 = chnk_d.shape
    
    #___________________________________________________________________________
    # only need the additional dimension at the point where the function is initialised
    if   np.ndim(chnk_d) == 2: 
        chnk_lat = chnk_lat[ 0, :] # 2D --> now is 1D again
    elif np.ndim(chnk_d) == 3:
        chnk_lat = chnk_lat[0, 0, :] # 3D --> now is 1D again
        chnk_wA  = chnk_wA[ 0, :, :] # 3D --> now is 2D again
    
    # Use np.digitize to find bin indices for longitudes and latitudes
    idx_lat = np.digitize(chnk_lat, lat_bins)-1  # Adjust to get 0-based index
    nlat    = len(lat_bins)-1
    
    # Initialize binned data storage 
    if   np.ndim(chnk_d) == 3: 
        # Replace NaNs with 0 value to summation issues
        ntime, nlev, nnod = chnk_d.shape
        chnk_wA     = np.where(np.isnan(chnk_d[0,:,:]), 0, chnk_wA)
        chnk_d      = np.where(np.isnan(chnk_d), 0, chnk_d)
        binned_d    = np.zeros((2, ntime, nlev, nlat), dtype=np.float32)
        # binned_d[0,...] - data*area weight
        # binned_d[1,...] - area weight sum
        
    elif np.ndim(chnk_d) == 2: 
        # Replace NaNs with 0 value to summation issues
        nlev, nnod  = chnk_d.shape
        chnk_wA     = np.where(np.isnan(chnk_d), 0, chnk_wA)
        chnk_d      = np.where(np.isnan(chnk_d), 0, chnk_d)
        binned_d    = np.zeros((2, nlev, nlat), dtype=np.float32)  
        # binned_d[0,...] - data*area weight
        # binned_d[1,...] - area weight sum
        
    # Precompute mask outside the loop
    idx_valid = (idx_lat >= 0) & (idx_lat < nlat)

    # Apply mask before looping
    idx_lat   = idx_lat[idx_valid   ]
    chnk_wA   = chnk_wA[:, idx_valid]
    nnod      = len(idx_lat)
    
    # Sum data based on binned indices with time dimension: [2, ntime, nlat, nlev]
    if   np.ndim(chnk_d) == 3:    
        chnk_d  = chnk_d[ :, :, idx_valid]
        for nod_i in range(0,nnod):
            jj = idx_lat[nod_i]
            binned_d[0, :, :, jj] = binned_d[0, :, :, jj] + chnk_d[ :, :, nod_i] * chnk_wA[:, nod_i]
            binned_d[1, :, :, jj] = binned_d[1, :, :, jj] + chnk_wA[   :, nod_i]
    
    # Sum data based on binned indices withou time dimension: [2, nlat, nlev]
    elif np.ndim(chnk_d) == 2:  
        chnk_d  = chnk_d[:,  idx_valid]
        for nod_i in range(0,nnod):
            jj = idx_lat[nod_i]
            binned_d[0, :, jj] = binned_d[0, :, jj] + chnk_d[ :, nod_i] * chnk_wA[:, nod_i]
            binned_d[1, :, jj] = binned_d[1, :, jj] + chnk_wA[:, nod_i]
    
    #___________________________________________________________________________
    return binned_d.flatten()
    #return np.array([n11, n21, n31 ]).flatten()


    
# ___CALC BOTTOM TOPO PATCH____________________________________________________
#|                                                                             |
#|                                                                             |
#|_____________________________________________________________________________|
def calc_bottom_patch(data, lat_bin, idx_iz, idxin):
    """
    --> compute idealized AMOC bottom path based on 3d data bathymetry structure
    
    Parameters:
    
        :data:          xarray MOC data object 
        
        :lat_bin:       xarray Array with latitudinal bins
        
        :idx_iz:        xarray Array with 2d bottom zlevel index. Can be for 
                        elements or vertices 
                        
            ::
        
                idx_iz = xr.DataArray(mesh.e_iz, dims=['elem']), or 
                idx_iz = xr.DataArray(mesh.n_iz, dims=['nod2'])
        
        :idxin:         bool index array for regional basin selection
        
    Returns:
    
        :data:          xarray MOC data object with additional coord: botnice 
    
    ____________________________________________________________________________
    """
    idx_z = data['depth'][idx_iz]
    idx_z = idx_z.isel({ list(idx_z.dims)[0] : idxin})
    idx_z = idx_z.groupby(lat_bin)
    #___________________________________________________________________________
    # maximum bottom topography for MOC
    botmax= idx_z.max()
    data  = data.assign_coords(botmax = botmax)
    
    #___________________________________________________________________________
    # optiocal nicer bottom topography for MOC
    botnic= idx_z.quantile(1-0.20, skipna=True).drop_vars(['quantile'])
    
    # smooth bottom topography patch
    #filt=np.array([1,2,3,2,1])
    filt=np.array([1,2,1])
    filt=filt/np.sum(filt)
    aux = np.concatenate( (np.ones((filt.size,))*botnic.data[0],botnic.data,np.ones((filt.size,))*botnic.data[-1] ) )
    aux = np.convolve(aux,filt,mode='same')
    botnic.data = aux[filt.size:-filt.size]
    del(aux)
    
    data  = data.assign_coords(botnice= botnic)
    
    #___________________________________________________________________________
    # index for max bottom index for every lat bin 
    #idx_iz    = idx_iz.isel({list(idx_z.dims)[0] : idxin})
    #idx_iz    = nodeiz.groupby(lat_bin).max()
    #data      = data.assign_coords(botmaxi=idx_iz)
    
    #___________________________________________________________________________
    return(data)
