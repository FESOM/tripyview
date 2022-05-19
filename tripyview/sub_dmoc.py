import numpy as np
import time as time
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

from .colormap_c2c    import *
from .sub_index import *
from .sub_moc import *


#+___CALCULATE MERIDIONAL OVERTURNING IN DENSITY COORDINATES___________________+
#| Global MOC, Atlantik MOC, Indo-Pacific MOC, Indo MOC                        |
#|                                                                             |
#+_____________________________________________________________________________+
def load_dmoc_data(mesh, datapath, descript, year, which_transf, std_dens, n_area=None, e_area=None, 
                   do_info=True, do_tarithm='mean', add_trend=False, **kwargs, ):
    #___________________________________________________________________________
    # ensure that attributes are preserved  during operations with yarray 
    xr.set_options(keep_attrs=True)
    
    #___________________________________________________________________________
    # Load triangle and cluster area if not given 
    if n_area is None or e_area is None:
        if do_info==True: print(' --> load triangle and cluster area from diag .nc file')
        fname = 'fesom.mesh.diag.nc'
        # check for directory with diagnostic file
        if   os.path.isfile( os.path.join(datapath, fname) ): 
            dname = datapath
        elif os.path.isfile( os.path.join( os.path.join(os.path.dirname(os.path.normpath(datapath)),'1/'), fname) ): 
            dname = os.path.join(os.path.dirname(os.path.normpath(datapath)),'1/')
        elif os.path.isfile( os.path.join(mesh.path,fname) ): 
            dname = mesh.path
        else:
            raise ValueError('could not find directory with...mesh.diag.nc file')    
        
        # load diag file
        meshdiag = xr.open_dataset(os.path.join(dname,fname))
        
        # only need cluster area from the surface since density classes dont know 
        # any information aboutthe bottom 
        if n_area is None: n_area = meshdiag['nod_area'].isel(nz=0) 
        if e_area is None: e_area = meshdiag['elem_area']
    
    #___________________________________________________________________________
    # Load netcdf data
    if do_info==True: print(' --> create xarray dataset and load std_* data')
    # create xarray dataset to combine dat for dmoc computation
    data_DMOC = xr.Dataset()
    
    # add surface transformations 
    if which_transf=='srf' or which_transf=='inner': # add surface fluxes
        data_DMOC = xr.merge([data_DMOC, 
                              load_data_fesom2(mesh, datapath, vname='std_heat_flux', year=year, descript=descript , do_info=do_info, do_ie2n=False, do_tarithm=do_tarithm)]) 
        
        data_DMOC = xr.merge([data_DMOC, 
                              load_data_fesom2(mesh, datapath, vname='std_frwt_flux', year=year, descript=descript , do_info=do_info, do_ie2n=False, do_tarithm=do_tarithm)])
        
        data_DMOC = xr.merge([data_DMOC, 
                              load_data_fesom2(mesh, datapath, vname='std_rest_flux', year=year, descript=descript , do_info=do_info, do_ie2n=False, do_tarithm=do_tarithm)])
        
    # add volume trend  
    if add_trend:  
        data_DMOC = xr.merge([data_DMOC, 
                              load_data_fesom2(mesh, datapath, vname='std_dens_dVdT', year=year, descript=descript , do_info=do_info, do_ie2n=False, do_tarithm=do_tarithm)]) 
        
    # add vertical mean z-coordinate position of density classes
    data_zpos = load_data_fesom2(mesh, datapath, vname='std_dens_Z'   , year=year, descript=descript , do_info=do_info, do_ie2n=False, do_tarithm=do_tarithm)
    data_zpos = data_zpos*e_area
    data_DMOC = xr.merge([data_DMOC, data_zpos]) 
    del(data_zpos)
    
    # add divergence of density classes --> diapycnal velocity
    data_div  = load_data_fesom2(mesh, datapath, vname='std_dens_DIV' , year=year, descript=descript , do_info=do_info, do_ie2n=False, do_tarithm=do_tarithm) 

    # integrated divergence below isopcnal  --> size: [nod2d x ndens] --> size: [elem x ndens]
    data_div  = data_div/n_area
    
    # have to do it via assign otherwise cant write [elem x ndens] into [nod2d x ndens] 
    # array an save the attributes in the same time
    if 'time' in list(data_div.dims):
        data_div  = data_div.assign( std_dens_DIV=data_div[list(data_div.keys())[0]][:, xr.DataArray(mesh.e_i, dims=["elem",'n3']), :].mean(dim="n3", keep_attrs=True) )
    else:
        data_div  = data_div.assign( std_dens_DIV=data_div[list(data_div.keys())[0]][xr.DataArray(mesh.e_i, dims=["elem",'n3']),:].mean(dim="n3", keep_attrs=True) )
    
    # drop nod2 dimensional coordiantes become later replaced with elemental coordinates
    data_div  = data_div.drop(['nod2','lon','lat'])
    data_div  = data_div*e_area
    data_DMOC = xr.merge([data_DMOC, data_div])    
    del(data_div)
    
    # add coordinates to xarray data set
    data_DMOC = data_DMOC.assign_coords({'ndens' :("ndens",std_dens),
                                         'lon'   :("elem",mesh.n_x[mesh.e_i].sum(axis=1)/3.0),
                                         'lat'   :("elem",mesh.n_y[mesh.e_i].sum(axis=1)/3.0),
                                         'elem_A':("elem",e_area.values)})
    
    #___________________________________________________________________________
    # return combined xarray dataset object
    return(data_DMOC)
    

#+___CALCULATE MERIDIONAL OVERTURNING IN DENSITY COORDINATES___________________+
#| Global MOC, Atlantik MOC, Indo-Pacific MOC, Indo MOC                        |
#|                                                                             |
#+_____________________________________________________________________________+
def calc_dmoc(mesh, data_dMOC, dlat=1.0, which_moc='gmoc', do_info=True, do_checkbasin=False, **kwargs, ):
    
    
    # number of sigma2 density levels 
    std_dens     = data_dMOC['ndens'].values
    ndens        = len(std_dens)
    wd, w        = np.diff(std_dens), np.zeros(ndens)
    w[0 ], w[-1] = wd[0   ]/2., wd[-1  ]/2.
    w[1:-1]      = (wd[0:-1]+wd[1:])/2. # drho @  std_dens level boundary
    weight_dens  = xr.DataArray(w, dims=["ndens"])

    #______________________________________________________________________________________________________
    # compute index for basin domain limitation
    idxin     = calc_basindomain_fast(mesh, which_moc=which_moc, do_onelem=True)

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

    #______________________________________________________________________________________________________
    # scale surface density fluxes
    if 'std_heat_flux' in list(data_dMOC.keys()):
        data_dMOC['std_heat_flux'] = data_dMOC['std_heat_flux'] / weight_dens * 1.0e-6 * 1024.0
    if 'std_frwt_flux' in list(data_dMOC.keys()):
        data_dMOC['std_frwt_flux'] = data_dMOC['std_frwt_flux'] / weight_dens * 1.0e-6 * 1024.0
    if 'std_rest_flux' in list(data_dMOC.keys()):
        data_dMOC['std_rest_flux'] = data_dMOC['std_rest_flux'] / weight_dens * 1.0e-6 * 1024.0
    # scale volume change over time
    if 'std_dens_dVdT' in list(data_dMOC.keys()):
        data_dMOC['std_dens_dVdT'] = data_dMOC['std_dens_dVdT'] * 1.0e-6
    # scale integrated divergence below isopcnal
    data_dMOC['std_dens_DIV']  = data_dMOC['std_dens_DIV'] * 1.0e-6

    #______________________________________________________________________________________________________
    # create meridional bins
    if do_info==True: print(' --> create latitudinal bins')
    #dlat      = 1.0
    e_y       = data_dMOC['lat'].values.copy()
    lat       = np.arange(np.floor(e_y.min())-dlat, np.ceil( e_y.max())+dlat, dlat)
    lat_i     = (( e_y-lat[0])/dlat ).astype('int')

    #______________________________________________________________________________________________________
    # number of latitudinal bins & number of density levels & number time slices
    dens = data_dMOC['ndens'].values
    nlat, ndens = lat.size, dens.size
    list_dimname, list_dimsize = ['ndens', 'nlat'], [ndens, nlat]
    if 'time' in list(data_dMOC.dims): 
        time = data_dMOC['time'].values
        nti  = time.size
        list_dimname, list_dimsize = ['time', 'ndens', 'nlat'], [nti, ndens, nlat]
    
    # allocate moc_... dataset
    #data_vars = {'dmoc_fh'  :(['ndens','nlat'], np.zeros((ndens,nlat)), {'long_name':'Transformation from heat flux', 'units':'Sv'}), 
                 #'dmoc_fw'  :(['ndens','nlat'], np.zeros((ndens,nlat)), {'long_name':'Transformation from freshwater flux', 'units':'Sv'}), 
                 #'dmoc_fr'  :(['ndens','nlat'], np.zeros((ndens,nlat)), {'long_name':'Transformation from surface salinity restoring', 'units':'Sv'}), 
                 #'dmoc_dvdt':(['ndens','nlat'], np.zeros((ndens,nlat)), {'long_name':'Transformation from volume change', 'units':'Sv'}), 
                 #'dmoc_zpos':(['ndens','nlat'], np.zeros((ndens,nlat)), {'long_name':'Density MOC Z position', 'units':'m'}), 
                 #'dmoc'     :(['ndens','nlat'], np.zeros((ndens,nlat)), {'long_name':'Density MOC', 'units':'Sv'})}
    data_vars = dict()
    # setup dmoc_fh xarray in dataset, rescue attributes from data_MOC
    if 'std_heat_flux' in list(data_dMOC.keys()):
        aux_attr = data_dMOC['std_heat_flux'].attrs
        aux_attr['long_name'], aux_attr['units'] = 'Transformation from heat flux', 'Sv'
        data_vars['dmoc_fh'  ] = (list_dimname, np.zeros(list_dimsize), aux_attr) 
    # setup dmoc_fw xarray in dataset, rescue attributes from data_MOC
    if 'std_frwt_flux' in list(data_dMOC.keys()):
        aux_attr = data_dMOC['std_frwt_flux'].attrs
        aux_attr['long_name'], aux_attr['units'] = 'Transformation from freshwater flux', 'Sv'
        data_vars['dmoc_fw'  ] = (list_dimname, np.zeros(list_dimsize), aux_attr) 
    # setup dmoc_fr xarray in dataset, rescue attributes from data_MOC
    if 'std_rest_flux' in list(data_dMOC.keys()):
        aux_attr = data_dMOC['std_rest_flux'].attrs
        aux_attr['long_name'], aux_attr['units'] = 'Transformation from surface salinity restoring', 'Sv'
        data_vars['dmoc_fr'  ] = (list_dimname, np.zeros(list_dimsize), aux_attr)
    # setup dmoc_dvdt xarray in dataset, rescue attributes from data_MOC
    if 'std_dens_dVdT' in list(data_dMOC.keys()):
        aux_attr = data_dMOC['std_dens_dVdT'].attrs
        aux_attr['long_name'], aux_attr['units'] = 'Transformation from volume change', 'Sv'
        data_vars['dmoc_dvdt']=(list_dimname, np.zeros(list_dimsize), aux_attr)
    # setup dmoc_zpos xarray in dataset, rescue attributes from data_MOC
    data_vars['dmoc_zpos'] = (list_dimname, np.zeros(list_dimsize), {'long_name':'Density MOC Z position', 'units':'m'})
    # setup dmoc xarray in dataset, rescue attributes from data_MOC
    aux_attr = data_dMOC['std_dens_DIV'].attrs
    aux_attr['long_name'], aux_attr['units'] = 'Density MOC', 'Sv'
    data_vars['dmoc'     ] = (list_dimname, np.zeros(list_dimsize), aux_attr)
    
    # define coordinates
    if 'time' in list(data_dMOC.dims): 
        coords = {'ndens': (['ndens'], dens),
                  'nlat' : (['nlat' ], lat ),
                  'time' : (['time' ], time)}
    else:
        coords = {'ndens': (['ndens'], dens),
                  'nlat' : (['nlat' ], lat )}
    
    # create DMOC dataset
    dMOC = xr.Dataset(data_vars=data_vars, coords=coords,attrs=data_dMOC.attrs)
    del(data_vars, coords, aux_attr)

    #______________________________________________________________________________________________________
    # do zonal sum over latitudinal bins 
    if do_info==True: print(' --> do latitudinal bining')
    for bini in range(lat_i.min(), lat_i.max()):
        
        # compute density MOC binning over time
        if 'time' in list(data_dMOC.dims): 
            # sum over latitudinal bins
            if 'dmoc_fh' in list(dMOC.keys()):
                dMOC['dmoc_fh'  ].data[:,:, bini] = data_dMOC['std_heat_flux'].isel(elem=lat_i==bini).sum(dim='elem')
            if 'dmoc_fw' in list(dMOC.keys()):    
                dMOC['dmoc_fw'  ].data[:,:, bini] = data_dMOC['std_frwt_flux'].isel(elem=lat_i==bini).sum(dim='elem')
            if 'dmoc_fr' in list(dMOC.keys()):
                dMOC['dmoc_fr'  ].data[:,:, bini] = data_dMOC['std_rest_flux'].isel(elem=lat_i==bini).sum(dim='elem')
            if 'dmoc_dvdt' in list(dMOC.keys()):    
                dMOC['dmoc_dvdt'].data[:,:, bini] = data_dMOC['std_dens_dVdT'].isel(elem=lat_i==bini).sum(dim='elem')
            dMOC['dmoc'     ].data[:,:, bini] = data_dMOC['std_dens_DIV' ].isel(elem=lat_i==bini).sum(dim='elem')
            
            # compute mean z-position of density levels over bins
            aux  = data_dMOC['std_dens_Z'].isel(elem=lat_i==bini)
            tvol = data_dMOC['elem_A'].isel(elem=lat_i==bini).expand_dims({'time':time, 'ndens':std_dens}).transpose()
            tvol = tvol.where(aux<-1) # False condition is set nan
            aux  = aux.where(aux<-1,)  # False condition is set nan
            dMOC['dmoc_zpos'].data[:,:, bini] = aux.sum(dim='elem', skipna=True)/tvol.sum(dim='elem', skipna=True)
        
        # compute density MOC binning over time mean
        else:    
            # sum over latitudinal bins
            if 'dmoc_fh' in list(dMOC.keys()):
                dMOC['dmoc_fh'  ].data[:, bini] = data_dMOC['std_heat_flux'].isel(elem=lat_i==bini).sum(dim='elem')
            if 'dmoc_fw' in list(dMOC.keys()):    
                dMOC['dmoc_fw'  ].data[:, bini] = data_dMOC['std_frwt_flux'].isel(elem=lat_i==bini).sum(dim='elem')
            if 'dmoc_fr' in list(dMOC.keys()):
                dMOC['dmoc_fr'  ].data[:, bini] = data_dMOC['std_rest_flux'].isel(elem=lat_i==bini).sum(dim='elem')
            if 'dmoc_dvdt' in list(dMOC.keys()):    
                dMOC['dmoc_dvdt'].data[:, bini] = data_dMOC['std_dens_dVdT'].isel(elem=lat_i==bini).sum(dim='elem')
            dMOC['dmoc'     ].data[:, bini] = data_dMOC['std_dens_DIV' ].isel(elem=lat_i==bini).sum(dim='elem')
            
            # compute mean z-position of density levels over bins
            aux  = data_dMOC['std_dens_Z'].isel(elem=lat_i==bini)
            tvol = data_dMOC['elem_A'].isel(elem=lat_i==bini).expand_dims({'ndens':std_dens}).transpose()
            tvol = tvol.where(aux<-1) # False condition is set nan
            aux  = aux.where(aux<-1,)  # False condition is set nan
            dMOC['dmoc_zpos'].data[:, bini] = aux.sum(dim='elem', skipna=True)/tvol.sum(dim='elem', skipna=True)
        del(aux,tvol)

    #______________________________________________________________________________________________________
    # cumulative sum over latitudes
    if do_info==True: print(' --> do cumsum over latitudes')
    # dMOC['dmoc_fh'  ] = dMOC['dmoc_fh'  ].cumsum(dim='nlat', skipna=True)
    # dMOC['dmoc_fw'  ] = dMOC['dmoc_fw'  ].cumsum(dim='nlat', skipna=True)
    # dMOC['dmoc_fr'  ] = dMOC['dmoc_fr'  ].cumsum(dim='nlat', skipna=True)
    # dMOC['dmoc_dvdt'] = dMOC['dmoc_dvdt'].cumsum(dim='nlat', skipna=True)
    # dMOC['dmoc'     ] = dMOC['dmoc'     ].cumsum(dim='nlat', skipna=True)
    var_list = list(dMOC.keys())
    var_list.remove('dmoc_zpos')
    for var in var_list:
        dMOC[ var ] = dMOC[ var ].cumsum(dim='nlat', skipna=True)
    
    #______________________________________________________________________________________________________
    # cumulative sum over depth 
    if do_info==True: print(' --> do cumsum over depth (bottom-->top)')
    dMOC[ 'dmoc' ] = dMOC[ 'dmoc' ].reindex(ndens=dMOC['ndens'][::-1]).cumsum(dim='ndens', skipna=True).reindex(ndens=dMOC['ndens'])

    #______________________________________________________________________________________________________
    # substract northern boundary
    if do_info==True: print(' --> normalize to northern boundary')
    # for li in range(nlat):
    #     dMOC['dmoc_fh'  ].data[:, li] = dMOC['dmoc_fh'  ].data[:, li] - dMOC['dmoc_fh'  ].data[:, -1]
    #     dMOC['dmoc_fw'  ].data[:, li] = dMOC['dmoc_fw'  ].data[:, li] - dMOC['dmoc_fw'  ].data[:, -1]
    #     dMOC['dmoc_fr'  ].data[:, li] = dMOC['dmoc_fr'  ].data[:, li] - dMOC['dmoc_fr'  ].data[:, -1]
    #     dMOC['dmoc_dvdt'].data[:, li] = dMOC['dmoc_dvdt'].data[:, li] - dMOC['dmoc_dvdt'].data[:, -1]
    #     dMOC['dmoc'     ].data[:, li] = dMOC['dmoc'     ].data[:, li] - dMOC['dmoc'     ].data[:, -1]
    for var in var_list:
        for li in range(nlat):
            if 'time' in list(data_dMOC.dims): 
                dMOC[ var ].data[:,:, li] = dMOC[ var ].data[:,:, li] - dMOC[ var ].data[:,:, -1]
            else:    
                dMOC[ var ].data[:, li] = dMOC[ var ].data[:, li] - dMOC[ var ].data[:, -1]
    
    #______________________________________________________________________________________________________
    return(dMOC)



#+___PLOT MERIDIONAL OVERTRUNING CIRCULATION  _________________________________+
#|                                                                             |
#+_____________________________________________________________________________+
def plot_dmoc(data, which_moc='gmoc', which_transf='dmoc', figsize=[12, 6], 
              n_rc=[1, 1], do_grid=True, cinfo=None,
              cbar_nl=8, cbar_orient='vertical', cbar_label=None, cbar_unit=None,
              do_bottom=True, color_bot=[0.6, 0.6, 0.6], 
              pos_fac=1.0, pos_gap=[0.01, 0.02], do_save=None, save_dpi=600, 
              do_contour=True, do_clabel=True, title='descript', do_rescale=None,
              do_yrescale=True, do_zcoord=False, do_check=True, 
              pos_extend=[0.075, 0.075, 0.90, 0.95] ):
    #____________________________________________________________________________
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
    cinfo = do_setupcinfo(cinfo, data, do_rescale, do_dmoc=which_transf)
    
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
    for ii in range(0,ndata):
        
        #_______________________________________________________________________
        # plot dmoc in z-coordinates
        if do_zcoord:
            # use depth information of depth of density classes at latitude
            data_y = data[ii]['dmoc_zpos'].values[1:-1,:].copy()
            data_y[data_y>=-1.0]=np.nan
            
            # do dirty trick here !!! make sure that below the deepest depth at 
            # every latitude that there is no nan value or a value shallower than
            # the deepest depth left
            nlat  =  data[ii]['nlat'].values.size
            ndens =  data_y.shape[0]
            for kk in range(nlat):
                min_datay = np.nanmin(data_y[:,kk])
                for jj in list(range(0,ndens)[::-1]): # to bottom to top
                    if data_y[jj,kk]==min_datay:break
                    if np.isnan(data_y[jj,kk]) or data_y[jj,kk]>min_datay: data_y[jj,kk]=min_datay
            del(min_datay)  
            # same but for the surface
            for kk in range(nlat):
                max_datay = np.nanmax(data_y[:,kk])
                for jj in list(range(0,ndens)): 
                    if data_y[jj,kk]==max_datay:break
                    if np.isnan(data_y[jj,kk]) or data_y[jj,kk]<max_datay: data_y[jj,kk]=max_datay        
            del(max_datay)        
            # do [ndens x nlat] matrix for latitudes
            data_x   = data[ii]['nlat'].values.copy()
            data_x   = repmat(data_x,data[ii]['ndens'].values[1:-1].size,1)
            
            ## do nearest neighbour interpolation of remaining nan values in depth_y
            xx, yy = np.meshgrid(data_x[0,:], data[ii]['ndens'].values[1:-1])
            data_y = np.ma.masked_invalid(data_y)
            data_y = interpolate.griddata((xx[~data_y.mask], yy[~data_y.mask]), data_y[~data_y.mask].ravel(),
                                            (xx, yy), method='nearest')
            del(xx,yy)
            
        #_______________________________________________________________________
        # plot dmoc in density-coordinates  
        else:
            data_x, data_y = data[ii]['nlat'].values.copy(), data[ii]['ndens'].values[1:-1].copy()
        
        #_______________________________________________________________________
        # What  should be plotted: density MOC, Surface Transformation or Inner
        # Transformation
        if   which_transf == 'dmoc':
            data_plot = data[ii]['dmoc'].values[1:-1,:].copy()
            
            #___________________________________________________________________
            # PLOT DMOC INFO: maximum/minimum dmoc sigma2 and depth
            if do_check:
                #print(data[ii])
                idxmax = data[ii]['dmoc'].argmax(dim=["ndens", "nlat"])
                #print(idxmax)
                idxmin = data[ii]['dmoc'].argmin(dim=["ndens", "nlat"])
                dmoc_max, dmoc_min = data[ii]['dmoc'].isel(idxmax).data, data[ii]['dmoc'].isel(idxmin).data
                if not do_zcoord:
                    s_max, l_max, d_max = data_y[idxmax['ndens'].data], data_x[idxmax['nlat'].data], data[ii]['dmoc_zpos'].isel(idxmax).data
                    s_min, l_min, d_min = data_y[idxmin['ndens'].data], data_x[idxmin['nlat'].data], data[ii]['dmoc_zpos'].isel(idxmin).data
                else:     
                    s_max = data[ii]['ndens'].isel(ndens=idxmax['ndens']).data
                    l_max = data_x[idxmax['ndens'].data, idxmax['nlat'].data]
                    d_max = data_y[idxmax['ndens'].data, idxmax['nlat'].data]
                    s_min = data[ii]['ndens'].isel(ndens=idxmin['ndens']).data
                    l_min = data_x[idxmin['ndens'].data, idxmin['nlat'].data]
                    d_min = data_y[idxmin['ndens'].data, idxmin['nlat'].data]
                print('DMOC_max={:5.1f} (sigma2={:5.2f}kg/m^3, depth={:5.0f}m, lat={:5.1f}°N)'.format(dmoc_max, s_max, d_max, l_max))
                print('DMOC_min={:5.1f} (sigma2={:5.2f}kg/m^3, depth={:5.0f}m, lat={:5.1f}°N)'.format(dmoc_min, s_min, d_min, l_min))                    
        
        elif which_transf == 'srf' : 
            data_plot = -(data[ii]['dmoc_fh'].values[1:-1,:].copy()+ \
                          data[ii]['dmoc_fw'].values[1:-1,:].copy()+ \
                          data[ii]['dmoc_fr'].values[1:-1,:].copy())
            
        elif which_transf == 'inner':
            data_plot = data[ii]['dmoc'].values[1:-1,:].copy()    + \
                        (data[ii]['dmoc_fh'].values[1:-1,:].copy()+ \
                         data[ii]['dmoc_fw'].values[1:-1,:].copy()+ \
                         data[ii]['dmoc_fr'].values[1:-1,:].copy())
                    
        data_plot[data_plot<cinfo['clevel'][ 0]] = cinfo['clevel'][ 0]+np.finfo(np.float32).eps
        data_plot[data_plot>cinfo['clevel'][-1]] = cinfo['clevel'][-1]-np.finfo(np.float32).eps
        
        #_______________________________________________________________________
        # if plot in density-coordinates first scale to regular y-axes, and flip  
        # the  y-axes of data_y and data_plot since ax.invert_yaxis() is not working 
        # with share plt.subplot(..., sharey=True)
        if not do_zcoord: 
            if do_yrescale: data_y = dens2reg(data_y)
            data_plot, data_y = data_plot[::-1,:], data_y[::-1]
           
        #_______________________________________________________________________
        # plot DATA
        hp=ax[ii].contourf(data_x, data_y, data_plot, levels=cinfo['clevel'], 
                           extend='both', cmap=cinfo['cmap'])
        
        if do_contour: 
            tickl    = cinfo['clevel']
            ncbar_l  = len(tickl)
            idx_cref = np.where(cinfo['clevel']==cinfo['cref'])[0]
            idx_cref = np.asscalar(idx_cref)
            nstep    = ncbar_l/(cbar_nl)
            nstep    = np.max([np.int(np.floor(nstep)),1])
            
            idx = np.arange(0, ncbar_l, 1)
            idxb = np.ones((ncbar_l,), dtype=bool)                
            idxb[idx_cref::nstep]  = False
            idxb[idx_cref::-nstep] = False
            idx_yes = idx[idxb==False]
            
            aux_clvl = cinfo['clevel'][idx_yes]
            aux_clvl = aux_clvl[aux_clvl!=cinfo['cref']]
            cont=ax[ii].contour(data_x, data_y, data_plot, 
                                levels=aux_clvl, colors='k', linewidths=[0.5]) #linewidths=[0.5,0.25])
            if do_clabel: 
                ax[ii].clabel(cont, cont.levels[np.where(cont.levels!=cinfo['cref'])], 
                            inline=1, inline_spacing=1, fontsize=6, fmt='%1.1f Sv')
            ax[ii].contour(data_x, data_y, data_plot, 
                                levels=[0.0], colors='k', linewidths=[1.25]) #linewidths=[0.5,0.25])
        
        #_______________________________________________________________________
        # plot bottom representation in case of z-coordinates
        if do_zcoord: 
            data_bot = np.nanmin(data_y, axis=0)
            ax[ii].plot(data_x[0,:], data_bot, color='k')
            ax[ii].fill_between(data_x[0,:], data_bot, data_bot.min(), color=color_bot, zorder=2)#,alpha=0.95)
        
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
        for im in ax[ii].get_images(): im.set_clim(cinfo['clevel'][ 0], cinfo['clevel'][-1])
        
        #_______________________________________________________________________
        # plot grid lines 
        if do_grid: ax[ii].grid(color='k', linestyle='-', linewidth=0.25,alpha=1.0)
        
        #_______________________________________________________________________
        # set description string
        if title is not None: 
            if not do_zcoord: 
                txtx = data_x[0]+(data_x[-1]-data_x[0])*0.025
                txty = data_y[0]+(data_y[-1]-data_y[0])*0.025    
            else:
                txtx = data_x.min()+(data_x.max()-data_x.min())*0.025
                txty = data_y.min()+(data_y.max()-data_y.min())*0.025    
            
            if   isinstance(title,str) : 
                # if title string is 'descript' than use descript attribute from 
                # data to set plot title 
                if title=='descript' and ('descript' in data[ii]['dmoc'].attrs.keys() ):
                    txts = data[ii]['dmoc'].attrs['descript']
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
            if collist[ii]==0: ax[ii].set_ylabel('${\sigma}_{2}$ potential density [kg/m³]',fontsize=12)
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
    cbar = fig.colorbar(hp, orientation=cbar_orient, ax=ax, ticks=cinfo['clevel'], 
                      extendrect=False, extendfrac=None,
                      drawedges=True, pad=0.025, shrink=1.0)
    
    # do formatting of colorbar 
    cbar = do_cbar_formatting(cbar, do_rescale, cbar_nl, fontsize)
    
    # do labeling of colorbar
    if   which_moc=='gmoc' : cbar_label = 'Global Meridional \n Overturning Circulation [Sv]'
    elif which_moc=='amoc' : cbar_label = 'Atlantic Meridional \n Overturning Circulation [Sv]'
    elif which_moc=='aamoc': cbar_label = 'Arctic-Atlantic Meridional \n Overturning Circulation [Sv]'
    elif which_moc=='pmoc' : cbar_label = 'Pacific Meridional \n Overturning Circulation [Sv]'
    elif which_moc=='ipmoc': cbar_label = 'Indo-Pacific Meridional \n Overturning Circulation [Sv]'
    elif which_moc=='imoc' : cbar_label = 'Indo Meridional \n Overturning Circulation [Sv]'
    if 'str_ltim' in data[0]['dmoc'].attrs.keys():
        cbar_label = cbar_label+'\n'+data[0]['dmoc'].attrs['str_ltim']
        
    if which_transf=='dmoc':    
        cbar.set_label('Density - '+cbar_label, size=fontsize+2)
    elif which_transf=='srf':    
        cbar.set_label('Srf. Transf. - '+cbar_label, size=fontsize+2)
    elif which_transf=='inner':    
        cbar.set_label('Inner. Transf. - '+cbar_label, size=fontsize+2)
        
    #___________________________________________________________________________
    # repositioning of axes and colorbar
    ax, cbar = do_reposition_ax_cbar(ax, cbar, rowlist, collist, pos_fac, pos_gap, 
                                     title=None, extend=pos_extend)
    
    plt.show()
    fig.canvas.draw()
    
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, fig, dpi=save_dpi)
    
    #___________________________________________________________________________
    return(fig, ax, cbar)



#+___PLOT MERIDIONAL OVERTRUNING CIRCULATION TIME-SERIES_______________________+
#|                                                                             |
#+_____________________________________________________________________________+
def plot_dmoc_tseries(time, moct_list, input_names, which_cycl=None, which_lat=['max'], 
                       which_moc='amoc', do_allcycl=False, ymaxstep=1, xmaxstep=5,
                       str_descript='', str_time='', figsize=[], 
                       do_save=None, save_dpi=600, do_pltmean=True, do_pltstd=False ):    
    
    import matplotlib.patheffects as path_effects
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator

    if len(figsize)==0: figsize=[13,6.5]
    fig,ax= plt.figure(figsize=figsize),plt.gca()
    
    #___________________________________________________________________________
    # setup colormap
    if do_allcycl: 
        if which_cycl is not None:
            cmap = categorical_cmap(np.int32(len(moct_list)/which_cycl), which_cycl, cmap="tab10")
        else:
            cmap = categorical_cmap(len(moct_list), 1, cmap="tab10")
    else:
        cmap = categorical_cmap(len(moct_list), 1, cmap="tab10")
    
    #___________________________________________________________________________
    ii=0
    for ii_ts, (tseries, tname) in enumerate(zip(moct_list, input_names)):
        
        if tseries.ndim>1: tseries = tseries.squeeze()
        
        if np.mod(ii_ts+1,which_cycl)==0 or do_allcycl==False:
            
            hp=ax.plot(time,tseries, 
                    linewidth=2, label=tname, color=cmap.colors[ii_ts,:], 
                    marker='o', markerfacecolor='w', markersize=5, #path_effects=[path_effects.SimpleLineShadow(offset=(1.5,-1.5),alpha=0.3),path_effects.Normal()],
                    zorder=2)
                
            if do_pltmean: 
                # plot mean value with trinagle 
                plt.plot(time[0]-(time[-1]-time[0])*0.0120, tseries.mean(),
                        marker='<', markersize=8, markeredgecolor='k', markeredgewidth=0.5,
                        color=hp[0].get_color(),clip_box=False,clip_on=False, zorder=3)
            if do_pltstd:
                # plot std. range
                plt.plot(time[0]-(time[-1]-time[0])*0.015, tseries.mean()+tseries.std(),
                        marker='^', markersize=6, markeredgecolor='k', markeredgewidth=0.5,
                        color=hp[0].get_color(),clip_box=False,clip_on=False, zorder=3)
                
                plt.plot(time[0]-(time[-1]-time[0])*0.015, tseries.mean()-tseries.std(),
                        marker='v', markersize=6, markeredgecolor='k', markeredgewidth=0.5,
                        color=hp[0].get_color(),clip_box=False,clip_on=False, zorder=3)
        
        else:
            hp=ax.plot(time, tseries, 
                   linewidth=2, label=tname, color=cmap.colors[ii_ts,:],
                   zorder=1) #marker='o', markerfacecolor='w', 
                   # path_effects=[path_effects.SimpleLineShadow(offset=(1.5,-1.5),alpha=0.3),path_effects.Normal()])
        
    #___________________________________________________________________________
    if which_lat[ii]=='max':
        str_label='max {:s}: 45°N<=lat<=60°N'.format(which_moc.upper(),which_lat[ii])
    else:
        str_label='{:s} at: {:2.1f}°N'.format(which_moc.upper(),which_lat[ii])
    
    #___________________________________________________________________________
    ax.legend(shadow=True, fancybox=True, frameon=True, #mode='None', 
              bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
              #bbox_to_anchor=(1.04, 1.0), ncol=1) #loc='lower right', 
    ax.set_xlabel('Time [years]',fontsize=12)
    ax.set_ylabel('Density {:s} in [Sv]'.format(which_moc.upper()),fontsize=12)
    ax.set_title(str_label, fontsize=12, fontweight='bold')
    
    #___________________________________________________________________________
    xmajor_locator = MultipleLocator(base=xmaxstep) # this locator puts ticks at regular intervals
    ymajor_locator = MultipleLocator(base=ymaxstep) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.yaxis.set_major_locator(ymajor_locator)

    xminor_locator = AutoMinorLocator(5)
    yminor_locator = AutoMinorLocator(4)
    ax.yaxis.set_minor_locator(yminor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    
    plt.grid(which='major')
    plt.xlim(time[0]-(time[-1]-time[0])*0.015,time[-1]+(time[-1]-time[0])*0.015)    
    
    #___________________________________________________________________________
    plt.show()
    fig.canvas.draw()
    
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, fig, dpi=save_dpi)
    
    #___________________________________________________________________________
    return(fig,ax)


