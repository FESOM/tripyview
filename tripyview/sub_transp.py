import os
import numpy             as np
import time              as time
import xarray            as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .sub_colormap import *
from .sub_mesh     import vec_r2g
from .sub_plot     import *
from .sub_utility  import *

#+___COMPUTE MERIDIONAL HEATFLUX FROOM TRACER ADVECTION TROUGH BINNING_________+
#|                                                                             |
#+_____________________________________________________________________________+
def calc_mhflx(mesh, data, lat, edge, edge_tri, edge_dxdy_l, edge_dxdy_r):
    #___________________________________________________________________________
    vname_list = list(data.keys())
    vname, vname2 = vname_list[0], vname_list[1]
    
    #___________________________________________________________________________
    # Create xarray dataset
    list_dimname, list_dimsize = ['nlat'], [lat.size]
    
    data_vars = dict()
    aux_attr  = data[vname].attrs
    aux_attr['long_name'], aux_attr['units'] = 'Meridional Heat Transport', 'PW'
    data_vars['mhflx'] = (list_dimname, np.zeros(list_dimsize), aux_attr) 
    # define coordinates
    coords    = {'nlat' : (['nlat' ], lat )}
    # create dataset
    mhflx     = xr.Dataset(data_vars=data_vars, coords=coords, attrs=data.attrs)
    del(data_vars, coords, aux_attr)
    
    #___________________________________________________________________________
    # factors for heatflux computation
    rho0 = 1030 # kg/m^3
    cp   = 3850 # J/kg/K
    inPW = 1.0e-15
    
    #___________________________________________________________________________
    # coordinates of triangle centroids
    e_x  = mesh.n_x[mesh.e_i].sum(axis=1)/3.0
    e_y  = mesh.n_y[mesh.e_i].sum(axis=1)/3.0
    
    #___________________________________________________________________________
    # do zonal sum over latitudinal bins 
    for bini, lat_i in enumerate(lat):
        # indices of edges crossed by lat_i
        ind  = ((mesh.n_y[edge[0,:]]-lat_i)*(mesh.n_y[edge[1,:]]-lat_i) < 0.)
        ind2 =  (mesh.n_y[edge[0,:]] <= lat_i)
        if not np.any(ind): continue
        
        #_______________________________________________________________________
        edge_dxdy_l[:, ind2]=-edge_dxdy_l[:, ind2]
        edge_dxdy_r[:, ind2]=-edge_dxdy_r[:, ind2]
        
        #_______________________________________________________________________
        # here must be a rotation of tu1,tv1; tu2,tv2 dx1,dy1; dx2,dy2 from rot-->geo
        dx1, dx2 = edge_dxdy_l[0,ind], edge_dxdy_r[0,ind] 
        dy1, dy2 = edge_dxdy_l[1,ind], edge_dxdy_r[1,ind]
        tu1, tv1 = data[vname].values[edge_tri[0, ind], :], data[vname2].values[edge_tri[0, ind], :]
        tu2, tv2 = data[vname].values[edge_tri[1, ind], :], data[vname2].values[edge_tri[1, ind], :]
        
        # can not rotate them together
        #tu1, tv1 = vec_r2g(mesh.abg, e_x[edge_tri[0, ind]], e_y[edge_tri[0, ind]], tu1, tv1)
        #dx1, dy1 = vec_r2g(mesh.abg, e_x[edge_tri[0, ind]], e_y[edge_tri[0, ind]], dx1, dy1)
        #tu2, tv2 = vec_r2g(mesh.abg, e_x[edge_tri[1, ind]], e_y[edge_tri[1, ind]], tu2, tv2)
        #dx2, dy2 = vec_r2g(mesh.abg, e_x[edge_tri[1, ind]], e_y[edge_tri[1, ind]], dx2, dy2)
        tv1, tv2 = tv1.T, tv2.T
        tu1, tu2 = tu1.T, tu2.T
        #del(dum, dy1, dy2, tu1, tu2)
        
        #_______________________________________________________________________
        # integrate along latitude bin--> int(t*u)dx 
        tv_dx    = -np.nansum(tu1*dx1 + tu2*dx2 + tv1*dy1 + tv2*dy2, axis=1)
        #tv_dx    = np.nansum(tv1*np.abs(dx1) + tv2*np.abs(dx2), axis=1)
        
        #_______________________________________________________________________
        # integrate vertically --> int()dz
        mhflx['mhflx'].data[bini] = np.sum(tv_dx * np.abs(mesh.zlev[1:]-mesh.zlev[:-1])) *rho0*cp*inPW
        
        #_______________________________________________________________________
        edge_dxdy_l[:, ind2]=-edge_dxdy_l[:, ind2]
        edge_dxdy_r[:, ind2]=-edge_dxdy_r[:, ind2]
        
    #___________________________________________________________________________
    return(mhflx)


#+___COMPUTE MERIDIONAL HEATFLUX FROOM TRACER ADVECTION TROUGH BINNING_________+
#|                                                                             |
#+_____________________________________________________________________________+
def calc_mhflx_box(mesh, data, box_list, edge, edge_tri, edge_dxdy_l, edge_dxdy_r, 
                   datat=None, dlat=1.0, do_checkbasin=True, do_buflay=True):
    #___________________________________________________________________________
    vname_list = list(data.keys())
    vname, vname2 = vname_list[0], vname_list[1]
    u = data[vname].values.T.copy()
    v = data[vname2].values.T.copy()
    
    # in case you only wrote out u, v and temp instead of u*temp and v*temp
    if datat != None: 
        vnamet = list(datat.keys())[0]
        temp=datat[vnamet].values.T.copy()
        
    #___________________________________________________________________________
    # factors for heatflux computation
    rho0 = 1030 # kg/m^3
    cp   = 3850 # J/kg/K
    inPW = 1.0e-15
    
    #___________________________________________________________________________
    # coordinates of triangle centroids
    e_x  = mesh.n_x[mesh.e_i].sum(axis=1)/3.0
    e_y  = mesh.n_y[mesh.e_i].sum(axis=1)/3.0
    
    #___________________________________________________________________________
    # Loop over boxes
    list_mhflx=list()
    for box in box_list:
        if not isinstance(box, shp.Reader) and not box =='global' and not box==None :
            if len(box)==2: boxname, box = box[1], box[0]
        elif isinstance(box, shp.Reader):
            #boxname = box.shapeName.split('/')[-1].replace('_',' ')
            boxname = box.shapeName.split('/')[-1]
            boxname = boxname.split('_')[0].replace('_',' ')
            print(boxname)
        elif box =='global':    
            boxname = 'global'
        
        #_______________________________________________________________________
        # compute box mask index for nodes
        #n_idxin=do_boxmask(mesh,box,do_elem=False)
        if boxname=='global':
            n_idxin = np.ones(mesh.n2dn,dtype='bool')
        else:    
            e_idxin = do_boxmask(mesh,box,do_elem=True)
            e_i     = mesh.e_i[e_idxin,:]
            e_i     = np.unique(e_i.flatten())
            n_idxin = np.zeros(mesh.n2dn,dtype='bool')
            n_idxin[e_i]=True
            del(e_i, e_idxin)
            
        #_______________________________________________________________________
        # create meridional bins
        #lat  = np.arange(np.floor(mesh.n_y[n_idxin].min())-dlat/2, np.ceil(mesh.n_y[n_idxin].max())+dlat/2, dlat)
        lat  = np.arange(np.ceil(mesh.n_y[n_idxin].min())-dlat/2, np.floor(mesh.n_y[n_idxin].max())+dlat/2, dlat)
 
        #___________________________________________________________________
        if do_buflay and boxname!='global':
            # add a buffer layer of selected triangles --> sometimes it can be be that 
            # the shapefile is to small in this case boundary triangles might not get 
            # selected
            e_idxin = n_idxin[mesh.e_i].max(axis=1)
            e_i = mesh.e_i[e_idxin,:]
            e_i = np.unique(e_i.flatten())
            n_idxin[e_i]=True
            del(e_i, e_idxin)
 
        #__________________________________________________________________________
        # Create xarray dataset
        list_dimname, list_dimsize = ['nlat'], [lat.size]
        data_vars = dict()
        aux_attr  = data[vname].attrs
        #aux_attr['long_name']  = f'{boxname} Meridional Heat Transport'
        #aux_attr['short_name'] = f'{boxname} Merid. Heat Transp.'
        aux_attr['long_name']  = f'Meridional Heat Transport'
        aux_attr['short_name'] = f'Merid. Heat Transp.'
        aux_attr['boxname'] = boxname
        aux_attr['units']      = 'PW'
        data_vars['mhflx'] = (list_dimname, np.zeros(list_dimsize), aux_attr) 
        # define coordinates
        coords    = {'nlat' : (['nlat' ], lat )}
        # create dataset
        mhflx     = xr.Dataset(data_vars=data_vars, coords=coords, attrs=data.attrs)
        del(data_vars, coords, aux_attr)
    
        #___________________________________________________________________________
        # do zonal sum over latitudinal bins 
        #n_check= np.zeros(mesh.n2dn,dtype='bool')
        for bini, lat_i in enumerate(lat):
            # indices of edges crossed by lat_i
            if boxname=='global':
                ind  = ((mesh.n_y[edge[0,:]]-lat_i)*(mesh.n_y[edge[1,:]]-lat_i)<=0.0)
            else:    
                ind  = ((mesh.n_y[edge[0,:]]-lat_i)*(mesh.n_y[edge[1,:]]-lat_i)<=0.0) & ((n_idxin[edge[0,:]]==True) | (n_idxin[edge[1,:]]==True))
            ind2 = (mesh.n_y[edge[0,:]]<=lat_i) # & ((n_idxin[edge[0,:]]==True) | (n_idxin[edge[1,:]]==True))
            if not np.any(ind): continue
            
            #print(lat_i, np.sum(ind))
            #n_check[edge[0, ind]]=True
            #n_check[edge[1, ind]]=True
            #_______________________________________________________________________
            edge_dxdy_l[:, ind2]=-edge_dxdy_l[:, ind2]
            edge_dxdy_r[:, ind2]=-edge_dxdy_r[:, ind2]
            
            ##_______________________________________________________________________
            u1 , v1  = u[:, edge_tri[0,ind]], v[:, edge_tri[0,ind]]
            u2 , v2  = u[:, edge_tri[1,ind]], v[:, edge_tri[1,ind]]
            ny1, nx1 = edge_dxdy_l[1,ind]   , edge_dxdy_l[0,ind]
            ny2, nx2 = edge_dxdy_r[1,ind]   , edge_dxdy_r[0,ind]
            
            ##___________________________________________________________________
            ## extra rotation is not necessary !
            #u1, v1 = vec_r2g(mesh.abg, e_x[edge_tri[0, ind]], e_y[edge_tri[0, ind]], u1.T, v1.T)
            #u2, v2 = vec_r2g(mesh.abg, e_x[edge_tri[1, ind]], e_y[edge_tri[1, ind]], u2.T, v2.T)
            #u1, v1 = u1.T, v1.T
            #u2, v2 = u2.T, v2.T
            #nx1  , ny1   = vec_r2g(mesh.abg, e_x[edge_tri[0, ind]], e_y[edge_tri[0, ind]], nx1  , ny1)
            #nx2  , ny2   = vec_r2g(mesh.abg, e_x[edge_tri[1, ind]], e_y[edge_tri[1, ind]], nx2  , ny2)
            
            #___________________________________________________________________
            u1dy1, v1dx1=nx1*u1, ny1*v1
            u2dy2, v2dx2=nx2*u2, ny2*v2
            
            #___________________________________________________________________
            # compute u*t, v*t if data wasnt already ut,vt
            if datat != None: 
                etemp = (temp[:,edge[0, ind]] + temp[:, edge[1, ind]])*0.5
                u1dy1, v1dx1 = u1dy1*etemp, v1dx1*etemp
                u2dy2, v2dx2 = u2dy2*etemp, v2dx2*etemp
                
            #___________________________________________________________________
            # integrate along latitudinal bin 
            u1dy1, v1dx1 = np.nansum(u1dy1,axis=1), np.nansum(v1dx1,axis=1)
            u2dy2, v2dx2 = np.nansum(u2dy2,axis=1), np.nansum(v2dx2,axis=1)
            HT=-(u1dy1+v1dx1+u2dy2+v2dx2)
            
            #_______________________________________________________________________
            # integrate vertically --> int()dz
            mhflx['mhflx'].data[bini] = np.sum(np.diff(-mesh.zlev)*HT)*rho0*cp*inPW
            
            #_______________________________________________________________________
            edge_dxdy_l[:, ind2]=-edge_dxdy_l[:, ind2]
            edge_dxdy_r[:, ind2]=-edge_dxdy_r[:, ind2]
        
        #if do_checkbasin:
            #from matplotlib.tri import Triangulation
            #tri = Triangulation(np.hstack((mesh.n_x,mesh.n_xa)), np.hstack((mesh.n_y,mesh.n_ya)), np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia)))
            #plt.figure()
            #plt.triplot(tri, color='k')
            ##if do_onelem:
                ##plt.triplot(tri.x, tri.y, tri.triangles[ np.hstack((idxin[mesh.e_pbnd_0], idxin[mesh.e_pbnd_a])) ,:], color='r')
            ##else:
            #plt.plot(mesh.n_x[n_check], mesh.n_y[n_check], 'or', linestyle='None', markersize=2)
            #plt.title('Basin selection')
            #plt.show()
        
        #STOP
        #_______________________________________________________________________
        if len(box_list)==1: list_mhflx = mhflx
        else               : list_mhflx.append(mhflx)
    #___________________________________________________________________________
    return(list_mhflx)



#+___COMPUTE MERIDIONAL HEATFLUX FROOM TRACER ADVECTION TROUGH BINNING_________+
#|                                                                             |
#+_____________________________________________________________________________+
def calc_gmhflx(mesh, data, lat):
    #___________________________________________________________________________
    vname = list(data.keys())[0]
    
    #___________________________________________________________________________
    # Create xarray dataset
    list_dimname, list_dimsize = ['nlat'], [lat.size]
    data_vars = dict()
    aux_attr  = data[vname].attrs
    aux_attr['long_name'], aux_attr['units'] = 'Global Meridional Heat Transport', 'PW'
    data_vars['gmhflx'] = (list_dimname, np.zeros(list_dimsize), aux_attr) 
    # define coordinates
    coords    = {'nlat' : (['nlat' ], lat )}
    # create dataset
    ghflx     = xr.Dataset(data_vars=data_vars, coords=coords, attrs=data.attrs)
    del(data_vars, coords, aux_attr)
    
    #___________________________________________________________________________
    # factors for heatflux computation
    inPW = 1.0e-15
    
    #___________________________________________________________________________
    data[vname] = data[vname]*data['w_A']
    
    #___________________________________________________________________________
    # do zonal sum over latitudinal bins 
    dlat = lat[1]-lat[0]
    lat_i = (( mesh.n_y-lat[0])/dlat ).astype('int')    
    for bini in range(lat_i.min(), lat_i.max()):
        # sum over latitudinal bins
        ghflx['gmhflx'].data[bini] = data[vname].isel(nod2=lat_i==bini).sum(dim='nod2')*inPW

    #___________________________________________________________________________    
    # do cumulative sum over latitudes    
    ghflx['gmhflx'] = -ghflx['gmhflx'].cumsum(dim='nlat', skipna=True) 
    
    #___________________________________________________________________________
    return(ghflx)


#+___COMPUTE MERIDIONAL HEATFLUX FROM TRACER ADVECTION TROUGH BINNING_________+
#|                                                                             |
#+_____________________________________________________________________________+
def calc_gmhflx_box(mesh, data, box_list, dlat=1.0):
    list_ghflx=list()
    #___________________________________________________________________________
    vname       = list(data.keys())[0]
    data[vname] = data[vname]*data['w_A']
    # factors for heatflux computation
    inPW = 1.0e-15
    
    #___________________________________________________________________________
    # Loop over boxes
    for box in box_list:
        if not isinstance(box, shp.Reader) and not box =='global' and not box==None :
            if len(box)==2: boxname, box = box[1], box[0]
        elif isinstance(box, shp.Reader):
            #boxname = box.shapeName.split('/')[-1].replace('_',' ')
            boxname = box.shapeName.split('/')[-1]
            boxname = boxname.split('_')[0].replace('_',' ')
            print(boxname)
        elif box =='global':    
            boxname = 'global'
            
        #_______________________________________________________________________
        # compute box mask index for nodes 
        n_idxin=do_boxmask(mesh,box)
        
        #_______________________________________________________________________
        # do zonal sum over latitudinal bins 
        lat   = np.arange(np.floor(mesh.n_y[n_idxin].min())+dlat/2, 
                          np.ceil( mesh.n_y[n_idxin].max())-dlat/2, 
                          dlat)
        lat_i = (( mesh.n_y[n_idxin]-lat[0])/dlat ).astype('int')    
        
        #_______________________________________________________________________
        # Create xarray dataset
        list_dimname, list_dimsize = ['nlat'], [lat.size]
        data_vars = dict()
        aux_attr  = data[vname].attrs
        #aux_attr['long_name']  = f'{boxname} Meridional Heat Transport'
        #aux_attr['short_name'] = f'{boxname} Merid. Heat Transp.'
        aux_attr['long_name']  = f'Meridional Heat Transport'
        aux_attr['short_name'] = f'Merid. Heat Transp.'
        aux_attr['boxname']    = boxname
        aux_attr['units']      = 'PW'
        data_vars['gmhflx'] = (list_dimname, np.zeros(list_dimsize), aux_attr) 
        # define coordinates
        coords    = {'nlat' : (['nlat' ], lat )}
        # create dataset
        ghflx     = xr.Dataset(data_vars=data_vars, coords=coords, attrs=data.attrs)
        del(data_vars, coords, aux_attr)
        
        #_______________________________________________________________________
        data_box = data[vname].isel(nod2=n_idxin)
        for bini in range(lat_i.min(), lat_i.max()):
            # sum over latitudinal bins
            ghflx['gmhflx'].data[bini] = data_box.isel(nod2=lat_i==bini).sum(dim='nod2')*inPW
        
        #_______________________________________________________________________
        # do cumulative sum over latitudes    
        ghflx['gmhflx'] = -ghflx['gmhflx'].cumsum(dim='nlat', skipna=True) 
    
        #_______________________________________________________________________
        if len(box_list)==1: list_ghflx = ghflx
        else               : list_ghflx.append(ghflx)
        
    #___________________________________________________________________________
    return(list_ghflx)



#+___COMPUTE HORIZONTAL BAROTROPIC STREAM FUNCTION TROUGH BINNING______________+
#|                                                                             |
#+_____________________________________________________________________________+
def calc_hbarstreamf(mesh, data, lon, lat, edge, edge_tri, edge_dxdy_l, edge_dxdy_r):
    
    #___________________________________________________________________________
    vname_list = list(data.keys())
    vname, vname2 = vname_list[0], vname_list[1]
    
    #___________________________________________________________________________
    # Create xarray dataset
    list_dimname, list_dimsize = ['nlon','nlat'], [lon.size, lat.size]
    
    data_vars = dict()
    aux_attr  = data[vname].attrs
    aux_attr['long_name'], aux_attr['units'] = 'Horizontal. Barotropic \n Streamfunction', 'Sv'
    aux_attr['short_name']= 'Horiz. Barotr. Streamf.'
    data_vars['hbstreamf'] = (list_dimname, np.zeros(list_dimsize), aux_attr) 
    # define coordinates
    coords    = {'lon' : (['nlon' ], lon ), 'lat' : (['nlat' ], lat ), }
    
    # create dataset
    hbstreamf = xr.Dataset(data_vars=data_vars, coords=coords, attrs=data.attrs)
    del(data_vars, coords, aux_attr)
    
    #___________________________________________________________________________
    # factors for heatflux computation
    inSv = 1.0e-6
    #mask = np.zeros((list_dimsize))
    
    #___________________________________________________________________________
    # loop over longitudinal bins 
    for ix, lon_i in enumerate(lon):
        ind  = ((mesh.n_x[edge[0,:]]-lon_i)*(mesh.n_x[edge[1,:]]-lon_i) <= 0.) & (abs(mesh.n_x[edge[0,:]]-lon_i)<50.) & (abs(mesh.n_x[edge[1,:]]-lon_i)<50.)
        ind2 =  (mesh.n_x[edge[0,:]] <= lon_i)
            
        #_______________________________________________________________________
        edge_dxdy_l[:, ind2]=-edge_dxdy_l[:, ind2]
        edge_dxdy_r[:, ind2]=-edge_dxdy_r[:, ind2]
        
        #_______________________________________________________________________
        u1dxl = (edge_dxdy_l[0,ind] * data[vname ].values[edge_tri[0,ind],:].T)
        v1dyl = (edge_dxdy_l[1,ind] * data[vname2].values[edge_tri[0,ind],:].T)
        u2dxr = (edge_dxdy_r[0,ind] * data[vname ].values[edge_tri[1,ind],:].T)
        v2dyr = (edge_dxdy_r[1,ind] * data[vname2].values[edge_tri[1,ind],:].T)
        AUX   =-(u1dxl+v1dyl+u2dxr+v2dyr)*inSv
        del(u1dxl, v1dyl, u2dxr, v2dyr)
        
        #_______________________________________________________________________
        # loop over latitudinal bins 
        for iy in range(0, lat.size-1):
            iind=(mesh.n_y[edge[:,ind]].mean(axis=0)>lat[iy]) & (mesh.n_y[edge[:,ind]].mean(axis=0)<=lat[iy+1])
            hbstreamf['hbstreamf'].data[ix, iy] = np.nansum(np.diff(-mesh.zlev)*np.nansum(AUX[:,iind],axis=1))
            #if not np.any(iind): continue
            #mask[ix,iy] = 1
            
        #_______________________________________________________________________
        edge_dxdy_l[:, ind2]=-edge_dxdy_l[:, ind2]
        edge_dxdy_r[:, ind2]=-edge_dxdy_r[:, ind2]
        del(ind, ind2)
        
    #___________________________________________________________________________
    hbstreamf['hbstreamf'] =-hbstreamf['hbstreamf'].cumsum(dim='nlat', skipna=True)#+150.0 
    hbstreamf['hbstreamf'] = hbstreamf['hbstreamf'].transpose()
    hbstreamf['hbstreamf'].data = hbstreamf['hbstreamf'].data-hbstreamf['hbstreamf'].data[-1,:]
    
    # impose periodic boundary condition
    hbstreamf['hbstreamf'].data[:,-1] = hbstreamf['hbstreamf'].data[:,-2]
    hbstreamf['hbstreamf'].data[:,0] = hbstreamf['hbstreamf'].data[:,1]
    #mask[-1,:] = mask[0,:]
    
    # set land sea mask 
    #hbstreamf['hbstreamf'].data[mask.T==0] = np.nan
    
    #del(mask)
    #___________________________________________________________________________
    return(hbstreamf)



#+___COMPUTE HORIZONTAL BAROTROPIC STREAM FUNCTION TROUGH BINNING______________+
#|                                                                             |
#+_____________________________________________________________________________+
# try to take full advantage of xarray and dask
def calc_hbarstreamf_fast(mesh, data, lon, lat, do_info=True, do_parallel=True, n_workers=10):
    
    #___________________________________________________________________________
    # Create xarray dataset for hor. bar. streamf
    vname, vname2 = 'u', 'v'
    
    # define variable attributes    
    data_attr              = data[vname].attrs
    data_attr['long_name' ]= 'Horizontal. Barotropic \n Streamfunction'
    data_attr['short_name']= 'Horiz. Barotr. Streamf.'
    data_attr['units'     ]= 'Sv'
    
    # define variable 
    data_vars              = dict()
    dims                   = ['nlat', 'nlon']
    dims_size              = [lat.size-1, lon.size-1]
    data_vars['hbstreamf'] = (dims, np.zeros(dims_size, dtype='float32'), data_attr) 
    
    # define coordinates
    lon, lat = lon.astype('float32'), lat.astype('float32')
    coords    = {'lon' : (['nlon' ], (lon[1:]+lon[:-1])/2 ), 
                 'lat' : (['nlat' ], (lat[1:]+lat[:-1])/2 ), }
    
    # create dataset
    hbstreamf = xr.Dataset(data_vars=data_vars, coords=coords, attrs=data.attrs)
    del(data_vars, coords, data_attr)
    
    #___________________________________________________________________________
    # factors for volume flux computation
    inSv = 1.0e-6
    
    #___________________________________________________________________________
    # define function for longitudinal binning --> should be possible to parallelize
    # this loop since each lon bin is independent
    def hbstrfbin_over_lon(lon_i, lat, data):
        #_______________________________________________________________________
        # compute which edge is cutted by the binning line along longitude
        # to select the  lon bin cutted edges with where is by far the fastest option
        # compared to using ...
        # idx_lonbin  = (  (data.edge_x[0,:]-lon_i)*(data.edge_x[1,:]-lon_i) <= 0.) & \
        #                  (abs(data.edge_x[0,:]-lon_i)<50.) & (abs(data.edge_x[1,:]-lon_i)<50. )
        # data_lonbin = data.groupby(idx_lonbin)[True]
        # --> groupby is here a factor 5-6 slower than using isel+np.where
        data_lonbin = data.isel(edg_n=np.where(  
                        ((data.edge_x[0,:]-lon_i)*(data.edge_x[1,:]-lon_i) <= 0.) & \
                        (abs(data.edge_x[0,:]-lon_i)<50.) & (abs(data.edge_x[1,:]-lon_i)<50. ))[0])
        
        # change direction of edge to make it consistent
        idx_direct = data_lonbin.edge_x[0,:]<=lon_i
        data_lonbin.edge_dx_lr[:,idx_direct] = -data_lonbin.edge_dx_lr[:,idx_direct]
        data_lonbin.edge_dy_lr[:,idx_direct] = -data_lonbin.edge_dy_lr[:,idx_direct]
        del(idx_direct)
        
        # compute transport u,v --> u*dx,v*dy
        data_lonbin['u'] = data_lonbin['u'] * data_lonbin['edge_dx_lr'] * inSv * (-1)
        data_lonbin['v'] = data_lonbin['v'] * data_lonbin['edge_dy_lr'] * inSv * (-1)
        
        # sum already over vflux contribution from left and right triangle 
        data_lonbin['u'] = data_lonbin['u'].sum(dim='n2', skipna=True) * data_lonbin['dz']
        data_lonbin['v'] = data_lonbin['v'].sum(dim='n2', skipna=True) * data_lonbin['dz']
        
        #_______________________________________________________________________
        # loop over latitudinal bins, here somehow using groupby_bins is faster 
        # than using a for loop with np.where(...)...
        # for iy, lat_i in enumerate(lat):
        #     data_latbin = data_lonbin.isel(edg_n=np.where( 
        data_latbin = data_lonbin.groupby_bins('edge_my',lat)
        del(data_lonbin)
        data_latbin = data_latbin.sum(skipna=True).sum(dim='nz1', skipna=True)
        return(data_latbin['u'] + data_latbin['v'])
    
    #___________________________________________________________________________
    # do serial loop over longitudinal bins
    if not do_parallel:
        ts = ts1 = clock.time()
        if do_info: print('\n ___loop over longitudinal bins___'+'_'*90, end='\n')
        for ix, lon_i in enumerate(hbstreamf.lon):
            #_______________________________________________________________________
            if do_info: print('{:+06.1f}|'.format(lon_i), end='')
            if np.mod(ix+1,15)==0 and do_info:
                print(' > time: {:2.1f} sec.'.format((clock.time()-ts1)), end='\n')
                ts1 = clock.time()
            hbstreamf['hbstreamf'][:,ix] = hbstrfbin_over_lon(lon_i) #, lat, data)
    
    # do parallel loop over longitudinal bins        
    else:
        ts = ts1 = clock.time()
        if do_info: print('\n ___parallel loop over longitudinal bins___'+'_'*1, end='\n')
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_workers)(delayed(hbstrfbin_over_lon)(lon_i, lat, data) for lon_i in hbstreamf.lon)
        hbstreamf['hbstreamf'][:,:] = xr.concat(results, dim='nlon').transpose()
        
    #___________________________________________________________________________
    hbstreamf['hbstreamf'] =-hbstreamf['hbstreamf'].cumsum(dim='nlat', skipna=True)#+150.0 
    #hbstreamf['hbstreamf'] = hbstreamf['hbstreamf'].transpose()
    hbstreamf['hbstreamf'].data = hbstreamf['hbstreamf'].data-hbstreamf['hbstreamf'].data[-1,:]
    
    # impose periodic boundary condition
    hbstreamf['hbstreamf'].data[:,-1] = hbstreamf['hbstreamf'].data[:,-2]
    hbstreamf['hbstreamf'].data[:, 0] = hbstreamf['hbstreamf'].data[:, 1]
    if do_info: print(' --> total elasped time: {:3.3f} min.'.format((clock.time()-ts)/60))      
    
    #___________________________________________________________________________
    return(hbstreamf)




#+___PLOT MERIDIONAL HEAT FLUX OVER LATITUDES__________________________________+
#|                                                                             |
#+_____________________________________________________________________________+
def plot_mhflx(mhflx_list, input_names, sect_name=None, str_descript='', str_time='', figsize=[], 
               do_allcycl=False, do_save=None, save_dpi=300,):   
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator
    if len(figsize)==0: figsize=[7,3.5]
    fig,ax= plt.figure(figsize=figsize),plt.gca()
    
    #___________________________________________________________________________
    # setup colormap
    #if do_allcycl: 
        #if which_cycl is not None:
            #cmap = categorical_cmap(np.int32(len(mhflx_list)/which_cycl), which_cycl, cmap="tab10")
        #else:
            #cmap = categorical_cmap(len(mhflx_list), 1, cmap="tab10")
    #else:
    cmap = categorical_cmap(len(mhflx_list), 1, cmap="tab10")
    lstyle = ['-','--','-.',':']
    str_units, str_ltim = None, None
    xmin, xmax = np.inf, -np.inf
    #___________________________________________________________________________
    for ii_ts, (datap, datap_name) in enumerate(zip(mhflx_list, input_names)):
        if isinstance(datap,list)  :
            for jj_ts, datap1 in enumerate(datap):
                vname = list(datap1.keys())[0]
                boxname   = datap1[vname].attrs['boxname']
                
                if 'units'    in datap1[vname].attrs.keys(): str_units = datap1[vname].attrs['units']
                if 'str_ltim' in datap1[vname].attrs.keys(): str_ltim  = datap1[vname].attrs['str_ltim']
                #_______________________________________________________________
                datap_x, datap_y = datap1['nlat'].values, datap1[vname].values
                hp=ax.plot(datap_x, datap_y, 
                        linewidth=1, linestyle=lstyle[jj_ts], label=f"{datap_name} {boxname}", color=cmap.colors[ii_ts,:], 
                        marker='None', markerfacecolor='w', markersize=5, 
                        zorder=2)
                #_______________________________________________________________
                xmin = np.min([xmin, datap_x.min()])
                xmax = np.max([xmax, datap_x.max()])
                
        else:
            vname = list(datap.keys())[0]
            if 'units'    in datap[vname].attrs.keys(): str_units = datap[vname].attrs['units']
            if 'str_ltim' in datap[vname].attrs.keys(): str_ltim  = datap[vname].attrs['str_ltim']
            #___________________________________________________________________
            datap_x, datap_y = datap['nlat'].values, datap[vname].values
            hp=ax.plot(datap_x, datap_y, 
                    linewidth=1, label=datap_name, color=cmap.colors[ii_ts,:], 
                    marker='None', markerfacecolor='w', markersize=5, 
                    zorder=2)
            #_______________________________________________________________
            xmin = np.min([xmin, datap_x.min()])
            xmax = np.max([xmax, datap_x.max()])     
            
    #___________________________________________________________________________
    #ax.legend(shadow=True, fancybox=True, frameon=True, bbox_to_anchor=(1.02,0.5), loc="center left", borderaxespad=0)
    ax.legend(shadow=True, fancybox=True, frameon=True, loc="lower right")
    ax.set_xlabel('Latitude [deg]',fontsize=12)
    if   vname == 'gmhflx': y_label = 'Global Meridional Heat Transport'
    elif vname == 'mhflx' : y_label = 'Meridional Heat Transport'
    
    if str_units is not None: y_label = y_label + ' [' + str_units +']'
    if str_ltim  is not None: y_label = y_label + '\n'+str_ltim
    ax.set_ylabel(y_label, fontsize=12)  
        
    #___________________________________________________________________________
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=20))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    plt.grid(which='major')
    plt.xlim(xmin-(xmax-xmin)*0.015,xmax+(xmax-xmin)*0.015)    
        
    #___________________________________________________________________________
    plt.show()
    fig.canvas.draw()
    
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, fig, dpi=save_dpi)
    
    #___________________________________________________________________________
    return(fig, ax)



#+___PLOT TIME SERIES OF TRANSPORT THROUGH SECTION_____________________________+
#|                                                                             |
#+_____________________________________________________________________________+
def plot_vflx_tseries(time, tseries_list, input_names, sect_name, which_cycl=None, 
                       do_allcycl=False, do_concat=False, str_descript='', str_time='', figsize=[], 
                       do_save=None, save_dpi=600, do_pltmean=True, do_pltstd=False,
                       ymaxstep=5, xmaxstep=5):    
    
    import matplotlib.patheffects as path_effects
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator

    if len(figsize)==0: figsize=[13,6.5]
    if do_concat: figsize[0] = figsize[0]*2
    fig,ax= plt.figure(figsize=figsize),plt.gca()
    
    #___________________________________________________________________________
    # setup colormap
    if do_allcycl: 
        if which_cycl is not None:
            cmap = categorical_cmap(np.int32(len(tseries_list)/which_cycl), which_cycl, cmap="tab10")
        else:
            cmap = categorical_cmap(len(tseries_list), 1, cmap="tab10")
    else:
        if do_concat: do_concat=False
        cmap = categorical_cmap(len(tseries_list), 1, cmap="tab10")
    
    #___________________________________________________________________________
    ii, ii_cycle = 0, 1
    if which_cycl is None: aux_which_cycl = 1
    else                 : aux_which_cycl = which_cycl
    
    for ii_ts, (tseries, tname) in enumerate(zip(tseries_list, input_names)):
        
        if tseries.ndim>1: tseries = tseries.squeeze()
        auxtime = time.copy()
        if np.mod(ii_ts+1,aux_which_cycl)==0 or do_allcycl==False:
            
            if do_concat: auxtime = auxtime + (time[-1]-time[0]+1)*(ii_cycle-1)
            hp=ax.plot(auxtime,tseries, 
                   linewidth=1.5, label=tname, color=cmap.colors[ii_ts,:], 
                   marker='o', markerfacecolor='w', markersize=5, #path_effects=[path_effects.SimpleLineShadow(offset=(1.5,-1.5),alpha=0.3),path_effects.Normal()],
                   zorder=2)
            
            if do_pltmean: 
                # plot mean value with triangle 
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
            if do_concat: auxtime = auxtime + (time[-1]-time[0]+1)*(ii_cycle-1)
            hp=ax.plot(auxtime, tseries, 
                   linewidth=1.5, label=tname, color=cmap.colors[ii_ts,:],
                   zorder=1) #marker='o', markerfacecolor='w', 
                   # path_effects=[path_effects.SimpleLineShadow(offset=(1.5,-1.5),alpha=0.3),path_effects.Normal()])
        
        ii_cycle=ii_cycle+1
        if ii_cycle>aux_which_cycl: ii_cycle=1
        
    #___________________________________________________________________________
    ax.legend(shadow=True, fancybox=True, frameon=True, #mode='None', 
              bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
              #bbox_to_anchor=(1.04, 1.0), ncol=1) #loc='lower right', 
    ax.set_xlabel('Time [years]',fontsize=12)
    ax.set_ylabel('{:s} in [Sv]'.format('Transport'),fontsize=12)
    ax.set_title(sect_name, fontsize=12, fontweight='bold')
    
    #___________________________________________________________________________
    if do_concat: xmaxstep=20
    xmajor_locator = MultipleLocator(base=xmaxstep) # this locator puts ticks at regular intervals
    ymajor_locator = MultipleLocator(base=ymaxstep) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.yaxis.set_major_locator(ymajor_locator)
    
    if not do_concat:
        xminor_locator = AutoMinorLocator(5)
        yminor_locator = AutoMinorLocator(4)
        ax.yaxis.set_minor_locator(yminor_locator)
        ax.xaxis.set_minor_locator(xminor_locator)
    
    plt.grid(which='major')
    if not do_concat:
        plt.xlim(time[0]-(time[-1]-time[0])*0.015,time[-1]+(time[-1]-time[0])*0.015)    
    else:    
        plt.xlim(time[0]-(time[-1]-time[0])*0.015,time[-1]+(time[-1]-time[0]+1)*(aux_which_cycl-1)+(time[-1]-time[0])*0.015)    
    
    #___________________________________________________________________________
    plt.show()
    fig.canvas.draw()
    
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, fig, dpi=save_dpi)
    
    #___________________________________________________________________________
    return(fig,ax)



def plot_hbstreamf(mesh, data, input_names, cinfo=None, box=None, proj='pc', figsize=[9,4.5], 
                n_rc=[1,1], do_grid=False, do_plot='tcf', do_rescale=False,
                do_reffig=False, ref_cinfo=None, ref_rescale=False,
                cbar_nl=8, cbar_orient='vertical', cbar_label=None, cbar_unit=None,
                do_lsmask='fesom', do_bottom=True, color_lsmask=[0.6, 0.6, 0.6], 
                color_bot=[0.75,0.75,0.75],  title=None,
                pos_fac=1.0, pos_gap=[0.02, 0.02], pos_extend=None, do_save=None, save_dpi=600,
                linecolor='k', linewidth=0.5, ):
    #____________________________________________________________________________
    fontsize = 12
    rescale_str = None
    rescale_val = 1.0
    
    #___________________________________________________________________________
    # make matrix with row colum index to know where to put labels
    rowlist = np.zeros((n_rc[0],n_rc[1]))
    collist = np.zeros((n_rc[0],n_rc[1]))       
    for ii in range(0,n_rc[0]): rowlist[ii,:]=ii
    for ii in range(0,n_rc[1]): collist[:,ii]=ii
    rowlist = rowlist.flatten()
    collist = collist.flatten()
    
    #___________________________________________________________________________
    # create box if not exist
    if box is None or box=="None": box = [ -180, 180, -90, 90 ]
    
    #___________________________________________________________________________
    ticknr=7
    tickstep = np.array([0.5,1.0,2.0,2.5,5.0,10.0,15.0,20.0,30.0,45.0, 360])
    idx     = int(np.argmin(np.abs( tickstep-(box[1]-box[0])/ticknr )))
    if np.abs(box[1]-box[0])==360:
        xticks   = np.arange(-180, 180+1, tickstep[idx])
    else:    
        xticks   = np.arange(box[0], box[1]+1, tickstep[idx]) 
    idx     = int(np.argmin(np.abs( tickstep-(box[3]-box[2])/ticknr )))
    if np.abs(box[3]-box[2])==180:
        yticks   = np.arange(-90, 90+1, tickstep[idx])  
    else:   
        yticks   = np.arange(box[2], box[3]+1, tickstep[idx])  
    xticks = np.unique(np.hstack((box[0], xticks, box[1])))
    yticks = np.unique(np.hstack((box[2], yticks, box[3])))
     
    #___________________________________________________________________________
    # Create projection
    if proj=='pc':
        which_proj=ccrs.PlateCarree()
        which_transf = None
    elif proj=='merc':
        which_proj=ccrs.Mercator()    
        which_transf = ccrs.PlateCarree()
    elif proj=='nps':    
        which_proj=ccrs.NorthPolarStereo()    
        which_transf = ccrs.PlateCarree()
        if box[2]<0: box[2]=0
    elif proj=='sps':        
        which_proj=ccrs.SouthPolarStereo()    
        which_transf = ccrs.PlateCarree()
        if box[3]>0: box[2]=0
    elif proj=='rob':        
        which_proj=ccrs.Robinson()    
        which_transf = ccrs.PlateCarree() 
        
    #___________________________________________________________________________    
    # create figure and axes
    fig, ax = plt.subplots( n_rc[0],n_rc[1], figsize=figsize, 
                            subplot_kw =dict(projection=which_proj),
                            gridspec_kw=dict(left=0.06, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05,),
                            constrained_layout=False)

    #___________________________________________________________________________    
    # flatt axes if there are more than 1
    if isinstance(ax, np.ndarray): ax = ax.flatten()
    else:                          ax = [ax] 
    nax = len(ax)

    #___________________________________________________________________________
    # data must be list filled with xarray data
    if not isinstance(data, list): data = [data]
    ndata = len(data)
    
    #___________________________________________________________________________
    # set up color info
    if do_reffig:
        ref_cinfo = do_setupcinfo(ref_cinfo, [data[0]], ref_rescale, do_hbstf=True)
        cinfo     = do_setupcinfo(cinfo    , data[1:] , do_rescale , do_hbstf=True)
    else:    
        cinfo     = do_setupcinfo(cinfo, data, do_rescale, do_hbstf=True)

    #___________________________________________________________________________
    # setup normalization log10, symetric log10, None
    which_norm = do_compute_scalingnorm(cinfo, do_rescale)
    if do_reffig:
        which_norm_ref = do_compute_scalingnorm(ref_cinfo, ref_rescale)
    
    #___________________________________________________________________________
    # loop over axes
    hpall=list()
    for ii in range(0,ndata):
        #_______________________________________________________________________
        # set axes extent
        ax[ii].set_extent(box, crs=ccrs.PlateCarree())
        
        #_______________________________________________________________________
        # periodic augment data
        vname = list(data[ii].keys())[0]
        data_plot = data[ii][vname].data.copy()
        data_x    = data[ii]['lon']
        data_y    = data[ii]['lat']
        
        #_______________________________________________________________________
        if do_reffig: 
            if ii==0: cinfo_plot, which_norm_plot = ref_cinfo, which_norm_ref
            else    : cinfo_plot, which_norm_plot = cinfo    , which_norm
        else        : cinfo_plot, which_norm_plot = cinfo    , which_norm
        
        #_______________________________________________________________________
        # plot tri contourf/pcolor
        if   do_plot=='tpc':
            hp=ax[ii].pcolor(data_x, data_y, data_plot,
                             shading='flat',
                             cmap=cinfo_plot['cmap'],
                             vmin=cinfo_plot['clevel'][0], vmax=cinfo_plot['clevel'][ -1],
                             norm = which_norm_plot)
            
        elif do_plot=='tcf': 
            # supress warning message when compared with nan
            with np.errstate(invalid='ignore'):
                data_plot[data_plot<cinfo_plot['clevel'][ 0]] = cinfo_plot['clevel'][ 0]
                data_plot[data_plot>cinfo_plot['clevel'][-1]] = cinfo_plot['clevel'][-1]
            
            hp=ax[ii].contourf(data_x, data_y, data_plot, 
                               transform=which_transf,
                               norm=which_norm_plot,
                               levels=cinfo_plot['clevel'], cmap=cinfo_plot['cmap'], extend='both')
        hpall.append(hp)  
        
        #_______________________________________________________________________
        # add mesh land-sea mask
        ax[ii] = do_plotlsmask(ax[ii],mesh, do_lsmask, box, which_proj,
                               color_lsmask=color_lsmask, edgecolor=linecolor, linewidth=linewidth)
        
        #_______________________________________________________________________
        # add gridlines
        ax[ii] = do_add_gridlines(ax[ii], rowlist[ii], collist[ii], 
                                  xticks, yticks, proj, which_proj)
        
        #_______________________________________________________________________
        # set title and axes labels
        if title is not None: 
            # is title  string:
            if   isinstance(title,str) : 
                # if title string is 'descript' than use descript attribute from 
                # data to set plot title 
                if title=='descript' and ('descript' in data[ii][ vname ].attrs.keys() ):
                    ax[ii].set_title(data[ii][ vname ].attrs['descript'], fontsize=fontsize+2)
                    
                else:
                    ax[ii].set_title(title, fontsize=fontsize+2)
            # is title list of string        
            elif isinstance(title,list): ax[ii].set_title(title[ii], fontsize=fontsize+2)
    nax_fin = ii+1

    #___________________________________________________________________________
    # delete axes that are not needed
    #for jj in range(nax_fin, nax): fig.delaxes(ax[jj])
    for jj in range(ndata, nax): fig.delaxes(ax[jj])
    if nax != nax_fin-1: ax = ax[0:nax_fin]
   
    #___________________________________________________________________________
    # create colorbar
    if do_reffig==False:
        cbar = plt.colorbar(hp, orientation=cbar_orient, ax=ax, ticks=cinfo['clevel'], 
                        extendrect=False, extendfrac=None,
                        drawedges=True, pad=0.025, shrink=1.0,)                      
        
        # do formatting of colorbar 
        cbar = do_cbar_formatting(cbar, do_rescale, cbar_nl, fontsize, cinfo['clevel'])
        
        # do labeling of colorbar
        if cbar_label is None: cbar_label = data[nax_fin-1][vname].attrs['long_name']
        if cbar_unit  is None: cbar_label = cbar_label+' ['+data[0][vname].attrs['units']+']'
        else:                  cbar_label = cbar_label+' ['+cbar_unit+']'
        if 'str_ltim' in data[0][vname].attrs.keys():
            cbar_label = cbar_label+', '+data[0][vname].attrs['str_ltim']
        if 'str_ldep' in data[0][vname].attrs.keys():
            cbar_label = cbar_label+data[0][vname].attrs['str_ldep']
        cbar.set_label(cbar_label, size=fontsize+2)
    else:
        cbar=list()
        for ii, aux_ax in enumerate(ax): 
            cbar_label =''
            if ii==0: 
                aux_cbar = plt.colorbar(hpall[ii], orientation=cbar_orient, ax=aux_ax, ticks=ref_cinfo['clevel'], 
                            extendrect=False, extendfrac=None, drawedges=True, pad=0.025, shrink=1.0,)  
                aux_cbar = do_cbar_formatting(aux_cbar, ref_rescale, cbar_nl, fontsize, ref_cinfo['clevel'])
            else:     
                aux_cbar = plt.colorbar(hpall[ii], orientation=cbar_orient, ax=aux_ax, ticks=cinfo['clevel'], 
                            extendrect=False, extendfrac=None, drawedges=True, pad=0.025, shrink=1.0,)  
                aux_cbar = do_cbar_formatting(aux_cbar, do_rescale, cbar_nl, fontsize, cinfo['clevel'])
                #cbar_label ='anom. '
            # do labeling of colorbar
            #if cbar_label is None : 
            if   'short_name' in data[ii][vname].attrs:
                cbar_label = cbar_label+data[ii][ vname ].attrs['short_name']
            elif 'long_name' in data[ii][vname].attrs:
                cbar_label = cbar_label+data[ii][ vname ].attrs['long_name']
            if cbar_unit  is None : cbar_label = cbar_label+' ['+data[ii][ vname ].attrs['units']+']'
            else                  : cbar_label = cbar_label+' ['+cbar_unit+']'
            if 'str_ltim' in data[ii][vname].attrs.keys():
                cbar_label = cbar_label+'\n'+data[ii][vname].attrs['str_ltim']
            aux_cbar.set_label(cbar_label, size=fontsize+2)
            cbar.append(aux_cbar)    
    
    #___________________________________________________________________________
    # repositioning of axes and colorbar
    if do_reffig==False:
        ax, cbar = do_reposition_ax_cbar(ax, cbar, rowlist, collist, pos_fac, 
                                        pos_gap, title=title, proj=proj, extend=pos_extend)
    fig.canvas.draw()
    
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, fig, dpi=save_dpi, transparent=True)
    plt.show(block=False)
    
    #___________________________________________________________________________
    return(fig, ax, cbar)












