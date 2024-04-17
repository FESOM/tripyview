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
    gattr = data.attrs
    gattrs['proj'] = 'index+xy'
    aux_attr  = data[vname].attrs
    aux_attr['long_name'], aux_attr['units'] = 'Meridional Heat Transport', 'PW'
    data_vars['mhflx'] = (list_dimname, np.zeros(list_dimsize), aux_attr) 
    # define coordinates
    coords    = {'nlat' : (['nlat' ], lat )}
    # create dataset
    mhflx     = xr.Dataset(data_vars=data_vars, coords=coords, attrs=gattr)
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
        list_dimname, list_dimsize = ['lat'], [lat.size]
        data_vars = dict()
        gattrs = data.attrs
        gattrs['proj'] = 'index+xy'
        aux_attr  = data[vname].attrs
        #aux_attr['long_name']  = f'{boxname} Meridional Heat Transport'
        #aux_attr['short_name'] = f'{boxname} Merid. Heat Transp.'
        aux_attr['long_name']  = f'Meridional Heat Transport'
        aux_attr['short_name'] = f'Merid. Heat Transp.'
        aux_attr['boxname'] = boxname
        aux_attr['units']      = 'PW'
        data_vars['mhflx'] = (list_dimname, np.zeros(list_dimsize), aux_attr) 
        # define coordinates
        coords    = {'lat' : (['lat' ], lat )}
        # create dataset        
        mhflx     = xr.Dataset(data_vars=data_vars, coords=coords, attrs=gattrs)
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
def calc_mhflx_box_fast(mesh, data, box_list, dlat=1.0, do_info=True, do_checkbasin=False, 
                        do_compute=False, do_load=True, do_persist=False, do_buflay=True, 
                        do_parallel=False, n_workers=10, 
                        ):
    #___________________________________________________________________________
    vname_list = list(data.keys())
    vnamet = None
    if 'temp' in vname_list:
        vnamet = 'temp'
        vname_list.remove('temp')
    for vi in vname_list:
        if 'u' in vi: vnameu=vi
        if 'v' in vi: vnamev=vi
    #vnameu, vnamev = vname_list[0], vname_list[1]
    
    # save global and local variable attributes
    gattrs = data.attrs
    gattrs['proj'] = 'index+xy'
    vattr = data[vnameu].attrs

    #___________________________________________________________________________
    # factors for heatflux computation
    rho0 = 1030 # kg/m^3
    cp   = 3850 # J/kg/K
    inPW = 1.0e-15
    
    #_______________________________________________________________________
    # define function for longitudinal binning --> should be possible to 
    # parallelize this loop since each lon bin is independent
    def sum_over_latbin(lat_i, data_box, vnameu, vnamev, vnamet):
        #___________________________________________________________________
        # compute which edge is cutted by the binning line along latitude
        # to select the  lat bin cutted edges with where is by far the fastest option
        # compared to using ...
        # idx_latbin  = (  (data.edge_y[0,:]-lat_i)*(data.edge_y[1,:]-lat_i) <= 0.) & \
        #                 )
        # data_lonbin = data.groupby(idx_lonbin)[True]
        # --> groupby is here a factor 5-6 slower than using isel+np.where
        data_latbin = data_box.isel(edg_n=np.where(  
                    (data_box.edge_y[0,:]-lat_i)*(data_box.edge_y[1,:]-lat_i) <= 0.0)[0])
        
        # change direction of edge to make it consistent
        idx_direct = data_latbin.edge_y[0,:]<=lat_i
        data_latbin.edge_dx_lr[:,idx_direct] = -data_latbin.edge_dx_lr[:,idx_direct]
        data_latbin.edge_dy_lr[:,idx_direct] = -data_latbin.edge_dy_lr[:,idx_direct]
        del(idx_direct)
        
        # make sure that value of right boundary triangle is zero when boundary edge 
        data_latbin[vnameu][1, data_latbin.edge_tri[1,:]<0 ,:] = 0.0
        data_latbin[vnamev][1, data_latbin.edge_tri[1,:]<0 ,:] = 0.0
        
        # compute transport u,v --> u*dx,v*dy
        data_latbin[vnameu] = data_latbin[vnameu] * data_latbin['edge_dx_lr'] 
        data_latbin[vnamev] = data_latbin[vnamev] * data_latbin['edge_dy_lr'] 
        
        # compute u*t, v*t if data wasnt already ut,vt
        if vnamet is not None:
            data_latbin[vnameu] = data_latbin[vnameu]*(data_latbin[vnamet][0,:]+data_latbin[vnamet][1,:])*0.5
            data_latbin[vnamev] = data_latbin[vnamev]*(data_latbin[vnamet][0,:]+data_latbin[vnamet][1,:])*0.5
        
        # multiply with layer thickness
        data_latbin[vnameu] = data_latbin[vnameu] * data_latbin['dz'] 
        data_latbin[vnamev] = data_latbin[vnamev] * data_latbin['dz']
         
        # sum already over vflux contribution from left and right triangle 
        # and over cutted edges 
        data_latbin[vnameu] = data_latbin[vnameu].sum(dim=['n2','edg_n'], skipna=True)
        data_latbin[vnamev] = data_latbin[vnamev].sum(dim=['n2','edg_n'], skipna=True)
        
        # integrate vertically
        data_latbin = data_latbin.sum(dim='nz1', skipna=True) * rho0*cp*inPW*(-1)
            
        return(data_latbin[vnameu] + data_latbin[vnamev])
        
    #___________________________________________________________________________
    # Loop over boxes/regions/shapefiles ...
    list_mhflx=list()
    for box in box_list:
        #_______________________________________________________________________
        if not isinstance(box, shp.Reader):
            if len(box)==2: boxname, box = box[1], box[0]
            if box is None or box=='global': boxname='global'
        else:     
            boxname = os.path.basename(box.shapeName)
            boxname = boxname.split('_')[0].replace('_',' ')  
        
        #_______________________________________________________________________
        # select box area
        if box=='global':
            data_box = data
            n_idxin = np.ones(mesh.n2dn,dtype='bool')
        else:     
            #___________________________________________________________________
            # compute box mask index for nodes
            #  --> sometimes it can be be that the shapefile is to small in this case 
            #  boundary triangles might not get selected therefore we if any node
            #  points of an edge triangle is within the shapefile
            e_idxin = do_boxmask(mesh,box,do_elem=True)
            e_i     = mesh.e_i[e_idxin,:]
            e_i     = np.unique(e_i.flatten())
            n_idxin = np.zeros(mesh.n2dn,dtype='bool')
            n_idxin[e_i]=True
            n_idxin_b = n_idxin.copy()
            del(e_i, e_idxin)
            if do_buflay:
                # add a buffer layer of selected triangles --> sometimes it can be be that 
                # the shapefile is to small in this case boundary triangles might not get 
                # selected. Therefor extend the selection of edges 
                e_idxin = n_idxin[mesh.e_i].max(axis=1)
                e_i = mesh.e_i[e_idxin,:]
                e_i = np.unique(e_i.flatten())
                n_idxin_b[e_i]=True
                del(e_i, e_idxin)
            idx_IN   = n_idxin_b[data.edges].any(axis=0)
            data_box = data.isel({'edg_n':idx_IN})
            #___________________________________________________________________
            if do_checkbasin:
                from matplotlib.tri import Triangulation
                tri = Triangulation(np.hstack((mesh.n_x,mesh.n_xa)), np.hstack((mesh.n_y,mesh.n_ya)), np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia)))
                plt.figure()
                ax = plt.gca()
                plt.triplot(tri, color='k')
                plt.plot(data_box.edge_x[0,:], data_box.edge_y[0,:], '*r', linestyle='None', markersize=1)
                plt.plot(data_box.edge_x[1,:], data_box.edge_y[1,:], '*r', linestyle='None', markersize=1)
                plt.title('Basin selection')
                plt.show()
            
        #_______________________________________________________________________
        # do zonal sum over latitudinal bins 
        lat_min  = np.ceil( mesh.n_y[n_idxin].min())  
        lat_max  = np.floor(mesh.n_y[n_idxin].max())
        lat      = np.arange(lat_min+dlat/2, lat_max-dlat/2, dlat )
        
        #_______________________________________________________________________
        # Create xarray dataset
        list_dimname, list_dimsize = ['nlat'], [lat.size]
        data_vars = dict()
        # define variable attributes 
        vattr['long_name' ] = f'Meridional Heat Transport'
        vattr['short_name'] = f'Merid. Heat Transp.'
        vattr['boxname'   ] = boxname
        vattr['units'     ] = 'PW'
        # define data_vars dict, coordinate dict, as well as list of dimension name 
        # and size 
        data_vars, coords, dim_n, dim_s,  = dict(), dict(), list(), list()
        if 'time' in list(data_box.dims): dim_n.append('time')
        dim_n.append('lat'); 
        for dim_ni in dim_n:
            if   dim_ni=='time': dim_s.append(data_box.sizes['time']); coords['time' ]=(['time'], data_box['time'].data ) 
            elif dim_ni=='lat' : dim_s.append(lat.size          ); coords['lat'  ]=(['lat' ], lat          ) 
        data_vars['mhflx'] = (dim_n, np.zeros(dim_s, dtype='float32')*np.nan, vattr) 
        # create dataset
        mhflx     = xr.Dataset(data_vars=data_vars, coords=coords, attrs=gattrs)
        del(data_vars, coords, dim_n, dim_s, lat)
        
        #_______________________________________________________________________
        # do serial loop over bins
        if not do_parallel:
            ts = ts1 = clock.time()
            if do_info: print('\n ___loop over latitudinal bins___'+'_'*90, end='\n')
            for iy, lat_i in enumerate(mhflx.lat):
                #_______________________________________________________________
                if do_info: print('{:+06.1f}|'.format(lat_i), end='')
                if np.mod(iy+1,15)==0 and do_info:
                    print(' > time: {:2.1f} sec.'.format((clock.time()-ts1)), end='\n')
                    ts1 = clock.time()
                if 'time' in data_box.dims: mhflx['mhflx'][:,iy] = sum_over_latbin(lat_i, data_box, vnameu, vnamev, vnamet)
                else                      : mhflx['mhflx'][  iy] = sum_over_latbin(lat_i, data_box, vnameu, vnamev, vnamet)       
        
        # do parallel loop over bins        
        else:
            ts = ts1 = clock.time()
            if do_info: print('\n ___parallel loop over latitudinal bins___'+'_'*1, end='\n')
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_workers)(delayed(sum_over_latbin)(lat_i, data_box, vnameu, vnamev, vnamet) for lat_i in mhflx.lat)
            if 'time' in data_box.dims: mhflx['mhflx'][:,:] = xr.concat(results, dim='lat').transpose('time','lat')
            else                      : mhflx['mhflx'][  :] = xr.concat(results, dim='lat')
     
        #_______________________________________________________________________
        if do_compute: mhflx = mhflx.compute()
        if do_load   : mhflx = mhflx.load()
        if do_persist: mhflx = mhflx.persist()
        
        #_______________________________________________________________________
        if len(box_list)==1: list_mhflx = mhflx
        else               : list_mhflx.append(mhflx)
        
        #_______________________________________________________________________
        del(mhflx)
        
    #___________________________________________________________________________
    return(list_mhflx)


#+___COMPUTE MERIDIONAL HEATFLUX FROOM TRACER ADVECTION TROUGH BINNING_________+
#|                                                                             |
#+_____________________________________________________________________________+
def calc_mhflx_box_fast_lessmem(mesh, data, datat, mdiag, box_list, dlat=1.0, do_info=True, do_checkbasin=False, 
                        do_compute=False, do_load=True, do_persist=False, do_buflay=True, 
                        do_parallel=False, n_workers=10, 
                        ):
    #___________________________________________________________________________
    vname_list = list(data.keys())
    vnamet = None
    if 'temp' in vname_list:
        vnamet = 'temp'
        vname_list.remove('temp')
    for vi in vname_list:
        if 'u' in vi: vnameu=vi
        if 'v' in vi: vnamev=vi
    #vnameu, vnamev = vname_list[0], vname_list[1]
    
    # save global and local variable attributes
    gattr = data.attrs
    vattr = data[vnameu].attrs

    #___________________________________________________________________________
    # factors for heatflux computation
    rho0 = 1030 # kg/m^3
    cp   = 3850 # J/kg/K
    inPW = 1.0e-15
    
    #_______________________________________________________________________
    # define function for longitudinal binning --> should be possible to 
    # parallelize this loop since each lon bin is independent
    def sum_over_latbin(lat_i, mdiag, data, datat):
        #___________________________________________________________________
        # compute which edge is cutted by the binning line along latitude
        # to select the  lat bin cutted edges with where is by far the fastest option
        # compared to using ...
        # idx_latbin  = (  (data.edge_y[0,:]-lat_i)*(data.edge_y[1,:]-lat_i) <= 0.) & \
        #                 )
        # data_lonbin = data.groupby(idx_lonbin)[True]
        # --> groupby is here a factor 5-6 slower than using isel+np.where
        mdiag_latbin = mdiag.isel(edg_n=np.where(  
                    (mdiag.edge_y[0,:]-lat_i)*(mdiag.edge_y[1,:]-lat_i) <= 0.0)[0])
        
        # change direction of edge to make it consistent
        idx_direct = mdiag_latbin.edge_y[0,:]<=lat_i
        mdiag_latbin.edge_dx_lr[:,idx_direct] = -mdiag_latbin.edge_dx_lr[:,idx_direct]
        mdiag_latbin.edge_dy_lr[:,idx_direct] = -mdiag_latbin.edge_dy_lr[:,idx_direct]
        del(idx_direct)
        
        # make sure that value of right boundary triangle is zero when boundary edge 
        data_latbin   = data.isel(elem=mdiag_latbin.edge_tri)
        list_vname    = list(data_latbin.keys())
        vnameu, vnamv = list_vname[0], list_vname[1]
        data_latbin[vnameu][1, mdiag_latbin.edge_tri[1,:]<0 ,:] = 0.0
        data_latbin[vnamev][1, mdiag_latbin.edge_tri[1,:]<0 ,:] = 0.0
        
        # compute transport u,v --> u*dx,v*dy
        data_latbin[vnameu] = data_latbin[vnameu] * mdiag_latbin['edge_dx_lr'] 
        data_latbin[vnamev] = data_latbin[vnamev] * mdiag_latbin['edge_dy_lr'] 
        
        # compute u*t, v*t if data wasnt already ut,vt
        if datat is not None:
            datat_latbin = datat.isel(nod2=mdiag_latbin.edges)
            vnamet       = list(datat_latbin.keys())[0]
            data_latbin[vnameu] = data_latbin[vnameu]*(datat_latbin[vnamet][0,:]+datat_latbin[vnamet][1,:])*0.5
            data_latbin[vnamev] = data_latbin[vnamev]*(datat_latbin[vnamet][0,:]+datat_latbin[vnamet][1,:])*0.5
            del(datat_latbin)
        
        # multiply with layer thickness
        data_latbin[vnameu] = data_latbin[vnameu] * data_latbin['dz'] 
        data_latbin[vnamev] = data_latbin[vnamev] * data_latbin['dz']
         
        # sum already over vflux contribution from left and right triangle 
        # and over cutted edges 
        data_latbin[vnameu] = data_latbin[vnameu].sum(dim=['n2','edg_n'], skipna=True)
        data_latbin[vnamev] = data_latbin[vnamev].sum(dim=['n2','edg_n'], skipna=True)
        
        # integrate vertically
        data_latbin = data_latbin.sum(dim='nz1', skipna=True) * rho0*cp*inPW*(-1)
            
        return(data_latbin[vnameu] + data_latbin[vnamev])
        
    #___________________________________________________________________________
    # Loop over boxes/regions/shapefiles ...
    list_mhflx=list()
    for box in box_list:
        #_______________________________________________________________________
        if not isinstance(box, shp.Reader):
            if len(box)==2: boxname, box = box[1], box[0]
            if box is None or box=='global': boxname='global'
        else:     
            boxname = os.path.basename(box.shapeName)
            boxname = boxname.split('_')[0].replace('_',' ')  
        
        #_______________________________________________________________________
        # select box area
        datat_box = None
        if box=='global':
            mdiag_box = mdiag
            n_idxin = np.ones(mesh.n2dn,dtype='bool')
        else:     
            #___________________________________________________________________
            # compute box mask index for nodes
            #  --> sometimes it can be be that the shapefile is to small in this case 
            #  boundary triangles might not get selected therefore we if any node
            #  points of an edge triangle is within the shapefile
            e_idxin = do_boxmask(mesh,box,do_elem=True)
            e_i     = mesh.e_i[e_idxin,:]
            e_i     = np.unique(e_i.flatten())
            n_idxin = np.zeros(mesh.n2dn,dtype='bool')
            n_idxin[e_i]=True
            n_idxin_b = n_idxin.copy()
            del(e_i, e_idxin)
            if do_buflay:
                # add a buffer layer of selected triangles --> sometimes it can be be that 
                # the shapefile is to small in this case boundary triangles might not get 
                # selected. Therefor extend the selection of edges 
                e_idxin = n_idxin[mesh.e_i].max(axis=1)
                e_i = mesh.e_i[e_idxin,:]
                e_i = np.unique(e_i.flatten())
                n_idxin_b[e_i]=True
                del(e_i, e_idxin)
            idx_IN   = n_idxin_b[mdiag.edges].any(axis=0)
            mdiag_box = mdiag.isel({'edg_n':idx_IN})
            #___________________________________________________________________
            if do_checkbasin:
                from matplotlib.tri import Triangulation
                tri = Triangulation(np.hstack((mesh.n_x,mesh.n_xa)), np.hstack((mesh.n_y,mesh.n_ya)), np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia)))
                plt.figure()
                ax = plt.gca()
                plt.triplot(tri, color='k')
                plt.plot(data_box.edge_x[0,:], data_box.edge_y[0,:], '*r', linestyle='None', markersize=1)
                plt.plot(data_box.edge_x[1,:], data_box.edge_y[1,:], '*r', linestyle='None', markersize=1)
                plt.title('Basin selection')
                plt.show()
            
        #_______________________________________________________________________
        # do zonal sum over latitudinal bins 
        lat_min  = np.ceil( mesh.n_y[n_idxin].min())  
        lat_max  = np.floor(mesh.n_y[n_idxin].max())
        lat      = np.arange(lat_min+dlat/2, lat_max-dlat/2, dlat )
        
        #_______________________________________________________________________
        # Create xarray dataset
        list_dimname, list_dimsize = ['nlat'], [lat.size]
        data_vars = dict()
        # define variable attributes 
        vattr['long_name' ] = f'Meridional Heat Transport'
        vattr['short_name'] = f'Merid. Heat Transp.'
        vattr['boxname'   ] = boxname
        vattr['units'     ] = 'PW'
        # define data_vars dict, coordinate dict, as well as list of dimension name 
        # and size 
        data_vars, coords, dim_n, dim_s,  = dict(), dict(), list(), list()
        if 'time' in list(data.dims): dim_n.append('time')
        dim_n.append('lat'); 
        for dim_ni in dim_n:
            if   dim_ni=='time': dim_s.append(data.sizes['time']); coords['time' ]=(['time'], data['time'].data ) 
            elif dim_ni=='lat' : dim_s.append(lat.size          ); coords['lat'  ]=(['lat' ], lat          ) 
        data_vars['mhflx'] = (dim_n, np.zeros(dim_s, dtype='float32')*np.nan, vattr) 
        # create dataset
        mhflx     = xr.Dataset(data_vars=data_vars, coords=coords, attrs=gattr)
        del(data_vars, coords, dim_n, dim_s, lat)
        
        #_______________________________________________________________________
        # do serial loop over bins
        if not do_parallel:
            ts = ts1 = clock.time()
            if do_info: print('\n ___loop over latitudinal bins___'+'_'*90, end='\n')
            for iy, lat_i in enumerate(mhflx.lat):
                #_______________________________________________________________
                if do_info: print('{:+06.1f}|'.format(lat_i), end='')
                if np.mod(iy+1,15)==0 and do_info:
                    print(' > time: {:2.1f} sec.'.format((clock.time()-ts1)), end='\n')
                    ts1 = clock.time()
                if 'time' in data.dims: mhflx['mhflx'][:,iy] = sum_over_latbin(lat_i, mdiag_box, data, datat)
                else                  : mhflx['mhflx'][  iy] = sum_over_latbin(lat_i, mdiag_box, data, datat)       
        
        # do parallel loop over bins        
        else:
            ts = ts1 = clock.time()
            if do_info: print('\n ___parallel loop over latitudinal bins___'+'_'*1, end='\n')
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_workers)(delayed(sum_over_latbin)(lat_i, mdiag_box, data, datat) for lat_i in mhflx.lat)
            if 'time' in data.dims: mhflx['mhflx'][:,:] = xr.concat(results, dim='lat').transpose('time','lat')
            else                  : mhflx['mhflx'][  :] = xr.concat(results, dim='lat')
     
        #_______________________________________________________________________
        if do_compute: mhflx = mhflx.compute()
        if do_load   : mhflx = mhflx.load()
        if do_persist: mhflx = mhflx.persist()
        
        #_______________________________________________________________________
        if len(box_list)==1: list_mhflx = mhflx
        else               : list_mhflx.append(mhflx)
        
        #_______________________________________________________________________
        del(mhflx)
        
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
    gattrs = data.attrs
    gattrs['proj']          = 'index+xy'
    aux_attr  = data[vname].attrs
    aux_attr['long_name'], aux_attr['units'] = 'Global Meridional Heat Transport', 'PW'
    data_vars['gmhflx'] = (list_dimname, np.zeros(list_dimsize), aux_attr) 
    # define coordinates
    coords    = {'nlat' : (['nlat' ], lat )}
    # create dataset
    ghflx     = xr.Dataset(data_vars=data_vars, coords=coords, attrs=gattrs)
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
def calc_gmhflx_box(mesh, data, box_list, dlat=1.0, do_info=True, 
                    do_compute=False, do_load=True, do_persist=False, 
                    do_parallel=False, n_workers=10, 
                   ):
    #___________________________________________________________________________
    vname = list(data.keys())[0]
    
    # save global and local variable attributes
    gattrs = data.attrs
    gattrs['proj'] = 'index+xy'
    vattr = data[vname].attrs
    
    # factors for heatflux computation
    inPW = 1.0e-15
    
    if 'nod2' in list(data.dims) : dimh, do_elem = 'nod2', False
    if 'elem' in list(data.dims) : dimh, do_elem = 'elem', True
    
    #___________________________________________________________________________
    # define subroutine for binning over latitudes, allows for parallelisation
    def sum_over_lat(lat_i, lat_bin, data, dimh):
        #_______________________________________________________________________
        # compute which vertice is within the latitudinal bin
        # --> groupby is here a factor 5-6 slower than using isel+np.where
        data_latbin = data.isel({dimh:np.where(lat_bin==lat_i)[0]})
        data_latbin = data_latbin.sum(dim=dimh, skipna=True)
        return(data_latbin)
        
    #___________________________________________________________________________
    # Loop over boxes
    list_gmhflx=list()
    for box in box_list:
        #_______________________________________________________________________
        if not isinstance(box, shp.Reader):
            if len(box)==2: boxname, box = box[1], box[0]
            if box is None or box=='global': boxname='global'
        else:     
            boxname = os.path.basename(box.shapeName).replace('_',' ')  
           
        #_______________________________________________________________________
        # compute  mask index
        idx_IN   = xr.DataArray(do_boxmask(mesh, box, do_elem=do_elem), dims=dimh).chunk({dimh:data.chunksizes[dimh]})
        
        #_______________________________________________________________________
        # select box area
        data_box = data.isel({dimh:idx_IN})
        
        #_______________________________________________________________________
        # multiply 2D data with area weight
        data_box = data_box[vname]*data_box['w_A']
        data_box = data_box.load()
        
        #_______________________________________________________________________
        # do zonal sum over latitudinal bins 
        lat_bin  = xr.DataArray(data=np.round(data_box.lat/dlat)*dlat, dims='nod2', name='lat')
        lat      = np.arange(lat_bin.min(), lat_bin.max()+dlat, dlat)
        
        #_______________________________________________________________________
        # Create xarray dataset
        tm1= clock.time()
        # define variable attributes  
        vattr['long_name' ] = f'Meridional Heat Transport'
        vattr['short_name'] = f'Merid. Heat Transp.'
        vattr['boxname'   ] = boxname
        vattr['units'     ] = 'PW'
        # define data_vars dict, coordinate dict, as well as list of dimension name 
        # and size 
        data_vars, coords, dim_n, dim_s,  = dict(), dict(), list(), list()
        if 'time' in list(data.dims): dim_n.append('time')
        dim_n.append('lat'); 
        for dim_ni in dim_n:
            if   dim_ni=='time': dim_s.append(data.sizes['time']); coords['time' ]=(['time'], data['time'].data ) 
            elif dim_ni=='lat' : dim_s.append(lat.size          ); coords['lat'  ]=(['lat' ], lat          ) 
        data_vars['gmhflx'] = (dim_n, np.zeros(dim_s, dtype='float32'), vattr) 
        # create dataset
        gmhflx = xr.Dataset(data_vars=data_vars, coords=coords, attrs=gattrs)
        del(data_vars, coords, dim_n, dim_s, lat)
        
        #___________________________________________________________________________
        # do serial loop over latitudinal bins
        if not do_parallel:
            if do_info: print('\n ___loop over latitudinal bins___'+'_'*90, end='\n')
            for iy, lat_i in enumerate(gmhflx.lat):
                if 'time' in data.dims: gmhflx['gmhflx'][:,iy] = sum_over_lat(lat_i, lat_bin, data_box, dimh)
                else                  : gmhflx['gmhflx'][  iy] = sum_over_lat(lat_i, lat_bin, data_box, dimh)
        
        # do parallel loop over latitudinal bins
        else:
            if do_info: print('\n ___parallel loop over longitudinal bins___'+'_'*1, end='\n')
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_workers)(delayed(sum_over_lat)(lat_i, lat_bin, data_box, dimh) for lat_i in gmhflx.lat)
            if 'time' in data.dims: gmhflx['gmhflx'][:,:] = xr.concat(results, dim='lat').transpose('time','lat')
            else                  : gmhflx['gmhflx'][  :] = xr.concat(results, dim='lat')
        del(data_box, lat_bin)
        
        #_______________________________________________________________________
        gmhflx['gmhflx'] = gmhflx['gmhflx'] * inPW
        
        #_______________________________________________________________________
        # do cumulative sum over latitudes    
        gmhflx['gmhflx'] = -gmhflx['gmhflx'].cumsum(dim='lat', skipna=True) 
        
        #___________________________________________________________________________
        if do_compute: gmhflx = gmhflx.compute()
        if do_load   : gmhflx = gmhflx.load()
        if do_persist: gmhflx = gmhflx.persist()
        
        #_______________________________________________________________________
        if len(box_list)==1: list_gmhflx = gmhflx
        else               : list_gmhflx.append(gmhflx)
        
        #_______________________________________________________________________
        del(gmhflx)
        
    #___________________________________________________________________________
    return(list_gmhflx)



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
            hbstreamf['hbstreamf'][:,ix] = hbstrfbin_over_lon(lon_i, lat, data)
    
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



#+___COMPUTE HORIZONTAL BAROTROPIC STREAM FUNCTION TROUGH BINNING______________+
#|                                                                             |
#+_____________________________________________________________________________+
# try to take full advantage of xarray and dask
def calc_hbarstreamf_fast_lessmem(mesh, data, mdiag, lon, lat, do_info=True, do_parallel=True, n_workers=10, client=None):
    
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
    def hbstrfbin_over_lon(lon_i, lat, data, mdiag):
        #_______________________________________________________________________
        # compute which edge is cutted by the binning line along longitude
        # to select the  lon bin cutted edges with where is by far the fastest option
        # compared to using ...
        # idx_lonbin  = (  (data.edge_x[0,:]-lon_i)*(data.edge_x[1,:]-lon_i) <= 0.) & \
        #                  (abs(data.edge_x[0,:]-lon_i)<50.) & (abs(data.edge_x[1,:]-lon_i)<50. )
        # data_lonbin = data.groupby(idx_lonbin)[True]
        # --> groupby is here a factor 5-6 slower than using isel+np.where
        #warnings.filterwarnings("ignore", category=UserWarning, message="Sending large graph of size")
        #warnings.filterwarnings("ignore", category=UserWarning, message="Large object of size \\d+\\.\\d+ detected in task graph")
        mdiag_lonbin = mdiag.isel(edg_n=np.where(  
                        ((mdiag.edge_x[0,:]-lon_i)*(mdiag.edge_x[1,:]-lon_i) <= 0.) & \
                        (abs(mdiag.edge_x[0,:]-lon_i)<50.) & (abs(mdiag.edge_x[1,:]-lon_i)<50. ))[0]).load()
        
        #data_lonbin = data.isel(elem=xr.DataArray(mdiag_lonbin.edge_tri, dims=['n2', 'edg_n'])) #, nz1=data.nzi.load())
        data_lonbin = data.isel(elem=mdiag_lonbin.edge_tri) #, nz1=data.nzi.load())
        data_lonbin = data_lonbin.assign_coords(edge_my=mdiag_lonbin.edge_my)
        #warnings.resetwarnings()
        
        # change direction of edge to make it consistent
        idx_direct = mdiag_lonbin.edge_x[0,:]<=lon_i
        mdiag_lonbin.edge_dx_lr[:,idx_direct] = -mdiag_lonbin.edge_dx_lr[:,idx_direct]
        mdiag_lonbin.edge_dy_lr[:,idx_direct] = -mdiag_lonbin.edge_dy_lr[:,idx_direct]
        del(idx_direct)
        
        # compute transport u,v --> u*dx,v*dy
        data_lonbin['u'] = data_lonbin['u'] * mdiag_lonbin['edge_dx_lr'] * inSv * (-1)
        data_lonbin['v'] = data_lonbin['v'] * mdiag_lonbin['edge_dy_lr'] * inSv * (-1)
        del(mdiag_lonbin)
        
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
            if do_info: print('{:+06.1f}|'.format(lon_i), end='')
            if np.mod(ix+1,15)==0 and do_info:
                print(' > time: {:2.1f} sec.'.format((clock.time()-ts1)), end='\n')
                ts1 = clock.time()
            hbstreamf['hbstreamf'][:,ix] = hbstrfbin_over_lon(lon_i, lat, data, mdiag)
    
    # do parallel loop over longitudinal bins        
    else:
        ts = ts1 = clock.time()
        if do_info: print('\n ___parallel loop over longitudinal bins___'+'_'*1, end='\n')
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_workers)(delayed(hbstrfbin_over_lon)(lon_i, lat, data, mdiag) for lon_i in hbstreamf.lon)
        hbstreamf['hbstreamf'][:,:] = xr.concat(results, dim='nlon').transpose()
        
    #___________________________________________________________________________
    hbstreamf['hbstreamf'] =-hbstreamf['hbstreamf'].cumsum(dim='nlat', skipna=True)#+150.0 
    hbstreamf['hbstreamf'].data = hbstreamf['hbstreamf'].data-hbstreamf['hbstreamf'].data[-1,:]
    
    # impose periodic boundary condition
    hbstreamf['hbstreamf'].data[:,-1] = hbstreamf['hbstreamf'].data[:,-2]
    hbstreamf['hbstreamf'].data[:, 0] = hbstreamf['hbstreamf'].data[:, 1]
    if do_info: print(' --> total elasped time: {:3.3f} min.'.format((clock.time()-ts)/60))      
    
    #___________________________________________________________________________
    return(hbstreamf)


