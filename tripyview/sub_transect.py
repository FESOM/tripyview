# Patrick Scholz, 23.01.2018
import sys
import os
import numpy as np
from   scipy.signal import convolve2d
import copy as  cp
from   shapely.geometry   import Point, Polygon, MultiPolygon, shape
from   shapely.vectorized import contains
import shapefile as shp
import json
import geopandas as gpd
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from   matplotlib.ticker import MultipleLocator, AutoMinorLocator, ScalarFormatter
import pyfesom2 as pf

from   .sub_mesh           import * 
from   .sub_data           import *
from   .sub_plot           import *
from   .sub_utility        import *
from   .sub_colormap       import *

def load_transect_fesom2(mesh, data, transect_list, do_compute=True, ):
    
    #___________________________________________________________________________
    # str_anod    = ''
    index_list  = []
    idxin_list  = []
    cnt         = 0
    
    #___________________________________________________________________________
    # loop over box_list
    for transect in transect_list:
        
        #_______________________________________________________________________
        # select data  points  closest to transect points --> pyfesom2
        idx_nodes = pf.tunnel_fast1d(mesh.n_y, mesh.n_x, transect_list[0]['ipm'])
        index_list.append( data.isel(nod2=idx_nodes.astype(int)) )
        
        index_list[cnt] = index_list[cnt].assign_coords(lon=("lon",transect_list[0]['ipm'][0,:]))
        index_list[cnt] = index_list[cnt].assign_coords(lat=("lat",transect_list[0]['ipm'][1,:]))
        index_list[cnt] = index_list[cnt].assign_coords(dst=("dst",transect_list[0]['ipmd']))
        
        #_______________________________________________________________________
        if do_compute: index_list[cnt] = index_list[cnt].compute()
        
        #_______________________________________________________________________
        vname = list(index_list[cnt].keys())
        if transect['name'] is not None: 
            index_list[cnt][vname[0]].attrs['transect_name'] = transect['name']
        else:
            index_list[cnt][vname[0]].attrs['transect_name'] = 'None'
        index_list[cnt][vname[0]].attrs['lon'] = transect['lon']    
        index_list[cnt][vname[0]].attrs['lat'] = transect['lat']    
        #_______________________________________________________________________
        cnt = cnt + 1
    #___________________________________________________________________________
    return(index_list)


def load_zmeantransect_fesom2(mesh, data, box_list, dlat=0.5, boxname=None, do_harithm='mean', 
                      do_compute=True, do_outputidx=False, diagpath=None, do_onelem=False, 
                      do_info=False, do_smooth=True, 
                       **kwargs,):
    
    #___________________________________________________________________________
    # str_anod    = ''
    index_list  = []
    idxin_list  = []
    cnt         = 0
    
    #___________________________________________________________________________
    # loop over box_list
    vname = list(data.keys())[0]
    if   'nz'  in list(data[vname].dims): which_ddim, ndi, depth = 'nz' , mesh.nlev  , mesh.zlev     
    elif 'nz1' in list(data[vname].dims): which_ddim, ndi, depth = 'nz1', mesh.nlev-1, mesh.zmid
    
    
    for box in box_list:
        if not isinstance(box, shp.Reader) and not box =='global' and not box==None :
            if len(box)==2: boxname, box = box[1], box[0]
        
        #_______________________________________________________________________
        # compute box mask index for nodes 
        n_idxin=do_boxmask(mesh,box)
        if do_onelem: 
            e_idxin = n_idxin[mesh.e_i].sum(axis=1)>=1  
        
        #___________________________________________________________________________
        # do zonal mean calculation either on nodes or on elements        
        # keep in mind that node area info is changing over depth--> therefor load from file 
        fname = data[vname].attrs['runid']+'.mesh.diag.nc'            
        if diagpath is None:
            if   os.path.isfile( os.path.join(data[vname].attrs['datapath'], fname) ): 
                dname = data[vname].attrs['datapath']
            elif os.path.isfile( os.path.join( os.path.join(os.path.dirname(os.path.normpath(data[vname].attrs['datapath'])),'1/'), fname) ): 
                dname = os.path.join(os.path.dirname(os.path.normpath(data[vname].attrs['datapath'])),'1/')
            elif os.path.isfile( os.path.join(mesh.path,fname) ): 
                dname = mesh.path
            else:
                raise ValueError('could not find directory with...mesh.diag.nc file')
            
            diagpath = os.path.join(dname,fname)
            if do_info: print(' --> found diag in directory:{}', diagpath)
        else:
            if os.path.isfile(os.path.join(diagpath,fname)):
                diagpath = os.path.join(diagpath,fname)
            elif os.path.isfile(os.path.join(os.path.join(os.path.dirname(os.path.normpath(diagpath)),'1/'),fname)) :
                diagpath = os.path.join(os.path.join(os.path.dirname(os.path.normpath(diagpath)),'1/'),fname)
        #___________________________________________________________________________
        # compute area weighted vertical velocities on elements
        if do_onelem:
            #_______________________________________________________________________
            # load elem area from diag file
            if ( os.path.isfile(diagpath)):
                mat_area = xr.open_mfdataset(diagpath, parallel=True, **kwargs)['elem_area']
                mat_area = mat_area.isel(elem=e_idxin).compute()   
                mat_area = mat_area.expand_dims({which_ddim:depth}).transpose()
                mat_iz   = xr.open_mfdataset(diagpath, parallel=True, **kwargs)['nlevels']-1
                mat_iz   = mat_iz.isel(elem=e_idxin).compute()   
            else: 
                raise ValueError('could not find ...mesh.diag.nc file')
            
            #_______________________________________________________________________
            # create meridional bins
            e_y   = mesh.n_y[mesh.e_i[e_idxin,:]].sum(axis=1)/3.0
            lat   = np.arange(np.floor(e_y.min())+dlat/2, 
                            np.ceil( e_y.max())-dlat/2, 
                            dlat)
            lat_i = (( e_y-lat[0])/dlat ).astype('int')
            
            #_______________________________________________________________________    
            # mean over elements + select MOC basin 
            if 'time' in list(data.dims): 
                wdim = ['time','elem',which_ddim]
                wdum = data[vname].data[:, mesh.e_i[e_idxin,:], :].sum(axis=2)/3.0 * 1e-6
            else                        : 
                wdim = ['elem',which_ddim]
                wdum = data[vname].data[mesh.e_i[e_idxin,:], :].sum(axis=1)/3.0 * 1e-6
            mat_mean = xr.DataArray(data=wdum, dims=wdim)
            mat_mean = mat_mean.fillna(0.0)
            del wdim, wdum
            
            #_______________________________________________________________________
            # calculate area weighted mean
            if 'time' in list(data.dims):
                nt = data['time'].values.size
                for nti in range(nt):
                    mat_mean.data[nti,:,:] = np.multiply(mat_mean.data[nti,:,:], mat_area.data)    
                    
                # be sure ocean floor is setted to zero 
                for di in range(0,ndi): 
                    mat_mean.data[:, np.where(di>=mat_iz)[0], di]=0.0
                
            else:
                mat_mean.data = np.multiply(mat_mean.data, mat_area.data)
                
                # be sure ocean floor is setted to zero 
                for di in range(0,ndi): 
                    mat_mean.data[np.where(di>=mat_iz)[0], di]=0.0
            del mat_area
        
        # compute area weighted vertical velocities on vertices
        else:     
            #_______________________________________________________________________
            # load vertice cluster area from diag file
            if ( os.path.isfile(diagpath)):
                mat_area = xr.open_mfdataset(diagpath, parallel=True, **kwargs)['nod_area'].transpose() 
                if   'nod_n' in list(mat_area.dims): mat_area = mat_area.isel(nod_n=n_idxin).compute()  
                elif 'nod2'  in list(mat_area.dims): mat_area = mat_area.isel(nod2=n_idxin).compute()     
                
                mat_iz   = xr.open_mfdataset(diagpath, parallel=True, **kwargs)['nlevels_nod2D']-1
                if   'nod_n' in list(mat_area.dims): mat_iz   = mat_iz.isel(nod_n=n_idxin).compute()
                elif 'nod2'  in list(mat_area.dims): mat_iz   = mat_iz.isel(nod2=n_idxin).compute()   
            else: 
                raise ValueError('could not find ...mesh.diag.nc file')
            
            # data are on mid depth levels
            if which_ddim=='nz1': mat_area = mat_area[:,:-1]
            
            #_______________________________________________________________________
            # create meridional bins
            lat   = np.arange(np.floor(mesh.n_y[n_idxin].min())+dlat/2, 
                            np.ceil( mesh.n_y[n_idxin].max())-dlat/2, 
                            dlat)
            lat_i = ( (mesh.n_y[n_idxin]-lat[0])/dlat ).astype('int')
                
            #_______________________________________________________________________    
            # select MOC basin 
            mat_mean = data[vname].isel(nod2=n_idxin)
            isnan = np.isnan(mat_mean.values)
            mat_mean.values[isnan] = 0.0
            mat_area.values[isnan] = 0.0
            del(isnan)
            #mat_mean = mat_mean.fillna(0.0)
            
            #_______________________________________________________________________
            # calculate area weighted mean
            if 'time' in list(data.dims):
                nt = data['time'].values.size
                for nti in range(nt):
                    mat_mean.data[nti,:,:] = np.multiply(mat_mean.data[nti,:,:], mat_area.data)    
                    
                # be sure ocean floor is setted to zero 
                for di in range(0,ndi): 
                    mat_mean.data[:, np.where(di>=mat_iz)[0], di]=0.0
            else:
                mat_mean.data = np.multiply(mat_mean.data, mat_area.data)
                
                # be sure ocean floor is setted to zero 
                for di in range(0,ndi): 
                    mat_mean.data[np.where(di>=mat_iz)[0], di]=0.0
                
        #___________________________________________________________________________
        # This approach is five time faster than the original from dima at least for
        # COREv2 mesh but needs probaply a bit more RAM
        if 'time' in list(data.dims): 
            aux_zonmean  = np.zeros([nt, ndi, lat.size])
            aux_zonarea  = np.zeros([nt, ndi, lat.size])
        else                        : 
            aux_zonmean  = np.zeros([ndi, lat.size])
            aux_zonarea  = np.zeros([ndi, lat.size])
        bottom  = np.zeros([lat.size,])
        numbtri = np.zeros([lat.size,])
    
        #  switch topo beteen computation on nodes and elements
        if do_onelem: topo    = np.float16(mesh.zlev[mesh.e_iz[e_idxin]])
        else        : topo    = np.float16(mesh.n_z[n_idxin])
        
        # this is more or less required so bottom patch looks aceptable
        topo[np.where(topo>-30.0)[0]]=np.nan  
        
        # loop over meridional bins
        if 'time' in list(data.dims):
            for bini in range(lat_i.min(), lat_i.max()):
                numbtri[bini]= np.sum(lat_i==bini)
                aux_zonmean[:,:, bini]=mat_mean[:,lat_i==bini,:].sum(axis=1)
                aux_zonarea[:,:, bini]=mat_area[:,lat_i==bini,:].sum(axis=1)
                bottom[bini] = np.nanpercentile(topo[lat_i==bini],15)
                
            
            # kickout outer bins where eventually no triangles are found
            idx    = numbtri>0
            aux_zonmean = aux_zonmean[:,:,idx]
            aux_zonarea = aux_zonarea[:,:,idx]
            del(mat_mean, mat_area, topo)
            
        else:        
            for bini in range(lat_i.min(), lat_i.max()):
                numbtri[bini]= np.sum(lat_i==bini)
                aux_zonmean[:, bini]=mat_mean[lat_i==bini,:].sum(axis=0)
                aux_zonarea[:, bini]=mat_area[lat_i==bini,:].sum(axis=0)
                #bottom[bini] = np.nanpercentile(topo[lat_i==bini],15)
                bottom[bini] = np.nanpercentile(topo[lat_i==bini],10)
                
            # kickout outer bins where eventually no triangles are found
            idx    = numbtri>0
            aux_zonmean = aux_zonmean[:,idx]
            aux_zonarea = aux_zonarea[:,idx]
            del(mat_mean, mat_area, topo)
            
        bottom = bottom[idx]
        lat    = lat[idx]
        aux_zonmean[aux_zonarea!=0]= aux_zonmean[aux_zonarea!=0]/aux_zonarea[aux_zonarea!=0]
        
        #___________________________________________________________________________
        if do_smooth: 
            filt=np.array([1,2,1])
            filt=filt/np.sum(filt)
            filt=filt[np.newaxis,:]
            aux_zonmean[aux_zonarea==0] = 0.0
            aux_zonmean = convolve2d(aux_zonmean, filt, mode='same', boundary='symm') 
        
        #___________________________________________________________________________
        aux_zonmean[aux_zonarea==0]= np.nan
        del(aux_zonarea)
        
        #___________________________________________________________________________
        # smooth bottom line a bit 
        filt=np.array([1,2,3,2,1])
        filt=filt/np.sum(filt)
        aux = np.concatenate( (np.ones((filt.size,))*bottom[0],bottom,np.ones((filt.size,))*bottom[-1] ) )
        aux = np.convolve(aux,filt,mode='same')
        bottom = aux[filt.size:-filt.size]
        del aux, filt
        
        
        
        #___________________________________________________________________________
        # Create Xarray Datasert for moc_basins
        # copy global attributes from dataset
        global_attr = data.attrs
        
        # copy local attributes from dataset
        local_attr  = data[vname].attrs
        if 'long_name' in local_attr:
            local_attr['long_name'] = " zonal mean {}".format(local_attr['long_name']) 
        else:
            local_attr['long_name'] = " zonal mean {}".format(vname) 
        
        # create coordinates
        if 'time' in list(data.dims):
            coords    = {'depth' : ([which_ddim], depth), 
                        'lat'   : (['ny'], lat      ), 
                        'bottom': (['ny'], bottom   ),
                        'time'  : (['time'], data['time'].values)}
            dims = ['time', which_ddim, 'ny']
        else:    
            coords    = {'depth' : ([which_ddim], depth), 
                        'lat'   : (['ny'], lat      ), 
                        'bottom': (['ny'], bottom   )}
            dims = [which_ddim,'ny']
            # create coordinates
        data_vars = {vname   : (dims, aux_zonmean, local_attr)} 
        index_list.append( xr.Dataset(data_vars=data_vars, coords=coords, attrs=global_attr) )
        
        #_______________________________________________________________________
        if box is None or box is 'global': 
            index_list[cnt][vname].attrs['transect_name'] = 'global zonal mean'
        elif isinstance(box, shp.Reader):
            str_name = box.shapeName.split('/')[-1].replace('_',' ')
            index_list[cnt][vname].attrs['transect_name'] = '{} zonal mean'.format(str_name.lower())
        
        #_______________________________________________________________________
        cnt = cnt + 1
    #___________________________________________________________________________
    if do_outputidx:
        return(index_list, idxin_list)
    else:
        return(index_list)
    
#
#
#_______________________________________________________________________________
def analyse_transects(input_transect, which_res='res', res=1.0, npts=500):
    
    transect_list = []
    # loop oover transects in list
    for transec_lon,transec_lat, transec_name in input_transect:
        
        ip, ipm, ipd, ipmd, pm_nvecm, pm_evec, idx_nodes = [],[],[],[],[],[],[]
        # loop oover transect points
        for ii in range(0,len(transec_lon)-1):
            #___________________________________________________________________
            P1    = [transec_lon[ii  ], transec_lat[ii  ]]
            P2    = [transec_lon[ii+1], transec_lat[ii+1]]
            #___________________________________________________________________
            # unit vector of line
            evec  = np.array([P2[0]-P1[0], P2[1]-P1[1]])
            evecn = (evec[0]**2+evec[1]**2)**0.5
            evec  = evec/evecn
            if which_res=='npts':
                evecnl = np.linspace(0,evecn,npts)
                loop_pts = npts
            elif which_res=='res':
                evecnl = np.arange(0,evecn,res)
                loop_pts=evecnl.size
                
            # normal vector 
            nvec = np.array([-evec[1],evec[0]])  
            
            #___________________________________________________________________
            # interpolation points
            dum_ip      = np.vstack(( P1[0]+evecnl*evec[0], P1[1]+evecnl*evec[1]))
            
            # interpolation mid points
            evecnlpm    = evecnl[:-1] + (evecnl[1:]-evecnl[:-1])/2.0
            dum_ipm     = np.vstack(( P1[0]+evecnlpm*evec[0], P1[1]+evecnlpm*evec[1]))
            del(evecnlpm)
            
            # compute dr in km
            Rearth      = 6371.0
            x,y,z       = grid_cart3d(np.radians(dum_ip[0,:]), np.radians(dum_ip[1,:]), R=Rearth)
            dr          = Rearth*np.arccos( (x[:-1]*x[1:] + y[:-1]*y[1:] + z[:-1]*z[1:])/(Rearth**2) )
            x,y,z       = grid_cart3d(np.radians(dum_ipm[0,:]), np.radians(dum_ipm[1,:]), R=Rearth)
            drm         = Rearth*np.arccos( (x[:-1]*x[1:] + y[:-1]*y[1:] + z[:-1]*z[1:])/(Rearth**2) )
            del(x,y,z)
            
            # compute distance from start point for corner and mid points
            if ii==0: dstart = 0.0 
            else    : dstart = ipd[-1]
            dum_ipd     = np.cumsum(np.hstack((dstart, dr)))
            dum_ipmd    = np.cumsum(np.hstack((dstart+dr[0]/2, drm)))
            
            # all normal and unit vector at mis points
            dum_pm_nvec = np.vstack( ( np.ones((loop_pts-1,))*nvec[0], np.ones((loop_pts-1,))*nvec[1]) )  
            dum_pm_evec = np.vstack( ( np.ones((loop_pts-1,))*evec[0], np.ones((loop_pts-1,))*evec[1]) )  
            
            
            #___________________________________________________________________
            # collect points from section of transect
            if ii==0: 
                ip, ipm, ipd, ipmd, pm_nvec, pm_evec = dum_ip, dum_ipm, dum_ipd, dum_ipmd, dum_pm_nvec, dum_pm_evec
            else:    
                ip          = np.hstack((ip, dum_ip))
                ipm         = np.hstack((ipm, dum_ipm))
                ipd         = np.hstack((ipd, dum_ipd))
                ipmd        = np.hstack((ipmd, dum_ipmd))
                pm_nvec     = np.hstack((pm_nvec, dum_pm_nvec))
                pm_Evec     = np.hstack((pm_evec, dum_pm_evec))
            del(dum_ip, dum_ipm, dum_ipd, dum_ipmd, dum_pm_nvec, dum_pm_evec)
        #_______________________________________________________________________
        transect = dict()
        transect['lon']     = transec_lon
        transect['lat']     = transec_lat
        transect['name']    = transec_name
        transect['ip']      = ip
        transect['ipm']     = ipm
        transect['ipd']     = ipd
        transect['ipmd']    = ipmd
        transect['pm_nvec'] = pm_nvec
        transect['pm_evec'] = pm_evec
        transect_list.append(transect)
    #___________________________________________________________________________
    return(transect_list)


 
#+___PLOT MERIDIONAL OVERTRUNING CIRCULATION  _________________________________+
#|                                                                             |
#+_____________________________________________________________________________+
def plot_transects(data, transects, figsize=[12, 6], 
              n_rc=[1, 1], do_grid=True, cinfo=None, do_rescale=False,
              cbar_nl=8, cbar_orient='vertical', cbar_label=None, cbar_unit=None,
              do_bottom=True, max_dep=[], color_bot=[0.6, 0.6, 0.6], 
              pos_fac=1.0, pos_gap=[0.02, 0.02], do_save=None, save_dpi=600, 
              do_contour=True, do_clabel=True, title='descript', 
              pos_extend=[0.05, 0.08, 0.95,0.95], do_ylog=True,
            ):
    #____________________________________________________________________________
    fontsize = 12
    rescale_str = None
    
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
    fig, ax = plt.subplots( n_rc[0],n_rc[1], figsize=figsize, 
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
    cinfo = do_setupcinfo(cinfo, data, do_rescale, do_index=True)
    
    #_______________________________________________________________________
    # setup normalization log10, symetric log10, None
    which_norm = do_compute_scalingnorm(cinfo, do_rescale)
    
    #___________________________________________________________________________
    # loop over axes
    for ii in range(0,ndata):
        
        #_______________________________________________________________________
        # limit data to color range
        vname= list(data[ii][0].keys())[0]
        data_plot = data[ii][0][vname].values.transpose().copy()
        
        #_______________________________________________________________________
        # setup x-coord and y-coord
        if   np.unique(data[ii][0]['lon'].values).size==1: xcoord, str_xlabel = data[ii][0]['lat'].values, 'Latitude [deg]'
        elif np.unique(data[ii][0]['lat'].values).size==1: xcoord, str_xlabel = data[ii][0]['lon'].values, 'Longitude [deg]'
        else:                                              xcoord, str_xlabel = data[ii][0]['dst'].values, 'Distance from start point [km]'
        
        if 'nz1' in list(data[ii][0].dims):
            depth, str_ylabel = data[ii][0]['nz1'].values, 'Depth [m]'
        elif 'nz' in list(data[ii][0].dims):
            depth, str_ylabel = data[ii][0]['nz'].values, 'Depth [m]'    
        
        #_______________________________________________________________________
        # be sure there are no holes
        data_plot[data_plot<cinfo['clevel'][ 0]] = cinfo['clevel'][ 0]+np.finfo(np.float32).eps
        data_plot[data_plot>cinfo['clevel'][-1]] = cinfo['clevel'][-1]-np.finfo(np.float32).eps
        
        #_______________________________________________________________________
        # plot MOC
        hp=ax[ii].contourf(xcoord, depth, data_plot, 
                           levels=cinfo['clevel'], extend='both', cmap=cinfo['cmap'],
                           norm = which_norm)
        if do_contour: 
            tickl    = cinfo['clevel']
            ncbar_l  = len(tickl)
            idx_cref = np.where(cinfo['clevel']==cinfo['cref'])[0]
            idx_cref = np.asscalar(idx_cref)
            nstep    = ncbar_l/cbar_nl
            nstep    = np.max([np.int(np.floor(nstep)),1])
            
            idx = np.arange(0, ncbar_l, 1)
            idxb = np.ones((ncbar_l,), dtype=bool)                
            idxb[idx_cref::nstep]  = False
            idxb[idx_cref::-nstep] = False
            idx_yes = idx[idxb==False]
            
            cont=ax[ii].contour(xcoord, depth, data_plot,
                            levels=cinfo['clevel'][idx_yes], colors='k', linewidths=[0.5],
                            norm = which_norm) #linewidths=[0.5,0.25])
            #if do_clabel: 
                #ax[ii].clabel(cont, cont.levels, inline=1, inline_spacing=1, fontsize=6, fmt='%1.1f Sv')
                #ax[ii].clabel(cont, cont.levels[np.where(cont.levels!=cinfo['cref'])], 
                            #inline=1, inline_spacing=1, fontsize=6, fmt='%1.1f Sv')
            
        #___________________________________________________________________
        ylim = np.sum(~np.isnan(data_plot),axis=0).max()-1
        if ylim<depth.shape[0]-1: ylim=ylim+1
        if np.isscalar(max_dep)==False: max_dep=depth[ylim]
        
        # plot bottom patch
        aux = np.ones(data_plot.shape,dtype='int16')
        aux[np.isnan(data_plot)]=0
        aux = aux.sum(axis=0)
        aux[aux!=0]=aux[aux!=0]-1
        bottom = np.abs(depth[aux])
        
        # smooth bottom patch
        filt=np.array([1,2,1]) #np.array([1,2,3,2,1])
        filt=filt/np.sum(filt)
        aux = np.concatenate( (np.ones((filt.size,))*bottom[0],bottom,np.ones((filt.size,))*bottom[-1] ) )
        aux = np.convolve(aux,filt,mode='same')
        bottom = aux[filt.size:-filt.size]
        # plot bottom patch    
        ax[ii].fill_between(xcoord, bottom, max_dep,color=[0.5,0.5,0.5])#,alpha=0.95)
        ax[ii].plot(xcoord, bottom, color='k')
        #_______________________________________________________________________
        # fix color range
        for im in ax[ii].get_images(): im.set_clim(cinfo['clevel'][ 0], cinfo['clevel'][-1])
        
        #_______________________________________________________________________
        # plot grid lines 
        if do_grid: ax[ii].grid(color='k', linestyle='-', linewidth=0.25,alpha=1.0)
        
        #_______________________________________________________________________
        # set description string plus x/y labels
        isnotnan = np.isnan(data_plot[:,0])==False
        isnotnan = isnotnan.sum()-1
        
        if title is not None: 
            #txtx, txty = xcoord[0]+(xcoord[-1]-xcoord[0])*0.025, depth[isnotnan]-(depth[isnotnan]-depth[0])*0.025                    
            txtx, txty = xcoord[0]+(xcoord[-1]-xcoord[0])*0.015, max_dep-(max_dep-depth[0])*0.015
            if   isinstance(title,str) : 
                # if title string is 'descript' than use descript attribute from 
                # data to set plot title 
                if title=='descript' and ('descript' in data[ii][0][vname].attrs.keys() ):
                    txts = data[ii][0][vname].attrs['descript']
                else:
                    txts = title
            # is title list of string        
            elif isinstance(title,list):   
                txts = title[ii]
            ax[ii].text(txtx, txty, txts, fontsize=12, fontweight='bold', horizontalalignment='left')
        
        #_______________________________________________________________________
        # set x/y label
        if collist[ii]==0        : ax[ii].set_ylabel(str_ylabel, fontsize=12)
        if rowlist[ii]==n_rc[0]-1: ax[ii].set_xlabel(str_xlabel, fontsize=12)
        
        #_______________________________________________________________________
        if do_ylog: 
            ax[ii].grid(True,which='major')
            #ax[ii].set_yscale('log')
            ax[ii].set_yscale('function', functions=(forward, inverse))
            #yticklog = np.array([5,10,25,50,100,250,500,1000,2000,4000,6000])
            yticklog = np.array([10,25,50,100,250,500,1000,2000,4000,6000])
            ax[ii].set_yticks(yticklog)
            if depth[0]==0: ax[ii].set_ylim(depth[1],max_dep)
            else          : ax[ii].set_ylim(depth[0],max_dep)
            ax[ii].invert_yaxis()
            
        else:
            ax[ii].set_ylim(depth[0],max_dep)
            ax[ii].invert_yaxis()
            ax[ii].grid(True,which='major')
            
        #ax[ii].set_yticks([5,10,25,50,100,250,500,1000,2000,4000,6000])
        ax[ii].get_yaxis().set_major_formatter(ScalarFormatter())
        
    nax_fin = ii+1
    
    #___________________________________________________________________________
    # set superior title
    if 'transect_name' in data[ii][0][vname].attrs.keys():
        fig.suptitle( data[ii][0][vname].attrs['transect_name'], x=0.5, y=1.04, fontsize=16, 
                     horizontalalignment='center', verticalalignment='bottom')
    
    #___________________________________________________________________________
    # delete axes that are not needed
    #for jj in range(nax_fin, nax): fig.delaxes(ax[jj])
    for jj in range(ndata, nax): fig.delaxes(ax[jj])
    if nax != nax_fin-1: ax = ax[0:nax_fin]
    
    #___________________________________________________________________________
    # delete axes that are not needed
    cbar = fig.colorbar(hp, orientation=cbar_orient, ax=ax, ticks=cinfo['clevel'], 
                      extendrect=False, extendfrac=None,
                      drawedges=True, pad=0.025, shrink=1.0)
    
    # do formatting of colorbar 
    cbar = do_cbar_formatting(cbar, do_rescale, cbar_nl, fontsize, cinfo['clevel'])
    
    # do labeling of colorbar
    if cbar_label is None : cbar_label = data[0][0][ vname ].attrs['long_name']
    if cbar_unit  is None : cbar_label = cbar_label+' ['+data[0][0][ vname ].attrs['units']+']'
    else                  : cbar_label = cbar_label+' ['+cbar_unit+']'
    if 'str_ltim' in data[0][0][vname].attrs.keys():
        cbar_label = cbar_label+'\n'+data[0][0][vname].attrs['str_ltim']
    cbar.set_label(cbar_label, size=fontsize+2)
    
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


#+___PLOT MERIDIONAL OVERTRUNING CIRCULATION  _________________________________+
#|                                                                             |
#+_____________________________________________________________________________+
def plot_zmeantransects(data, figsize=[12, 6], 
              n_rc=[1, 1], do_grid=True, cinfo=None, do_rescale=False,
              cbar_nl=8, cbar_orient='vertical', cbar_label=None, cbar_unit=None,
              do_bottom=True, max_dep=[], color_bot=[0.6, 0.6, 0.6], 
              pos_fac=1.0, pos_gap=[0.02, 0.02], do_save=None, save_dpi=600, 
              do_contour=True, do_clabel=False, title='descript', do_ylog=True,
              pos_extend=[0.05, 0.08, 0.95,0.95],
            ):
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
    fig, ax = plt.subplots( n_rc[0],n_rc[1], figsize=figsize, 
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
    cinfo = do_setupcinfo(cinfo, data, do_rescale, do_index=True)
    
    #_______________________________________________________________________
    # setup normalization log10, symetric log10, None
    which_norm = do_compute_scalingnorm(cinfo, do_rescale)
    
    #___________________________________________________________________________
    # loop over axes
    for ii in range(0,ndata):
        
        #_______________________________________________________________________
        # limit data to color range
        vname= list(data[ii][0].keys())[0]
        data_plot = data[ii][0][vname].values.copy()
        
        #_______________________________________________________________________
        # setup x-coorod and y-coord
        lat  , str_xlabel = data[ii][0]['lat'].values   , 'Latitude [deg]'
        depth, str_ylabel = data[ii][0]['depth'].values , 'Depth [m]'
        depth = np.abs(depth)
        
        #_______________________________________________________________________
        # be sure there are no holes
        data_plot[data_plot<cinfo['clevel'][ 0]] = cinfo['clevel'][ 0]+np.finfo(np.float32).eps
        data_plot[data_plot>cinfo['clevel'][-1]] = cinfo['clevel'][-1]-np.finfo(np.float32).eps
        
        #_______________________________________________________________________
        # plot zonal mean data
        hp=ax[ii].contourf(lat, depth, data_plot, levels=cinfo['clevel'], extend='both', cmap=cinfo['cmap'],
                           norm = which_norm)
        if do_contour: 
            tickl    = cinfo['clevel']
            ncbar_l  = len(tickl)
            idx_cref = np.where(cinfo['clevel']==cinfo['cref'])[0]
            idx_cref = np.asscalar(idx_cref)
            nstep    = ncbar_l/cbar_nl
            nstep    = np.max([np.int(np.floor(nstep)),1])
            
            idx = np.arange(0, ncbar_l, 1)
            idxb = np.ones((ncbar_l,), dtype=bool)                
            idxb[idx_cref::nstep]  = False
            idxb[idx_cref::-nstep] = False
            idx_yes = idx[idxb==False]
            cont=ax[ii].contour(lat, depth, data_plot, levels=cinfo['clevel'][idx_yes], colors='k', linewidths=[0.5],
                                norm = which_norm) #linewidths=[0.5,0.25])
            
            if do_clabel: 
                ax[ii].clabel(cont, cont.levels, inline=1, inline_spacing=1, fontsize=6, fmt='%1.1f',zorder=1)
                #ax[ii].clabel(cont, cont.levels[np.where(cont.levels!=cinfo['cref'])], 
                            #inline=1, inline_spacing=1, fontsize=6, fmt='%1.1f Sv')
            
        #___________________________________________________________________
        # plot bottom patch   
        if do_bottom:
            aux = np.ones(data_plot.shape,dtype='int16')
            aux[np.isnan(data_plot)]=0
            aux = aux.sum(axis=0)
            aux[aux!=0]=aux[aux!=0]-1
            bottom = depth[aux]
            # smooth bottom patch
            filt=np.array([1,2,3,2,1]) #np.array([1,2,1])
            filt=filt/np.sum(filt)
            aux = np.concatenate( (np.ones((filt.size,))*bottom[0],bottom,np.ones((filt.size,))*bottom[-1] ) )
            aux = np.convolve(aux,filt,mode='same')
            bottom = aux[filt.size:-filt.size]
            bottom = np.maximum(bottom, data[ii][0]['bottom'].values)
            
            ax[ii].fill_between(lat, bottom, depth[-1],color=[0.5,0.5,0.5], zorder=2)#,alpha=0.95)
            ax[ii].plot(lat, bottom, color='k')
        
        #_______________________________________________________________________
        # fix color range
        for im in ax[ii].get_images(): im.set_clim(cinfo['clevel'][ 0], cinfo['clevel'][-1])
        
        #_______________________________________________________________________
        # plot grid lines 
        if do_grid: ax[ii].grid(color='k', linestyle='-', linewidth=0.25,alpha=1.0)
        
        #_______________________________________________________________________
        # set description string plus x/y labels
        if title is not None: 
            txtx, txty = lat[0]+(lat[-1]-lat[0])*0.015, depth[-1]-(depth[-1]-depth[0])*0.015
            if   isinstance(title,str) : 
                # if title string is 'descript' than use descript attribute from 
                # data to set plot title 
                if title=='descript' and ('descript' in data[ii][0][vname].attrs.keys() ):
                    txts = data[ii][0][vname].attrs['descript']
                else:
                    txts = title
            # is title list of string        
            elif isinstance(title,list):   
                txts = title[ii]
            ax[ii].text(txtx, txty, txts, fontsize=12, fontweight='bold', horizontalalignment='left')
        
        #_______________________________________________________________________
        # set x/y label
        if collist[ii]==0        : ax[ii].set_ylabel(str_ylabel, fontsize=12)
        if rowlist[ii]==n_rc[0]-1: ax[ii].set_xlabel(str_xlabel, fontsize=12)
        
        #_______________________________________________________________________
        if do_ylog: 
            if depth[0]==0: ax[ii].set_ylim(depth[1],depth[-1])
            else          : ax[ii].set_ylim(depth[0],depth[-1])
            ax[ii].set_yscale('function', functions=(forward, inverse))
            yticklog = np.array([10,25,50,100,250,500,1000,2000,4000,6000])
            ax[ii].set_yticks(yticklog)
            ax[ii].invert_yaxis()
            ax[ii].grid(True,which='major')
            
        else:
            ax[ii].set_ylim(depth[0],depth[-1])
            ax[ii].invert_yaxis()
            ax[ii].grid(True,which='major')
        
        #ax[ii].set_yscale('log')
        #ax[ii].set_yticks([5,10,25,50,100,250,500,1000,2000,4000,6000])
        ax[ii].get_yaxis().set_major_formatter(ScalarFormatter())
        
    nax_fin = ii+1
    
    #___________________________________________________________________________
    # set superior title
    if 'transect_name' in data[ii][0][vname].attrs.keys():
        fig.suptitle( data[ii][0][vname].attrs['transect_name'], x=0.5, y=1.04, fontsize=16, 
                     horizontalalignment='center', verticalalignment='bottom')
    
    #___________________________________________________________________________
    # delete axes that are not needed
    #for jj in range(nax_fin, nax): fig.delaxes(ax[jj])
    for jj in range(ndata, nax): fig.delaxes(ax[jj])
    if nax != nax_fin-1: ax = ax[0:nax_fin]
    
    #___________________________________________________________________________
    # delete axes that are not needed
    cbar = fig.colorbar(hp, orientation=cbar_orient, ax=ax, ticks=cinfo['clevel'], 
                      extendrect=False, extendfrac=None,
                      drawedges=True, pad=0.025, shrink=1.0)
    
    # do formatting of colorbar 
    cbar = do_cbar_formatting(cbar, do_rescale, cbar_nl, fontsize, cinfo['clevel'])
    
    # do labeling of colorbar
    if cbar_label is None : cbar_label = data[0][0][ vname ].attrs['long_name']
    if cbar_unit  is None : cbar_label = cbar_label+' ['+data[0][0][ vname ].attrs['units']+']'
    else                  : cbar_label = cbar_label+' ['+cbar_unit+']'
    if 'str_ltim' in data[0][0][vname].attrs.keys():
        cbar_label = cbar_label+'\n'+data[0][0][vname].attrs['str_ltim']
    cbar.set_label(cbar_label, size=fontsize+2)
    
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



# ___DO ANOMALY________________________________________________________________
#| compute anomaly between two xarray Datasets                                 |
#| ___INPUT_________________________________________________________________   |
#| data1        :   xarray dataset object                                      |
#| data2        :   xarray dataset object                                      |
#| ___RETURNS_______________________________________________________________   |
#| anom         :   xarray dataset object, data1-data2                         |
#|_____________________________________________________________________________|
def do_transectanomaly(index1,index2):
    
    anom_index = list()
    
    #___________________________________________________________________________
    for idx1,idx2 in zip(index1, index2):
    
        # copy datasets object 
        anom_idx = idx1.copy()
        
        idx1_vname = list(idx1.keys())
        idx2_vname = list(idx2.keys())
        #for vname in list(anom.keys()):
        for vname, vname2 in zip(idx1_vname, idx2_vname):
            # do anomalous data 
            anom_idx[vname].data = idx1[vname].data - idx2[vname2].data
            
            # do anomalous attributes 
            attrs_data1 = idx1[vname].attrs
            attrs_data2 = idx2[vname2].attrs
            
            for key in attrs_data1.keys():
                if (key in attrs_data1.keys()) and (key in attrs_data2.keys()):
                    if key in ['long_name']:
                        anom_idx[vname].attrs[key] = 'anomalous '+anom_idx[vname].attrs[key] 
                    elif key in ['units',]: 
                        continue
                    elif key in ['descript']: 
                        if len(idx1[vname].attrs[key])+len(idx2[vname2].attrs[key])>30:
                            anom_idx[vname].attrs[key]  = idx1[vname].attrs[key]+'\n - '+idx2[vname2].attrs[key]
                        else:     
                            anom_idx[vname].attrs[key]  = idx1[vname].attrs[key]+' - '+idx2[vname2].attrs[key]
                        
                    elif key in ['do_rescale']: 
                        anom_idx[vname].attrs[key]  = idx1[vname].attrs[key]    
                
                    elif idx1[vname].attrs[key] != idx2[vname2].attrs[key]:
                        anom_idx[vname].attrs[key]  = idx1[vname].attrs[key]+' - '+idx2[vname2].attrs[key]
                    
        #___________________________________________________________________________
        anom_index.append(anom_idx)
    #___________________________________________________________________________
    return(anom_index)
