# Patrick, Scholz 02.09.2018
import numpy as np
import time
import os
from netCDF4 import Dataset
import xarray as xr
import matplotlib
matplotlib.rcParams['contour.negative_linestyle']= 'solid'
import matplotlib.pyplot as plt
import matplotlib.patches as Polygon
from colormap_c2c    import *
import matplotlib.path as mpltPath
from matplotlib.tri import Triangulation
from numba import jit, njit, prange
from sub_index import *
import shapefile as shp

#+___CALCULATE MERIDIONAL OVERTURNING FROM VERTICAL VELOCITIES_________________+
#| Global MOC, Atlantik MOC, Indo-Pacific MOC, Indo MOC                        |
#|                                                                             |
#+_____________________________________________________________________________+
def calc_xmoc(mesh, data, dlat=0.5, which_moc='gmoc', do_onelem=True, 
              do_info=True, diagpath=None, **kwargs,
             ):
    #_________________________________________________________________________________________________
    t1=time.time()
    if do_info==True: print('_____calc. '+which_moc.upper()+' from vertical velocities via meridional bins_____')
        
    #___________________________________________________________________________
    # calculate/use index for basin domain limitation
    if which_moc=='gmoc':
        if do_onelem: e_idxin = np.ones((mesh.n2de,), dtype=bool)
        else        : n_idxin = np.ones((mesh.n2dn,), dtype=bool)
    else:    
        tt1=time.time()
        box_list = list()
        
        #_______________________________________________________________________
        # set proper directory so he can find the moc basin shape files
        try: dname = os.environ['PATH_TRIPYVIEW']
        except: dname=''    
        mocbaspath=os.path.join(dname,'src/shapefiles/moc_basins/')
        
        #_______________________________________________________________________
        # amoc2 ... calculate amoc without arctic
        if   which_moc=='amoc':
            # for calculation of amoc mesh focus must be on 0 degree longitude
            box_list.append( shp.Reader(os.path.join(mocbaspath,'Atlantic_MOC.shp') ))
            
        #_______________________________________________________________________
        # amoc ... calculate amoc including arctic
        elif which_moc=='aamoc':
            # for calculation of amoc mesh focus must be on 0 degree longitude
            box_list.append( [-180.0,180.0,65.0,90.0] )
            box_list.append( shp.Reader(os.path.join(mocbaspath,'Atlantic_MOC.shp') ))
        #_______________________________________________________________________
        # pmoc ... calculate pacific moc
        elif which_moc=='pmoc':
            # for calculation of pmoc mesh focus must be on 180 degree longitude
            box_list.append( shp.Reader(os.path.join(mocbaspath,'Pacific_MOC.shp') ))
        
        #_______________________________________________________________________
        # ipmoc ... calculate indo-pacific moc
        elif which_moc=='ipmoc':
            # for calculation of pmoc mesh focus must be on 180 degree longitude
            box_list.append( shp.Reader(os.path.join(mocbaspath,'IndoPacific_MOC.shp') ))
            
        #_______________________________________________________________________
        # imoc ... calculate indian ocean moc
        elif which_moc=='imoc':
            box_list.append( shp.Reader(os.path.join(mocbaspath,'Indian_MOC.shp') ))
        
        #_______________________________________________________________________
        else: raise ValueError("The option which_moc={} is not supported.".format(str(which_moc)))
        
        #_______________________________________________________________________
        # compute vertice index for in box 
        n_idxin = np.zeros((mesh.n2dn,), dtype=bool)
        for box in box_list:
            n_idxin = np.logical_or(n_idxin, do_boxmask(mesh, box, do_elem=False))
        
        if do_onelem: 
            # e_idxin = n_idxin[np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia))].sum(axis=1)>=1
            e_idxin = n_idxin[mesh.e_i].sum(axis=1)>=1    
    
    #___________________________________________________________________________
    #e_idxin = n_idxin[np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia))].sum(axis=1)>=1    
    #tri = Triangulation(np.hstack((mesh.n_x,mesh.n_xa)), np.hstack((mesh.n_y,mesh.n_ya)),
                        #np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia)))
    #e_idxin = np.hstack(( e_idxin[mesh.e_pbnd_0], e_idxin[mesh.e_pbnd_a]))
    #fig = plt.figure(figsize=[6,3])
    #plt.triplot(tri.x, tri.y, tri.triangles[e_idxin,:], linewidth=0.2)
    #plt.axis('scaled')
    #plt.title('Basin limited domain')
    #plt.show()
    #STOP
    
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
        
    # compute area weighted vertical velocities on elements
    if do_onelem:
        #_______________________________________________________________________
        # load elem area from diag file
        if ( os.path.isfile(diagpath)):
            mat_area = xr.open_mfdataset(diagpath, parallel=True, **kwargs)['elem_area']
            mat_area = mat_area.isel(elem=e_idxin).compute()   
            mat_area = mat_area.expand_dims({'nz':mesh.zlev}).transpose()
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
            wdim = ['time','elem','nz']
            wdum = data['w'].data[:, mesh.e_i[e_idxin,:], :].sum(axis=2)/3.0 * 1e-6
        else                        : 
            wdim = ['elem','nz']
            wdum = data['w'].data[mesh.e_i[e_idxin,:], :].sum(axis=1)/3.0 * 1e-6
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
            for di in range(0,mesh.nlev): 
                mat_mean.data[:, np.where(di>=mat_iz)[0], di]=0.0
            
        else:
            mat_mean.data = np.multiply(mat_mean.data, mat_area.data)
            
            # be sure ocean floor is setted to zero 
            for di in range(0,mesh.nlev): 
                mat_mean.data[np.where(di>=mat_iz)[0], di]=0.0
        del mat_area
    
    # compute area weighted vertical velocities on vertices
    else:     
        #_______________________________________________________________________
        # load vertice cluster area from diag file
        if ( os.path.isfile(diagpath)):
            mat_area = xr.open_mfdataset(diagpath, parallel=True, **kwargs)['nod_area'].transpose() 
            mat_area = mat_area.isel(nod2=n_idxin).compute()  
            mat_iz   = xr.open_mfdataset(diagpath, parallel=True, **kwargs)['nlevels_nod2D']-1
            mat_iz   = mat_iz.isel(nod2=n_idxin).compute()   
        else: 
            raise ValueError('could not find ...mesh.diag.nc file')
        
        #_______________________________________________________________________
        # create meridional bins
        lat   = np.arange(np.floor(mesh.n_y[n_idxin].min())+dlat/2, 
                          np.ceil( mesh.n_y[n_idxin].max())-dlat/2, 
                          dlat)
        lat_i = ( (mesh.n_y[n_idxin]-lat[0])/dlat ).astype('int')
            
        #_______________________________________________________________________    
        # select MOC basin 
        mat_mean = data['w'].isel(nod2=n_idxin)*1e-6
        mat_mean = mat_mean.fillna(0.0)
        
        #_______________________________________________________________________
        # calculate area weighted mean
        if 'time' in list(data.dims):
            nt = data['time'].values.size
            for nti in range(nt):
                mat_mean.data[nti,:,:] = np.multiply(mat_mean.data[nti,:,:], mat_area.data)    
                
            # be sure ocean floor is setted to zero 
            for di in range(0,mesh.nlev): 
                mat_mean.data[:, np.where(di>=mat_iz)[0], di]=0.0
                
        else:
            mat_mean.data = np.multiply(mat_mean.data, mat_area.data)
            
            # be sure ocean floor is setted to zero 
            for di in range(0,mesh.nlev): 
                mat_mean.data[np.where(di>=mat_iz)[0], di]=0.0
        del mat_area
    
    #___________________________________________________________________________
    # This approach is five time faster than the original from dima at least for
    # COREv2 mesh but needs probaply a bit more RAM
    if 'time' in list(data.dims): auxmoc  = np.zeros([nt, mesh.nlev, lat.size])
    else                        : auxmoc  = np.zeros([mesh.nlev, lat.size])
    bottom  = np.zeros([lat.size,])
    numbtri = np.zeros([lat.size,])
    
    #  switch topo beteen computation on nodes and elements
    if do_onelem: topo    = np.float16(mesh.zlev[mesh.e_iz[e_idxin]])
    else        : topo    = np.float16(mesh.n_z[n_idxin])
    
    # this is more or less required so bottom patch looks aceptable
    if which_moc=='pmoc':
        topo[np.where(topo>-30.0)[0]]=np.nan  
    else:
        #topo[np.where(topo>-100.0)[0]]=np.nan  
        topo[np.where(topo>-30.0)[0]]=np.nan  
    
    # loop over meridional bins
    if 'time' in list(data.dims):
        for bini in range(lat_i.min(), lat_i.max()):
            numbtri[bini]= np.sum(lat_i==bini)
            auxmoc[:,:, bini]=mat_mean[:,lat_i==bini,:].sum(axis=1)
            bottom[bini] = np.nanpercentile(topo[lat_i==bini],15)
        
        # kickout outer bins where eventually no triangles are found
        idx    = numbtri>0
        auxmoc = auxmoc[:,:,idx]
        
    else:        
        for bini in range(lat_i.min(), lat_i.max()):
            numbtri[bini]= np.sum(lat_i==bini)
            auxmoc[:, bini]=mat_mean[lat_i==bini,:].sum(axis=0)
            bottom[bini] = np.nanpercentile(topo[lat_i==bini],15)
            
        # kickout outer bins where eventually no triangles are found
        idx    = numbtri>0
        auxmoc = auxmoc[:,idx]
        
    bottom = bottom[idx]
    lat    = lat[idx]
        
    # do cumulative summation to finally calculate moc
    if 'time' in list(data.dims):
        auxmoc = np.flip(auxmoc, axis=2)
        auxmoc = -auxmoc.cumsum(axis=2)
        auxmoc = np.flip(auxmoc, axis=2)
    else:
        auxmoc = np.fliplr(auxmoc)
        auxmoc = -auxmoc.cumsum(axis=1)
        auxmoc = np.fliplr(auxmoc)    
    
    
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
    local_attr  = data['w'].attrs
    local_attr['which_moc'] = which_moc
    
    # create coordinates
    if 'time' in list(data.dims):
        coords    = {'depth' : (['nz'], mesh.zlev), 
                     'lat'   : (['ny'], lat      ), 
                     'bottom': (['ny'], bottom   ),
                     'time'  : (['time'], data['time'].values)}
        mocdims = ['time', 'nz', 'ny']
    else:    
        coords    = {'depth' : (['nz'], mesh.zlev), 
                     'lat'   : (['ny'], lat      ), 
                     'bottom': (['ny'], bottom   )}
        mocdims = ['nz','ny']
        # create coordinates
    data_vars = {'moc'   : (mocdims, auxmoc, local_attr)} 
    moc = xr.Dataset(data_vars=data_vars, coords=coords, attrs=global_attr)
    
    #___________________________________________________________________________
    # write some infos 
    t2=time.time()
    if do_info==True: 
        print(' --> total time:{:.3f} s'.format(t2-t1))
        if 'time' not in list(data.dims):
            if which_moc in ['amoc', 'aamoc', 'gmoc']:
                maxv = moc.isel(nz=moc['depth']<=-700 , ny=moc['lat']>0.0)['moc'].max().values
                minv = moc.isel(nz=moc['depth']<=-2500, ny=moc['lat']>-50.0)['moc'].min().values
                print(' max. NADW_{:s} = {:.2f} Sv'.format(moc['moc'].attrs['descript'],maxv))
                print(' max. AABW_{:s} = {:.2f} Sv'.format(moc['moc'].attrs['descript'],minv))
            elif which_moc in ['pmoc', 'ipmoc']:
                minv = moc.isel(nz=moc['depth']<=-2000, ny=moc['lat']>-50.0)['moc'].min().values
                print(' max. AABW_{:s} = {:.2f} Sv'.format(moc['moc'].attrs['descript'],minv))
    
    #___________________________________________________________________________
    return(moc)
    
 
#+___PLOT MERIDIONAL OVERTRUNING CIRCULATION  _________________________________+
#|                                                                             |
#+_____________________________________________________________________________+
def plot_xmoc(data, which_moc='gmoc', figsize=[12, 6], 
              n_rc=[1, 1], do_grid=True, cinfo=None,
              cbar_nl=8, cbar_orient='vertical', cbar_label=None, cbar_unit=None,
              do_bottom=True, color_bot=[0.6, 0.6, 0.6], 
              pos_fac=1.0, pos_gap=[0.01, 0.01], do_save=None, save_dpi=600, 
              do_contour=True, do_clabel=True, title='descript',
              pos_extend=[0.06, 0.08, 0.95,0.95] ):
    #____________________________________________________________________________
    fontsize = 12
    str_rescale = None
    
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
    if cinfo is None: cinfo=dict()
    # check if dictionary keys exist, if they do not exist fill them up 
    cfac = 1
    if 'cfac' in cinfo.keys(): cfac = cinfo['cfac']
    if (('cmin' not in cinfo.keys()) or ('cmax' not in cinfo.keys())) and ('crange' not in cinfo.keys()):
        cmin, cmax = np.Inf, -np.Inf
        for data_ii in data:
            cmin = np.min([cmin, data_ii['moc'].isel(nz=data_ii['depth']<=-700).min().values ])
            cmax = np.max([cmax, data_ii['moc'].isel(nz=data_ii['depth']<=-700).max().values ])
            cmin, cmax = cmin*cfac, cmax*cfac
        if 'cmin' not in cinfo.keys(): cinfo['cmin'] = cmin
        if 'cmax' not in cinfo.keys(): cinfo['cmax'] = cmax    
    if 'crange' in cinfo.keys():
        # cinfo['cmin'], cinfo['cmax'], cinfo['cref'] = cinfo['crange'][0], cinfo['crange'][1], cinfo['crange'][2]
        cinfo['cmin'], cinfo['cmax'], cinfo['cref'] = cinfo['crange'][0], cinfo['crange'][1], 0.0
    else:
        if (cinfo['cmin'] == cinfo['cmax'] ): raise ValueError (' --> can\'t plot! data are everywhere: {}'.format(str(cinfo['cmin'])))
        cref = cinfo['cmin'] + (cinfo['cmax']-cinfo['cmin'])/2
        if 'cref' not in cinfo.keys(): cinfo['cref'] = 0
    if 'cnum' not in cinfo.keys(): cinfo['cnum'] = 15
    if 'cstr' not in cinfo.keys(): cinfo['cstr'] = 'blue2red'
    cinfo['cmap'],cinfo['clevel'] = colormap_c2c(cinfo['cmin'],cinfo['cmax'],cinfo['cref'],cinfo['cnum'],cinfo['cstr'])

    #___________________________________________________________________________
    # loop over axes
    ndi, nli, nbi =0, 0, 0
    for ii in range(0,ndata):
        
        #_______________________________________________________________________
        # limit data to color range
        data_plot = data[ii]['moc'].values
        lat       = data[ii]['lat'].values
        depth     = data[ii]['depth'].values
        
        data_plot[data_plot<cinfo['clevel'][ 0]] = cinfo['clevel'][ 0]+np.finfo(np.float32).eps
        data_plot[data_plot>cinfo['clevel'][-1]] = cinfo['clevel'][-1]-np.finfo(np.float32).eps
        
        #_______________________________________________________________________
        # plot MOC
        hp=ax[ii].contourf(lat, depth, data_plot, 
                           levels=cinfo['clevel'], extend='both', cmap=cinfo['cmap'])
        
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
            
            cont=ax[ii].contour(lat, depth, data_plot, 
                            levels=cinfo['clevel'][idx_yes], colors='k', linewidths=[0.5]) #linewidths=[0.5,0.25])
            if do_clabel: 
                ax[ii].clabel(cont, cont.levels[np.where(cont.levels!=cinfo['cref'])], 
                            inline=1, inline_spacing=1, fontsize=6, fmt='%1.1f Sv')
            ax[ii].contour(lat, depth, data_plot, 
                            levels=[0.0], colors='k', linewidths=[1.25]) #linewidths=[0.5,0.25])
            
        if do_bottom:
            bottom    = data[ii]['bottom'].values
            ax[ii].plot(lat, bottom, color='k')
            ax[ii].fill_between(lat, bottom, depth[-1], color=color_bot, zorder=2)#,alpha=0.95)
        
        #_______________________________________________________________________
        # fix color range
        for im in ax[ii].get_images(): im.set_clim(cinfo['clevel'][ 0], cinfo['clevel'][-1])
        
        #_______________________________________________________________________
        # plot grid lines 
        if do_grid: ax[ii].grid(color='k', linestyle='-', linewidth=0.25,alpha=1.0)
        
        #_______________________________________________________________________
        # set description string plus x/y labels
        if title is not None: 
            txtx, txty = lat[0]+(lat[-1]-lat[0])*0.025, depth[-1]-(depth[-1]-depth[0])*0.025                    
            if   isinstance(title,str) : 
                # if title string is 'descript' than use descript attribute from 
                # data to set plot title 
                if title=='descript' and ('descript' in data[ii]['moc'].attrs.keys() ):
                    txts = data[ii]['moc'].attrs['descript']
                else:
                    txts = title
            # is title list of string        
            elif isinstance(title,list):   
                txts = title[ii]
            ax[ii].text(txtx, txty, txts, fontsize=14, fontweight='bold', horizontalalignment='left')
        
        if collist[ii]==0        : ax[ii].set_ylabel('Depth [m]',fontsize=12)
        if rowlist[ii]==n_rc[0]-1: ax[ii].set_xlabel('Latitudes [deg]',fontsize=12)
        
    nax_fin = ii+1
    
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
    cbar.ax.tick_params(labelsize=fontsize)
    
    if which_moc=='gmoc'   : cbar_label = 'Global Meridional Overturning Circulation [Sv]'
    elif which_moc=='amoc' or which_moc=='aamoc':
        cbar_label = 'Atlantic Meridional Overturning Circulation [Sv]'
    elif which_moc=='pmoc' : cbar_label = 'Pacific Meridional Overturning Circulation [Sv]'
    elif which_moc=='ipmoc': cbar_label = 'Indo-Pacific Meridional Overturning Circulation [Sv]'
    elif which_moc=='imoc' : cbar_label = 'Indo Meridional Overturning Circulation [Sv]'
    if 'str_ltim' in data[0]['moc'].attrs.keys():
        cbar_label = cbar_label+'\n'+data[0]['moc'].attrs['str_ltim']
    cbar.set_label(cbar_label, size=fontsize+2)
    
    #___________________________________________________________________________
    # kickout some colormap labels if there are to many
    cbar = do_cbar_label(cbar, cbar_nl, cinfo)
    
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
def plot_xmoc_tseries(time,moc_t,which_lat=['max'],which_moc='amoc',str_descript='',str_time='',figsize=[]):    
    import matplotlib.patheffects as path_effects
    from matplotlib.ticker import AutoMinorLocator

    if len(figsize)==0: figsize=[13,4]
    fig,ax= plt.figure(figsize=figsize),plt.gca()
    
    for ii in range(len(which_lat)):
        if which_lat[ii]=='max':
            str_label='max {:s}: 30°N<=lat<=45°N'.format(which_moc.upper(),which_lat[ii])
        else:
            str_label='max {:s} at: {:2.1f}°N'.format(which_moc.upper(),which_lat[ii])
        hp=ax.plot(time,moc_t[:,ii],\
                   linewidth=2,label=str_label,marker='o',markerfacecolor='w',\
                   path_effects=[path_effects.SimpleLineShadow(offset=(1.5,-1.5),alpha=0.3),path_effects.Normal()])
        # plot mean value with trinagle 
        plt.plot(time[0]-(time[-1]-time[0])*0.015,moc_t[:,ii].mean(),\
                 marker='<',markersize=8,markeredgecolor='k',markeredgewidth=0.5,\
                 color=hp[0].get_color(),zorder=3,clip_box=False,clip_on=False)
        
        # plot std. range
        plt.plot(time[0]-(time[-1]-time[0])*0.015,moc_t[:,ii].mean()+moc_t[:,ii].std(),\
                 marker='^',markersize=6,markeredgecolor='k',markeredgewidth=0.5,\
                 color=hp[0].get_color(),zorder=3,clip_box=False,clip_on=False)
        
        plt.plot(time[0]-(time[-1]-time[0])*0.015,moc_t[:,ii].mean()-moc_t[:,ii].std(),\
                 marker='v',markersize=6,markeredgecolor='k',markeredgewidth=0.5,\
                 color=hp[0].get_color(),zorder=3,clip_box=False,clip_on=False)
        
    ax.legend(loc='lower right', shadow=True,fancybox=True,frameon=True,mode='None')
    ax.set_xlabel('Time [years]',fontsize=12)
    ax.set_ylabel('{:s} in [Sv]'.format(which_moc.upper()),fontsize=12)
    minor_locator = AutoMinorLocator(5)
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.xaxis.set_minor_locator(minor_locator)
    plt.grid(which='major')
    plt.xticks(np.arange(1940,2015,5))
    plt.xlim(time[0]-(time[-1]-time[0])*0.015,time[-1]+(time[-1]-time[0])*0.015)    
    plt.show(block=False)
    
    fig.canvas.draw()
    return(fig,ax)


#+___CALCULATE BASIN LIMITED DOMAIN____________________________________________+
#| to calculate the regional moc (amoc,pmoc,imoc) the domain needs be limited  |
#| to corresponding basin. here the elemental index of the triangels in the    |
#| closed basin is calcualted                                                  |
#+_____________________________________________________________________________+
def calc_basindomain(mesh,box_moc,do_output=False):
    
    if do_output==True: print(' --> calculate basin limited domain',end='')
    t1=time.time()
    
    #___________________________________________________________________________
    # 1st. pre-limit ocean domain by pre defined boxes for atlantic, 
    # indo-pacific ... basin
    box_moc = np.matrix(box_moc)
    allbox_idx  =np.zeros((mesh.n2dea,),dtype=bool)
    for bi in range(0,box_moc.shape[0]):
        #_______________________________________________________________________
        box_idx = mesh.nodes_2d_xg[mesh.elem_2d_i].sum(axis=1)/3.0<box_moc[bi,0]
        box_idx = np.logical_or(box_idx,mesh.nodes_2d_xg[mesh.elem_2d_i].sum(axis=1)/3.0>box_moc[bi,1])
        box_idx = np.logical_or(box_idx,mesh.nodes_2d_yg[mesh.elem_2d_i].sum(axis=1)/3.0<box_moc[bi,2])
        box_idx = np.logical_or(box_idx,mesh.nodes_2d_yg[mesh.elem_2d_i].sum(axis=1)/3.0>box_moc[bi,3])
        allbox_idx[np.where(box_idx==False)[0]]=True
    
    # kick out all the elements that are not located within predefined box
    allbox_elem = mesh.elem_2d_i[allbox_idx==True,:]
    
    #fig = plt.figure(figsize=[10,5])
    #plt.triplot(mesh.nodes_2d_xg,mesh.nodes_2d_yg,mesh.elem_2d_i[allbox_idx==True,:],linewidth=0.2)
    #plt.axis('scaled')
    #plt.title('Box limited domain')
    #plt.show()
    #fig.canvas.draw()
    
    #___________________________________________________________________________
    # 2nd. select main ocean basin cluster for either atlantic or pacific by finding 
    # the outer coastline of the main basin
    edge    = np.concatenate((allbox_elem[:,[0,1]], allbox_elem[:,[0,2]], allbox_elem[:,[1,2]]),axis=0)
    edge    = np.sort(edge,axis=1) 
    
    # python  sortrows algorythm --> matlab equivalent
    # twice as fast as list sorting
    sortidx = np.lexsort((edge[:,0],edge[:,1]))
    edge    = edge[sortidx,:].squeeze()
    edge    = np.array(edge)
    
    # list sorting
    #edge    = edge.tolist()
    #edge.sort()
    #edge    = np.array(edge)
        
    idx     = np.diff(edge,axis=0)==0
    idx     = np.all(idx,axis=1)
    idx     = np.logical_or(np.concatenate((idx,np.array([False]))),np.concatenate((np.array([False]),idx)))
        
    # all outer ocean edges that belong to preselected boxes defined domain
    bnde    = edge[idx==False,:]
    nbnde    = bnde.shape[0];
    
    # find most northern ocean edge and start from there
    maxi = np.argmax(mesh.nodes_2d_yg[bnde].sum(axis=1)/2)
    
    #fig = plt.figure(figsize=[10,5])
    #plt.plot(mesh.nodes_2d_xg[np.unique(bnde.flatten())],mesh.nodes_2d_yg[np.unique(bnde.flatten())],'*')
    #plt.plot(mesh.nodes_2d_xg[np.unique(bnde[maxi].flatten())],mesh.nodes_2d_yg[np.unique(bnde[maxi].flatten())],'*',color='red')
    #plt.axis('scaled')
    #plt.title('Outer boundary edges')
    #plt.show()
    #fig.canvas.draw()
    
    #___________________________________________________________________________
    # start with on outer coastline edge and find the next edge that is 
    # connected and so forth, like that build entire outer coastline of the main 
    # basin
    
    run_cont        = np.zeros((1,nbnde+1))*np.nan
    run_cont[0,:2]  = bnde[maxi,:] # initialise the first landmask edge
    #run_bnde        = bnde[1:,:] # remaining edges that still need to be distributed
    run_bnde        = np.delete(bnde, (maxi), axis=0)
    count_init      = 1;
    init_ind        = run_cont[0,0];
    ind_lc_s        = 0;
    
    ocebasin_polyg = []
    for ii in range(0,nbnde):
        #_______________________________________________________________________
        # search for next edge that contains the last node index from 
        # run_cont
        kk_rc = np.column_stack(np.where( run_bnde==np.int(run_cont[0,count_init]) ))
        kk_r  = kk_rc[:,0]
        kk_c  = kk_rc[:,1]
        count_init  = count_init+1
        if len(kk_c)==0 : break        
        
        #_______________________________________________________________________
        if kk_c[0] == 0 :
            run_cont[0,count_init] = run_bnde[kk_r[0],1]
        else:
            run_cont[0,count_init] = run_bnde[kk_r[0],0]
            
        #_______________________________________________________________________
        # if a land sea mask polygon is closed
        if  np.any(run_bnde[kk_r[0],:] == init_ind):
            count_init  = count_init+1
            
            aux_lx = mesh.nodes_2d_xg[np.int64(run_cont[0,0:count_init])];
            aux_ly = mesh.nodes_2d_yg[np.int64(run_cont[0,0:count_init])];
            aux_xy = np.zeros((count_init,2))
            aux_xy[:,0] = aux_lx
            aux_xy[:,1] = aux_ly
            ocebasin_polyg=aux_xy
            del aux_lx; del aux_ly; del aux_xy
            
            ind_lc_s = ind_lc_s+count_init+1;
            
            count_init = count_init+1
            aux_ind  = np.arange(0,run_bnde.shape[0],1)
            run_bnde = run_bnde[aux_ind!=kk_r[0],:]
            if np.size(run_bnde)==0:
                break
                
            #___________________________________________________________________
            run_cont        = np.zeros((1,nbnde))*np.nan
            run_cont[0,:2]  = run_bnde[0,:]
            run_bnde        = run_bnde[1:,:]
            count_init=1;
        else:
            aux_ind =np.arange(0,run_bnde.shape[0],1)
            run_bnde=run_bnde[aux_ind!=kk_r[0],:]
            
    #___________________________________________________________________________
    # check which preselected triangle centroids are within main ocean basin 
    # polygon 
    ptsc = list(zip(mesh.nodes_2d_xg[allbox_elem].sum(axis=1)/3,mesh.nodes_2d_yg[allbox_elem].sum(axis=1)/3))
    
    #___________________________________________________________________________
    # Option (1)
    # python Matplotlib mplPath seems to be faster at least by a factor of 2 when
    # compared to ray_tracing method
    #print(' >> use mpltPath ',end='')
    #path = mpltPath.Path(ocebasin_polyg)
    #inside_ocebasin = path.contains_points(ptsc)
    
    # Option (2)
    # determine points in polygon by ray tracing method
    #print(' >> use rtracing ',end='')
    #inside_ocebasin = [calc_ray_tracing(point[0], point[1], np.array(ocebasin_polyg)) for point in ptsc]
    #inside_ocebasin = np.array(inside_ocebasin)
    
    # Option (3)
    # determine points in polygon by parallel (numba optimized) ray tracing 
    # method --> considerable faster for large meshes
    print(' >> use rtracing parallel ',end='')
    inside_ocebasin = calc_ray_tracing_parallel(np.array(ptsc),np.array(ocebasin_polyg),np.zeros((len(ptsc),),dtype=bool))
    
    #___________________________________________________________________________
    # write out regional indices with respect to the global elemental array
    allbox_tidx = np.where(allbox_idx==True)[0]
    allbox_fin = allbox_tidx[inside_ocebasin==True]
    
    #fig = plt.figure(figsize=[10,5])
    #plt.triplot(mesh.nodes_2d_xg,mesh.nodes_2d_yg,mesh.elem_2d_i[allbox_fin,:],linewidth=0.2)
    #plt.axis('scaled')
    #plt.title('Basin limited domain')
    #plt.show()
    #fig.canvas.draw()
    
    #___________________________________________________________________________
    t2=time.time()
    print(" >> time: {:.3f} s".format(t2-t1))   
    
    return(allbox_fin)


#+___CALCULATE BASIN LIMITED DOMAIN___________________________________________________________________+
#| to calculate the regional moc (amoc,pmoc,imoc) the domain needs be limited to corresponding basin.
#| here the elemental index of the triangels in the closed basin is calcualted
#+____________________________________________________________________________________________________+
def calc_basindomain_slow(mesh,box_moc,do_output=False):
    
    if do_output==True: print('     --> calculate regional basin limited domain')
    box_moc = np.matrix(box_moc)
    for bi in range(0,box_moc.shape[0]):
        #_____________________________________________________________________________________________
        box_idx = mesh.nodes_2d_xg[mesh.elem_2d_i].sum(axis=1)/3.0<box_moc[bi,0]
        box_idx = np.logical_or(box_idx,mesh.nodes_2d_xg[mesh.elem_2d_i].sum(axis=1)/3.0>box_moc[bi,1])
        box_idx = np.logical_or(box_idx,mesh.nodes_2d_yg[mesh.elem_2d_i].sum(axis=1)/3.0<box_moc[bi,2])
        box_idx = np.logical_or(box_idx,mesh.nodes_2d_yg[mesh.elem_2d_i].sum(axis=1)/3.0>box_moc[bi,3])
        box_idx = np.where(box_idx==False)[0]
        box_elem2di = mesh.elem_2d_i[box_idx,:]

        #_____________________________________________________________________________________________
        # calculate edge indices of box limited domain
        edge_12     = np.sort(np.array(box_elem2di[:,[0,1]]),axis=1)
        edge_23     = np.sort(np.array(box_elem2di[:,[1,2]]),axis=1)
        edge_31     = np.sort(np.array(box_elem2di[:,[2,0]]),axis=1)
        edge_triidx = np.arange(0,box_elem2di.shape[0],1)

        #_____________________________________________________________________________________________
        # start with seed triangle
        seed_pts     = [box_moc[bi,0]+(box_moc[bi,1]-box_moc[bi,0])/2.0,box_moc[bi,2]+(box_moc[bi,3]-box_moc[bi,2])/2.0]
        seed_triidx  = np.argsort((mesh.nodes_2d_xg[box_elem2di].sum(axis=1)/3.0-seed_pts[0])**2 + (mesh.nodes_2d_yg[box_elem2di].sum(axis=1)/3.0-seed_pts[1])**2,axis=-0)[0]
        seed_elem2di = box_elem2di[seed_triidx,:]
        seed_edge    = np.concatenate((seed_elem2di[:,[0,1]], seed_elem2di[:,[1,2]], seed_elem2di[:,[2,0]]),axis=0)     
        seed_edge    = np.sort(seed_edge,axis=1) 
        
        # already delete seed triangle and coresbonding edges from box limited domain list
        edge_triidx = np.delete(edge_triidx,seed_triidx)
        edge_12     = np.delete(edge_12,seed_triidx,0)
        edge_23     = np.delete(edge_23,seed_triidx,0)
        edge_31     = np.delete(edge_31,seed_triidx,0)

        #_____________________________________________________________________________________________
        # do iterative search of which triangles are connected to each other and form cluster
        t1 = time.time()
        tri_merge_idx = np.zeros((box_elem2di.shape[0],),dtype='int')
        tri_merge_count = 0
        for ii in range(0,10000): 
            #print(ii,tri_merge_count,seed_edge.shape[0])
        
            # determine which triangles contribute to edge
            triidx12 = ismember_rows(seed_edge,edge_12)
            triidx23 = ismember_rows(seed_edge,edge_23)
            triidx31 = ismember_rows(seed_edge,edge_31)
        
            # calculate new seed edges
            seed_edge = np.concatenate((edge_23[triidx12,:],edge_31[triidx12,:],\
                                        edge_12[triidx23,:],edge_31[triidx23,:],\
                                        edge_12[triidx31,:],edge_23[triidx31,:]))
            
            # collect all found connected triagles    
            triidx = np.concatenate((triidx12,triidx23,triidx31))
            triidx = np.unique(triidx)
            
            # break out of iteration loop 
            if triidx.size==0: break 
                
            # add found trinagles to final domain list    
            tri_merge_idx[tri_merge_count:tri_merge_count+triidx.size]=edge_triidx[triidx]
            tri_merge_count = tri_merge_count+triidx.size
            
            # delete already found trinagles and edges from list
            edge_triidx = np.delete(edge_triidx,triidx)
            edge_12     = np.delete(edge_12,triidx,0)
            edge_23     = np.delete(edge_23,triidx,0)
            edge_31     = np.delete(edge_31,triidx,0)
    
            del triidx,triidx12,triidx23,triidx31
        
        tri_merge_idx = tri_merge_idx[:tri_merge_count-1]
        t2=time.time()
        if do_output==True: print('         elpased time:'+str(t2-t1)+'s')
        
        #_____________________________________________________________________________________________
        # calculate final domain limited trinagle cluster element index
        if bi==0:
            box_idx_fin = box_idx[tri_merge_idx]
        else:
            box_idx_fin = np.concatenate((box_idx_fin,box_idx[tri_merge_idx]))
        
    return(box_idx_fin)


#+___EQUIVALENT OF MATLAB ISMEMBER FUNCTION___________________________________________________________+
#|                                                                                                    |
#+____________________________________________________________________________________________________+
def ismember_rows(a, b):
    return np.flatnonzero(np.in1d(b[:,0], a[:,0]) & np.in1d(b[:,1], a[:,1]))


#+___RAY TRACING METHOD TO CHECK IF POINT IS IN POLYGON________________________+
#| see...https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-
#| checking-if-a-point-is-inside-a-polygon-in-python
#| 
#+_____________________________________________________________________________+
@jit(nopython=True)
def calc_ray_tracing(x,y,poly):
    n = len(poly)
    inside = False
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside
#+___RAY TRACING METHOD PARALLEL TO CHECK IF POINT IS IN POLYGON_______________+
#| see...https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-|
#| checking-if-a-point-is-inside-a-polygon-in-python                           |
#|                                                                             |
#+_____________________________________________________________________________+
#@njit(parallel=True,nopython=True)
@njit(parallel=True)
def calc_ray_tracing_parallel(pts,poly,inside):
    n = len(poly)
    npts=len(pts)
    for j in prange(npts):
        x,y = pts[j]
        p1x,p1y = poly[0]
        for i in range(n+1):
            p2x,p2y = poly[i % n]
            if y > min(p1y,p2y):
                if y <= max(p1y,p2y):
                    if x <= max(p1x,p2x):
                        if p1y != p2y:
                            xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xints:
                            inside[j] = not inside[j]
            p1x,p1y = p2x,p2y

    return inside
