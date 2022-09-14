# Patrick, Scholz 02.09.2018
import numpy as np
import time as time
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

#+___CALCULATE MERIDIONAL OVERTURNING FROM VERTICAL VELOCITIES_________________+
#| Global MOC, Atlantik MOC, Indo-Pacific MOC, Indo MOC                        |
#|                                                                             |
#+_____________________________________________________________________________+
def calc_xmoc(mesh, data, dlat=0.5, which_moc='gmoc', do_onelem=False, 
              do_info=True, diagpath=None, **kwargs,
             ):
    #_________________________________________________________________________________________________
    t1=time.time()
    if do_info==True: print('_____calc. '+which_moc.upper()+' from vertical velocities via meridional bins_____')
        
    #___________________________________________________________________________
    # calculate/use index for basin domain limitation
    idxin = calc_basindomain_fast(mesh, which_moc=which_moc, do_onelem=do_onelem)
    
    #___________________________________________________________________________
    # e_idxin = n_idxin[np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia))].sum(axis=1)>=1    
    # tri = Triangulation(np.hstack((mesh.n_x,mesh.n_xa)), np.hstack((mesh.n_y,mesh.n_ya)),
    #                     np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia)))
    # e_idxin = np.hstack(( e_idxin[mesh.e_pbnd_0], e_idxin[mesh.e_pbnd_a]))
    # fig = plt.figure(figsize=[6,3])
    # plt.triplot(tri.x, tri.y, tri.triangles[e_idxin,:], linewidth=0.2)
    # plt.plot(mesh.n_x[n_idxin], mesh.n_y[n_idxin],'*')
    # plt.axis('scaled')
    # plt.title('Basin limited domain')
    # plt.show()
    # STOP
    
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
        which_hdim='elem'
        #_______________________________________________________________________
        # load elem area from diag file
        if ( os.path.isfile(diagpath)):
            mat_area = xr.open_mfdataset(diagpath, parallel=True, **kwargs)['elem_area']
            mat_area = mat_area.isel(elem=idxin).compute()   
            mat_area = mat_area.expand_dims({'nz':mesh.zlev}).transpose()
            mat_iz   = xr.open_mfdataset(diagpath, parallel=True, **kwargs)['nlevels']-1
            mat_iz   = mat_iz.isel(elem=idxin).compute()   
        else: 
            raise ValueError('could not find ...mesh.diag.nc file')
        
        #_______________________________________________________________________
        # create meridional bins
        e_y   = mesh.n_y[mesh.e_i[idxin,:]].sum(axis=1)/3.0
        lat   = np.arange(np.floor(e_y.min())+dlat/2, 
                          np.ceil( e_y.max())-dlat/2, 
                          dlat)
        lat_i = (( e_y-lat[0])/dlat ).astype('int')
        
        #_______________________________________________________________________    
        # mean over elements + select MOC basin 
        if 'time' in list(data.dims): 
            wdim = ['time','elem','nz']
            wdum = data['w'].data[:, mesh.e_i[idxin,:], :].sum(axis=2)/3.0 * 1e-6
        else                        : 
            wdim = ['elem','nz']
            wdum = data['w'].data[mesh.e_i[idxin,:], :].sum(axis=1)/3.0 * 1e-6
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
        which_hdim='nod2'
        #_______________________________________________________________________
        # load vertice cluster area from diag file
        if ( os.path.isfile(diagpath)):
            mat_area = xr.open_mfdataset(diagpath, parallel=True, chunks=dict(data.chunks), **kwargs)['nod_area']
            mat_area = mat_area.isel(nod2=idxin).drop('nz').load().transpose() 
        else: 
            raise ValueError('could not find ...mesh.diag.nc file')
        
        #_______________________________________________________________________    
        # select MOC basin 
        mat_mean = data['w'].isel(nod2=idxin)
        
        #_______________________________________________________________________
        # calculate area weighted mean
        #mat_mean.data = mat_mean*mat_area*1e-6
        mat_mean = mat_mean*mat_area*1e-6
        del mat_area
        
        #_______________________________________________________________________
        # load all necessary data int o memory do this befor binning loop 
        # otherwise binning loop takes forever
        mat_mean = mat_mean.load()
        
        #_______________________________________________________________________
        # create meridional bins
        lat   = np.arange(np.floor(mesh.n_y[idxin].min())+dlat/2, 
                          np.ceil( mesh.n_y[idxin].max())-dlat/2, 
                          dlat)
        lat_i = ( (mesh.n_y[idxin]-lat[0])/dlat ).astype('int')
        
    #___________________________________________________________________________
    # This approach is five time faster than the original from dima at least for
    # COREv2 mesh but needs probaply a bit more RAM
    if 'time' in list(data.dims): 
        nt = data.dims['time']
        auxmoc  = np.zeros([nt, mesh.nlev, lat.size])
    else                        : 
        auxmoc  = np.zeros([mesh.nlev, lat.size])
    bottom  = np.zeros([lat.size,])
    numbtri = np.zeros([lat.size,])
    
    #  switch topo beteen computation on nodes and elements
    if do_onelem: topo    = np.float16(mesh.zlev[mesh.e_iz[idxin]])
    else        : topo    = np.float16(mesh.n_z[idxin])
    
    # this is more or less required so bottom patch looks aceptable
    if which_moc=='pmoc':
        topo[np.where(topo>-30.0)[0]]=np.nan  
    else:
        #topo[np.where(topo>-100.0)[0]]=np.nan  
        topo[np.where(topo>-30.0)[0]]=np.nan  
    
    # loop over meridional bins
    if 'time' in list(data.dims):
        for bini in range(lat_i.min(), lat_i.max()):
            numbtri[bini]    = np.sum(lat_i==bini)
            #auxmoc[:,:, bini]= mat_mean[:,lat_i==bini,:].sum(axis=1)
            auxmoc[:,:, bini]= mat_mean.isel({which_hdim:lat_i==bini}).sum(dim=which_hdim)
            bottom[bini]     = np.nanpercentile(topo[lat_i==bini],15)
        
        # kickout outer bins where eventually no triangles are found
        idx    = numbtri>0
        auxmoc = auxmoc[:,:,idx]
        
    else:        
        for bini in range(lat_i.min(), lat_i.max()):
            numbtri[bini]   = np.sum(lat_i==bini)
            #auxmoc[:, bini] = mat_mean[lat_i==bini,:].sum(axis=0)
            auxmoc[:, bini] = mat_mean.isel({which_hdim:lat_i==bini}).sum(dim=which_hdim)
            bottom[bini]    = np.nanpercentile(topo[lat_i==bini],15)
            
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
              n_rc=[1, 1], do_grid=True, cinfo=None, do_rescale=None, 
              do_reffig=False, ref_cinfo=None, ref_rescale=None,
              cbar_nl=8, cbar_orient='vertical', cbar_label=None, cbar_unit=None,
              do_bottom=True, color_bot=[0.6, 0.6, 0.6], 
              pos_fac=1.0, pos_gap=[0.01, 0.01], do_save=None, save_dpi=600, 
              do_contour=True, do_clabel=True, title='descript', 
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
    if do_reffig:
        ref_cinfo = do_setupcinfo(ref_cinfo, [data[0]], ref_rescale, do_moc=True)
        cinfo     = do_setupcinfo(cinfo    , data[1:] , do_rescale , do_moc=True)
    else:
        cinfo     = do_setupcinfo(cinfo    , data     , do_rescale , do_moc=True)
        
    #___________________________________________________________________________
    # loop over axes
    ndi, nli, nbi =0, 0, 0
    hpall=list()
    for ii in range(0,ndata):
        
        #_______________________________________________________________________
        # limit data to color range
        data_plot = data[ii]['moc'].values
        lat       = data[ii]['lat'].values
        depth     = data[ii]['depth'].values
        
        #_______________________________________________________________________
        if do_reffig: 
            if ii==0: cinfo_plot = ref_cinfo
            else    : cinfo_plot = cinfo
        else: cinfo_plot = cinfo
        
        #_______________________________________________________________________
        data_plot[data_plot<cinfo_plot['clevel'][ 0]] = cinfo_plot['clevel'][ 0]+np.finfo(np.float32).eps
        data_plot[data_plot>cinfo_plot['clevel'][-1]] = cinfo_plot['clevel'][-1]-np.finfo(np.float32).eps
        
        #_______________________________________________________________________
        # plot MOC
        hp=ax[ii].contourf(lat, depth, data_plot, 
                           levels=cinfo_plot['clevel'], extend='both', cmap=cinfo_plot['cmap'])
        hpall.append(hp)
        
        if do_contour: 
            tickl    = cinfo_plot['clevel']
            ncbar_l  = len(tickl)
            idx_cref = np.where(cinfo_plot['clevel']==cinfo_plot['cref'])[0]
            idx_cref = np.asscalar(idx_cref)
            nstep    = ncbar_l/cbar_nl
            nstep    = np.max([np.int(np.floor(nstep)),1])
            
            idx = np.arange(0, ncbar_l, 1)
            idxb = np.ones((ncbar_l,), dtype=bool)                
            idxb[idx_cref::nstep]  = False
            idxb[idx_cref::-nstep] = False
            idx_yes = idx[idxb==False]
            
            cont=ax[ii].contour(lat, depth, data_plot, 
                            levels=cinfo_plot['clevel'][idx_yes], colors='k', linewidths=[0.5]) #linewidths=[0.5,0.25])
            if do_clabel: 
                ax[ii].clabel(cont, cont.levels[np.where(cont.levels!=cinfo_plot['cref'])], 
                            inline=1, inline_spacing=1, fontsize=6, fmt='%1.1f Sv')
            ax[ii].contour(lat, depth, data_plot, 
                            levels=[0.0], colors='k', linewidths=[1.25]) #linewidths=[0.5,0.25])
            
        if do_bottom:
            bottom    = data[ii]['bottom'].values
            ax[ii].plot(lat, bottom, color='k')
            ax[ii].fill_between(lat, bottom, depth[-1], color=color_bot, zorder=2)#,alpha=0.95)
        
        #_______________________________________________________________________
        # fix color range
        for im in ax[ii].get_images(): im.set_clim(cinfo_plot['clevel'][ 0], cinfo_plot['clevel'][-1])
        
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
            ax[ii].text(txtx, txty, txts, fontsize=12, fontweight='bold', horizontalalignment='left')
        
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
    if do_reffig==False:
        cbar = fig.colorbar(hp, orientation=cbar_orient, ax=ax, ticks=cinfo['clevel'], 
                        extendrect=False, extendfrac=None,
                        drawedges=True, pad=0.025, shrink=1.0)
        
        # do formatting of colorbar 
        cbar = do_cbar_formatting(cbar, do_rescale, cbar_nl, fontsize, cinfo['clevel'])
        
        # do labeling of colorbar
        #if n_rc[0]==1:
            #if   which_moc=='gmoc' : cbar_label = 'Global Meridional \n Overturning Circulation [Sv]'
            #elif which_moc=='amoc' : cbar_label = 'Atlantic Meridional \n Overturning Circulation [Sv]'
            #elif which_moc=='aamoc': cbar_label = 'Arctic-Atlantic Meridional \n Overturning Circulation [Sv]'
            #elif which_moc=='pmoc' : cbar_label = 'Pacific Meridional \n Overturning Circulation [Sv]'
            #elif which_moc=='ipmoc': cbar_label = 'Indo-Pacific Meridional \n Overturning Circulation [Sv]'
            #elif which_moc=='imoc' : cbar_label = 'Indo Meridional \n Overturning Circulation [Sv]'
        #else:    
            #if   which_moc=='gmoc' : cbar_label = 'Global Meridional Overturning Circulation [Sv]'
            #elif which_moc=='amoc' : cbar_label = 'Atlantic Meridional Overturning Circulation [Sv]'
            #elif which_moc=='aamoc': cbar_label = 'Arctic-Atlantic Meridional Overturning Circulation [Sv]'
            #elif which_moc=='pmoc' : cbar_label = 'Pacific Meridional Overturning Circulation [Sv]'
            #elif which_moc=='ipmoc': cbar_label = 'Indo-Pacific Meridional Overturning Circulation [Sv]'
            #elif which_moc=='imoc' : cbar_label = 'Indo Meridional Overturning Circulation [Sv]'
        if   which_moc=='gmoc' : cbar_label = 'Global MOC [Sv]'
        elif which_moc=='amoc' : cbar_label = 'Atlantic MOC [Sv]'
        elif which_moc=='aamoc': cbar_label = 'Arctic-Atlantic MOC [Sv]'
        elif which_moc=='pmoc' : cbar_label = 'Pacific MOC [Sv]'
        elif which_moc=='ipmoc': cbar_label = 'Indo-Pacific MOC [Sv]'
        elif which_moc=='imoc' : cbar_label = 'Indo MOC [Sv]'    
        if 'str_ltim' in data[0]['moc'].attrs.keys():
            cbar_label = cbar_label+'\n'+data[0]['moc'].attrs['str_ltim']
        cbar.set_label(cbar_label, size=fontsize+2)
        
    else:    
        cbar=list()
        for ii, aux_ax in enumerate(ax): 
            cbar_label = ''
            if ii==0:
                aux_cbar = fig.colorbar(hpall[ii], orientation=cbar_orient, ax=aux_ax, ticks=ref_cinfo['clevel'], 
                                        extendrect=False, extendfrac=None, drawedges=True, pad=0.025, shrink=1.0)
                aux_cbar = do_cbar_formatting(aux_cbar, ref_rescale, cbar_nl, fontsize, ref_cinfo['clevel'])
            else:
                aux_cbar = fig.colorbar(hpall[ii], orientation=cbar_orient, ax=aux_ax, ticks=cinfo['clevel'], 
                                        extendrect=False, extendfrac=None, drawedges=True, pad=0.025, shrink=1.0)
                aux_cbar = do_cbar_formatting(aux_cbar, do_rescale, cbar_nl, fontsize, cinfo['clevel'])
                #cbar_label = 'anomalous '
                cbar_label = 'anom. '
            # do labeling of colorbar
            #if n_rc[0]==1:
                #if   which_moc=='gmoc' : cbar_label = 'Global Meridional \n Overturning Circulation [Sv]'
                #elif which_moc=='amoc' : cbar_label = 'Atlantic Meridional \n Overturning Circulation [Sv]'
                #elif which_moc=='aamoc': cbar_label = 'Arctic-Atlantic Meridional \n Overturning Circulation [Sv]'
                #elif which_moc=='pmoc' : cbar_label = 'Pacific Meridional \n Overturning Circulation [Sv]'
                #elif which_moc=='ipmoc': cbar_label = 'Indo-Pacific Meridional \n Overturning Circulation [Sv]'
                #elif which_moc=='imoc' : cbar_label = 'Indo Meridional \n Overturning Circulation [Sv]'
            #else:    
                #if   which_moc=='gmoc' : cbar_label = 'Global Meridional Overturning Circulation [Sv]'
                #elif which_moc=='amoc' : cbar_label = 'Atlantic Meridional Overturning Circulation [Sv]'
                #elif which_moc=='aamoc': cbar_label = 'Arctic-Atlantic Meridional Overturning Circulation [Sv]'
                #elif which_moc=='pmoc' : cbar_label = 'Pacific Meridional Overturning Circulation [Sv]'
                #elif which_moc=='ipmoc': cbar_label = 'Indo-Pacific Meridional Overturning Circulation [Sv]'
                #elif which_moc=='imoc' : cbar_label = 'Indo Meridional Overturning Circulation [Sv]'
            if   which_moc=='gmoc' : cbar_label = cbar_label+'Global MOC [Sv]'
            elif which_moc=='amoc' : cbar_label = cbar_label+'Atlantic MOC [Sv]'
            elif which_moc=='aamoc': cbar_label = cbar_label+'Arctic-Atlantic MOC [Sv]'
            elif which_moc=='pmoc' : cbar_label = cbar_label+'Pacific MOC [Sv]'
            elif which_moc=='ipmoc': cbar_label = cbar_label+'Indo-Pacific MOC [Sv]'
            elif which_moc=='imoc' : cbar_label = cbar_label+'Indo MOC [Sv]'    
            if 'str_ltim' in data[0]['moc'].attrs.keys():
                cbar_label = cbar_label+'\n'+data[0]['moc'].attrs['str_ltim']
                #cbar_label = cbar_label+', '+data[0]['moc'].attrs['str_ltim']
            aux_cbar.set_label(cbar_label, size=fontsize+2)
            cbar.append(aux_cbar)
    
    #___________________________________________________________________________
    # repositioning of axes and colorbar
    if do_reffig==False:
        ax, cbar = do_reposition_ax_cbar(ax, cbar, rowlist, collist, pos_fac, pos_gap, 
                                     title=None, extend=pos_extend)
    fig.canvas.draw()
    
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, fig, dpi=save_dpi)
    plt.show(block=False)
    
    #___________________________________________________________________________
    return(fig, ax, cbar)



#+___PLOT MERIDIONAL OVERTRUNING CIRCULATION TIME-SERIES_______________________+
#|                                                                             |
#+_____________________________________________________________________________+
def plot_xmoc_tseries(time, moct_list, input_names, which_cycl=None, which_lat=['max'], 
                       which_moc='amoc', do_allcycl=False, do_concat=False, ymaxstep=1, xmaxstep=5,
                       str_descript='', str_time='', figsize=[], do_rapid=None, 
                       do_save=None, save_dpi=600, do_pltmean=True, do_pltstd=False ):    
    
    import matplotlib.patheffects as path_effects
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator

    if len(figsize)==0: figsize=[13,6.5]
    if do_concat: figsize[0] = figsize[0]*2
    fig,ax= plt.figure(figsize=figsize),plt.gca()
    
    #___________________________________________________________________________
    # setup colormap
    if do_allcycl: 
        if which_cycl is not None:
            cmap = categorical_cmap(np.int32(len(moct_list)/which_cycl), which_cycl, cmap="tab10")
        else:
            cmap = categorical_cmap(len(moct_list), 1, cmap="tab10")
    else:
        if do_concat: do_concat=False
        cmap = categorical_cmap(len(moct_list), 1, cmap="tab10")
    
    #___________________________________________________________________________
    ii=0
    ii_cycle=1
    for ii_ts, (tseries, tname) in enumerate(zip(moct_list, input_names)):
        
        if tseries.ndim>1: tseries = tseries.squeeze()
        auxtime = time.copy()
        if np.mod(ii_ts+1,which_cycl)==0 or do_allcycl==False:
            
            if do_concat: auxtime = auxtime + (time[-1]-time[0]+1)*(ii_cycle-1)
            hp=ax.plot(auxtime,tseries, 
                    linewidth=1.5, label=tname, color=cmap.colors[ii_ts,:], 
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
            if do_concat: auxtime = auxtime + (time[-1]-time[0]+1)*(ii_cycle-1)
            hp=ax.plot(auxtime, tseries, 
                   linewidth=1.5, label=tname, color=cmap.colors[ii_ts,:],
                   zorder=1) #marker='o', markerfacecolor='w', 
                   # path_effects=[path_effects.SimpleLineShadow(offset=(1.5,-1.5),alpha=0.3),path_effects.Normal()])
            #if do_concat:
                #hp[0].set_marker('o')
                #hp[0].set_markerfacecolor('w')
                #hp[0].set_markersize(5)
                
        ii_cycle=ii_cycle+1
        if ii_cycle>which_cycl: ii_cycle=1
    
    #___________________________________________________________________________
    # add Rapid moc data @26.5°
    if do_rapid is not None: 
        path = do_rapid
        rapid26 = xr.open_dataset(path)['moc_mar_hc10']
        rapid26_ym = rapid26.groupby('time.year').mean('time', skipna=True)
        
        time_rapid = rapid26_ym.year
        if do_allcycl: 
            time_rapid = time_rapid + (which_cycl-1)*(time[-1]-time[0]+1)
        hpr=plt.plot(time_rapid,rapid26_ym.data,
                linewidth=2, label='Rapid @ 26.5°N', color='k', marker='o', markerfacecolor='w', 
                markersize=5, zorder=2)
        
        if do_pltmean: 
            # plot mean value with trinagle 
            plt.plot(time[0]-(time[-1]-time[0])*0.0120, rapid26_ym.data.mean(),
                     marker='<', markersize=8, markeredgecolor='k', markeredgewidth=0.5,
                     color=hpr[0].get_color(), clip_box=False,clip_on=False, zorder=3)
        if do_pltstd:
            # plot std. range
            plt.plot(time[0]-(time[-1]-time[0])*0.015, rapid26_ym.data.mean()+rapid26_ym.data.std(),
                    marker='^', markersize=6, markeredgecolor='k', markeredgewidth=0.5,
                    color=hpr[0].get_color(),clip_box=False,clip_on=False, zorder=3)
                
            plt.plot(time[0]-(time[-1]-time[0])*0.015, rapid26_ym.data.mean()-rapid26_ym.data.std(),
                    marker='v', markersize=6, markeredgecolor='k', markeredgewidth=0.5,
                    color=hpr[0].get_color(),clip_box=False,clip_on=False, zorder=3)    
        del(rapid26)
    
    #___________________________________________________________________________
    if which_lat[ii]=='max':
        str_label='max {:s}: 30°N<=lat<=45°N'.format(which_moc.upper(),which_lat[ii])
    else:
        str_label='{:s} at: {:2.1f}°N'.format(which_moc.upper(),which_lat[ii])
    
    #___________________________________________________________________________
    ax.legend(shadow=True, fancybox=True, frameon=True, #mode='None', 
              bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
              #bbox_to_anchor=(1.04, 1.0), ncol=1) #loc='lower right', 
    ax.set_xlabel('Time [years]',fontsize=12)
    ax.set_ylabel('{:s} in [Sv]'.format(which_moc.upper()),fontsize=12)
    ax.set_title(str_label, fontsize=12, fontweight='bold')
    
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
        plt.xlim(time[0]-(time[-1]-time[0])*0.015,time[-1]+(time[-1]-time[0]+1)*(which_cycl-1)+(time[-1]-time[0])*0.015)    
    #___________________________________________________________________________
    plt.show()
    fig.canvas.draw()
    
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, fig, dpi=save_dpi)
    
    #___________________________________________________________________________
    return(fig,ax)





