import os
import numpy             as np
import time              as time
import xarray            as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .sub_colormap import *
from .sub_mesh     import vec_r2g
from .sub_plot     import *

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
        dum, tv1 = vec_r2g(mesh.abg, e_x[edge_tri[0, ind]], e_y[edge_tri[0, ind]], tu1, tv1)
        dx1, dum = vec_r2g(mesh.abg, e_x[edge_tri[0, ind]], e_y[edge_tri[0, ind]], dx1, dy1)
        dum, tv2 = vec_r2g(mesh.abg, e_x[edge_tri[1, ind]], e_y[edge_tri[1, ind]], tu2, tv2)
        dx2, dum = vec_r2g(mesh.abg, e_x[edge_tri[1, ind]], e_y[edge_tri[1, ind]], dx2, dy2)
        tv1, tv2 = tv1.T, tv2.T
        del(dum, dy1, dy2, tu1, tu2)
        
        #_______________________________________________________________________
        # integrate along latitude bin--> int(t*u)dx 
        tv_dx    = np.nansum(tv1*np.abs(dx1) + tv2*np.abs(dx2), axis=1)
        
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
    data_vars['hbstreamf'] = (list_dimname, np.zeros(list_dimsize), aux_attr) 
    # define coordinates
    coords    = {'nlon' : (['nlon' ], lon ), 'nlat' : (['nlat' ], lat ), }
    
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
    if do_allcycl: 
        if which_cycl is not None:
            cmap = categorical_cmap(np.int32(len(mhflx_list)/which_cycl), which_cycl, cmap="tab10")
        else:
            cmap = categorical_cmap(len(mhflx_list), 1, cmap="tab10")
    else:
        cmap = categorical_cmap(len(mhflx_list), 1, cmap="tab10")
    
    #___________________________________________________________________________
    for ii_ts, (data, data_name) in enumerate(zip(mhflx_list, input_names)):
        
        vname = list(data.keys())[0]
        #_______________________________________________________________________
        data_x, data_y = data['nlat'].values, data[vname].values
        hp=ax.plot(data_x, data_y, 
                linewidth=1, label=data_name, color=cmap.colors[ii_ts,:], 
                marker='None', markerfacecolor='w', markersize=5, 
                zorder=2)
                 
    #___________________________________________________________________________
    ax.legend(shadow=True, fancybox=True, frameon=True, bbox_to_anchor=(1.02,0.5), loc="center left", borderaxespad=0)
    ax.set_xlabel('Latitude [deg]',fontsize=12)
    if   vname == 'gmhflx': y_label = 'Global Meridional Heat Transport'
    elif vname == 'mhflx' : y_label = 'Meridional Heat Transport'
    
    if 'units' in mhflx_list[0][vname].attrs.keys():
        y_label = y_label + ' [' + mhflx_list[0][vname].attrs['units'] +']'
    if 'str_ltim' in mhflx_list[0][vname].attrs.keys():
        y_label = y_label+'\n'+mhflx_list[0][vname].attrs['str_ltim']
    ax.set_ylabel(y_label, fontsize=12)  
        
    #___________________________________________________________________________
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=20))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    plt.grid(which='major')
    plt.xlim(data_x[0]-(data_x[-1]-data_x[0])*0.015,data_x[-1]+(data_x[-1]-data_x[0])*0.015)    
        
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
    ii=0
    ii_cycle=1
    for ii_ts, (tseries, tname) in enumerate(zip(tseries_list, input_names)):
        
        if tseries.ndim>1: tseries = tseries.squeeze()
        auxtime = time.copy()
        if np.mod(ii_ts+1,which_cycl)==0 or do_allcycl==False:
            
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
        if ii_cycle>which_cycl: ii_cycle=1
        
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
        plt.xlim(time[0]-(time[-1]-time[0])*0.015,time[-1]+(time[-1]-time[0]+1)*(which_cycl-1)+(time[-1]-time[0])*0.015)    
    
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
        data_x    = data[ii]['nlon']
        data_y    = data[ii]['nlat']
        
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
            if   'short_name' in data[ii][0][vname].attrs:
                cbar_label = cbar_label+data[ii][0][ vname ].attrs['short_name']
            elif 'long_name' in data[ii][0][vname].attrs:
                cbar_label = cbar_label+data[ii][0][ vname ].attrs['long_name']
            if cbar_unit  is None : cbar_label = cbar_label+' ['+data[ii][0][ vname ].attrs['units']+']'
            else                  : cbar_label = cbar_label+' ['+cbar_unit+']'
            if 'str_ltim' in data[ii][0][vname].attrs.keys():
                cbar_label = cbar_label+'\n'+data[ii][0][vname].attrs['str_ltim']
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












