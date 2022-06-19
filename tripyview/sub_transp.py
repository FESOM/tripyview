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
    if   vname == 'gmhflx': y_label = 'Global Meridional Heat Transport [TW]'
    elif vname == 'mhflx' : y_label = 'Meridional Heat Transport [TW]'
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
