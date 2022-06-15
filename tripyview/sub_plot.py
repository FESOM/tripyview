import os
import sys
import numpy as np
import time as time
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
from cartopy.mpl.gridliner import Gridliner

from mpl_toolkits.axes_grid1 import make_axes_locatable

#import matplotlib.pylab as plt
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation,TriAnalyzer
import matplotlib.ticker as mticker
import matplotlib.path   as mpath
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

from .sub_mesh     import *
from .sub_data     import *
from .colormap_c2c import *
   

# ___PLOT HORIZONTAL FESOM2 DATA SLICES________________________________________
#|                                                                             |
#|      *** PLOT HORIZONTAL FESOM2 DATA SLICES --> BASED ON CARTOPY ***        |
#|                                                                             |
#|_____________________________________________________________________________|
def plot_hslice(mesh, data, cinfo=None, box=None, proj='pc', figsize=[9,4.5], 
                n_rc=[1,1], do_grid=False, do_plot='tcf', do_rescale=True,
                cbar_nl=8, cbar_orient='vertical', cbar_label=None, cbar_unit=None,
                do_lsmask='fesom', do_bottom=True, color_lsmask=[0.6, 0.6, 0.6], 
                color_bot=[0.75,0.75,0.75],  title=None,
                pos_fac=1.0, pos_gap=[0.02, 0.02], pos_extend=None, do_save=None, save_dpi=600,
                linecolor='k', linewidth=0.5, ):
    """
    ---> plot FESOM2 horizontal data slice:
    ___INPUT:___________________________________________________________________
    mesh        :   fesom2 mesh object,  with all mesh information 
    data        :   xarray dataset object, or list of xarray dataset object
    cinfo       :   None, dict() (default: None), dictionary with colorbar 
                    formation. Information that are given are used others are 
                    computed. cinfo dictionary entries can me: 
                    > cinfo['cmin'], cinfo['cmax'], cinfo['cref'] ... scalar min, 
                    max, reference value
                    > cinfo['crange'] ...  list with [cmin, cmax, cref] overrides 
                    scalar values 
                    > cinfo['cnum'] ... minimum number of colors
                    > cinfo['cstr'] ... name of colormap see in sub_colormap_c2c.py
                    > cinfo['cmap'] ... colormap object ('wbgyr', 'blue2red, 'jet' ...)
                    > cinfo['clevel'] ... color level array
    box         :   None, list (default: None) regional limitation of plot [lonmin,
                    lonmax, latmin, latmax]
    proj        :   str, (default: 'pc') which projection should be used, 'pc'=
                    ccrs.PlateCarree(), 'merc'=ccrs.Mercator(), 'nps'= 
                    ccrs.NorthPolarStereo(), 'sps'=ccrs.SouthPolarStereo(), 
                    'rob'=ccrs.Robinson()
    fig_size    :   list (default:[9,4.5] ), list with figure width and figure
                    height [w, h]
    n_rc        :   list, (default: [1,1]) if plotting multiple data panels, give
                    number of rows and number of columns in to plot, just works when 
                    also data are a list of xarray dataset object
    do_grid     :   bool, (default:False) overlay the fesom2 surface grid over 
                    data
    do_plot     :   str, (default: 'tcf'), make plotts either as tricontourf plot
                    ('tcf', much faster than pcolor) or tripcolor plot ('tpc')
    do_rescale  :   None, bool, str (default:True) rescale data and writes 
                    rescaling string into colorbar labels
                    If: None    ... no rescaling is applied
                        True    ... rescale to multiple of 10 or 1/10
                        'log10' ... rescale towards log10
    cbar_nl     :   int, (default:8) minumum number of colorbar label to show 
    cbar_orient :   str, (default:'vertical') should colorbar be either 'vertical'
                    or 'horizontal' oriented
    cbar_label  :   None, str, (default: None) if: None the cbar label is taken from 
                    the description attribute of the data, if: str cbar_label
                    is overwritten by string 
    cbar_unit   :   None, str, (default: None) if: None the cbar unit string is 
                    taken from the units attribute of the data, if: str cbar_unir
                    is overwritten by string   
    do_lsmask   :   None, str (default: 'fesom') plot land-sea mask.  
                    If:  None   ... no land sea mask is used, 
                        'fesom' ... overlay fesom shapefile land-sea mask using
                                    color color_lsmask
                        'stock' ... use cartopy stock image as land sea mask
                        'bluemarble' ... use bluemarble image as land sea mask
                        'etopo' ... use etopo image as land sea mask
    do_bottom   :   bool, (default:True) highlight nan bottom topography 
                    with gray color defined by color_bot
    color_lsmask:   list, (default: [0.6, 0.6, 0.6]) RGB facecolor value for fesom
                    shapefile land-sea mask patch
    color_bot   :   list, (default: [0.8, 0.8, 0.8]) RGB facecolor value for  
                    nan bottom topography
    title       :   None, str,(default:None) give every plot panel a title string
                    IF: None       ... no title is plotted
                        'descript' ... use data 'descript' attribute for title string
                        'string'   ... use given string as title 
    pos_fac     :   float, (default:1.0) multiplication factor  to increase/decrease
                    width and height of plotted  panel
    pos_gap     :   list, (default: [0.02, 0.02]) gives width and height of gaps 
                    between multiple panels 
    do_save     :   None, str (default:None) if None figure will by not saved, if 
                    string figure will be saved, strings must give directory and 
                    filename  where to save.
    linecolor   :   str, list, (default:'k') either color string or RGB list set
                    color of coastline 
    linewidth   :   float, (default:0.2) sets linewidth of coastline
    ___RETURNS:_________________________________________________________________
    fig        :    returns figure handle 
    ax         :    returns list with axes handle 
    cbar       :    returns colorbar handle
    ____________________________________________________________________________
    """
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
    if box is None or box=="None": box = [ -180+mesh.focus, 180+mesh.focus, -90, 90 ]
    
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
    # create lon, lat ticks 
    xticks,yticks = do_ticksteps(mesh, box)
    
    #___________________________________________________________________________    
    # create figure and axes
    fig, ax = plt.subplots( n_rc[0],n_rc[1],
                                figsize=figsize, 
                                subplot_kw =dict(projection=which_proj),
                                gridspec_kw=dict(left=0.06, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05,),
                                constrained_layout=False)
    
    #___________________________________________________________________________    
    # flatt axes if there are more than 1
    if isinstance(ax, np.ndarray): ax = ax.flatten()
    else:                          ax = [ax] 
    nax = len(ax)
    
    #___________________________________________________________________________
    # create mesh triangulation
    tri = Triangulation(np.hstack((mesh.n_x,mesh.n_xa)),
                        np.hstack((mesh.n_y,mesh.n_ya)),
                        np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia)))
    
    # Limit points to projection box
    if proj=='nps' or proj=='sps' or 'pc':
        e_idxbox = grid_cutbox_e(tri.x, tri.y, tri.triangles, box, which='soft')
    else:    
        points = which_transf.transform_points(which_proj, 
                                                tri.x[tri.triangles].sum(axis=1)/3, 
                                                tri.y[tri.triangles].sum(axis=1)/3)
        
        xpts, ypts = points[:,0].flatten().tolist(), points[:,1].flatten().tolist()
        
        crs_pts = list(zip(xpts,ypts))
        fig_pts = ax[0].transData.transform(crs_pts)
        ax_pts  = ax[0].transAxes.inverted().transform(fig_pts)
        x, y =  ax_pts[:,0], ax_pts[:,1]
        e_idxbox = (x>=-0.05) & (x<=1.05) & (y>=-0.05) & (y<=1.05)
    tri.triangles = tri.triangles[e_idxbox,:]    
    
    #___________________________________________________________________________
    # data must be list filled with xarray data
    if not isinstance(data, list): data = [data]
    ndata = len(data)
    
    #___________________________________________________________________________
    # set up color info
    #vname = list(data[ii].keys())
    #if data[ii][ vname[0] ].size==mesh.n2dn:
    cinfo = do_setupcinfo(cinfo, data, do_rescale, mesh=mesh, tri=tri, do_cweights=mesh.n_area)
    #else:
        #cinfo = do_setupcinfo(cinfo, data, do_rescale, mesh=mesh, tri=tri, do_cweights=mesh.e_area)
        
    #_______________________________________________________________________
    # setup normalization log10, symetric log10, None
    which_norm = do_compute_scalingnorm(cinfo, do_rescale)
        
    #___________________________________________________________________________
    # loop over axes
    hpall=list()
    #for ii in range(0,nax):
    for ii in range(0,ndata):
        #_______________________________________________________________________
        # if there are more axes allocated than data evailable 
        #if ii>=ndata: continue
        
        #_______________________________________________________________________
        # set axes extent
        ax[ii].set_extent(box, crs=ccrs.PlateCarree())
        
        #_______________________________________________________________________
        # periodic augment data
        vname = list(data[ii].keys())
        data_plot = data[ii][ vname[0] ].data.copy()
        
        if   data_plot.size==mesh.n2dn:
            data_plot = np.hstack((data_plot,data_plot[mesh.n_pbnd_a]))
        elif data_plot.size==mesh.n2de:
            data_plot = np.hstack((data_plot[mesh.e_pbnd_0],data_plot[mesh.e_pbnd_a]))

        #_______________________________________________________________________
        # kick out triangles with Nan cut elements to box size        
        isnan   = np.isnan(data_plot)
        if data_plot.size == mesh.n2dea:
            e_idxok = isnan==False
        else:
            e_idxok = np.any(isnan[tri.triangles], axis=1)==False
        
        #_______________________________________________________________________
        # add color for ocean bottom
        if do_bottom and np.any(e_idxok==False):
            hbot = ax[ii].triplot(tri.x, tri.y, tri.triangles[e_idxok==False,:], color=color_bot)
        
        #_______________________________________________________________________
        # plot tri contourf/tripcolor
        if   do_plot=='tpc' or data_plot.size == mesh.n2dea:
            # plot over elements
            if data_plot.size == mesh.n2dea:
                hp=ax[ii].tripcolor(tri.x, tri.y, tri.triangles[e_idxok,:], data_plot[e_idxok],
                #hp=ax[ii].tripcolor(tri.x, tri.y, tri.triangles[:,:], data_plot[:],
                                    #transform=which_transf,
                                    shading='flat',
                                    cmap=cinfo['cmap'],
                                    vmin=cinfo['clevel'][0], vmax=cinfo['clevel'][ -1],
                                    norm = which_norm)
            # plot over vertices    
            else:
                hp=ax[ii].tripcolor(tri.x, tri.y, tri.triangles[e_idxok,:], data_plot,
                                    shading='flat',
                                    cmap=cinfo['cmap'],
                                    vmin=cinfo['clevel'][0], vmax=cinfo['clevel'][ -1],
                                    norm = which_norm)
            
        elif do_plot=='tcf': 
            # supress warning message when compared with nan
            with np.errstate(invalid='ignore'):
                data_plot[data_plot<cinfo['clevel'][ 0]] = cinfo['clevel'][ 0]
                data_plot[data_plot>cinfo['clevel'][-1]] = cinfo['clevel'][-1]
            
            hp=ax[ii].tricontourf(tri.x, tri.y, tri.triangles[e_idxok,:], data_plot, 
                                transform=which_transf,
                                norm=which_norm,
                                levels=cinfo['clevel'], cmap=cinfo['cmap'], extend='both')
            
        hpall.append(hp)        
        #_______________________________________________________________________
        # add grid mesh on top
        if do_grid: ax[ii].triplot(tri.x, tri.y, tri.triangles[:,:], #tri.triangles[e_idxok,:], 
                                   color='k', linewidth=0.2, alpha=0.75) 
                                   #transform=which_transf)
        
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
                if title=='descript' and ('descript' in data[ii][vname[0]].attrs.keys() ):
                    ax[ii].set_title(data[ii][ vname[0] ].attrs['descript'], fontsize=fontsize+2)
                    
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
    cbar = plt.colorbar(hp, orientation=cbar_orient, ax=ax, ticks=cinfo['clevel'], 
                      extendrect=False, extendfrac=None,
                      drawedges=True, pad=0.025, shrink=1.0,)                      
    
    # do formatting of colorbar 
    cbar = do_cbar_formatting(cbar, do_rescale, cbar_nl, fontsize, cinfo['clevel'])
    
    # do labeling of colorbar
    if cbar_label is None: cbar_label = data[nax_fin-1][ vname[0] ].attrs['long_name']
    #if cbar_unit  is None: cbar_label = cbar_label+' ['+data[nax_fin-1][ vname[0] ].attrs['units']+']'
    if cbar_unit  is None: cbar_label = cbar_label+' ['+data[0][ vname[0] ].attrs['units']+']'
    else:                  cbar_label = cbar_label+' ['+cbar_unit+']'
    if 'str_ltim' in data[0][vname[0]].attrs.keys():
        cbar_label = cbar_label+'\n'+data[0][vname[0]].attrs['str_ltim']
    if 'str_ldep' in data[0][vname[0]].attrs.keys():
        cbar_label = cbar_label+data[0][vname[0]].attrs['str_ldep']
    cbar.set_label(cbar_label, size=fontsize+2)
    
    #___________________________________________________________________________
    # repositioning of axes and colorbar
    ax, cbar = do_reposition_ax_cbar(ax, cbar, rowlist, collist, pos_fac, 
                                     pos_gap, title=title, proj=proj, extend=pos_extend)
    fig.canvas.draw()
    
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, fig, dpi=save_dpi, transparent=True)
    plt.show(block=False)
    
    #___________________________________________________________________________
    return(fig, ax, cbar)
    


# ___PLOT HORIZONTAL FESOM2 DATA SLICES________________________________________
#|                                                                             |
#|      *** PLOT HORIZONTAL FESOM2 DATA SLICES --> BASED ON CARTOPY ***        |
#|                                                                             |
#|_____________________________________________________________________________|
def plot_hvec(mesh, data, cinfo=None, box=None, proj='pc', figsize=[9,4.5], 
                n_rc=[1,1], do_grid=False, do_plot='tcf', do_rescale=False,
                cbar_nl=8, cbar_orient='vertical', cbar_label=None, cbar_unit=None,
                do_lsmask='fesom', do_bottom=True, color_lsmask=[0.6, 0.6, 0.6], 
                color_bot=[0.8,0.8,0.8],  title=None,
                pos_fac=1.0, pos_gap=[0.02, 0.02], do_save=None, save_dpi=600, linecolor='k', 
                linewidth=0.5, hsize = 20, do_normalize=True, do_topo=False, do_density=None,
                pos_extend=None,):
    """
    ---> plot FESOM2 horizontal data slice:
    ___INPUT:___________________________________________________________________
    mesh        :   fesom2 mesh object,  with all mesh information 
    data        :   xarray dataset object, or list of xarray dataset object
    cinfo       :   None, dict() (default: None), dictionary with colorbar 
                    formation. Information that are given are used others are 
                    computed. cinfo dictionary entries can me: 
                    > cinfo['cmin'], cinfo['cmax'], cinfo['cref'] ... scalar min, 
                    max, reference value
                    > cinfo['crange'] ...  list with [cmin, cmax, cref] overrides 
                    scalar values 
                    > cinfo['cnum'] ... minimum number of colors
                    > cinfo['cstr'] ... name of colormap see in sub_colormap_c2c.py
                    > cinfo['cmap'] ... colormap object ('wbgyr', 'blue2red, 'jet' ...)
                    > cinfo['clevel'] ... color level array
    box         :   None, list (default: None) regional limitation of plot [lonmin,
                    lonmax, latmin, latmax]
    proj        :   str, (default: 'pc') which projection should be used, 'pc'=
                    ccrs.PlateCarree(), 'merc'=ccrs.Mercator(), 'nps'= 
                    ccrs.NorthPolarStereo(), 'sps'=ccrs.SouthPolarStereo(), 
                    'rob'=ccrs.Robinson()
    fig_size    :   list (default:[9,4.5] ), list with figure width and figure
                    height [w, h]
    n_rc        :   list, (default: [1,1]) if plotting multiple data panels, give
                    number of rows and number of columns in to plot, just works when 
                    also data are a list of xarray dataset object
    do_grid     :   bool, (default:False) overlay the fesom2 surface grid over 
                    data
    do_plot     :   str, (default: 'tcf'), make plotts either as tricontourf plot
                    ('tcf', much faster than pcolor) or tripcolor plot ('tpc')
    do_rescale  :   None, bool, str (default:True) rescale data and writes 
                    rescaling string into colorbar labels
                    If: None    ... no rescaling is applied
                        True    ... rescale to multiple of 10 or 1/10
                        'log10' ... rescale towards log10
    cbar_nl     :   int, (default:8) minumum number of colorbar label to show 
    cbar_orient :   str, (default:'vertical') should colorbar be either 'vertical'
                    or 'horizontal' oriented
    cbar_label  :   None, str, (default: None) if: None the cbar label is taken from 
                    the description attribute of the data, if: str cbar_label
                    is overwritten by string 
    cbar_unit   :   None, str, (default: None) if: None the cbar unit string is 
                    taken from the units attribute of the data, if: str cbar_unir
                    is overwritten by string   
    do_lsmask   :   None, str (default: 'fesom') plot land-sea mask.  
                    If:  None   ... no land sea mask is used, 
                        'fesom' ... overlay fesom shapefile land-sea mask using
                                    color color_lsmask
                        'stock' ... use cartopy stock image as land sea mask
                        'bluemarble' ... use bluemarble image as land sea mask
                        'etopo' ... use etopo image as land sea mask
    do_bottom   :   bool, (default:True) highlight nan bottom topography 
                    with gray color defined by color_bot
    color_lsmask:   list, (default: [0.6, 0.6, 0.6]) RGB facecolor value for fesom
                    shapefile land-sea mask patch
    color_bot   :   list, (default: [0.8, 0.8, 0.8]) RGB facecolor value for  
                    nan bottom topography
    title       :   None, str,(default:None) give every plot panel a title string
                    IF: None       ... no title is plotted
                        'descript' ... use data 'descript' attribute for title string
                        'string'   ... use given string as title 
    pos_fac     :   float, (default:1.0) multiplication factor  to increase/decrease
                    width and height of plotted  panel
    pos_gap     :   list, (default: [0.02, 0.02]) gives width and height of gaps 
                    between multiple panels 
    do_save     :   None, str (default:None) if None figure will by not saved, if 
                    string figure will be saved, strings must give directory and 
                    filename  where to save.
    linecolor   :   str, list, (default:'k') either color string or RGB list set
                    color of coastline 
    linewidth   :   float, (default:0.2) sets linewidth of coastline
    ___RETURNS:_________________________________________________________________
    fig        :    returns figure handle 
    ax         :    returns list with axes handle 
    cbar       :    returns colorbar handle
    ____________________________________________________________________________
    """
    fontsize = 12
    str_rescale = None
    
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
    if box is None: box = [ -180+mesh.focus, 180+mesh.focus, -90, 90 ]
    
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
    elif proj=='sps':        
        which_proj=ccrs.SouthPolarStereo()    
        which_transf = ccrs.PlateCarree()
    elif proj=='rob':        
        which_proj=ccrs.Robinson()    
        which_transf = ccrs.PlateCarree()    
    
    #___________________________________________________________________________    
    # create lon, lat ticks 
    xticks,yticks = do_ticksteps(mesh, box)
    
    #___________________________________________________________________________    
    # create figure and axes
    fig, ax = plt.subplots( n_rc[0],n_rc[1],
                                figsize=figsize, 
                                subplot_kw =dict(projection=which_proj),
                                gridspec_kw=dict(left=0.06, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05,),
                                constrained_layout=False,)
                                #sharex='all', sharey='all' )
    
    #___________________________________________________________________________    
    # flatt axes if there are more than 1
    if isinstance(ax, np.ndarray): ax = ax.flatten()
    else:                          ax = [ax] 
    nax = len(ax)
    
    #___________________________________________________________________________
    # create mesh triangulation
    tri = Triangulation(np.hstack((mesh.n_x,mesh.n_xa)),
                        np.hstack((mesh.n_y,mesh.n_ya)),
                        np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia)))
    
    # Limit points to projection box
    if proj=='nps' or proj=='sps' or 'pc':
        e_idxbox = grid_cutbox_e(tri.x, tri.y, tri.triangles, box, which='soft')
    else:    
        points = which_transf.transform_points(which_proj, 
                                                tri.x[tri.triangles].sum(axis=1)/3, 
                                                tri.y[tri.triangles].sum(axis=1)/3)
        
        xpts, ypts = points[:,0].flatten().tolist(), points[:,1].flatten().tolist()
        
        crs_pts = list(zip(xpts,ypts))
        fig_pts = ax[0].transData.transform(crs_pts)
        ax_pts  = ax[0].transAxes.inverted().transform(fig_pts)
        x, y =  ax_pts[:,0], ax_pts[:,1]
        e_idxbox = (x>=-0.05) & (x<=1.05) & (y>=-0.05) & (y<=1.05)
    tri.triangles = tri.triangles[e_idxbox,:]    
    
    #___________________________________________________________________________
    # data must be list filled with xarray data
    if not isinstance(data, list): data = [data]
    ndata = len(data)
        
    #___________________________________________________________________________
    # set up color info 
    cinfo = do_setupcinfo(cinfo, data, tri, mesh, do_rescale, do_vec=True)
    
    #_______________________________________________________________________    
    # setup normalization log10, symetric log10, None
    which_norm = do_compute_scalingnorm(cinfo, do_rescale)   
        
    #___________________________________________________________________________
    # loop over axes
    for ii in range(0,ndata):
        
        #_______________________________________________________________________
        # add color for bottom bottom
        if do_bottom : 
            ax[ii].background_patch.set_facecolor(color_bot)
        
        #_______________________________________________________________________
        # set axes extent
        ax[ii].set_extent(box, crs=ccrs.PlateCarree())
        
        #_______________________________________________________________________
        # periodic augment data
        vname = list(data[ii].keys())
        data_plot_u = data[ii][ vname[0] ].data
        data_plot_v = data[ii][ vname[1] ].data
        data_plot_n = np.sqrt(data_plot_u**2 + data_plot_v**2)
        
        data_plot_u = np.hstack((data_plot_u,data_plot_u[mesh.n_pbnd_a]))
        data_plot_v = np.hstack((data_plot_v,data_plot_v[mesh.n_pbnd_a]))
        data_plot_n = np.hstack((data_plot_n,data_plot_n[mesh.n_pbnd_a]))
        
        data_plot_u, data_plot_v = data_plot_u/data_plot_n, data_plot_v/data_plot_n
        data_plot_n[data_plot_n<cinfo['clevel'][0]]  = cinfo['clevel'][0] #+np.finfo(np.float32).eps
        data_plot_n[data_plot_n>cinfo['clevel'][-1]] = cinfo['clevel'][-1]#-np.finfo(np.float32).eps
        data_plot_u, data_plot_v = data_plot_u*data_plot_n, data_plot_v*data_plot_n
        
        
        if do_normalize:
            data_plot_u = data_plot_u/data_plot_n
            data_plot_v = data_plot_v/data_plot_n
            
        #_______________________________________________________________________
        # kick out triangles with Nan cut elements to box size        
        isok   = np.isnan(data_plot_n)==False
        #e_idxok = np.any(isok[tri.triangles], axis=1)==True
        
        if do_density is not None:
            r0      = 1/(np.sqrt(mesh.n_area))
            isp     = np.random.rand(mesh.n2dn)>r0/np.max(r0)*do_density #1.5
            isok    = np.logical_and(isok,np.hstack((isp,isp[mesh.n_pbnd_a])) )
        
        #_______________________________________________________________________
        # plot tri contourf/tripcolor
        hfac=3
        hp=ax[ii].quiver(tri.x[     isok], tri.y[      isok], 
                        data_plot_u[isok], data_plot_v[isok],
                        data_plot_n[isok],
                        transform=which_transf,
                        cmap = cinfo['cmap'], 
                        edgecolor='k', linewidth=0.15,
                        units='xy', angles='xy', scale_units='xy', scale=1/hsize,
                        width = 0.10, #0.1,
                        headlength=hfac*max([hsize,1.0]),#hsize, 
                        headaxislength=hfac*max([hsize,1.0]), #hsize, 
                        headwidth=hfac*max([hsize,1.0])*0.8,# hsize*2/3,
                        zorder=10,
                        norm = which_norm)
        hp.set_clim([cinfo['clevel'][0],cinfo['clevel'][-1]])
        
        
        #_______________________________________________________________________
        # add grid mesh on top
        if do_topo: 
            fname = data[ii][vname[0]].attrs['runid']+'.mesh.diag.nc'
            dname = data[ii][vname[0]].attrs['datapath']
            diagpath = os.path.join(dname,fname)
            n_iz   = xr.open_mfdataset(diagpath, parallel=True)['nlevels_nod2D']-1
            data_plot = np.abs(mesh.zlev[n_iz])
            data_plot = np.hstack((data_plot,data_plot[mesh.n_pbnd_a]))
            
            levels = np.hstack((25, 50, 100, 150, 200, 250, np.arange(500,6000+1,500)))
            N = len(levels)
            vals = np.ones((N, 4))
            vals[:, 0] = np.linspace(0.2, 0.95, N)
            vals[:, 1] = np.linspace(0.2, 0.95, N)
            vals[:, 2] = np.linspace(0.2, 0.95, N)
            vals = np.flipud(vals)
            newcmp = ListedColormap(vals)
            ax[ii].tricontourf(tri.x, tri.y, tri.triangles, data_plot, 
                                levels=levels, cmap=newcmp, extend='both')
            ax[ii].tricontour(tri.x, tri.y, tri.triangles, data_plot, levels=levels, 
                              colors='k', linewidths=0.25, alpha=0.5)
        #_______________________________________________________________________
        # add grid mesh on top
        if do_grid: ax[ii].triplot(tri.x, tri.y, tri.triangles[e_idxok,:], #tri.triangles[:,:], #
                                   color='k', linewidth=0.2, alpha=0.75, zorder=1) 
                                   #transform=which_transf)
        
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
                if title=='descript':
                    ax[ii].set_title(data[ii][ vname[0] ].attrs['descript'], fontsize=fontsize+2)
                    
                else:
                    ax[ii].set_title(title, fontsize=fontsize+2)
            # is title list of string        
            elif isinstance(title,list): ax[ii].set_title(title[ii], fontsize=fontsize+2)
    nax_fin = ii+1        
    
    #___________________________________________________________________________
    # delete axes that are not needed
    #for jj in range(nax_fin, nax): fig.delaxes(ax[jj])
    for jj in range(ndata, nax): fig.delaxes(ax[jj])
    
    #___________________________________________________________________________
    # delete axes that are not needed
    cbar = fig.colorbar(hp, orientation=cbar_orient, ax=ax, ticks=cinfo['clevel'], 
                      extend='neither',extendrect=False, extendfrac=None,
                      drawedges=True, pad=0.025, shrink=1.0)
    
    # do formatting of colorbar 
    cbar = do_cbar_formatting(cbar, do_rescale, cbar_nl, fontsize, cinfo['clevel'])
    
    # do labeling of colorbar    
    if cbar_label is None: cbar_label = data[nax_fin-1][ vname[0] ].attrs['long_name']
    if cbar_unit  is None: cbar_label = cbar_label+' ['+data[nax_fin-1][ vname[0] ].attrs['units']+']'
    else:                  cbar_label = cbar_label+' ['+cbar_unit+']'
    if 'str_ltim' in data[0][vname[0]].attrs.keys():
        cbar_label = cbar_label+'\n'+data[0][vname[0]].attrs['str_ltim']
    if 'str_ldep' in data[0][vname[0]].attrs.keys():
        cbar_label = cbar_label+data[0][vname[0]].attrs['str_ldep']
    cbar.set_label(cbar_label, size=fontsize+2)
    
    #___________________________________________________________________________
    # repositioning of axes and colorbar
    ax, cbar = do_reposition_ax_cbar(ax, cbar, rowlist, collist, pos_fac, 
                                     pos_gap, title=title, proj=proj, extend=pos_extend)
    
    plt.show(block=False)
    fig.canvas.draw()
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, fig, dpi=save_dpi)
    
    #___________________________________________________________________________
    return(fig, ax, cbar)



def plot_hmesh(mesh, box=None, proj='pc', figsize=[9,4.5], 
                title=None, do_save=None, do_lsmask='fesom', color_lsmask=[0.6, 0.6, 0.6],
                linecolor='k', linewidth=0.2, linealpha=0.75, pos_extend=None,):
    """
    ---> plot FESOM2 horizontal mesh:
    ___INPUT:___________________________________________________________________
    mesh        :   fesom2 mesh object,  with all mesh information 
    box         :   None, list (default: None) regional limitation of plot [lonmin,
                    lonmax, latmin, latmax]
    proj        :   str, (default: 'pc') which projection should be used, 'pc'=
                    ccrs.PlateCarree(), 'merc'=ccrs.Mercator(), 'nps'= 
                    ccrs.NorthPolarStereo(), 'sps'=ccrs.SouthPolarStereo(), 
                    'rob'=ccrs.Robinson()
    fig_size    :   list (default:[9,4.5] ), list with figure width and figure
                    height [w, h]
    title       :   None, str,(default:None) give every plot panel a title string
                    IF: None       ... no title is plotted
                        'descript' ... use data 'descript' attribute for title string
                        'string'   ... use given string as title 
    do_save     :   None, str (default:None) if None figure will by not saved, if 
                    string figure will be saved, strings must give directory and 
                    filename  where to save.
    do_lsmask   :   None, str (default: 'fesom') plot land-sea mask.  
                    If:  None   ... no land sea mask is used, 
                        'fesom' ... overlay fesom shapefile land-sea mask using
                                    color color_lsmask
                        'stock' ... use cartopy stock image as land sea mask
                        'bluemarble' ... use bluemarble image as land sea mask
                        'etopo' ... use etopo image as land sea mask
    do_bottom   :   bool, (default:True) highlight nan bottom topography 
                    with gray color defined by color_bot
    color_lsmask:   list, (default: [0.6, 0.6, 0.6]) RGB facecolor value for fesom
                    shapefile land-sea mask patch
    linecolor   :   str, list, (default:'k') either color string or RGB list               
    linewidth   :   float, (default:0.2) linewidth of mesh
    linealpha   :   float, (default:0.75) alpha value of mesh
    ___RETURNS:_________________________________________________________________
    fig        :    returns figure handle 
    ax         :    returns list with axes handle 
    ____________________________________________________________________________
    """
    fontsize    = 12
    str_rescale = None
    n_rc        = [1,1]
    pos_fac     = 1.0
    pos_gap     = [0.02, 0.02]
    
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
    if box is None: box = [ -180+mesh.focus, 180+mesh.focus, -90, 90 ]
    
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
    elif proj=='sps':        
        which_proj=ccrs.SouthPolarStereo()    
        which_transf = ccrs.PlateCarree()
    elif proj=='rob':        
        which_proj=ccrs.Robinson()    
        which_transf = ccrs.PlateCarree()    
        
    #___________________________________________________________________________    
    # create lon, lat ticks 
    xticks,yticks = do_ticksteps(mesh, box)
    
    #___________________________________________________________________________    
    # create figure and axes
    fig, ax = plt.subplots( n_rc[0],n_rc[1],
                                figsize=figsize, 
                                subplot_kw =dict(projection=which_proj),
                                gridspec_kw=dict(left=0.06, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05,),
                                constrained_layout=False, )
    
    #___________________________________________________________________________    
    # flatt axes if there are more than 1
    if isinstance(ax, np.ndarray): ax = ax.flatten()
    else:                          ax = [ax] 
    nax = len(ax)
    
    #___________________________________________________________________________
    # create mesh triangulation
    tri = Triangulation(np.hstack((mesh.n_x,mesh.n_xa)),
                        np.hstack((mesh.n_y,mesh.n_ya)),
                        np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia)))
    
    # Limit points to projection box
    if proj=='nps' or proj=='sps' or 'pc':
        e_idxbox = grid_cutbox_e(tri.x, tri.y, tri.triangles, box, which='hard')
    else:    
        points = which_transf.transform_points(which_proj, 
                                                tri.x[tri.triangles].sum(axis=1)/3, 
                                                tri.y[tri.triangles].sum(axis=1)/3)
        
        xpts, ypts = points[:,0].flatten().tolist(), points[:,1].flatten().tolist()
        
        crs_pts = list(zip(xpts,ypts))
        fig_pts = ax[0].transData.transform(crs_pts)
        ax_pts  = ax[0].transAxes.inverted().transform(fig_pts)
        x, y =  ax_pts[:,0], ax_pts[:,1]
        e_idxbox = (x>=-0.05) & (x<=1.05) & (y>=-0.05) & (y<=1.05)
    tri.triangles = tri.triangles[e_idxbox,:]    
    
    #___________________________________________________________________________
    # loop over axes
    for ii in range(0,nax):
        
        #_______________________________________________________________________
        # set axes extent
        ax[ii].set_extent(box, crs=ccrs.PlateCarree())
        
        #_______________________________________________________________________
        # add grid mesh on top
        ax[ii].triplot(tri.x, tri.y, tri.triangles, 
                                   color=linecolor, linewidth=linewidth, 
                                   alpha=linealpha)
        
        #_______________________________________________________________________
        # add mesh land-sea mask
        ax[ii] = do_plotlsmask(ax[ii],mesh, do_lsmask, box, which_proj,
                               color_lsmask=color_lsmask, edgecolor=linecolor, 
                               linewidth=0.5)
        
        #_______________________________________________________________________
        # add gridlines
        ax[ii] = do_add_gridlines(ax[ii], rowlist[ii], collist[ii], 
                                  xticks, yticks, proj, which_proj)
       
        #_______________________________________________________________________
        # set title and axes labels
        if title is not None: 
            # is title  string:
            if   isinstance(title,str) : 
                ax[ii].set_title(title, fontsize=fontsize+2)
            # is title list of string        
            elif isinstance(title,list): 
                ax[ii].set_title(title[ii], fontsize=fontsize+2)
                
    nax_fin = ii+1        
    
    #___________________________________________________________________________
    # delete axes that are not needed
    for jj in range(nax_fin, nax): fig.delaxes(ax[jj])
    #___________________________________________________________________________
    # repositioning of axes and colorbar
    ax, cbar = do_reposition_ax_cbar(ax, None, rowlist, collist, pos_fac, 
                                     pos_gap, title=title, proj=proj, extend=pos_extend)
    
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, fig)
    
    #___________________________________________________________________________
    return(fig, ax)



# ___DO RESCALE DATA___________________________________________________________
#| rescale data towards multiple of 10  or 1/10 or usinjg log10                |
#| ___INPUT_________________________________________________________________   |
#| data         :   xarray dataset object                                      |
#| do_rescale   :   None, bool, str (default:True) rescale data and writes     |
#|                  rescaling string into colorbar labels                      |
#|                  If: None    ... no rescaling is applied                    |
#|                      True    ... rescale to multiple of 10 or 1/10          |
#|                      'log10' ... rescale towards log10                      |
#| ___RETURNS_______________________________________________________________   |
#| data         :   xarray dataset object                                      |
#| str_rescale  :   None, str if rescaling is applied returns string with      |
#|                  rescaling factor to be shown in colorbar                   |
#|_____________________________________________________________________________|     
def do_rescale_data(data,do_rescale):
    #___________________________________________________________________________
    # cutoff exponentials --> add therefore string to unit parameter
    str_rescale=None
    
    #___________________________________________________________________________
    if do_rescale==True:
        if np.nanmax(np.abs(data))<1e-2 and np.nanmax(np.abs(data))>0.0:
            scal = 10**(np.floor(np.log10(max(abs(np.nanmin(data)),abs(np.nanmax(data))))-1))
            data = data/scal
            str_rescale  = ' $ \cdot 10^{'+str(int(np.log10(scal)))+'} $'
        elif np.nanmax(np.abs(data))>1.0e4:
            scal = 10**(np.floor(np.log10(max(abs(np.nanmin(data)),abs(np.nanmax(data))))-1))
            data = data/scal
            str_rescale  = ' $ \cdot 10^{'+str(int(np.log10(scal)))+'} $'
            
    #___________________________________________________________________________
    elif do_rescale=='log10':
        #data[data!=0.0] = np.log10(data[data!=0.0])
        #data.rescale='log10'
        str_rescale  = ' log10() '
    
    #___________________________________________________________________________
    return(data,str_rescale)



def do_compute_scalingnorm(cinfo, do_rescale):
    #___________________________________________________________________________
    which_norm = None
    if   do_rescale =='log10':
            which_norm = mcolors.LogNorm(vmin=cinfo['clevel'][0], vmax=cinfo['clevel'][-1])
    elif do_rescale =='slog10':    
            which_norm = mcolors.SymLogNorm(np.min(np.abs(cinfo['clevel'][cinfo['clevel']!=0])),
                                            linscale=1.0, 
                                            vmin=cinfo['clevel'][0], vmax=cinfo['clevel'][-1], 
                                            clip=True)
    #___________________________________________________________________________
    return(which_norm)

    

# ___DO COLORMAP INFO__________________________________________________________
#| build up colormap dictionary                                                |
#| ___INPUT_________________________________________________________________   |
#| cinfo        :   None, dict() (default: None), dictionary with colorbar     |
#|                  formation. Information that are given are used others are  |
#|                  computed. cinfo dictionary entries can me:                 |
#|                  > cinfo['cmin'], cinfo['cmax'], cinfo['cref'] ... scalar   |
#|                    min, max, reference value                                |
#|                  > cinfo['crange'] ...  list with [cmin, cmax, cref]        |
#|                    overrides scalar values                                  |
#|                  > cinfo['cnum'] ... minimum number of colors               |
#|                  > cinfo['cstr'] ... name of colormap see in                |
#|                    sub_colormap_c2c.py                                      |
#|                  > cinfo['cmap'] ... colormap object ('wbgyr', 'blue2red,   |
#|                    'jet' ...)                                               |
#|                  > cinfo['clevel'] ... color level array                    |
#| data         :   xarray dataset object                                      |
#| tri          :   mesh triangulation object                                  |
#| mesh         :   fesom2 mesh object,                                        |
#| do_rescale   :   None, bool, str (default:True) rescale data and writes     |
#|                  rescaling string into colorbar labels                      |
#|                  If: None    ... no rescaling is applied                    |
#|                      True    ... rescale to multiple of 10 or 1/10          |
#|                      'log10' ... rescale towards log10                      |
#| ___RETURNS_______________________________________________________________   |
#| cinfo        :   color info dictionary                                      |
#|_____________________________________________________________________________|     
def do_setupcinfo(cinfo, data, do_rescale, mesh=None, tri=None, do_vec=False, 
                  do_index=False, do_moc=False, do_dmoc=None, do_cweights=None):
    #___________________________________________________________________________
    # set up color info 
    if cinfo is None: cinfo=dict()
    else            : cinfo=cinfo.copy()
    
    #___________________________________________________________________________
    # check if dictionary keys exist, if they do not exist fill them up 
    cfac = 1
    if 'cfac' in cinfo.keys(): cfac = cinfo['cfac']
    if 'chist'  not in cinfo.keys(): cinfo['chist']  = True
    if 'ctresh' not in cinfo.keys(): cinfo['ctresh'] = 0.995
    if (('cmin' not in cinfo.keys()) or ('cmax' not in cinfo.keys())) and ('crange' not in cinfo.keys()):
        #_______________________________________________________________________
        # loop over all the input data --> find out total cmin/cmax value
        cmin, cmax = np.Inf, -np.Inf
        for data_ii in data:
            if do_index: vname = list(data_ii[0].keys())
            else       : vname = list(data_ii.keys())
            
            #___________________________________________________________________
            if do_vec==False:
                if   do_index: data_plot = data_ii[0][ vname[0] ].data.copy()
                elif do_moc  : data_plot = data_ii['moc'].isel(nz=data_ii['depth']<=-700).data.copy()
                elif do_dmoc is not None  : 
                    if   do_dmoc=='dmoc'  : data_plot = data_ii['dmoc'].data.copy()
                    elif do_dmoc=='srf'   : data_plot = -(data_ii['dmoc_fh'].data.copy()+data_ii['dmoc_fw'].data.copy()+data_ii['dmoc_fr'].data.copy())
                    elif do_dmoc=='inner' : data_plot = data_ii['dmoc'].data.copy() + \
                                                        (data_ii['dmoc_fh'].data.copy()+data_ii['dmoc_fw'].data.copy()+data_ii['dmoc_fr'].data.copy())
                else         : data_plot = data_ii[ vname[0] ].data.copy()
            else:
                # compute norm when vector data
                data_plot = np.sqrt(data_ii[ vname[0] ].data.copy()**2 + data_ii[ vname[1] ].data.copy()**2)
            
            # for logarythmic rescaling cmin or cmax can not be zero
            if do_rescale=='log10' or do_rescale=='slog10': 
                data_plot[np.abs(data_plot)==0]=np.nan
                data_plot[np.abs(data_plot)<=1e-15]=np.nan
            
            #___________________________________________________________________
            if tri is None or do_index:
                #cmin = np.min([cmin,np.nanmin(data_plot) ])
                #cmax = np.max([cmax,np.nanmax(data_plot) ])
                #print('cmin, cmax = ', cmin, cmax)
                if cinfo['chist']:
                    auxcmin,auxcmax = do_climit_hist(data_plot.flatten(),ctresh=cinfo['ctresh'], cweights=do_cweights)
                    cmin, cmax = np.min([cmin,auxcmin]), np.max([cmax,auxcmax])
                    print('--> histo: cmin, cmax = ', cmin, cmax)
                else:    
                    cmin = np.min([cmin,np.nanmin(data_plot) ])
                    cmax = np.max([cmax,np.nanmax(data_plot) ])
                    print('cmin, cmax = ', cmin, cmax)
            else:    
                data_plot = np.hstack((data_plot,data_plot[mesh.n_pbnd_a]))
                #cmin = np.min([cmin,np.nanmin(data_plot[tri.triangles.flatten()]) ])
                #cmax = np.max([cmax,np.nanmax(data_plot[tri.triangles.flatten()]) ])
                #print('cmin, cmax = ', cmin, cmax)
                if cinfo['chist']:
                    if do_cweights is not None: 
                        do_cweights_in = do_cweights.copy()
                        if do_cweights_in.size==mesh.n2dn: do_cweights_in = np.hstack((do_cweights_in,do_cweights_in[mesh.n_pbnd_a]))
                        do_cweights_in = do_cweights_in[tri.triangles.flatten()]
                    else: 
                        do_cweights_in = do_cweights
                    auxcmin,auxcmax = do_climit_hist(data_plot[tri.triangles.flatten()], ctresh=cinfo['ctresh'], cweights=do_cweights_in)
                    cmin, cmax = np.min([cmin,auxcmin]), np.max([cmax,auxcmax])
                    print('--> histo: cmin, cmax = ', cmin, cmax)
                else:    
                    cmin = np.min([cmin,np.nanmin(data_plot[tri.triangles.flatten()]) ])
                    cmax = np.max([cmax,np.nanmax(data_plot[tri.triangles.flatten()]) ])
                    print('cmin, cmax = ', cmin, cmax)
            cmin, cmax = cmin*cfac, cmax*cfac
            
        #_______________________________________________________________________
        if 'climit' in cinfo.keys():
           cmin = np.max([cmin, cinfo['climit'][0]])
           cmax = np.min([cmax, cinfo['climit'][-1]])
        
        #_______________________________________________________________________
        # dezimal rounding of cmin and cmax
        if not do_rescale=='log10' and not do_rescale=='slog10':
            cdmin, cdmax = 0.0, 0.0
            if np.abs(np.mod(np.abs(cmin),1))!=0: cdmin = np.floor(np.log10(np.abs(np.mod(np.abs(cmin),1))))
            if np.abs(np.mod(np.abs(cmax),1))!=0: cdmax = np.floor(np.log10(np.abs(np.mod(np.abs(cmax),1))))
            cdez        = np.min([cdmin,cdmax])
            cmin, cmax  = np.around(cmin, -np.int32(cdez-1)), np.around(cmax, -np.int32(cdez-1))
            
        #_______________________________________________________________________
        # write cmin/cmax too cinfo dictionary
        if 'cmin' not in cinfo.keys(): cinfo['cmin'] = cmin
        if 'cmax' not in cinfo.keys(): cinfo['cmax'] = cmax  
        
    #___________________________________________________________________________    
    if 'crange' in cinfo.keys():
        cinfo['cmin'], cinfo['cmax'], cinfo['cref'] = cinfo['crange'][0], cinfo['crange'][1], cinfo['crange'][2]
        if do_rescale=='slog10': cinfo['cref'] = np.abs(cinfo['cref']) 
    else:
        if (cinfo['cmin'] == cinfo['cmax'] ): raise ValueError (' --> can\'t plot! data are everywhere: {}'.format(str(cinfo['cmin'])))
        cref = cinfo['cmin'] + (cinfo['cmax']-cinfo['cmin'])/2
        if 'cref' not in cinfo.keys(): 
            if do_rescale=='log10':
                # compute cref in decimal units and tranfer back to normal units 
                # afterwards
                cdmin = np.floor(np.log10(np.abs(cinfo['cmin'])))
                cdmax = np.floor(np.log10(np.abs(cinfo['cmax'])))
                cref = cdmin + (cdmax-cdmin)/2
                cinfo['cref'] = np.around(cref, -np.int32(np.floor(np.log10(np.abs(cref)))-1) )
                cinfo['cref'] = np.power(10.0,cinfo['cref'])
            elif do_rescale=='slog10':    
                # cref becomes cutoff value for logarithmic to liner transition in 
                # case of symetric log10
                cinfo['cref'] = np.power(10.0,-6)
            else:
                dez = 1
                while True:
                    new_cref = np.around(cref, -np.int32(np.floor(np.log10(np.abs(cref)))-dez) )
                    if new_cref>cinfo['cmin'] and new_cref<cinfo['cmax']:
                        break
                    else: 
                        dez=dez+1
                cinfo['cref'] = new_cref
        
    #___________________________________________________________________________    
    if 'cnum' not in cinfo.keys(): cinfo['cnum'] = 20
    if 'cstr' not in cinfo.keys(): cinfo['cstr'] = 'wbgyr'
    
    #___________________________________________________________________________    
    # compute clevels and cmap
    if do_rescale=='log10':
        # transfer cmin, cmax, cref into decimal units
        cdmin = np.floor(np.log10(np.abs(cinfo['cmin'])))
        cdmax = np.floor(np.log10(np.abs(cinfo['cmax'])))
        cdref = np.floor(np.log10(np.abs(cinfo['cref'])))
        
        print(cinfo)
        print(cdmin,cdmax,cdref)
        #compute levels in decimal units
        cinfo['cmap'],cinfo['clevel'],cinfo['cref'] = colormap_c2c(cdmin,cdmax,cdref,cinfo['cnum'],cinfo['cstr'])
        
        # transfer levels back to normal units
        cinfo['clevel'] = np.power(10.0,cinfo['clevel'])
        cinfo['cref']   = np.power(10.0,cinfo['cref'])
           
    elif do_rescale=='slog10':
        # transfer cmin, cmax, cref into decimal units
        cdmin = np.floor(np.log10(np.abs(cinfo['cmin'])))
        cdmax = np.floor(np.log10(np.abs(cinfo['cmax'])))
        cdref = np.floor(np.log10(np.abs(cinfo['cref'])))
        ddcmin, ddcmax = -(cdmin-cdref), (cdmax-cdref)
        cinfo['cmap'],cinfo['clevel'],cinfo['cref'] = colormap_c2c(ddcmin,ddcmax,0.0,cinfo['cnum'],cinfo['cstr'])
        
        # rescale clevels towards symetric logarithm
        isneg = cinfo['clevel']<0
        ispos = cinfo['clevel']>0
        cinfo['clevel'][isneg] = (np.abs(cinfo['clevel'][isneg]) + cdref)
        cinfo['clevel'][ispos] = (np.abs(cinfo['clevel'][ispos]) + cdref)
        cinfo['clevel'][isneg] = -np.power(10.0, cinfo['clevel'][isneg])
        cinfo['clevel'][ispos] = np.power(10.0, cinfo['clevel'][ispos])
        
    else:    
        if cinfo['cref'] == 0.0:
            if cinfo['cref'] > cinfo['cmax']: cinfo['cmax'] = cinfo['cref']+np.finfo(np.float32).eps
            if cinfo['cref'] < cinfo['cmin']: cinfo['cmin'] = cinfo['cref']-np.finfo(np.float32).eps
        cinfo['cmap'],cinfo['clevel'],cinfo['cref'] = colormap_c2c(cinfo['cmin'],cinfo['cmax'],cinfo['cref'],cinfo['cnum'],cinfo['cstr'])
        
    #___________________________________________________________________________
    print(cinfo)
    return(cinfo)    



#| compute min/max value range by histogram, computation of cumulativ distribution
#| function at certain cutoff treshold
#|_____________________________________________________________________________|  
def do_climit_hist(data_in, ctresh=0.99, cbin=1000, cweights=None):
    if cweights is None:
        hist, bin_e = np.histogram(data_in[~np.isnan(data_in)], bins=cbin, density=False,) #weights=mesh.n_area[isnotnan]/np.sum(mesh.n_area[isnotnan]), )
    else:
        hist, bin_e = np.histogram(data_in[~np.isnan(data_in)], bins=cbin, weights=cweights[~np.isnan(data_in)], density=True,) #weights=mesh.n_area[isnotnan]/np.sum(mesh.n_area[isnotnan]), )
    hist        = hist/hist.sum()
    bin_m       = bin_e[:-1]+(bin_e[:-1]-bin_e[1:])/2
    cmin        = bin_m[np.where(np.cumsum(hist[::-1])[::-1]>=ctresh)[0][-1]]
    cmax        = bin_m[np.where(np.cumsum(hist)            >=ctresh)[0][ 0]]
    return(cmin, cmax)



# ___DO PLOT LAND-SEA MASK_____________________________________________________
#| plot different land sea masks, based on patch: 'fesom', based on png image: |
#| 'bluemarble', 'stock' or 'etopo'                                            |
#| ___INPUT_________________________________________________________________   |
#| ax           :   actual axes handle                                         |
#| mesh         :   fesom2 mesh object                                         |
#| do_lsmask    :   None, str (default: 'fesom') plot land-sea mask.           |
#|                  If:  None   ... no land sea mask is used,                  |
#|                      'fesom' ... overlay fesom shapefile land-sea mask using|
#|                                  color color_lsmask                         |
#|                      'stock' ... use cartopy stock image as land sea mask   |
#|                      'bluemarble' ... use bluemarble image as land sea mask |
#|                      'etopo' ... use etopo image as land sea mask           |  
#| box          :   None, list (default: None) regional limitation of plot     |
#|                  [lonmin, lonmax, latmin, latmax]                           |
#| which_proj   :   cartopy projection handle                                  |
#| color_lsmask :   RGB list with color of fesom land-sea mask patch           |
#| edgecolor    :   edge color of fesom coastline                              |
#| linewidth    :   linewidth of fesom coastline                               |
#| resolution   :   str, (default:'low') resolution of background image when   |
#|                  using bluemarble or etopo can be either 'low' or           |
#|                  'high' --> see image.json file in src/backgrounds/         |
#| ___RETURNS_______________________________________________________________   |
#| ax           :   actual axes handle                                         |
#|_____________________________________________________________________________|  
def do_plotlsmask(ax, mesh, do_lsmask, box, which_proj, 
                  color_lsmask=[0.6, 0.6, 0.6], edgecolor='k', linewidth=0.5,
                  resolution='low'):
    #___________________________________________________________________________
    # add mesh land-sea mask
    if   do_lsmask is None: 
        return(ax)

    elif do_lsmask=='fesom':
        ax.add_geometries(mesh.lsmask_p, crs=ccrs.PlateCarree(), 
                              facecolor=color_lsmask, edgecolor=edgecolor ,linewidth=linewidth)
        
    elif do_lsmask=='stock':   
        ax.stock_img()
        ax.add_geometries(mesh.lsmask_p, crs=ccrs.PlateCarree(), 
                              facecolor='None', edgecolor=edgecolor, linewidth=linewidth)
            
    elif do_lsmask=='bluemarble': 
        # --> see original idea at http://earthpy.org/cartopy_backgroung.html#disqus_thread and 
        # https://stackoverflow.com/questions/67508054/improve-resolution-of-cartopy-map
        bckgrndir = os.getcwd()
        bckgrndir = os.path.normpath(bckgrndir+'/tripyview/backgrounds/')
        os.environ["CARTOPY_USER_BACKGROUNDS"] = bckgrndir
        ax.background_img(name=do_lsmask, resolution=resolution)
        ax.add_geometries(mesh.lsmask_p, crs=ccrs.PlateCarree(), 
                          facecolor='None', edgecolor=edgecolor, linewidth=linewidth)
            
    elif do_lsmask=='etopo':   
        # --> see original idea at http://earthpy.org/cartopy_backgroung.html#disqus_thread and 
        # https://stackoverflow.com/questions/67508054/improve-resolution-of-cartopy-map
        bckgrndir = os.getcwd()
        bckgrndir = os.path.normpath(bckgrndir+'/tripyview/backgrounds/')
        os.environ["CARTOPY_USER_BACKGROUNDS"] = bckgrndir
        ax.background_img(name=do_lsmask, resolution=resolution)    
        ax.add_geometries(mesh.lsmask_p, crs=ccrs.PlateCarree(), 
                          facecolor='None', edgecolor=edgecolor, linewidth=linewidth)
    else:
        raise ValueError(" > the do_lsmask={} is not supported, must be either 'fesom', 'stock', 'bluemarble' or 'etopo'! ")
        
    #___________________________________________________________________________
    return(ax)



# ___DO ADD CARTOPY GRIDLINES__________________________________________________
#| add cartopy grid lines, the functionality of cartopy regarding gridlines is |
#| still very limited, espesially regarding lon lat labels, so far onyl        |
#| ccrs.PlateCarree() fully supports labels                                    |
#| ___INPUT_________________________________________________________________   |
#| ax           :   actual axes handle                                         |
#| rowlist      :   list, with panel row indices                               |
#| collist      :   list, with panel column indices                            |
#| xticks       :   array, with xticks                                         |
#| yticks       :   array, yith xticks                                         |
#| proj         :   str, (default: 'pc') which projection should be used, 'pc'=|
#|                  ccrs.PlateCarree(), 'merc'=ccrs.Mercator(), 'nps'=         |
#|                  ccrs.NorthPolarStereo(), 'sps'=ccrs.SouthPolarStereo(),    |
#| which_proj   :   cartopy projection handle                                  |
#| ___RETURNS_______________________________________________________________   |
#| ax           :   actual axes handle                                         |
#|_____________________________________________________________________________|  
def do_add_gridlines(ax, rowlist, collist, xticks, yticks, proj, which_proj):
    
    maxr = np.max(rowlist)+1
    #_______________________________________________________________________
    # add gridlines
    if proj=='merc': 
        gl=ax.gridlines(color='black', linestyle='-', 
                            draw_labels=False, xlocs=xticks, ylocs=yticks,
                            alpha=0.25, )
        gl.xlabels_top, gl.ylabels_right = False, False
        
    elif proj=='pc':
            
        ax.set_xticks(xticks[1:-1], crs=ccrs.PlateCarree())
        ax.set_yticks(yticks[1:-1], crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter()) 
        gl=ax.gridlines(crs=which_proj, color='black', linestyle='-', 
                            draw_labels=False, 
                            xlocs=xticks, ylocs=yticks, 
                            alpha=0.25, )
        if rowlist!=maxr-1: ax.set_xticklabels([])
        if collist >0     : ax.set_yticklabels([])
            
    elif proj=='nps' or proj=='sps':
        ax.gridlines(color='black', linestyle='-', alpha=0.25, xlocs=xticks, ylocs=yticks,  )
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)
        
    elif proj=='rob':
        ax.gridlines(color='black', linestyle='-', alpha=0.25, xlocs=xticks, ylocs=yticks,)
    
    #___________________________________________________________________________
    return(ax)



# ___DO REDUCTION OF COLORBAR LABELS___________________________________________
#| reduce number of colorbar ticklabels, show minimum number cbar_nl of        |
#| colorbar labels                                                             |
#| ___INPUT_________________________________________________________________   |
#| cbar         :   actual colorbar handle                                     |
#| cbar_nl      :   int, (default:8) minimum number of colorbar labels to show |
#| cinfo        :   None, dict() (default: None), dictionary with colorbar     |
#|                  formation. Information that are given are used others are  |
#|                  computed. cinfo dictionary entries can me:                 |
#|                  > cinfo['cmin'], cinfo['cmax'], cinfo['cref'] ... scalar   |
#|                    min, max, reference value                                |
#|                  > cinfo['crange'] ...  list with [cmin, cmax, cref]        |
#|                    overrides scalar values                                  |
#|                  > cinfo['cnum'] ... minimum number of colors               |
#|                  > cinfo['cstr'] ... name of colormap see in                |
#|                    sub_colormap_c2c.py                                      |
#|                  > cinfo['cmap'] ... colormap object ('wbgyr', 'blue2red,   |
#|                    'jet' ...)                                               |
#|                  > cinfo['clevel'] ... color level array                    |
#| ___RETURNS_______________________________________________________________   |
#| cbar         :   actual colorbar handle                                     |                  
#|_____________________________________________________________________________|  
def do_cbar_label(cbar, cbar_nl, cinfo, do_vec=False):
    #___________________________________________________________________________
    # kickout some colormap labels if there are to many
    if cbar.orientation=='vertical': tickl = cbar.ax.get_yticklabels()
    else:                            tickl = cbar.ax.get_xticklabels()
    ncbar_l=len(tickl)
    idx_cref = np.where(cinfo['clevel']==cinfo['cref'])[0]
    idx_cref = np.asscalar(idx_cref)
    
    nstep = ncbar_l/cbar_nl
    nstep = np.max([np.int(np.floor(nstep)),1])
    #if nstep==0:nstep=1
    
    idx = np.arange(0,len(tickl),1)
    idxb = np.ones((len(tickl),), dtype=bool)                
    idxb[idx_cref::nstep]  = False
    idxb[idx_cref::-nstep] = False
    idx_not = idx[idxb==True]
    idx_yes = idx[idxb==False]
    
    for ii in list(idx_not): tickl[ii]=''
    if do_vec: 
        for ii in list(idx_yes): tickl[ii]='{:2.2f}'.format(cinfo['clevel'][ii])
    if cbar.orientation=='vertical': cbar.ax.set_yticklabels(tickl)
    else:                            cbar.ax.set_xticklabels(tickl)
    
    #___________________________________________________________________________
    return(cbar)



# ___DO FORMATING OF COLORBAR___________________________________________________
#|
#|
#| ___RETURNS_______________________________________________________________   |
#| cbar         :   actual colorbar handle                                     |   
#|_____________________________________________________________________________|  
def do_cbar_formatting(cbar, do_rescale, cbar_nl, fontsize, clocs, pw_lim=[-3,4]):
    # formatting of normal colorbar axis
    if not do_rescale == 'log10' and not do_rescale == 'slog10':
        formatter     = mticker.ScalarFormatter(useOffset=True, useMathText=True, useLocale=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((pw_lim[0], pw_lim[-1]))      
        cbar.formatter= formatter
        cbar.locator  = mticker.FixedLocator(clocs, nbins=cbar_nl)
        cbar.ax.yaxis.get_offset_text().set(size=fontsize, horizontalalignment='center')
        cbar.update_ticks()
    # formating for log and symlog colorbar axis    
    else:
        cbar.set_ticks(clocs[np.mod(np.log10(np.abs(clocs)),1)==0.0])
        cbar.ax.minorticks_off()
        cbar.update_ticks()
    cbar.ax.tick_params(labelsize=fontsize)

    #___________________________________________________________________________
    return(cbar)



# ___DO REPOSITIONING OF AXES PANELS AND COLORBAR______________________________
#| in case of multiple panels reposition the axes and colorbar for tighter     |
#| fit                                                                         |                  
#| ___INPUT_________________________________________________________________   |
#| ax           :   actual axes handle                                         |
#| cbar         :   actual colorbar handle                                     | 
#| rowlist      :   list, with panel row indices                               |
#| collist      :   list, with panel column indices                            |
#| pos_fac      :   float, (default:1.0) multiplication factor  to             |
#|                  increase/decrease width and height of plotted  panel       |
#| pos_gap      :   list, (default: [0.02, 0.02]) gives width and height       |
#|                  of gaps between multiple panels                            |
#| title        :   None, str,(default:None) give every plot panel a title     |
#|                  string                                                     |
#|                  IF: None       ... no title is plotted                     |
#|                      'descript' ... use data 'descript' attribute for title |
#|                                     string                                  |
#|                      'string'   ... use given string as title               |   
#| ___RETURNS_______________________________________________________________   |
#| ax           :   actual axes handle                                         |
#| cbar         :   actual colorbar handle                                     | 
#|_____________________________________________________________________________|  
def do_reposition_ax_cbar(ax, cbar, rowlist, collist, pos_fac, pos_gap, title=None, 
                          extend=None, proj=None):
    #___________________________________________________________________________
    # repositioning of axes and colorbar
    nax = len(ax)
    ax_pos = np.zeros((nax,4))
    for jj in range(0,nax):
        aux = ax[jj].get_position()
        ax_pos[jj,:] = np.array([aux.x0, aux.y0, aux.width, aux.height])
    maxr = rowlist.max()+1
    maxc = collist.max()+1
    #print(maxr,maxc)
    #print(ax_pos)
    
    #fac = pos_fac
    ##x0, y0, x1, y1 = 0.05, 0.05, 0.95, 0.95
    #x0, y0, x1, y1 = 0.1, 0.1, 0.9, 0.9
    #if cbar is not None:
        #if cbar.orientation=='horizontal': y0 = 0.21
    #w, h = ax_pos[:,2].min(), ax_pos[:,3].min()
    #w, h = w*fac, h*fac
    #wg, hg = pos_gap[0], pos_gap[1]
    #if title is not None: hg = hg+0.04
    
    fac = pos_fac
    wg, hg = pos_gap[0], pos_gap[1]
    x0, y0, x1, y1 = 0.075, 0.05, 0.825, 0.95
    if extend is not None:
        x0, y0, x1, y1 = extend[0], extend[1], extend[2], extend[3]
        #ax_pos[:,2] = x1-x0
        #ax_pos[:,3] = y1-y0
    
    #print(ax_pos[:,3], ax_pos[:,2])
    if cbar is not None:
        if cbar.orientation=='horizontal': y0 = y0+0.25
        
    dx = x1-x0-(maxc-1)*wg
    dy = y1-y0-(maxr-1)*hg
    #print('dx,dy=', dx, dy)
    wref  = dx/maxc
    href  = dy/maxr
    #print('wref,href=',wref,href)
    fac   = ax_pos[:,3].min()/ax_pos[:,2].min()
    w,h   = wref, wref*fac
    if h>href: 
        w,h   = href/fac, href
    #h  = dh*(ax_pos[:,3].min()/ax_pos[:,2].min())
    #print('w,h=',w,h,)
    
    #dy = y1-y0-(maxr-1)*hg
    #h  = dy/maxr
    #w  = h*ax_pos[:,2].min()/ax_pos[:,3].min()
    if proj in ['nps', 'sps']:
        if title is not None: hg = hg+0.01
    else:
        if title is not None: hg = hg+0.06
    if (h*maxr+hg*(maxr-1)+y0)>y1: fac = 1/(h*maxr+hg*(maxr-1)+y0)
    if (w*maxc+wg*(maxc-1)+x0)>x1: fac = 1/(w*maxc+wg*(maxc-1)+x0) 
    #w, h = w*fac, h*fac
    #print('w,h=',w,h,fac)
    
    for jj in range(nax-1,0-1,-1):
        ax[jj].set_position( [x0+(w+wg)*collist[jj], y0+(h+hg)*np.abs(rowlist[jj]-maxr+1), w, h] )
    
    if cbar is not None:
        cbar_pos = cbar.ax.get_position()
        if cbar.orientation=='vertical': 
            cbar.ax.set_position([x0+(w+wg)*maxc, y0, cbar_pos.width*0.75, h*maxr+hg*(maxr-1)])
            cbar.ax.set_aspect('auto')
        else: 
            #cbar.ax.set_position([x0, 0.125, w*maxc+wg*(maxc-1), cbar_pos.height/2])/
            cbar.ax.set_position([x0, 0.14, w*maxc+wg*(maxc-1), cbar_pos.height/2])
            cbar.ax.set_aspect('auto')
        
    #___________________________________________________________________________
    return(ax, cbar)    



# ___DO XTICK AND YTICKS_______________________________________________________
#| compute xticks and yticks based on minimum number of tick labesl            |                  
#| ___INPUT_________________________________________________________________   |
#| mesh         :   fesom2 mesh object                                         |
#| box          :   None, list (default: None) regional limitation of plot     |
#|                  [lonmin, lonmax, latmin, latmax]                           |
#| ticknr       :   int, (default:7) minimum number of lon and lat ticks       |                                     |
#| ___RETURNS_______________________________________________________________   |
#| xticks,yticks:   array, with optimal lon and lat ticks                   
#|_____________________________________________________________________________|  
def do_ticksteps(mesh, box, ticknr=7):
    #___________________________________________________________________________
    tickstep = np.array([0.5,1.0,2.0,2.5,5.0,10.0,15.0,20.0,30.0,45.0, 360])
    
    #___________________________________________________________________________
    idx     = int(np.argmin(np.abs( tickstep-(box[1]-box[0])/ticknr )))
    if np.abs(box[1]-box[0])==360:
        xticks   = np.arange(-180+mesh.focus, 180+1, tickstep[idx])
    else:    
        xticks   = np.arange(box[0], box[1]+1, tickstep[idx]) 
    idx     = int(np.argmin(np.abs( tickstep-(box[3]-box[2])/ticknr )))
    if np.abs(box[3]-box[2])==180:
        yticks   = np.arange(-90, 90+1, tickstep[idx])  
    else:   
        yticks   = np.arange(box[2], box[3]+1, tickstep[idx])  
        
    #___________________________________________________________________________
    xticks = np.unique(np.hstack((box[0], xticks, box[1])))
    yticks = np.unique(np.hstack((box[2], yticks, box[3])))
    #print(' > xtick: {}'.format(str(xticks)))    
    #print(' > ytick: {}'.format(str(yticks))) 
    
    #___________________________________________________________________________
    return(xticks,yticks)



# ___DO SAVE FIGURE____________________________________________________________
#| save figure to file                                                         |
#| ___INPUT_________________________________________________________________   |
#| do_save      :   None, str (default:None) if None figure will by not saved, |
#|                  if string figure will be saved, strings must give          |
#|                  directory and filename  where to save.                     |
#| dpi          :   int, (default:600), resolution of image                    |
#| transparent  :   bool, (default:True) in case of png make white background  |
#|                  transparent                                                |
#| pad_inches   :   float, (default:0.1) pad space around plot                 
#| ___RETURNS_______________________________________________________________   |
#| None                  
#|_____________________________________________________________________________|  
def do_savefigure(do_save, fig, dpi=300, transparent=False, pad_inches=0.1, **kw):
    if do_save is not None:
        #_______________________________________________________________________
        # extract filename from do_save
        sfname = os.path.basename(do_save)
        
        # extract file extensions
        sfformat = os.path.splitext(sfname)[1][1:]
        
        # extract dirname from do_save --> create directory if not exist
        sdname = os.path.dirname(do_save)
        sdname = os.path.expanduser(sdname)
        print(' > save figure: {}'.format(os.path.join(sdname,sfname)))
        
        #_______________________________________________________________________
        if not os.path.isdir(sdname): os.makedirs(sdname)
        
        #_______________________________________________________________________
        fig.savefig(os.path.join(sdname,sfname), format=sfformat, dpi=dpi, 
                    bbox_inches='tight', pad_inches=pad_inches,\
                    transparent=transparent, **kw)
