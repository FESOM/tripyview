import os
import sys
import numpy                    as np
import time                     as clock
import xarray                   as xr
import cartopy.crs              as ccrs
import cartopy.feature          as cfeature
from   cartopy.mpl.ticker       import (LongitudeFormatter, LatitudeFormatter)
from   cartopy.mpl.gridliner    import Gridliner
from   mpl_toolkits.axes_grid1  import make_axes_locatable
import matplotlib
#import matplotlib.pylab         as plt
import matplotlib.pyplot        as plt
from   matplotlib.tri           import Triangulation,TriAnalyzer
import matplotlib.ticker        as mticker
import matplotlib.path          as mpath
import matplotlib.colors        as mcolors
from   matplotlib.colors        import ListedColormap

from .sub_mesh     import *
from .sub_data     import *
from .sub_colormap import *

# ___PLOT HORIZONTAL FESOM2 DATA SLICES________________________________________
#|                                                                             |
#|      *** PLOT HORIZONTAL FESOM2 DATA SLICES --> BASED ON CARTOPY ***        |
#|                                                                             |
#|_____________________________________________________________________________|
def plot_hslice(mesh, data, cinfo=None, box=None, proj='pc', figsize=[9, 4.5], 
                n_rc=[1,1], do_grid=False, do_plot='tcf', do_rescale=True,
                cbar_nl=8, cbar_orient='vertical', cbar_label=None, cbar_unit=None,
                do_lsmask='fesom', do_bottom=True, color_lsmask=[0.6, 0.6, 0.6], 
                color_bot=[0.75,0.75,0.75],  title=None, do_ie2n = True, 
                pos_fac=1.0, pos_gap=[0.02, 0.02], pos_extend=None, do_save=None, save_dpi=600,
                linecolor='k', linewidth=0.5, do_reffig=False, ref_cinfo=None, ref_rescale=None,
                do_info=False, nargout=['fig', 'ax', 'cbar']):
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
    t1=clock.time()
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
    if proj=='channel':
        if box is None or box=="None": box = [np.hstack((mesh.n_x,mesh.n_xa)).min(), np.hstack((mesh.n_x,mesh.n_xa)).max(), np.hstack((mesh.n_y,mesh.n_ya)).min(), np.hstack((mesh.n_y,mesh.n_ya)).max()]
    else:    
        if box is None or box=="None": box = [ -180+mesh.focus, 180+mesh.focus, -90, 90 ]
    
    #___________________________________________________________________________
    # Create projection
    if   proj=='pc'     : which_proj, which_transf = ccrs.PlateCarree()     , ccrs.PlateCarree()
    elif proj=='merc'   : which_proj, which_transf = ccrs.Mercator()        , ccrs.PlateCarree()
    elif proj=='rob'    : which_proj, which_transf = ccrs.Robinson()        , ccrs.PlateCarree()
    #elif proj=='mol'    : which_proj, which_transf = ccrs.Mollweide()       , ccrs.PlateCarree()        
    elif proj=='eqearth': which_proj, which_transf = ccrs.EqualEarth(central_longitude=mesh.focus)      , ccrs.PlateCarree()        
    elif proj=='nps'    : 
        which_proj, which_transf = ccrs.NorthPolarStereo(), ccrs.PlateCarree()
        if box[2]<0: box[2]=0
    elif proj=='sps'    : 
        which_proj, which_transf = ccrs.SouthPolarStereo(), ccrs.PlateCarree()
        if box[3]>0: box[2]=0
    elif proj=='channel':
        #which_proj, which_transf = None, None
        which_proj, which_transf = ccrs.PlateCarree()     , ccrs.PlateCarree()
    else: raise ValueError('The projection {} is not supporrted!'.format(proj))    
    
    #___________________________________________________________________________    
    # create lon, lat ticks 
    xticks, yticks = do_ticksteps(mesh, box)

    #___________________________________________________________________________    
    # create figure and axes
    fig, ax = plt.subplots( n_rc[0],n_rc[1],
                                figsize=figsize, 
                                subplot_kw =dict(projection=which_proj),
                                gridspec_kw=dict(left=0.06, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05,),
                                constrained_layout=False, sharex=True, sharey=True)
    
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
    if proj in ['nps', 'sps', 'pc', 'channel']:
        e_idxbox = grid_cutbox_e(tri.x, tri.y, tri.triangles, box, which='soft')
    elif proj in ['rob', 'eqearth'] :   
        #box = [-179.750, 179.750, -90.0, 90.0]
        box[0], box[1] = box[0]+0.25, box[1]-0.25
        # otherwise produces strange artefacts when using robinson projection
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
        # print(x.min(), x.max(), y.min(), y.max())
        e_idxbox = (x>=-0.05) & (x<=1.05) & (y>=-0.05) & (y<=1.05)
    
    tri.triangles = tri.triangles[e_idxbox,:]    
    if do_info:
        print('--> do triangulation: ', clock.time()-t1) 
        t1 = clock.time()
    
    #___________________________________________________________________________
    # data must be list filled with xarray data
    if not isinstance(data, list): data = [data]
    ndata = len(data)
    
    #___________________________________________________________________________
    # set up color info
    if do_reffig:
        ref_cinfo = do_setupcinfo(ref_cinfo, [data[0]], ref_rescale, mesh=mesh, tri=tri)
        cinfo     = do_setupcinfo(cinfo,      data[1:], do_rescale, mesh=mesh, tri=tri)        
    else:
        cinfo = do_setupcinfo(cinfo, data, do_rescale, mesh=mesh, tri=tri)
    if do_info:
        print('--> do setup cinfo: ', clock.time()-t1)
        t1 = clock.time()
    
    #___________________________________________________________________________
    # setup normalization log10, symetric log10, None
    which_norm = do_compute_scalingnorm(cinfo, do_rescale)
    if do_reffig:     
        which_norm_ref = do_compute_scalingnorm(ref_cinfo, ref_rescale)
    if do_info:
        print('--> do scaling norm: ', clock.time()-t1)
        t1 = clock.time()
        
    #___________________________________________________________________________
    # do the mapping transformation outside of tricontourf is absolutely 
    # faster than let doing cartopy doing it within 
    #if proj=='channel': mappoints = np.column_stack((tri.x, tri.y))
    #else              : mappoints = which_proj.transform_points(which_transf, tri.x, tri.y)
    mappoints = which_proj.transform_points(which_transf, tri.x, tri.y)
    
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
        if not proj=='channel': 
            ax[ii].set_extent(box, crs=ccrs.PlateCarree())
        else:
            ax[ii].set_extent(box, crs=ccrs.PlateCarree())
            ax[ii].set_aspect('auto')
        
        #_______________________________________________________________________
        # periodic augment data
        vname     = list(data[ii].keys())
        data_plot = data[ii][ vname[0] ].data.copy()
        #data_plot = data[ii][ vname[0] ].values.copy()
            
        is_onvert = True
        if   data_plot.size==mesh.n2dn:
            is_onvert = True
            data_plot = np.hstack((data_plot,data_plot[mesh.n_pbnd_a]))
        elif data_plot.size==mesh.n2de:
            if not do_ie2n:
                is_onvert = False
                data_plot = np.hstack((data_plot[mesh.e_pbnd_0],data_plot[mesh.e_pbnd_a]))
                data_plot = data_plot[e_idxbox]
            else:
                # interpolate from elements to vertices --> cartopy plotting is faster
                is_onvert = True
                data_plot = grid_interp_e2n(mesh,data_plot)
                data_plot = np.hstack((data_plot,data_plot[mesh.n_pbnd_a]))
        
        if do_info: print('--> do augment data: ', clock.time()-t1) ; t1 = clock.time()
        
        #_______________________________________________________________________
        if do_reffig: 
            if ii==0: cinfo_plot, which_norm_plot = ref_cinfo, which_norm_ref
            else    : cinfo_plot, which_norm_plot = cinfo    , which_norm
        else        : cinfo_plot, which_norm_plot = cinfo    , which_norm
            
        #_______________________________________________________________________
        # kick out triangles with Nan cut elements to box size        
        isnan   = np.isnan(data_plot)
        if not is_onvert:
            e_idxok = isnan==False
        else:
            e_idxok = np.any(isnan[tri.triangles], axis=1)==False
        del(isnan)  
        
        #_______________________________________________________________________
        # add color for ocean bottom
        if do_bottom and np.any(e_idxok==False):
            hbot = ax[ii].triplot(mappoints[:,0], mappoints[:,1], tri.triangles[e_idxok==False,:], 
                                      color=color_bot)#, transform=which_transf)
            if do_info: print('--> do plot bottom patch: ', clock.time()-t1) ; t1 = clock.time()
            
        #_______________________________________________________________________
        # plot tri contourf/tripcolor
        if   do_plot=='tpc' or (do_plot=='tcf' and not is_onvert):
            if not is_onvert: data_plot = data_plot[e_idxok]
            print(data_plot.shape)
            hp=ax[ii].tripcolor(mappoints[:,0], mappoints[:,1], tri.triangles[e_idxok,:], data_plot,
                                    shading='flat',
                                    cmap=cinfo_plot['cmap'],
                                    vmin=cinfo_plot['clevel'][0], vmax=cinfo_plot['clevel'][ -1],
                                    norm = which_norm_plot) #transform=which_transf,
            if do_info: print('--> do tripcolor: ', clock.time()-t1) ; t1 = clock.time()                        
            
        elif do_plot=='tcf': 
            # supress warning message when compared with nan
            with np.errstate(invalid='ignore'):
                data_plot[data_plot<cinfo_plot['clevel'][ 0]] = cinfo_plot['clevel'][ 0]
                data_plot[data_plot>cinfo_plot['clevel'][-1]] = cinfo_plot['clevel'][-1]
                
            hp=ax[ii].tricontourf(mappoints[:,0], mappoints[:,1], tri.triangles[e_idxok,:], data_plot,
                                  levels=cinfo_plot['clevel'], cmap=cinfo_plot['cmap'], extend='both',
                                  norm=which_norm_plot)#, transform=which_transf) 
            if do_info: print('--> do tricontourf: ', clock.time()-t1) ; t1 = clock.time()        
        hpall.append(hp)
            
        #_______________________________________________________________________
        # add grid mesh on top
        if do_grid: 
            #ts = clock.time()
            ax[ii].triplot(mappoints[:,0], mappoints[:,1], tri.triangles[:,:], color='k', linewidth=0.1, alpha=0.75,zorder=5) #transform=which_transf)
            if do_info: print('--> do triplot: ', clock.time()-t1) ; t1 = clock.time()       
            
        #_______________________________________________________________________
        # add mesh land-sea mask
        ax[ii] = do_plotlsmask(ax[ii],mesh, do_lsmask, box, which_proj, color_lsmask=color_lsmask, edgecolor=linecolor, linewidth=linewidth)
        if do_info: print('--> do lsmask: ', clock.time()-t1) ; t1 = clock.time()     
        
        #_______________________________________________________________________
        # add gridlines
        ax[ii] = do_add_gridlines(ax[ii], rowlist[ii], collist[ii], xticks, yticks, proj, which_proj)
        if do_info: print('--> do gridlines: ', clock.time()-t1) ; t1 = clock.time()    
        
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
            
        fig.canvas.draw()    
        
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
        if cbar_label is None: 
            if   'short_name' in data[0][vname[0]].attrs:
                cbar_label = data[0][vname[0]].attrs['short_name']
            elif 'long_name' in data[0][vname[0]].attrs:
                cbar_label = data[0][vname[0]].attrs['long_name']
        #if cbar_unit  is None: cbar_label = cbar_label+' ['+data[0][ vname[0] ].attrs['units']+']'
        if cbar_unit  is None: cbar_label = cbar_label+' ['+data[0][ vname[0] ].attrs['units']+']'
        else:                  cbar_label = cbar_label+' ['+cbar_unit+']'
        if 'str_ltim' in data[0][vname[0]].attrs.keys():
            cbar_label = cbar_label+'\n'+data[0][vname[0]].attrs['str_ltim']
        if 'str_ldep' in data[0][vname[0]].attrs.keys():
            cbar_label = cbar_label+data[0][vname[0]].attrs['str_ldep']
        cbar.set_label(cbar_label, size=fontsize+2)
    else:
        cbar=list()
        for ii, aux_ax in enumerate(ax): 
            cbar_label=''
            if ii==0: 
                aux_cbar = plt.colorbar(hpall[ii], orientation=cbar_orient, ax=aux_ax, ticks=ref_cinfo['clevel'], 
                            extendrect=False, extendfrac=None, drawedges=True, pad=0.025, shrink=1.0,)  
                # do formatting of colorbar 
                aux_cbar = do_cbar_formatting(aux_cbar, ref_rescale, cbar_nl, fontsize, ref_cinfo['clevel'])
            else:     
                aux_cbar = plt.colorbar(hpall[ii], orientation=cbar_orient, ax=aux_ax, ticks=cinfo['clevel'], 
                            extendrect=False, extendfrac=None, drawedges=True, pad=0.025, shrink=1.0,)  
                # do formatting of colorbar 
                aux_cbar = do_cbar_formatting(aux_cbar, do_rescale, cbar_nl, fontsize, cinfo['clevel'])
                #cbar_label='anom. '
                
            # do labeling of colorbar
            # cbar_label=None
            # if cbar_label is None: 
            if   'short_name' in data[ii][vname[0]].attrs:
                cbar_label = cbar_label+data[ii][vname[0]].attrs['short_name']
            elif 'long_name' in data[ii][vname[0]].attrs:
                cbar_label = cbar_label+data[ii][vname[0]].attrs['long_name']
            if cbar_unit  is None: cbar_label = cbar_label+' ['+data[ii][ vname[0] ].attrs['units']+']'
            else:                  cbar_label = cbar_label+' ['+cbar_unit+']'
            if 'str_ltim' in data[ii][vname[0]].attrs.keys():
                cbar_label = cbar_label+'\n'+data[ii][vname[0]].attrs['str_ltim']
            if 'str_ldep' in data[ii][vname[0]].attrs.keys():
                cbar_label = cbar_label+data[ii][vname[0]].attrs['str_ldep']
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
    #print(' -AXES-> elapsed time: {} min'.format((clock.time()-ts)/60))
    
    #___________________________________________________________________________
    list_argout=[]
    if len(nargout)>0:
        for stri in nargout:
            if   stri == 'fig'    : list_argout.append(fig)
            elif stri == 'ax'     : list_argout.append(ax)
            elif stri == 'cbar'   : list_argout.append(cbar)
            elif stri == 'hpall'  : list_argout.append(hpall)
            elif stri == 'tri'    : list_argout.append(tri)
            elif stri == 'mappoints': list_argout.append(mappoints)
            elif stri == 'e_idxok': list_argout.append(e_idxok)
            elif stri == 'cinfo': list_argout.append(cinfo)
            
            mappoints
        return(list_argout)
    else:
        return
    

def plot_hslice_reg(mesh, data, input_names, cinfo=None, box=None, proj='pc', figsize=[9,4.5], 
                n_rc=[1,1], do_grid=False, do_plot='tcf', do_rescale=True,
                do_reffig=False, ref_cinfo=None, ref_rescale=None,
                cbar_nl=8, cbar_orient='vertical', cbar_label=None, cbar_unit=None,
                do_lsmask='fesom', do_bottom=True, color_lsmask=[0.6, 0.6, 0.6], 
                color_bot=[0.75,0.75,0.75],  title=None,
                pos_fac=1.0, pos_gap=[0.02, 0.02], pos_extend=None, do_save=None, save_dpi=600,
                linecolor='k', linewidth=0.5, ):
    
    fontsize = 12
    
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
                             norm = which_norm)
            
        elif do_plot=='tcf': 
            # supress warning message when compared with nan
            with np.errstate(invalid='ignore'):
                data_plot[data_plot<cinfo_plot['clevel'][ 0]] = cinfo_plot['clevel'][ 0]
                data_plot[data_plot>cinfo_plot['clevel'][-1]] = cinfo_plot['clevel'][-1]
            
            hp=ax[ii].contourf(data_x, data_y, data_plot, 
                               transform=which_transf,
                               norm=which_norm,
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
        if cbar_label is None: 
            if   'short_name' in data[0][vname].attrs:
                cbar_label = data[0][vname].attrs['short_name']
            elif 'long_name' in data[0][vname].attrs:
                cbar_label = data[0][vname].attrs['long_name']
        if cbar_unit  is None: cbar_label = cbar_label+' ['+data[0][vname].attrs['units']+']'
        else:                  cbar_label = cbar_label+' ['+cbar_unit+']'
        if 'str_ltim' in data[0][vname].attrs.keys():
            cbar_label = cbar_label+'\n'+data[0][vname].attrs['str_ltim']
            #cbar_label = cbar_label+', '+data[0][vname].attrs['str_ltim']
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
                cbar_label ='anom. '
            # do labeling of colorbar
            #if cbar_label is None: 
            if   'short_name' in data[ii][vname].attrs:
                cbar_label = cbar_label+data[ii][vname].attrs['short_name']
            elif 'long_name' in data[ii][vname].attrs:
                cbar_label = cbar_label+data[ii][vname].attrs['long_name']    
            if cbar_unit  is None: cbar_label = cbar_label+' ['+data[ii][vname].attrs['units']+']'
            else:                  cbar_label = cbar_label+' ['+cbar_unit+']'
            if 'str_ltim' in data[ii][vname].attrs.keys():
                cbar_label = cbar_label+'\n'+data[ii][vname].attrs['str_ltim']
                #cbar_label = cbar_label+', '+data[ii][vname].attrs['str_ltim']
            if 'str_ldep' in data[ii][vname].attrs.keys():
                cbar_label = cbar_label+data[ii][vname].attrs['str_ldep']
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



#+___PLOT MERIDIONAL OVERTRUNING CIRCULATION TIME-SERIES_______________________+
#|                                                                             |
#+_____________________________________________________________________________+
def plot_tseries(tseries_list, input_names, sect_name, which_cycl=None, 
                       do_allcycl=False, do_concat=False, str_descript='', str_time='', figsize=[], 
                       do_save=None, save_dpi=600, do_pltmean=True, do_pltstd=False,
                       ymaxstep=None, xmaxstep=5):    
    
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
    if which_cycl is None: aux_which_cycl = 1
    else                 : aux_which_cycl = which_cycl
    for ii_ts, (tseries, tname) in enumerate(zip(tseries_list, input_names)):
        
        time = tseries[0]['time.year'].values + (tseries[0]['time.dayofyear'].values-1)/365
        #_______________________________________________________________________
        if isinstance(tseries,list): tseries = tseries[0]
        
        if 'keys' in dir(tseries):
            vname = list(tseries.keys())[0]
            tseries = tseries[vname]
        
        #_______________________________________________________________________
        if tseries.ndim>1: tseries = tseries.squeeze()
        auxtime = time.copy()
        if np.mod(ii_ts+1,aux_which_cycl)==0 or do_allcycl==False:
            
            if do_concat: auxtime = auxtime + (time[-1]-time[0]+1)*(ii_cycle-1)
            hp=ax.plot(auxtime,tseries, 
                   linewidth=1.5, label=tname, color=cmap.colors[ii_ts,:], 
                   marker='None', markerfacecolor='w', markersize=5, #path_effects=[path_effects.SimpleLineShadow(offset=(1.5,-1.5),alpha=0.3),path_effects.Normal()],
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
              bbox_to_anchor=(1.01,0.5), loc="center left", borderaxespad=0)
              #bbox_to_anchor=(1.04, 1.0), ncol=1) #loc='lower right', 
    ax.set_xlabel('Time [years]',fontsize=12)
    ax.set_ylabel('{:s} in [{:s}]'.format(tseries.attrs['description'], tseries.attrs['units']),fontsize=12)
    ax.set_title(sect_name, fontsize=12, fontweight='bold')
    
    #___________________________________________________________________________
    if do_concat: xmaxstep=20
    xmajor_locator = MultipleLocator(base=xmaxstep) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(xmajor_locator)
    if ymaxstep is not None: 
        ymajor_locator = MultipleLocator(base=ymaxstep) # this locator puts ticks at regular intervals
        ax.yaxis.set_major_locator(ymajor_locator)
    #if not do_concat:
        #xminor_locator = AutoMinorLocator(5)
        #yminor_locator = AutoMinorLocator(4)
        #ax.yaxis.set_minor_locator(yminor_locator)
        #ax.xaxis.set_minor_locator(xminor_locator)
    
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
                  do_index=False, do_moc=False, do_dmoc=None, do_hbstf=False, boxidx=0):
    #___________________________________________________________________________
    # set up color info 
    if cinfo is None: cinfo=dict()
    else            : cinfo=cinfo.copy()
    do_cweights=None
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
            if do_index: vname = list(data_ii[boxidx].keys())
            else       : vname = list(data_ii.keys())
            
            #___________________________________________________________________
            if do_vec==False:
                if   do_index: data_plot = data_ii[boxidx][ vname[0] ].data.copy()
                elif do_moc  : data_plot = data_ii['zmoc'].isel(nz=np.abs(data_ii['depth'])>=700).values.copy()
                elif do_dmoc is not None  : 
                    if   do_dmoc=='dmoc'  : data_plot = data_ii['dmoc'].data.copy()
                    elif do_dmoc=='srf'   : data_plot = -(data_ii['dmoc_fh'].data.copy()+data_ii['dmoc_fw'].data.copy()+data_ii['dmoc_fr'].data.copy())
                    elif do_dmoc=='inner' : data_plot = data_ii['dmoc'].data.copy() + \
                                                        (data_ii['dmoc_fh'].data.copy()+data_ii['dmoc_fw'].data.copy()+data_ii['dmoc_fr'].data.copy())
                elif do_hbstf: data_plot = data_ii[ vname[0] ].data.copy() 
                else         : 
                    data_plot   = data_ii[ vname[0] ].data.copy()
                    if cinfo['chist']: do_cweights = data_ii['w_A'].data.copy()
            else:
                # compute norm when vector data
                data_plot = np.sqrt(data_ii[ vname[0] ].data.copy()**2 + data_ii[ vname[1] ].data.copy()**2)
                cinfo['chist']: do_cweights = data_ii['w_A'].data.copy()
            
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
                dez = 0
                if cref==0.0: 
                    cinfo['cref'] = cref
                else:
                    while True:
                        new_cref = np.around(cref, -np.int32(np.floor(np.log10(np.abs(cref)))-dez) )
                        #print(cref, new_cref, cinfo['cmin'], cinfo['cmax'])
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
    if cmin==cmax: 
        cmax        = bin_m[np.where(np.cumsum(hist)        >=ctresh)[0][1]]
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
    elif proj=='channel':
            
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
        
    elif proj in ['rob', 'mol', 'eqearth']:
        ax.gridlines(color='black', linestyle='-', alpha=0.25, xlocs=xticks, ylocs=yticks,
                     draw_labels=False)
    
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
    #idx_cref = np.asscalar(idx_cref)
    idx_cref = idx_cref.item()
    
    nstep = ncbar_l/cbar_nl
    nstep = np.max([np.int32(np.floor(nstep)),1])
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
    
    if len(clocs)>=48: cbar.dividers.set_color('None')
    
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
        #cbar.ax.minorticks_off()
        cbar.update_ticks()
        cbar.ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
        cbar.ax.yaxis.get_offset_text().set(horizontalalignment='right')
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
def do_ticksteps(mesh, box, ticknr=6):
    #___________________________________________________________________________
    tickstep = np.array([0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 360])

    #___________________________________________________________________________
    idx     = int(np.argmin(np.abs( tickstep-(box[1]-box[0])/ticknr )))
    if np.abs(box[1]-box[0])==360:
        xticks   = np.arange(-180+mesh.focus, 180+1, tickstep[idx])
    else:    
        xticks   = np.arange(box[0], box[1]+tickstep[idx], tickstep[idx]) 
    idx     = int(np.argmin(np.abs( tickstep-(box[3]-box[2])/ticknr )))
    if np.abs(box[3]-box[2])==180:
        yticks   = np.arange(-90, 90+1, tickstep[idx])  
    else:   
        yticks   = np.arange(box[2], box[3]+tickstep[idx], tickstep[idx])  
        
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
def do_savefigure(do_save, fig, dpi=300, transparent=False, pad_inches=0.1, do_info=True, **kw):
    if do_save is not None:
        #_______________________________________________________________________
        # extract filename from do_save
        sfname = os.path.basename(do_save)
        
        # extract file extensions
        sfformat = os.path.splitext(sfname)[1][1:]
        
        # extract dirname from do_save --> create directory if not exist
        sdname = os.path.dirname(do_save)
        sdname = os.path.expanduser(sdname)
        if do_info: print(' > save figure: {}'.format(os.path.join(sdname,sfname)))
        
        #_______________________________________________________________________
        if not os.path.isdir(sdname): os.makedirs(sdname)
        
        #_______________________________________________________________________
        fig.savefig(os.path.join(sdname,sfname), format=sfformat, dpi=dpi, 
                    bbox_inches='tight', pad_inches=pad_inches,\
                    transparent=transparent, **kw)




def set_cinfo(cstr, cnum, crange, cmin, cmax, cref, cfac, climit, chist, ctresh):
    cinfo=dict()   
    if cstr     is not None: cinfo['cstr'  ]=cstr
    if cnum     is not None: cinfo['cnum'  ]=cnum
    if crange   is not None: cinfo['crange']=crange
    if cmin     is not None: cinfo['cmin'  ]=cmin
    if cmax     is not None: cinfo['cmax'  ]=cmax
    if cref     is not None: cinfo['cref'  ]=cref
    if cfac     is not None: cinfo['cfac'  ]=cfac
    if climit   is not None: cinfo['climit']=climit    
    if chist    is not None: cinfo['chist' ]=chist
    if ctresh   is not None: cinfo['ctresh']=ctresh
    return(cinfo)
    
    
#
#
#_______________________________________________________________________________
# --> this based on work of Nils Bruegemann see 
# https://gitlab.dkrz.de/m300602/pyicon/-/blob/master/pyicon/pyicon_plotting.py
# i needed this to unify the ploting between icon and fesom for model comparison
# paper
def arrange_axes(nx, ny,
                 sharex = True, sharey = False,
                 xlabel = '', ylabel = '',
                 # labeling axes with e.g. (a), (b), (c)
                 do_axes_labels = True,
                 axlab_kw = dict(),
                 # colorbar
                 plot_cb = True,
                 # projection (e.g. for cartopy)
                 projection = None,
                 # aspect ratio of axes
                 asp = 1.,
                 sasp = 0.,  # for compability with older version of arrange_axes
                 # width and height of axes
                 wax = 'auto', hax = 4.,
                 # extra figure spaces (left, right, top, bottom)
                 dfigl= 0.0, dfigr=0.0, dfigt=0.0, dfigb=0.0,
                 # space aroung axes (left, right, top, bottom) 
                 daxl = 1.8, daxr =0.8, daxt =0.8, daxb =1.2, 
                 # space around colorbars (left, right, top, bottom) 
                 dcbl =-0.5, dcbr =1.4, dcbt =0.0, dcbb =0.5,
                 # width and height of colorbars
                 wcb = 0.5, hcb = 'auto',
                 # factors to increase widths and heights of axes and colorbars
                 fig_size_fac = 1.,
                 f_wax  =1., f_hax  =1., f_wcb  =1., f_hcb  =1.,
                 # factors to increase spaces (figure)
                 f_dfigl=1., f_dfigr=1., f_dfigt=1., f_dfigb=1.,
                 # factors to increase spaces (axes)
                 f_daxl =1., f_daxr =1., f_daxt =1., f_daxb =1.,
                 # factors to increase spaces (colorbars)
                 f_dcbl =1., f_dcbr =1., f_dcbt =1., f_dcbb =1.,
                 # font sizes of labels, titles, ticks
                 fs_label = 10., fs_title = 12., fs_ticks = 10.,
                 # font size increasing factor
                 f_fs = 1,
                 reverse_order = False,
                 nargout=['fig', 'hca', 'hcb'],
                ):

    # factor to convert cm into inch
    cm2inch = 0.3937

    if sasp!=0:
        print('::: Warning: You are using keyword ``sasp`` for setting the aspect ratio but you should switch to use ``asp`` instead.:::')
        asp = 1.*sasp

    # --- set hcb in case it is auto
    if isinstance(wax, str) and wax=='auto': wax = hax/asp

    # --- set hcb in case it is auto
    if isinstance(hcb, str) and hcb=='auto': hcb = hax

    # --- rename horizontal->bottom and vertical->right
    if isinstance(plot_cb, str) and plot_cb=='horizontal': plot_cb = 'bottom'
    if isinstance(plot_cb, str) and plot_cb=='vertical'  : plot_cb = 'right'
  
    # --- apply fig_size_fac
    # font sizes
    #f_fs *= fig_size_fac
    # factors to increase widths and heights of axes and colorbars
    f_wax *= fig_size_fac
    f_hax *= fig_size_fac
    #f_wcb *= fig_size_fac
    f_hcb *= fig_size_fac
    ## factors to increase spaces (figure)
    #f_dfigl *= fig_size_fac
    #f_dfigr *= fig_size_fac
    #f_dfigt *= fig_size_fac
    #f_dfigb *= fig_size_fac
    ## factors to increase spaces (axes)
    #f_daxl *= fig_size_fac
    #f_daxr *= fig_size_fac
    #f_daxt *= fig_size_fac
    #f_daxb *= fig_size_fac
    ## factors to increase spaces (colorbars)
    #f_dcbl *= fig_size_fac
    #f_dcbr *= fig_size_fac
    #f_dcbt *= fig_size_fac
    #f_dcbb *= fig_size_fac
  
    # --- apply font size factor
    fs_label *= f_fs
    fs_title *= f_fs
    fs_ticks *= f_fs

    # make vector of plot_cb if it has been true or false before
    # plot_cb can have values [{1}, 0] 
    # with meanings:
    #   1: plot cb; 
    #   0: do not plot cb
    plot_cb_right  = False
    plot_cb_bottom = False
    if isinstance(plot_cb, bool) and (plot_cb==True):
        plot_cb = np.ones((nx,ny))  
    elif isinstance(plot_cb, bool) and (plot_cb==False):
        plot_cb = np.zeros((nx,ny))
    elif isinstance(plot_cb, str) and plot_cb=='right':
        plot_cb = np.zeros((nx,ny))
        plot_cb_right = True
    elif isinstance(plot_cb, str) and plot_cb=='bottom':
        plot_cb = np.zeros((nx,ny))
        plot_cb_bottom = True
    else:
        plot_cb = np.array(plot_cb)
        if plot_cb.size!=nx*ny    : raise ValueError('Vector plot_cb has wrong length!')
        if plot_cb.shape[0]==nx*ny: plot_cb = plot_cb.reshape(ny,nx).transpose()
        elif plot_cb.shape[0]==ny : plot_cb = plot_cb.transpose()
  
    # --- make list of projections if it is not a list
    if not isinstance(projection, list): projection = [projection]*nx*ny
    
    # --- make arrays and multiply by f_*
    daxl = np.array([daxl]*nx)*f_daxl
    daxr = np.array([daxr]*nx)*f_daxr
    dcbl = np.array([dcbl]*nx)*f_dcbl
    dcbr = np.array([dcbr]*nx)*f_dcbr
    
    wax  = np.array([wax]*nx)*f_wax
    wcb  = np.array([wcb]*nx)*f_wcb
    
    daxt = np.array([daxt]*ny)*f_daxt
    daxb = np.array([daxb]*ny)*f_daxb
    dcbt = np.array([dcbt]*ny)*f_dcbt
    dcbb = np.array([dcbb]*ny)*f_dcbb
    
    hax  = np.array([hax]*ny)*f_hax
    hcb  = np.array([hcb]*ny)*f_hcb
  
    # --- adjust for shared axes
    if sharex: daxb[:-1] = 0.
    
    if sharey: daxl[1:] = 0.

    # --- adjust for one colorbar at the right or bottom
    if plot_cb_right:
        daxr_s = daxr[0]
        dcbl_s = dcbl[0]
        dcbr_s = dcbr[0]
        wcb_s  = wcb[0]
        hcb_s  = hcb[0]
        dfigr += dcbl_s+wcb_s+0.*dcbr_s+daxl[0]
    if plot_cb_bottom:
        hcb_s  = wcb[0]
        wcb_s  = wax[0]
        dcbb_s = dcbb[0]+daxb[-1]
        dcbt_s = dcbt[0]
        #hcb_s  = hcb[0]
        dfigb += dcbb_s+hcb_s+dcbt_s
  
    # --- adjust for columns without colorbar
    delete_cb_space = plot_cb.sum(axis=1)==0
    dcbl[delete_cb_space] = 0.0
    dcbr[delete_cb_space] = 0.0
    wcb[delete_cb_space]  = 0.0
    
    # --- determine ax position and fig dimensions
    x0 =   dfigl
    y0 = -(dfigt)
    
    pos_axcm = np.zeros((nx*ny,4))
    pos_cbcm = np.zeros((nx*ny,4))
    nn = -1
    y00 = y0
    x00 = x0
    for jj in range(ny):
        y0 += -(daxt[jj]+hax[jj])
        x0 = x00
        for ii in range(nx):
            nn += 1
            x0   += daxl[ii]
            pos_axcm[nn,:] = [x0, y0, wax[ii], hax[jj]]
            pos_cbcm[nn,:] = [x0+wax[ii]+daxr[ii]+dcbl[ii], y0, wcb[ii], hcb[jj]]
            x0   += wax[ii]+daxr[ii]+dcbl[ii]+wcb[ii]+dcbr[ii]
        y0   += -(daxb[jj])
    wfig = x0+dfigr
    hfig = y0-dfigb
  
    # --- transform from negative y axis to positive y axis
    hfig = -hfig
    pos_axcm[:,1] += hfig
    pos_cbcm[:,1] += hfig
    
    # --- convert to fig coords
    cm2fig_x = 1./wfig
    cm2fig_y = 1./hfig
    
    pos_ax = 1.*pos_axcm
    pos_cb = 1.*pos_cbcm
    
    pos_ax[:,0] = pos_axcm[:,0]*cm2fig_x
    pos_ax[:,2] = pos_axcm[:,2]*cm2fig_x
    pos_ax[:,1] = pos_axcm[:,1]*cm2fig_y
    pos_ax[:,3] = pos_axcm[:,3]*cm2fig_y
    
    pos_cb[:,0] = pos_cbcm[:,0]*cm2fig_x
    pos_cb[:,2] = pos_cbcm[:,2]*cm2fig_x
    pos_cb[:,1] = pos_cbcm[:,1]*cm2fig_y
    pos_cb[:,3] = pos_cbcm[:,3]*cm2fig_y

    # --- find axes center (!= figure center)
    x_ax_cent = pos_axcm[0,0] +0.5*(pos_axcm[-1,0]+pos_axcm[-1,2]-pos_axcm[0,0])
    y_ax_cent = pos_axcm[-1,1]+0.5*(pos_axcm[0,1] +pos_axcm[0,3] -pos_axcm[-1,1])
    
    # --- make figure and axes
    fig = plt.figure(figsize=(wfig*cm2inch, hfig*cm2inch), facecolor='white')
  
    hca = [0]*(nx*ny)
    hcb = [0]*(nx*ny)
    nn = -1
    for jj in range(ny):
        for ii in range(nx):
            nn+=1
            
            # --- axes
            hca[nn] = fig.add_subplot(position=pos_ax[nn,:], projection=projection[nn])
            hca[nn].set_position(pos_ax[nn,:])
            
            # --- colorbar
            if plot_cb[ii,jj] == 1:
                hcb[nn] = fig.add_subplot(position=pos_cb[nn,:])
                hcb[nn].set_position(pos_cb[nn,:])
            ax  = hca[nn]
            cax = hcb[nn] 
            
            # --- label
            ax.set_xlabel(xlabel, fontsize=fs_label)
            ax.set_ylabel(ylabel, fontsize=fs_label)
            #ax.set_title('', fontsize=fs_title)
            matplotlib.rcParams['axes.titlesize'] = fs_title
            ax.tick_params(labelsize=fs_ticks)
            if plot_cb[ii,jj] == 1:
                hcb[nn].tick_params(labelsize=fs_ticks)
            
            #ax.tick_params(pad=-10.0)
            #ax.xaxis.labelpad = 0
            #ax._set_title_offset_trans(float(-20))
            
            # --- axes ticks
            # delete labels for shared axes
            if sharex and jj!=ny-1:
                hca[nn].ticklabel_format(axis='x',style='plain',useOffset=False)
                hca[nn].tick_params(labelbottom=False)
                hca[nn].set_xlabel('')
            
            if sharey and ii!=0:
                hca[nn].ticklabel_format(axis='y',style='plain',useOffset=False)
                hca[nn].tick_params(labelleft=False)
                hca[nn].set_ylabel('')
            
            # ticks for colorbar 
            if plot_cb[ii,jj] == 1:
                hcb[nn].set_xticks([])
                hcb[nn].yaxis.tick_right()
                hcb[nn].yaxis.set_label_position("right")

    #--- needs to converted to fig coords (not cm)
    if plot_cb_right:
        nn = -1
        #pos_cb = np.array([(wfig-(dfigr+dcbr_s+wcb_s))*cm2fig_x, (y_ax_cent-0.5*hcb_s)*cm2fig_y, wcb_s*cm2fig_x, hcb_s*cm2fig_y])
        pos_cb = np.array([ (pos_axcm[-1,0]+pos_axcm[-1,2]+daxr_s+dcbl_s)*cm2fig_x, 
                            (y_ax_cent-0.5*hcb_s)*cm2fig_y, 
                            (wcb_s)*cm2fig_x, 
                            (hcb_s)*cm2fig_y 
                        ])
        hcb[nn] = fig.add_subplot(position=pos_cb)
        hcb[nn].tick_params(labelsize=fs_ticks)
        hcb[nn].set_position(pos_cb)
        hcb[nn].set_xticks([])
        hcb[nn].yaxis.tick_right()
        hcb[nn].yaxis.set_label_position("right")

    if plot_cb_bottom:
        nn = -1
        pos_cb = np.array([ (x_ax_cent-0.5*wcb_s)*cm2fig_x, 
                            (dcbb_s)*cm2fig_y, 
                            (wcb_s)*cm2fig_x, 
                            (hcb_s)*cm2fig_y
                        ])
        hcb[nn] = fig.add_subplot(position=pos_cb)
        hcb[nn].set_position(pos_cb)
        hcb[nn].tick_params(labelsize=fs_ticks)
        hcb[nn].set_yticks([])

    if reverse_order:
        isort = np.arange(nx*ny, dtype=int).reshape((ny,nx)).transpose().flatten()
        hca = list(np.array(hca)[isort]) 
        hcb = list(np.array(hcb)[isort])

    # add letters for subplots
    if (do_axes_labels) and (axlab_kw is not None):
        hca = axlab(hca, fontdict=axlab_kw)

    #___________________________________________________________________________
    list_argout=[]
    if len(nargout)>0:
        for stri in nargout:
            try:
                list_argout.append(eval(stri))
            except:
                print(f" -warning-> variable {stri} was not found, could not be added as output argument") 
            
        return(list_argout)
    else:
        return
    #return fig, hca, hcb


#
#
#_______________________________________________________________________________
# --> this based on work of Nils Bruegemann see 
# https://gitlab.dkrz.de/m300602/pyicon/-/blob/master/pyicon/pyicon_plotting.py
# i needed this to unify the ploting between icon and fesom for model comparison
# paper    
def axlab(hca, figstr=[], posx=[-0.00], posy=[1.05], fontdict=None):
  """
input:
----------
  hca:      list with axes handles
  figstr:   list with strings that label the subplots
  posx:     list with length 1 or len(hca) that gives the x-coordinate in ax-space
  posy:     list with length 1 or len(hca) that gives the y-coordinate in ax-space
last change:
----------
2015-07-21
  """

  # make list that looks like [ '(a)', '(b)', '(c)', ... ]
  if len(figstr)==0:
    #lett = "abcdefghijklmnopqrstuvwxyz"
    lett  = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
    lett += ["a2","b2","c2","d2","e2","f2","g2","h2","i2","j2","k2","l2","m2","n2","o2","p2","q2","r2","s2","t2","u2","v2","w2","x2","y2","z2"]
    lett = lett[0:len(hca)]
    figstr = ["z"]*len(hca)
    for nn, ax in enumerate(hca):
      figstr[nn] = "(%s)" % (lett[nn])
  
  if len(posx)==1:
    posx = posx*len(hca)
  if len(posy)==1:
    posy = posy*len(hca)
  
  # draw text
  for nn, ax in enumerate(hca):
    ht = hca[nn].text(posx[nn], posy[nn], figstr[nn], 
                      transform = hca[nn].transAxes, 
                      horizontalalignment = 'right',
                      fontdict=fontdict)
    # add text handle to axes to give possibility of changing text properties later
    # e.g. by hca[nn].axlab.set_fontsize(8)
    hca[nn].axlab = ht
#  for nn, ax in enumerate(hca):
#    #ax.set_title(figstr[nn]+'\n', loc='left', fontsize=10)
#    ax.set_title(figstr[nn], loc='left', fontsize=10)
  return hca
