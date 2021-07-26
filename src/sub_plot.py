import os
import sys
import numpy as np
import matplotlib.pylab as plt
#import matplotlib.pyplot as plt
import time as time
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
from cartopy.mpl.gridliner import Gridliner

from matplotlib.tri import Triangulation,TriAnalyzer
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sub_mesh import *
from sub_data import *
from colormap_c2c import *
import matplotlib.colors
import matplotlib.ticker as mticker
import matplotlib.path as mpath



def plot_hslice(mesh, data, cinfo=None, box=None, proj='pc', figsize=[9,4.5], 
                n_rc=[1,1], do_grid=False, 
                cbar_nl=8, cbar_orient='vertical', cbar_label=None, cbar_unit=None,
                do_lsmask='fesom', do_bottom=True, color_lsmask=[0.6, 0.6, 0.6], 
                color_bot=[0.8,0.8,0.8], do_plot='tcf', do_rescale=True, title=None,
                pos_fac=1.0, pos_gap=[0.02, 0.02]):
    
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
        e_idxbox = grid_cutbox(tri.x, tri.y, tri.triangles, box, which='hard')
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
        
    #___________________________________________________________________________
    # set up color info 
    cinfo = do_setupcinfo(cinfo, data, tri, mesh, do_rescale)
    
    #___________________________________________________________________________
    # loop over axes
    for ii in range(0,nax):
        
        #_______________________________________________________________________
        # add color for bottom bottom
        if do_bottom: ax[ii].background_patch.set_facecolor(color_bot)
        
        #_______________________________________________________________________
        # set axes extent
        ax[ii].set_extent(box, crs=ccrs.PlateCarree())
        
        #_______________________________________________________________________
        # periodic augment data
        vname = list(data[ii].keys())
        data_plot = data[ii][ vname[0] ].data
        data_plot, str_rescale= do_rescale_data(data_plot, do_rescale)
        data_plot = np.hstack((data_plot,data_plot[mesh.n_pbnd_a]))
        
        #_______________________________________________________________________
        # kick out triangles with Nan cut elements to box size        
        isnan   = np.isnan(data_plot)
        e_idxok = np.any(isnan[tri.triangles], axis=1)==False
        
        #_______________________________________________________________________
        # plot tri contourf/tripcolor
        if   do_plot=='tpc':
            hp=ax[ii].tripcolor(tri.x, tri.y, tri.triangles[e_idxok,:], data_plot,
                                transform=which_transf,
                                shading='flat',
                                cmap=cinfo['cmap'],
                                vmin=cinfo['clevel'][0], vmax=cinfo['clevel'][ -1])
        elif do_plot=='tcf': 
            data_plot[data_plot<cinfo['clevel'][ 0]] = cinfo['clevel'][ 0]
            data_plot[data_plot>cinfo['clevel'][-1]] = cinfo['clevel'][-1]
            hp=ax[ii].tricontourf(tri.x, tri.y, tri.triangles[e_idxok,:], data_plot, 
                                transform=which_transf, 
                                levels=cinfo['clevel'], cmap=cinfo['cmap'], extend='both')
        
        
        #_______________________________________________________________________
        # add grid mesh on top
        if do_grid: ax[ii].triplot(tri.x, tri.y, tri.triangles[e_idxok,:], 
                                   color='k', linewidth=0.2, alpha=0.75) 
                                   #transform=which_transf)
        
        #_______________________________________________________________________
        # add mesh land-sea mask
        ax[ii] = do_plotlsmask(ax[ii],mesh, do_lsmask, box, which_proj,
                               color_lsmask=color_lsmask, edgecolor='k', linewidth=0.5)
        
        #_______________________________________________________________________
        # add gridlines
        ax[ii] = do_add_gridlines(ax[ii], rowlist[ii], collist[ii], n_rc[0], proj, 
                                  xticks, yticks, which_proj)
       
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
    for jj in range(nax_fin, nax): fig.delaxes(ax[jj])
    
    #___________________________________________________________________________
    # delete axes that are not needed
    cbar = fig.colorbar(hp, orientation=cbar_orient, ax=ax, ticks=cinfo['clevel'], 
                      extend='neither',extendrect=False, extendfrac=None,
                      drawedges=True, pad=0.025, shrink=1.0)
    cbar.ax.tick_params(labelsize=fontsize)
    
    if cbar_label is None: cbar_label = data[nax_fin-1][ vname[0] ].attrs['long_name']
    if str_rescale is not None: cbar_label = cbar_label+str_rescale  
    if cbar_unit  is None: cbar_label = cbar_label+' ['+data[nax_fin-1][ vname[0] ].attrs['units']+']'
    else:                  cbar_label = cbar_label+' ['+cbar_unit+']'
    cbar.set_label(cbar_label, size=fontsize+2)
    
    #___________________________________________________________________________
    # kickout some colormap labels if there are to many
    cbar = do_cbar_label(cbar, cbar_nl, cinfo)
    
    #___________________________________________________________________________
    # repositioning of axes and colorbar
    ax, cbar = do_reposition_ax_cbar(ax, cbar, rowlist, collist, pos_fac, pos_gap, title=title)
    
    #___________________________________________________________________________
    return(fig, ax, which_proj, cbar)
    
    
    
def do_rescale_data(data,do_rescale):
    #___________________________________________________________________________
    # cutoff exponentials --> add therefore string to unit parameter
    str_rescale=None
    if do_rescale==True:
        if np.nanmax(np.abs(data))<1e-2 and np.nanmax(np.abs(data))>0.0:
            scal = 10**(np.floor(np.log10(max(abs(np.nanmin(data)),abs(np.nanmax(data))))-1))
            data = data/scal
            str_rescale  = ' $ \cdot 10^{'+str(int(np.log10(scal)))+'} $'
        elif np.nanmax(np.abs(data))>1.0e4:
            scal = 10**(np.floor(np.log10(max(abs(np.nanmin(data)),abs(np.nanmax(data))))-1))
            data = data/scal
            str_rescale  = ' $ \cdot 10^{'+str(int(np.log10(scal)))+'} $'
            
    elif do_rescale=='log10':
        data[data!=0.0] = np.log10(data[data!=0.0])
        data.rescale='log10'
        str_rescale  = ' log10() '
    
    #___________________________________________________________________________
    return(data,str_rescale)
    
    

def do_setupcinfo(cinfo, data, tri, mesh, do_rescale):
    #___________________________________________________________________________
    # set up color info 
    if cinfo is None: cinfo=dict()
    
    # check if dictionary keys exist, if they do not exist fill them up 
    if (('cmin' not in cinfo.keys()) or ('cmax' not in cinfo.keys())) and ('crange' not in cinfo.keys()):
        cmin, cmax = np.Inf, -np.Inf
        for data_ii in data:
            vname = list(data_ii.keys())
            data_plot = data_ii[ vname[0] ].data.copy()
            data_plot, str_rescale= do_rescale_data(data_plot, do_rescale)
            data_plot = np.hstack((data_plot,data_plot[mesh.n_pbnd_a]))
            cmin = np.min([cmin,np.nanmin(data_plot[tri.triangles.flatten()]) ])
            cmax = np.max([cmax,np.nanmax(data_plot[tri.triangles.flatten()]) ])
            
        if 'cmin' not in cinfo.keys(): cinfo['cmin'] = cmin
        if 'cmax' not in cinfo.keys(): cinfo['cmax'] = cmax    
    if 'crange' in cinfo.keys():
        cinfo['cmin'], cinfo['cmax'], cinfo['cref'] = cinfo['crange'][0], cinfo['crange'][1], cinfo['crange'][2]
        
    else:
        if (cinfo['cmin'] == cinfo['cmax'] ): raise ValueError (' --> can\'t plot! data are everywhere: {}'.format(str(cinfo['cmin'])))
        cref = cinfo['cmin'] + (cinfo['cmax']-cinfo['cmin'])/2
        if 'cref' not in cinfo.keys(): cinfo['cref'] = np.around(cref, -np.int32(np.floor(np.log10(np.abs(cref)))-1) )
        
    if 'cnum' not in cinfo.keys(): cinfo['cnum'] = 20
    if 'cstr' not in cinfo.keys(): cinfo['cstr'] = 'wbgyr'
    cinfo['cmap'],cinfo['clevel'] = colormap_c2c(cinfo['cmin'],cinfo['cmax'],cinfo['cref'],cinfo['cnum'],cinfo['cstr'])

    #___________________________________________________________________________
    return(cinfo)
    
    

def do_plotlsmask(ax, mesh, do_lsmask, box, which_proj, color_lsmask=[0.6, 0.6, 0.6], edgecolor='k', linewidth=0.5):
    #___________________________________________________________________________
    # add mesh land-sea mask
    if   do_lsmask=='fesom':
        ax.add_geometries(mesh.lsmask_p, crs=ccrs.PlateCarree(), 
                              facecolor=color_lsmask, edgecolor=edgecolor ,linewidth=linewidth)
        
    elif do_lsmask=='stock':   
        ax.stock_img()
        ax.add_geometries(mesh.lsmask_p, crs=ccrs.PlateCarree(), 
                              facecolor='None', edgecolor=edgecolor, linewidth=linewidth)
            
    elif do_lsmask=='bluemarble':   
        img = plt.imread('src/bluemarble.png')
        ax.imshow(img, origin='upper', extent=box, transform=which_proj)
        ax.add_geometries(mesh.lsmask_p, crs=ccrs.PlateCarree(), 
                              facecolor='None', edgecolor=edgecolor, linewidth=linewidth)
            
    elif do_lsmask=='etopo':   
        img = plt.imread('src/etopo1.png')
        ax.imshow(img, origin='upper', extent=box, transform=which_proj)
        ax.add_geometries(mesh.lsmask_p, crs=ccrs.PlateCarree(), 
                              facecolor='None', edgecolor=edgecolor, linewidth=linewidth)
            
    else:
        raise ValueError(" > the do_lsmask={} is not supported, must be either 'fesom', 'stock', 'bluemarble' or 'etopo'! ")
        
    #___________________________________________________________________________
    return(ax)



def do_add_gridlines(ax, rowlist, collist, maxr, proj, xticks, yticks, which_proj):
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



def do_cbar_label(cbar, cbar_nl, cinfo):
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
    idx = idx[idxb==True]
    for ii in list(idx): tickl[ii]=''
    if cbar.orientation=='vertical': cbar.ax.set_yticklabels(tickl)
    else:                            cbar.ax.set_xticklabels(tickl)
    
    #___________________________________________________________________________
    return(cbar)


def do_reposition_ax_cbar(ax, cbar, rowlist, collist, pos_fac, pos_gap, title=None):
    #___________________________________________________________________________
    # repositioning of axes and colorbar
    nax = len(ax)
    ax_pos = np.zeros((nax,4))
    for jj in range(0,nax):
        aux = ax[jj].get_position()
        ax_pos[jj,:] = np.array([aux.x0, aux.y0, aux.width, aux.height])
    
    fac = pos_fac
    #x0, y0, x1, y1 = 0.05, 0.05, 0.95, 0.95
    x0, y0, x1, y1 = 0.1, 0.1, 0.9, 0.9
    if cbar.orientation=='horizontal': y0 = 0.21
    w, h = ax_pos[:,2].min(), ax_pos[:,3].min()
    w, h = w*fac, h*fac
    wg, hg = pos_gap[0], pos_gap[1]
    if title is not None: hg = hg+0.04
    
    maxr = rowlist.max()+1
    maxc = collist.max()+1
    
    for jj in range(nax-1,0-1,-1):
        ax[jj].set_position( [x0+(w+wg)*collist[jj], y0+(h+hg)*np.abs(rowlist[jj]-maxr+1), w, h] )
    
    cbar_pos = cbar.ax.get_position()
    if cbar.orientation=='vertical': 
        cbar.ax.set_position([x0+(w+wg)*maxc, y0, cbar_pos.width*0.75, h*maxr+hg*(maxr-1)])
        cbar.ax.set_aspect('auto')
    else: 
        cbar.ax.set_position([x0, 0.11, w*maxc+wg*(maxc-1), cbar_pos.height/2])
        cbar.ax.set_aspect('auto')
        
    #___________________________________________________________________________
    return(ax, cbar)    



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
