# Patrick Scholz, 23.01.2018
import sys
import os
import numpy as np
import copy as  cp
from   shapely.geometry   import Point, Polygon, MultiPolygon, shape
from   shapely.vectorized import contains
import shapefile as shp
import json
import geopandas as gpd
import matplotlib.pylab as plt
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from   .sub_mesh           import * 
from   .sub_data           import *
from   .sub_plot           import *
from   .colormap_c2c       import *


def load_index_fesom2(mesh, data, box_list, boxname=None, do_harithm='median', 
                      do_compute=True, do_outputidx=False):
    
    #___________________________________________________________________________
    # str_anod    = ''
    index_list  = []
    idxin_list  = []
    cnt         = 0
    
    #___________________________________________________________________________
    # loop over box_list
    for box in box_list:
        
        if not isinstance(box, shp.Reader):
            if len(box)==2: boxname, box = box[1], box[0]
            if box is None or box=='global': boxname='global'
        
        #_______________________________________________________________________
        # compute  mask index
        idx_IN=do_boxmask(mesh,box)
        
        #_______________________________________________________________________
        # selected points in xarray dataset object and  average over selected 
        # points
        index_list.append( do_horiz_arithmetic(data.sel(nod2=idx_IN), do_harithm) )
        idxin_list.append(idx_IN)
        
        #_______________________________________________________________________
        if do_compute: index_list[cnt] = index_list[cnt].compute()
        
        #_______________________________________________________________________
        vname = list(index_list[cnt].keys())            
        if boxname is not None: 
            index_list[cnt][vname[0]].attrs['boxname'] = boxname
        elif isinstance(box, shp.Reader):
            index_list[cnt][vname[0]].attrs['boxname'] = os.path.basename(box.shapeName).replace('_',' ')  
        elif boxname is None or boxname=='global': 
            index_list[cnt][vname[0]].attrs['boxname'] = 'global'
        
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
def do_boxmask(mesh, box, do_elem=False):
    #___________________________________________________________________________
    if do_elem: mesh_x, mesh_y = mesh.n_x[mesh.e_i].sum(axis=1)/3.0, mesh.n_y[mesh.e_i].sum(axis=1)/3.0
    else      : mesh_x, mesh_y = mesh.n_x, mesh.n_y
    
    #___________________________________________________________________________
    # a rectangular box is given --> translate into shapefile object
    if  box is None or box is 'global': # if None do global
        idx_IN = np.ones((mesh_x.shape),dtype=bool)
        
    elif  (isinstance(box,list) or isinstance(box, np.ndarray)) and len(box)==4: 
        px     = [box[0], box[1], box[1], box[0], box[0]]
        py     = [box[2], box[2], box[3], box[3], box[2]]
        p      = Polygon(list(zip(px,py)))
        idx_IN = contains(p, mesh_x, mesh_y)
            
    # a polygon as list or ndarray is given --> translate into shape file object
    elif isinstance(box,list) and len(box)==2: 
        px, py = box[0], box[1]  
        p      = Polygon(list(zip(px,py)))  
        idx_IN = contains(p, mesh_x, mesh_y)
            
    elif isinstance(box, np.ndarray): 
        if box.shape[0]==2:
            px, py = list(box[0,:]), list(box[1,:])
            p      = Polygon(list(zip(px,py)))
            idx_IN = contains(p, mesh_x, mesh_y)
                
        else:
            raise  ValueError(' ndarray box has wrong format must be [2 x npts], yours is {}'.format(str(box.shape)))
            
    # a polygon as shapefile or shapefile collection is given
    elif (isinstance(box, (Polygon, MultiPolygon))):
        if   isinstance(box, Polygon): 
            idx_IN = contains(box, mesh_x, mesh_y)
                
        elif isinstance(box, MultiPolygon):
            idx_IN = np.zeros((mesh.n2dn,), dtype=bool)
            for p in box:
                auxidx = contains(p, mesh_x, mesh_y)
                idx_IN = np.logical_or(idx_IN, auxidx)
        
    elif (isinstance(box, shp.Reader)):
        idx_IN = np.zeros((mesh.n2dn,), dtype=bool)
        for shape in box.shapes(): 
            p      = Polygon(shape.points)
            auxidx = contains(p, mesh_x, mesh_y)
            idx_IN = np.logical_or(idx_IN, auxidx)
    # otherwise
    else:
        raise ValueError('the given box information to compute the index has no valid format')
        
    #___________________________________________________________________________
    return(idx_IN)

#
#
#_______________________________________________________________________________
def do_horiz_arithmetic(data, do_harithm):
    if do_harithm is not None:
        
        #_______________________________________________________________________
        if   do_harithm=='mean':
            data = data.mean(  dim="nod2", keep_attrs=True, skipna=True)
        elif do_harithm=='median':
            data = data.median(dim="nod2", keep_attrs=True, skipna=True)
        elif do_harithm=='std':
            data = data.std(   dim="nod2", keep_attrs=True, skipna=True) 
        elif do_harithm=='var':
            data = data.var(   dim="nod2", keep_attrs=True, skipna=True)       
        elif do_harithm=='max':
            data = data.max(   dim="nod2", keep_attrs=True, skipna=True)
        elif do_harithm=='min':
            data = data.min(   dim="nod2", keep_attrs=True, skipna=True)  
        elif do_harithm=='sum':
            data = data.sum(   dim="nod2", keep_attrs=True, skipna=True)      
        elif do_harithm=='None':
            ...
        else:
            raise ValueError(' the time arithmetic of do_tarithm={} is not supported'.format(str(do_tarithm))) 
    
    #___________________________________________________________________________
    return(data)

#
#
#_______________________________________________________________________________
def convert_geojson2shp(geojsonfilepath, shppath, do_plot=False):
    with open(geojsonfilepath) as f:
        #_______________________________________________________________________
        # loop over features in geojson file 
        features = json.load(f)["features"]
        for feature in features:
            
            #___________________________________________________________________
            # get geometry object 
            #geom = shapely.geometry.shape(feature["geometry"])
            geom = shape(feature["geometry"])
            
            #___________________________________________________________________
            # get name of geometry object 
            name = feature["properties"]["name"]
            name_sav = name.replace(' ','_')
            #___________________________________________________________________
            # initialse geopanda DataFrame 
            sf = gpd.GeoDataFrame()
            
            #___________________________________________________________________
            # write into geopanda DataFrame 
            if isinstance(geom, MultiPolygon):
                polygon = list(geom)
                for jj in range(0,len(polygon)):
                    sf.loc[jj,'geometry'] = Polygon(np.transpose(polygon[jj].exterior.xy))
                    sf.loc[jj,'location'] = '{}'.format(str(name))
                    
            else:
                sf.loc[0,'geometry'] = Polygon(np.transpose(geom.exterior.xy))
                sf.loc[0,'location'] = '{}'.format(str(name))
            
            #___________________________________________________________________
            # save geopanda DataFrame into shape file 
            shpfname = name_sav+'.shp'
            sf.to_file(os.path.join(shppath, shpfname))
    #___________________________________________________________________________
    if do_plot: sf.plot()
    #___________________________________________________________________________
    return

#
#
#_______________________________________________________________________________
def convert_box2shp(boxlist, boxnamelist, shppath):
    #___________________________________________________________________________
    # if boxlistname is list write one shapefile for each defined box, 
    if isinstance(boxnamelist, list):
        for box, boxname in zip(boxlist, boxnamelist):
            #___________________________________________________________________
            # a rectangular box is given --> translate into shapefile object
            if  (isinstance(box,list) or isinstance(box, np.ndarray)) and len(box)==4: 
                px     = [box[0], box[1], box[1], box[0], box[0]]
                py     = [box[2], box[2], box[3], box[3], box[2]]
                
            # a polygon as list or ndarray is given --> translate into shape file object
            elif isinstance(box,list) and len(box)==2: 
                px, py = box[0], box[1]  
                
            elif isinstance(box, np.ndarray): 
                if box.shape[0]==2:
                    px, py = list(box[0,:]), list(box[1,:])
                    
                else:
                    raise  ValueError(' ndarray box has wrong format must be [2 x npts], yours is {}'.format(str(box.shape)))
         
            #___________________________________________________________________
            name_sav = boxname.replace(' ','_')
            
            #___________________________________________________________________
            # initialse geopanda DataFrame 
            sf = gpd.GeoDataFrame()
            
            #___________________________________________________________________
            sf.loc[0,'geometry'] = Polygon(list(zip(px,py)))
            sf.loc[0,'location'] = '{}'.format(str(boxname))
            
            #___________________________________________________________________
            # save geopanda DataFrame into shape file 
            shpfname = name_sav+'.shp'
            sf.to_file(os.path.join(shppath, shpfname))
    
    #___________________________________________________________________________
    # elseif boxlistname is one string write all boxes in single shape shapefile
    else:    
        #_______________________________________________________________________
        name_sav = boxnamelist.replace(' ','_')
            
        #_______________________________________________________________________
        # initialse geopanda DataFrame 
        sf = gpd.GeoDataFrame()
        
        #_______________________________________________________________________
        for ii, box in enumerate(boxlist):
            
            #___________________________________________________________________
            # a rectangular box is given --> translate into shapefile object
            if  (isinstance(box,list) or isinstance(box, np.ndarray)) and len(box)==4: 
                px, py = [box[0], box[1], box[1], box[0], box[0]], [box[2], box[2], box[3], box[3], box[2]]
                
            # a polygon as list or ndarray is given --> translate into shape file object
            elif isinstance(box,list) and len(box)==2: 
                px, py = box[0], box[1]  
                
            elif isinstance(box, np.ndarray): 
                if box.shape[0]==2: px, py = list(box[0,:]), list(box[1,:])                    
                else: raise  ValueError(' ndarray box has wrong format must be [2 x npts], yours is {}'.format(str(box.shape)))
                
            #___________________________________________________________________
            sf.loc[ii,'geometry'] = Polygon(list(zip(px,py)))
            sf.loc[ii,'location'] = '{}-{}'.format(str(boxnamelist),str(ii))
                    
        #_______________________________________________________________________
        # save geopanda DataFrame into shape file 
        shpfname = name_sav+'.shp'
        sf.to_file(os.path.join(shppath, shpfname))
    
    #___________________________________________________________________________
    return
    
#
#
#_______________________________________________________________________________
def plot_index_region(mesh, idx_IN, box_list, which='hard'):
    from matplotlib.tri import Triangulation
    
    #___________________________________________________________________________
    # make triangulation
    tri       = Triangulation(np.hstack(( mesh.n_x, mesh.n_xa )),
                              np.hstack(( mesh.n_y, mesh.n_ya )),
                              np.vstack(( mesh.e_i[mesh.e_pbnd_0, :], mesh.e_ia ))) 
    
    #___________________________________________________________________________
    # plot basemesh
    plt.figure()
    #plt.triplot(tri,linewidth=0.2)
    
    nidx = len(idx_IN)
    for ii in range(0,nidx):
        isnan     = idx_IN[ii]
        aux_isnan = np.hstack((isnan, isnan[mesh.n_pbnd_a]))
        if   which == 'soft': isnan_tri = np.any(aux_isnan[tri.triangles], axis=1)
        elif which == 'hard': isnan_tri = np.all(aux_isnan[tri.triangles], axis=1)
        elif which == 'mid' : isnan_tri = (np.sum(aux_isnan[tri.triangles], axis=1)>1)
        plt.triplot(tri.x, tri.y, tri.triangles[isnan_tri,:], linewidth=0.2)
        
    #___________________________________________________________________________
    # loop over box_list
    for box in box_list:
        #_______________________________________________________________________
        # a rectangular box is given --> translate into shapefile object
        if  (isinstance(box,list) or isinstance(box, np.ndarray)) and len(box)==4: 
            px     = [box[0], box[1], box[1], box[0], box[0]]
            py     = [box[2], box[2], box[3], box[3], box[2]]
            p      = Polygon(list(zip(px,py)))
            plt.plot(*p.exterior.xy,color='k', linewidth=1.0)
            
        # a polygon as list or ndarray is given --> translate into shape file object
        elif isinstance(box,list) and len(box)==2: 
            px, py = box[0], box[1]  
            p      = Polygon(list(zip(px,py)))  
            plt.plot(*p.exterior.xy,color='k', linewidth=1.0)
            
        elif isinstance(box, np.ndarray): 
            if box.shape[0]==2:
                px, py = list(box[0,:]), list(box[1,:])
                p      = Polygon(list(zip(px,py)))
                plt.plot(*p.exterior.xy,color='k', linewidth=1.0)
                
            else:
                raise  ValueError(' ndarray box has wrong format must be [2 x npts], yours is {}'.format(str(box.shape)))
            
        # a polygon as shapefile or shapefile collection is given
        elif (isinstance(box, (Polygon, MultiPolygon))):
            if   isinstance(box, Polygon): plt.plot(*box.exterior.xy,color='k', linewidth=1.0)
                
            elif isinstance(box, MultiPolygon):
                for p in box: plt.plot(*p.exterior.xy,color='k', linewidth=1.0)
        
        elif (isinstance(box, shp.Reader)):
            for shape in box.shapes(): 
                p      = Polygon(shape.points)
                plt.plot(*p.exterior.xy,color='k', linewidth=1.0)
        # otherwise
        else:
            raise ValueError('the given box information to compute the index has no valid format')
            
    #___________________________________________________________________________
    return

#
#
#_______________________________________________________________________________
def plot_index_z(index_list, label_list, box_list, figsize=[12,8], n_rc=[1,1], 
                 linecolor_list=None, linestyle_list=None, cbar_label=None,
                 cbar_unit=None, do_save=None, linewidth=1.0, do_alpha=0.8, do_rescale=False):
        
    #___________________________________________________________________________
    # make matrix with row colum index to know where to put labels
    rowlist = np.zeros((n_rc[0],n_rc[1]))
    collist = np.zeros((n_rc[0],n_rc[1]))       
    for ii in range(0,n_rc[0]): rowlist[ii,:]=ii
    for ii in range(0,n_rc[1]): collist[:,ii]=ii
    rowlist = rowlist.flatten()
    collist = collist.flatten()
    
    #___________________________________________________________________________
    # create figure and axes
    fig, ax = plt.subplots(n_rc[0],n_rc[1], figsize=figsize, sharex=False, sharey=True,
                        gridspec_kw=dict(left=0.1, bottom=0.1, right=0.90, top=0.90, wspace=0.10, hspace=0.3,),
                                    )
    if isinstance(ax, np.ndarray): ax = ax.flatten()
    else:                          ax = [ax] 
    nax = len(ax)
    
    if not isinstance(index_list, list): index_list = [index_list]
    
    #___________________________________________________________________________
    nbi = len(box_list)
    ndi = len(index_list)
    # loop over boxes definitions
    for bi in range(0,len(box_list)):
        
        # loop over data
        for di in range(0,ndi):
                
            vname = list(index_list[di][bi].keys())
            val   = index_list[di][bi][vname[0]].values.copy()
            val, str_rescale= do_rescale_data(val, do_rescale)
            
            cname = list(index_list[di][bi].coords)
            dep = np.abs(index_list[di][bi].coords[cname[0]].values)
            
            linestyle = 'solid'
            if linestyle_list is not None: 
                if (not linestyle_list[di])==False: linestyle =  linestyle_list[di]
                
            if isinstance(linewidth, (list, np.ndarray)):
                lwidth = linewidth[di]
            else:
                lwidth = linewidth
        
            #ax[bi].plot(val,dep, color='k', linewidth=lwidth*1.05)
            if linecolor_list is None: 
                
                ax[bi].plot(val,dep, label=label_list[di], linestyle=linestyle, linewidth=lwidth, alpha=do_alpha)
            else:
                if isinstance(linecolor_list[di], (list, np.ndarray)):
                    if len(linecolor_list[di])==0:
                        ax[bi].plot(val,dep, label=label_list[di], linestyle=linestyle, linewidth=lwidth, alpha=do_alpha)
                    else:
                        ax[bi].plot(val,dep, color=linecolor_list[di], 
                                label=label_list[di], linestyle=linestyle, linewidth=lwidth, alpha=do_alpha)
                elif isinstance(linecolor_list[di], str):
                    ax[bi].plot(val,dep, color=linecolor_list[di], 
                                label=label_list[di], linestyle=linestyle, linewidth=lwidth, alpha=do_alpha)
                #if not linecolor_list[di]:
                    #ax[bi].plot(val,dep, label=label_list[di], linestyle=linestyle, linewidth=linewidth)
                #else:
                    
                    
        #_______________________________________________________________________
        # if bi==nbi-1 : 
        if bi==n_rc[1]-1 : 
            ax[bi].legend(loc='upper right', 
                          frameon=True, fancybox=True, shadow=True, fontsize=10, ncol=1,
                          labelspacing=1.0,
                          bbox_to_anchor=(2.3, 1.0)) #bbox_to_anchor=(1.5, 1.5))
        
        #_______________________________________________________________________
        if 'boxname' in index_list[di][bi][vname[0]].attrs.keys(): 
            str_title = index_list[di][bi][vname[0] ].attrs['boxname']
        elif (isinstance(box_list[bi], shp.Reader)): 
            str_title = os.path.basename(box_list[bi].shapeName)
        else:                                      
            str_title = '{}'.format(str(box_list[bi]))
        str_title.replace('_',' ')    
        ax[bi].set_title(str_title)
        
        #_______________________________________________________________________
        if collist[bi]==0:
            ax[bi].set_ylabel('depth [m]')
        
        #_______________________________________________________________________
        if cbar_label is None     : str_xlabel = index_list[di][bi][ vname[0] ].attrs['long_name']
        else                      : str_xlabel = cbar_label
        if str_rescale is not None: str_xlabel = str_xlabel+str_rescale  
        if cbar_unit  is None     : str_xlabel = str_xlabel+'\n ['+index_list[di][bi][ vname[0] ].attrs['units']+']'
        else                      : str_xlabel = str_xlabel+'\n ['+cbar_unit+']'    
        ax[bi].set_xlabel(str_xlabel)
        
        #_______________________________________________________________________
        if  do_rescale=='log10' and dep[0]==0: 
            ax[bi].set_ylim(dep[1],6000)
        elif dep[0]==0:     
            ax[bi].set_ylim(dep[1],6000)
        else:
            ax[bi].set_ylim(dep[0],6000)
        
        ax[bi].invert_yaxis()
        ax[bi].grid(True,which='major')
        
        ax[bi].set_yscale('log')
        ax[bi].set_yticks([5,10,25,50,100,250,500,1000,2000,4000,6000])
        ax[bi].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        
        ax[bi].xaxis.set_minor_locator(AutoMinorLocator())
    #___________________________________________________________________________
    # delete axes that are not needed
    for jj in range(bi+1, nax): fig.delaxes(ax[jj])    
        
    plt.show()
    
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, fig, )
    
    #___________________________________________________________________________
    return(fig, ax)
    

 
#+___PLOT MERIDIONAL OVERTRUNING CIRCULATION  _________________________________+
#|                                                                             |
#+_____________________________________________________________________________+
def plot_index_hovm(data, box_list, figsize=[12, 6], 
              n_rc=[1, 1], do_grid=True, cinfo=None,
              cbar_nl=8, cbar_orient='vertical', cbar_label=None, cbar_unit=None,
              do_bottom=True, color_bot=[0.6, 0.6, 0.6], 
              pos_fac=1.0, pos_gap=[0.02, 0.02], do_save=None, save_dpi=600, 
              do_contour=True, do_clabel=True, title='descript', do_rescale=False, 
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
        data_plot = data[ii][0][vname].values.copy()
        data_plot = data_plot.transpose()
        
        #_______________________________________________________________________
        # setup x-coorod and y-coord
        time      = data[ii][0]['time'].values
        if   'nz1' in data[ii][0].dims: depth = data[ii][0]['nz1' ].values
        elif 'nz'  in data[ii][0].dims: depth = data[ii][0]['nz'  ].values
        elif 'nz_1'in data[ii][0].dims: depth = data[ii][0]['nz_1'].values
        
        #_______________________________________________________________________
        # be sure there are no holes
        data_plot[data_plot<cinfo['clevel'][ 0]] = cinfo['clevel'][ 0]+np.finfo(np.float32).eps
        data_plot[data_plot>cinfo['clevel'][-1]] = cinfo['clevel'][-1]-np.finfo(np.float32).eps
        
        #_______________________________________________________________________
        # plot MOC
        hp=ax[ii].contourf(time, depth, data_plot, 
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
            
            cont=ax[ii].contour(time, depth, data_plot,
                            levels=cinfo['clevel'][idx_yes], colors='k', linewidths=[0.5],
                            norm = which_norm) #linewidths=[0.5,0.25])
            #if do_clabel: 
                #ax[ii].clabel(cont, cont.levels[np.where(cont.levels!=cinfo['cref'])], 
                            #inline=1, inline_spacing=1, fontsize=6, fmt='%1.1f Sv')
        
        #_______________________________________________________________________
        # fix color range
        for im in ax[ii].get_images(): im.set_clim(cinfo['clevel'][ 0], cinfo['clevel'][-1])
        
        #_______________________________________________________________________
        # plot grid lines 
        if do_grid: ax[ii].grid(color='k', linestyle='-', linewidth=0.25,alpha=1.0)
        
        #_______________________________________________________________________
        # set y limits
        isnotnan = np.isnan(data_plot[:,0])==False
        isnotnan = isnotnan.sum()-1
        ylim = [depth[0], depth[isnotnan]]
        if depth[0]==0: ylim[0] = depth[1]
        
        #_______________________________________________________________________
        # set description string plus x/y labels
        if title is not None: 
            txtx, txty = time[0]+(time[-1]-time[0])*0.015, ylim[1]-(ylim[1]-ylim[0])*0.05
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
        
        if 'boxname' in data[ii][0][vname].attrs.keys():
            txtx, txty, txts = time[0]+(time[-1]-time[0])*0.015, ylim[0]+(ylim[1]-ylim[0])*0.0001, data[ii][0][vname].attrs['boxname']
            ax[ii].text(txtx, txty, txts, fontsize=10, fontweight='bold', horizontalalignment='left', verticalalignment='top')
        
        if collist[ii]==0        : ax[ii].set_ylabel('Depth [m]',fontsize=12)
        if rowlist[ii]==n_rc[0]-1: ax[ii].set_xlabel('Time [years]',fontsize=12)
        
        #_______________________________________________________________________
        if do_ylog: 
            ax[ii].grid(True,which='major')
            #ax[ii].set_yscale('log')
            ax[ii].set_yscale('function', functions=(forward, inverse))
            #yticklog = np.array([5,10,25,50,100,250,500,1000,2000,4000,6000])
            yticklog = np.array([10,25,50,100,250,500,1000,2000,4000,6000])
            ax[ii].set_yticks(yticklog)
            ax[ii].set_ylim(ylim[0], ylim[1])
            ax[ii].invert_yaxis()
            
        else:
            ax[ii].set_ylim(depth[0], depth[isnotnan])
            ax[ii].invert_yaxis()
            ax[ii].grid(True,which='major')
        
        #ax[ii].set_yticks([5,10,25,50,100,250,500,1000,2000,4000,6000])
        ax[ii].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        
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
    
    # do formatting of colorbar 
    cbar = do_cbar_formatting(cbar, do_rescale, cbar_nl, fontsize)
    
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



# please see at:
# --> https://stackoverflow.com/questions/47222585/matplotlib-generic-colormap-from-tab10
# also https://www.py4u.net/discuss/222050
def categorical_cmap(nc, nsc, cmap="tab10", cmap2='nipy_spectral', continuous=False):
    from matplotlib.colors import ListedColormap
    #if nc > plt.get_cmap(cmap).N: cmap = "hsv"
        #raise ValueError("Too many categories for colormap.")
        
    if continuous:
        if nc > plt.get_cmap(cmap).N:
            ccolors = ListedColormap(plt.cm.get_cmap(cmap2, nc)(np.linspace(0,1,nc))).colors
        else:    
            ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
            
    else:
        if nc > plt.get_cmap(cmap).N:
            ccolors = ListedColormap(plt.cm.get_cmap(cmap2, nc)(np.arange(nc, dtype=int))).colors
        else:
            ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
            
            
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv,nsc).reshape(nsc,3)
        if not chsv[0]==0.0 and not chsv[1]==0.0:
            arhsv[:,1] = np.linspace(chsv[1],0.2,nsc)
            arhsv[:,2] = np.linspace(chsv[2],1.0,nsc)
        else:
            arhsv[:,2] = np.linspace(chsv[2],0.8,nsc)
        arhsv      = np.flipud(arhsv)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = rgb       
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap

# ___DO ANOMALY________________________________________________________________
#| compute anomaly between two xarray Datasets                                 |
#| ___INPUT_________________________________________________________________   |
#| data1        :   xarray dataset object                                      |
#| data2        :   xarray dataset object                                      |
#| ___RETURNS_______________________________________________________________   |
#| anom         :   xarray dataset object, data1-data2                         |
#|_____________________________________________________________________________|
def do_indexanomaly(index1,index2):
    
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
                    elif idx1[vname].attrs[key] != idx2[vname2].attrs[key]:
                        anom_idx[vname].attrs[key]  = idx1[vname].attrs[key]+' - '+idx2[vname2].attrs[key]
        
        #___________________________________________________________________________
        anom_index.append(anom_idx)
    #___________________________________________________________________________
    return(anom_index)


# Function x**(1/2)
def forward(x):
    return x**(1/2.5)
def inverse(x):
    return x**(2.5)
