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
from   sub_mesh           import * 
from   sub_data           import *
from   sub_plot           import *
from   colormap_c2c       import *
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def load_index_fesom2(mesh, data, box_list, boxname=None, do_harithm='mean', 
                      do_compute=True):
    
    #___________________________________________________________________________
    str_anod    = ''
    index_list  = []
    idxin_list  = []
    cnt         = 0
    
    #___________________________________________________________________________
    # loop over box_list
    for box in box_list:
        #_______________________________________________________________________
        # compute  mask index
        idx_IN=do_boxmask(mesh,box)
        
        #_______________________________________________________________________
        # selected points in xarray dataset object and  average over selected 
        # points
        index_list.append( do_horiz_arithmetic(data.sel(nod2=idx_IN), do_harithm) )
        idxin_list.append(idx_IN)
        str_anod = str(do_harithm)
    
        #_______________________________________________________________________
        if do_compute: index_list[cnt] = index_list[cnt].compute()
    
        #_______________________________________________________________________
        cnt = cnt + 1
    #___________________________________________________________________________
    return(index_list, idxin_list)
    
#
#
#_______________________________________________________________________________
def do_boxmask(mesh, box, do_elem=False):
    #___________________________________________________________________________
    if do_elem: mesh_x, mesh_y = mesh.n_x[mesh.e_i].sum(axis=1)/3.0, mesh.n_y[mesh.e_i].sum(axis=1)/3.0
    else      : mesh_x, mesh_y = mesh.n_x, mesh.n_y
    
    #___________________________________________________________________________
    # a rectangular box is given --> translate into shapefile object
    if  (isinstance(box,list) or isinstance(box, np.ndarray)) and len(box)==4: 
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
                 cbar_unit=None, do_save=None, linewidth=1.0):
        
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
            val   = index_list[di][bi][vname[0]].values
            
            cname = list(index_list[di][bi].coords)
            dep = index_list[di][bi].coords[cname[0]].values
        
            linestyle = 'solid'
            if linestyle_list is not None: 
                if (not linestyle_list[di])==False: linestyle =  linestyle_list[di]
        
            if linecolor_list is None: 
                ax[bi].plot(val,dep, label=label_list[di], linestyle=linestyle, linewidth=linewidth)
            else:
                if not linecolor_list[di]:
                    ax[bi].plot(val,dep, label=label_list[di], linestyle=linestyle, linewidth=linewidth)
                else:
                    ax[bi].plot(val,dep, color=linecolor_list[di], 
                                label=label_list[di], linestyle=linestyle, linewidth=linewidth)
                    
        #_______________________________________________________________________
        if bi==0 : ax[bi].legend(loc='best', frameon=True, shadow=True,fontsize=8)
        
        #_______________________________________________________________________
        if (isinstance(box_list[bi], shp.Reader)): str_title = box_list[bi].shapeName
        else:                                      str_title = '{}'.format(str(box_list[bi]))
        ax[bi].set_title(os.path.basename(box_list[bi].shapeName))
        
        #_______________________________________________________________________
        if collist[bi]==0:
            ax[bi].set_ylabel('depth [m]')
        
        #_______________________________________________________________________
        if cbar_label is None: 
            str_xlabel = index_list[di][bi][ vname[0] ].attrs['long_name']
        else:
            str_xlabel = cbar_label
        if cbar_unit  is None: str_xlabel = str_xlabel+' ['+index_list[di][bi][ vname[0] ].attrs['units']+']'
        else:                  str_xlabel = str_xlabel+' ['+cbar_unit+']'    
        ax[bi].set_xlabel(str_xlabel)
        
        #_______________________________________________________________________
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
    do_savefigure(do_save)
    
    #___________________________________________________________________________
    return(fig, ax)
    
