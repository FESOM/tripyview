# Patrick Scholz, 23.01.2018
import sys
import os
import numpy as np
import copy as  cp
from   shapely.geometry   import Point, Polygon, MultiPolygon, shape
import shapefile as shp
import json
import geopandas as gpd
import matplotlib.pylab as plt
from   matplotlib.ticker import MultipleLocator, AutoMinorLocator, ScalarFormatter

from   .sub_mesh           import * 
from   .sub_data           import *
from   .sub_utility        import *
from   .sub_plot           import *
from   .sub_colormap       import *


def load_index_fesom2(mesh, data, box_list, boxname=None, do_harithm='wmean', 
                      do_zarithm=None, do_outputidx=False, 
                      do_compute=False, do_load=True, do_persist=False, do_checkbasin=False):
    xr.set_options(keep_attrs=True)
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
        else:     
            boxname = os.path.basename(box.shapeName).replace('_',' ')  
           
        #_______________________________________________________________________
        # compute  mask index
        if   'nod2' in data.dims: 
            idx_IN=xr.DataArray(do_boxmask(mesh, box, do_elem=False), dims='nod2')
            # --> seems to be not allowed anymore in newer version of xarray + dask
            #     indexing array cant be chunked anymore
            #if any(data.chunks.values()): idx_IN = idx_IN.chunk({'nod2':data.chunksizes['nod2']})
            
        elif 'elem' in data.dims: 
            idx_IN=xr.DataArray(do_boxmask(mesh, box, do_elem=True), dims='elem')
            # --> seems to be not allowed anymore in newer version of xarray + dask
            #     indexing array cant be chunked anymore
            #if any(data.chunks.values()): idx_IN = idx_IN.chunk({'elem':data.chunksizes['elem']})
            
        #_______________________________________________________________________
        # check basin selection
        if do_checkbasin:
            from matplotlib.tri import Triangulation
            tri = Triangulation(np.hstack((mesh.n_x,mesh.n_xa)), np.hstack((mesh.n_y,mesh.n_ya)), np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia)))
            plt.figure()
            ax = plt.gca()
            plt.triplot(tri, color='k')
            if   'nod2' in data.dims: 
                plt.plot(mesh.n_x[idx_IN], mesh.n_y[idx_IN], '*r', linestyle='None', markersize=1)
            else: 
                plt.plot(mesh.n_x[tri.triangles[idx_IN,:]].sum(axis=1)/3.0, mesh.n_y[tri.triangles[idx_IN,:]].sum(axis=1)/3.0, '*r', linestyle='None', markersize=1)
            plt.title('Basin selection')
            plt.show()
                
        #_______________________________________________________________________
        # selected points in xarray dataset object and  average over selected 
        # points
        dim_name=[]
        if   'nod2' in data.dims: dim_name.append('nod2')
        elif 'elem' in data.dims: dim_name.append('elem')    
        if   'nz'   in data.dims: dim_name.append('nz')
        elif 'nz1'  in data.dims: dim_name.append('nz1')      
                
        #_______________________________________________________________________
        # do volume averaged mean
        if do_harithm=='wmean' and do_zarithm=='wmean':
            if   'nod2' in data.dims:
                weights = data['w_A']*data['w_z']
                data    = data.drop_vars(['w_A', 'w_z'])
                weights = weights/weights.sum(dim=dim_name, skipna=True)
                data    = data*weights.astype('float32', copy=False )
                index   = data.sum(dim=dim_name, keep_attrs=True, skipna=True)  
            elif 'elem' in data.dims:    
                STOP
        
        #_______________________________________________________________________
        # do volume integral
        elif do_harithm=='wint' and do_zarithm=='wint':
            if   'nod2' in data.dims:
                weights = data['w_A']*data['w_z']
                data    = data.drop_vars(['w_A', 'w_z'])
                data    = data*weights.astype('float32', copy=False )
                index   = data.sum(dim=dim_name, keep_attrs=True, skipna=True)  
            elif 'elem' in data.dims:    
                STOP  
        
        #_______________________________________________________________________
        # do horizontal/ vertical metrix that can be idependent from each other
        else:    
            #___________________________________________________________________
            if   'nod2' in data.dims:
                index = do_horiz_arithmetic(data.sel(nod2=idx_IN), do_harithm, 'nod2')
            elif 'elem' in data.dims:    
                index = do_horiz_arithmetic(data.sel(elem=idx_IN), do_harithm, 'elem')
                
            #___________________________________________________________________
            if   'nz1' in data.dims and do_harithm is not None:
                index = do_depth_arithmetic(index, do_zarithm, 'nz1')
            elif 'nz'  in data.dims and do_harithm is not None:        
                index = do_depth_arithmetic(index, do_zarithm, 'nz')
                    
        index_list.append(index)
        idxin_list.append(idx_IN)
        del(index)
        
        #_______________________________________________________________________
        warnings.filterwarnings("ignore", category=UserWarning, message="Sending large graph of size")
        warnings.filterwarnings("ignore", category=UserWarning, message="Large object of size")
        if do_compute: index_list[cnt] = index_list[cnt].compute()
        if do_load   : index_list[cnt] = index_list[cnt].load()
        if do_persist: index_list[cnt] = index_list[cnt].persist()
        warnings.resetwarnings()
            
        #_______________________________________________________________________
        # set additional attributes
        vname = list(index_list[cnt].data_vars)
        index_list[cnt][vname[0]].attrs['boxname'] = boxname
        proj = 'index'
        if 'nz1' in list(index_list[cnt].variables) or 'nz' in list(index_list[cnt].variables):
            proj = proj+'+depth'
            if 'nz1' in list(index_list[cnt].variables):
                index_list[cnt] = index_list[cnt].rename_vars({'nz1': 'depth'})
            elif 'nz' in list(index_list[cnt].variables):
                index_list[cnt] = index_list[cnt].rename_vars({'nz' : 'depth'})
                
        if 'time' in list(index_list[cnt].variables):
            proj = proj+'+time'
            
        if 'nod2' in list(index_list[cnt].variables) or 'elem' in list(index_list[cnt].variables):
            proj = proj+'+xy'
            
        index_list[cnt].attrs['proj'] = proj    
        
        
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
                    if   key in ['long_name']:
                        anom_idx[vname].attrs[key] = 'anom. '+anom_idx[vname].attrs[key].capitalize() 
                    elif key in ['short_name']:
                        anom_idx[vname].attrs[key] = 'anom. '+anom_idx[vname].attrs[key]        
                    elif key in ['units',]: 
                        continue
                    elif idx1[vname].attrs[key] != idx2[vname2].attrs[key]:
                        anom_idx[vname].attrs[key]  = idx1[vname].attrs[key]+' - '+idx2[vname2].attrs[key]
        
        #___________________________________________________________________________
        anom_index.append(anom_idx)
    #___________________________________________________________________________
    return(anom_index)
