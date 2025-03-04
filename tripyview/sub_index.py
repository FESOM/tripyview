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

#
#
#___COMPUTE INDEX OVER REGION 2D AND 3D_________________________________________
def load_index_fesom2(mesh                  , 
                      data                  , 
                      box_list              , 
                      boxname       = None  , 
                      do_harithm    ='wmean', 
                      do_zarithm    = None  ,
                      do_checkbasin = False , 
                      do_compute    = False , 
                      do_load       = True  , 
                      do_persist    = False ,
                      client        = None  ,
                      ):
    """
    --> compute index over region from 2d and 3d vertice data
    
    Parameters:
        
        :mesh:          fesom2 mesh object, with all mesh information

        :data:          xarray dataset object, or list of xarray dataset object with 3d vertice
                        data

        :box_list:      None, list (default: None)  list with regional box limitation for index computation box can be: 

                        - ['global']   ... compute global index 
                        - [shp.Reader] ... index region defined by shapefile 
                        - [ [lonmin,lonmax,latmin, latmax], boxname] index region defined by rect box 
                        - [ [ [px1,px2...], [py1,py2,...]], boxname] index region defined by polygon
                        - [ np.array(2 x npts), boxname] index region defined by polygon

        :do_harithm:    str, (default='wmean') which horizontal arithmetic should be applied 
                        over the index definition region
                        
                        - 'wmean' ... area weighted mean 
                        - 'wint'  ... area weighted integral
        
        :do_zarithm:    str, (default=None) which arithmetic should be applied in the vertical
                        
                        - 'wmean' ... depth weighted mean 
                        - 'wint'  ... depth weighted integral
                    
        :do_checkbasin: bool, (default=False) additional plot with selected region/
                        basin information
        
        :do_compute:    bool (default=False), do xarray dataset compute() at the end
                        data = data.compute(), creates a new dataobject the original
                        data object seems to persist
        
        :do_load:       bool (default=True), do xarray dataset load() at the end
                        data = data.load(), applies all operations to the original
                        dataset
                        
        :do_persist:    bool (default=False), do xarray dataset persist() at the end
                        data = data.persist(), keeps the dataset as dask array, keeps
                        the chunking    
                      
        :do_info:       bool (defalt=False), print variable info at the end 
                       
        :client:        dask client object (default=None)
        
    Returns:
    
    
    ____________________________________________________________________________
    """
    
    
    xr.set_options(keep_attrs=True)
    #___________________________________________________________________________
    # str_anod    = ''
    index_list  = []
    idxin_list  = []
    cnt         = 0
    
    #___________________________________________________________________________
    vname = list(data.keys())[0]
    dimn_v = None
    dimn_h, dimn_v = 'dum', 'dum'
    if   ('nod2' in data.dims): dimn_h, do_elem = 'nod2', False
    elif ('elem' in data.dims): dimn_h, do_elem = 'elem', False
    if   'nz'  in list(data[vname].dims): 
        dimn_v  = 'nz'    
    elif 'nz1' in list(data[vname].dims) or 'nz_1' in list(data[vname].dims): 
        dimn_v  = 'nz1'
        
    #___________________________________________________________________________
    # loop over box_list
    for box in box_list:
        if not isinstance(box, shp.Reader):
            if   len(box)==2: boxname, box = box[1], box[0]
            elif len(box)==4 and boxname==None: boxname = '[{:03.2f}...{:03.2f}°E, {:03.2f}...{:03.2f}°N]'.format(box[0],box[1],box[2],box[3])
            if box is None or box=='global': boxname='global'
        else:     
            # if we do the box polygon selection in paralel, i cant give into the 
            # box selection routine the Read shapefile handle since its not pickable
            # therfore i need to extract the shapefile polygon points before throwing it into 
            # the do_boxmask_dask routine 
            boxname = os.path.basename(box.shapeName).replace('_',' ')  
            shape_data = [shape.points for shape in box.shapes()]  # Extract raw data
            box = MultiPolygon([Polygon(shape) for shape in shape_data])
            
        #_______________________________________________________________________
        # compute  mask index to select index region 
        if box != 'global': 
            #idxin=xr.DataArray(do_boxmask(mesh, box, do_elem=do_elem), dims=dimn_h)
            idxin = da.map_blocks(  do_boxmask_dask,
                                    data['lon'].data,
                                    data['lat'].data,
                                    data['ispbnd'].data,
                                    box,
                                    dtype=bool).compute()
            idxin = idxin.compute() # ---> we can not index whit a dask array 
        else:
            idxin = None
        #_______________________________________________________________________
        # check basin selection
        if do_checkbasin and idxin is not None:
            from matplotlib.tri import Triangulation
            tri = Triangulation(np.hstack((mesh.n_x,mesh.n_xa)), np.hstack((mesh.n_y,mesh.n_ya)), np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia)))
            plt.figure()
            ax = plt.gca()
            plt.triplot(tri, color='k')
            if  ~ do_elem: 
                plt.plot(mesh.n_x[idxin], mesh.n_y[idxin], '*r', linestyle='None', markersize=1)
            else: 
                plt.plot(mesh.n_x[tri.triangles[idxin,:]].sum(axis=1)/3.0, mesh.n_y[tri.triangles[idxin,:]].sum(axis=1)/3.0, '*r', linestyle='None', markersize=1)
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
        # select index region from data
        if box != 'global': index = data.sel({dimn_h:idxin})
        else              : index = data    
            
        #_______________________________________________________________________
        # do volume averaged mean (apply horiz. and vertical)
        if   do_harithm=='wmean' and do_zarithm=='wmean':
            index = index.weighted(index['w_A']*index['w_z']).mean(dim=dim_name, keep_attrs=True, skipna=True)
                
        # do volume integraln  (apply horiz. and vertical)
        elif do_harithm=='wint' and do_zarithm=='wint':
            index = index.weighted(index['w_A']*index['w_z']).sum(dim=dim_name, keep_attrs=True, skipna=True)

        # do horizontal/ vertical metrix that can be idependent from each other
        else:    
            # only horizontal arithmetic 
            if   dimn_h in ['nod2', 'elem'] and do_harithm is not None:
                index = do_horiz_arithmetic(index, do_harithm, dimn_h)
                
            # only vertical arithmetic 
            if   dimn_v in ['nz', 'nz1','nz_1', 'ncat', 'ndens'] and do_zarithm is not None:
                index = do_depth_arithmetic(index, do_zarithm, dimn_v)
                    
        index_list.append(index)
        idxin_list.append(idxin)
        del(index, idxin)
        
        #_______________________________________________________________________
        warnings.filterwarnings("ignore", category=UserWarning, message="Sending large graph of size")
        warnings.filterwarnings("ignore", category=UserWarning, message="Large object of size")
        if   do_compute: index_list[cnt] = index_list[cnt].compute()
        elif do_load   : index_list[cnt] = index_list[cnt].load()
        elif do_persist: index_list[cnt] = index_list[cnt].persist()
        
        # additionally rebalancing the  memory load of workers 
        if client is not None: client.rebalance()
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
    return(index_list)


    
#
#
#_______________________________________________________________________________
def plot_index_region(mesh, idx_IN, box_list, which='hard'):
    """
    --> plot index definition region 
    
    Parameters:
        
        :mesh:          fesom2 tripyview mesh object,  with all mesh information 
        
        :idx_IN:        list with bool np.array which vertices are within index defintion 
                        region
        
        :box_list:      None, list (default: None)  list with regional box limitation for index computation box can be: 
        
                        - ['global']   ... compute global index 
                        - [shp.Reader] ... index region defined by shapefile 
                        - [ [lonmin,lonmax,latmin, latmax], boxname] index region defined by rect box 
                        - [ [ [px1,px2...], [py1,py2,...]], boxname] index region defined by polygon
                        - [ np.array(2 x npts), boxname] index region defined by polygon
        
        :which:         str, (default=hard)
                        - 'soft' plot triangles that at least one selected vertice in them
                        - 'hard' plot triangles that at all three selected vertice in them 
                        - 'mid'  plot triangles that have more than one selected vertice in them
        
    Returns:
        
        :None:
    
    ____________________________________________________________________________
    """
    
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
#___DO INDEX ANOMALY____________________________________________________________
def do_indexanomaly(index1,index2):
    """
    --> compute anomaly between two index xarray Datasets
    
    Parameters:
    
        :data1:   index xarray dataset object

        :data2:   index xarray dataset object

    Returns:
    
        :anom:   index xarray dataset object, data1-data2

    ____________________________________________________________________________
    """
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
