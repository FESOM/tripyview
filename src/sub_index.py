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
from   sub_mesh           import * 
from   sub_data           import *
from   colormap_c2c       import *


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
        #___________________________________________________________________________
        # a rectangular box is given --> translate into shapefile object
        if  (isinstance(box,list) or isinstance(box, np.ndarray)) and len(box)==4: 
            px     = [box[0], box[1], box[1], box[0], box[0]]
            py     = [box[2], box[2], box[3], box[3], box[2]]
            p      = Polygon(list(zip(px,py)))
            idx_IN = contains(p, mesh.n_x, mesh.n_y)
            
        # a polygon as list or ndarray is given --> translate into shape file object
        elif isinstance(box,list) and len(box)==2: 
            px, py = box[0], box[1]  
            p      = Polygon(list(zip(px,py)))  
            idx_IN = contains(p, mesh.n_x, mesh.n_y)
            
        elif isinstance(box, np.ndarray): 
            if box.shape[0]==2:
                px, py = list(box[0,:]), list(box[1,:])
                p      = Polygon(list(zip(px,py)))
                idx_IN = contains(p, mesh.n_x, mesh.n_y)
                
            else:
                raise  ValueError(' ndarray box has wrong format must be [2 x npts], yours is {}'.format(str(box.shape)))
            
        # a polygon as shapefile or shapefile collection is given
        elif (isinstance(box, (Polygon, MultiPolygon))):
            if   isinstance(box, Polygon): 
                idx_IN = contains(box, mesh.n_x, mesh.n_y)
                
            elif isinstance(box, MultiPolygon):
                idx_IN = np.zeros((mesh.n2dn,), dtype=bool)
                for p in box:
                    auxidx = contains(p, mesh.n_x, mesh.n_y)
                    idx_IN = np.logical_or(idx_IN, auxidx)
        
        elif (isinstance(box, shp.Reader)):
            idx_IN = np.zeros((mesh.n2dn,), dtype=bool)
            for shape in box.shapes(): 
                p      = Polygon(shape.points)
                auxidx = contains(p, mesh.n_x, mesh.n_y)
                idx_IN = np.logical_or(idx_IN, auxidx)
        # otherwise
        else:
            raise ValueError('the given box information to compute the index has no valid format')
        
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



def convert_geojson2shp(geojsonfilepath, shppath):
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
    return



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




        

    
    ##+___PLOT FESOM2.0 DATA IN INDEX BOX OVER DEPTH AND TIME DATA______________+
    ##|                                                                         |
    ##+_________________________________________________________________________+
    #def plot_index_t_x_z(self,numb=[],figsize=[],do_subplot=[],do_output=True,which_lines=[0,0,0]):
        #from set_inputarray import inputarray
        #fsize=16
        ##_______________________________________________________________________
        #if isinstance(numb,int)==True : numb = [numb]
        #if len(numb)==0 : numb=range(0,len(self.box_define))
        
        ##_______________________________________________________________________
        #for ii in numb:
            ##fig = plt.figure(figsize=(12, 6))
            ##ax1  = plt.gca()
            
            #if len(figsize)==0 : figsize=[10,5]
            ##___________________________________________________________________________
            ## plot is not part of subplot
            #if len(do_subplot)==0:
                #fig = plt.figure(figsize=figsize)
                #ax1 = plt.gca()
            #else:
                #fig=do_subplot[0]
                #ax1=do_subplot[1]
                #fig.sca(ax1)
            #resolution = 'c'
            #fsize = 14
            
            ##___________________________________________________________________
            #cnumb= self.cnumb
            ##cmin = np.nanmin(self.value[ii])
            ##cmax = np.nanmax(self.value[ii])
            #cmin, cmax = np.nanmin(self.value), np.nanmax(self.value)
            #cref = cmin + (cmax-cmin)/2
            #cref = np.around(cref, -np.int32(np.floor(np.log10(np.abs(cref)))-1) ) 
            ##cref =0.0
            
            ##___________________________________________________________________
            ## if anomaly data    
            #if self.anom==True: 
                ##cmax,cmin,cref = np.nanmax(self.value[ii]), np.nanmin(self.value[ii]), 0.0
                #cmax,cmin,cref = np.nanmax(self.value), np.nanmin(self.value), 0.0
                #self.cmap='blue2red'
            ##___________________________________________________________________
            ## if predefined color range    
            #if len(self.crange)!=0:
                #if len(self.crange)==2:
                    #cmin = np.float(self.crange[0])
                    #cmax = np.float(self.crange[1])
                    #cref = np.around(cref, -np.int32(np.floor(np.log10(np.abs(cref)))-1) ) 
                #elif len(self.crange)==3:
                    #cmin = np.float(self.crange[0])
                    #cmax = np.float(self.crange[1])
                    #cref = np.float(self.crange[2])
                #else:
                    #print(' this colorrange definition is not supported !!!')
                    #print('data.crange=[cmin,cmax] or data.crange=[cmin,cmax,cref]')
                
            #if do_output==True: print('[cmin,cmax,cref] = ['+str(cmin)+', '+str(cmax)+', '+str(cref)+']')
            #if do_output==True: print('[cnum]=',cnumb)
            #cmap0,clevel = colormap_c2c(cmin,cmax,cref,cnumb,self.cmap)
            #if do_output==True: print('clevel = ',clevel)
            
            #do_drawedges=True
            #if clevel.size>40: do_drawedges=False
            
            ## overwrite discrete colormap
            ##cmap0 = cmocean.cm.balance
            ##do_drawedges=False
            
            ##___________________________________________________________________
            ## make pcolor or contour plot 
            ##depth = self.depth[:-1] + (self.depth[1:]-self.depth[:-1])/2.0
            ##depth = self.zlev[:-1] + (self.zlev[1:]-self.zlev[:-1])/2.0
            #depth = -self.zlev
            #depthlim = np.sum(~np.isnan(self.value[0,:])).max()
            #if depthlim==depth.shape: depthlim=depthlim-1
            #yy,xx = np.meshgrid(depth,self.time)
            
            ##___________________________________________________________________
            #data_plot = np.copy(self.value)
            #data_plot[data_plot<clevel[0]]  = clevel[0]+np.finfo(np.float32).eps
            #data_plot[data_plot>clevel[-1]] = clevel[-1]-np.finfo(np.float32).eps
            
            ##___________________________________________________________________
            #if inputarray['which_plot']=='pcolor':
                #hp=ax1.pcolormesh(xx[:,0:depthlim],yy[:,0:depthlim],data_plot[:,0:depthlim],
                            #shading='flat',#flat
                            #antialiased=False,
                            #edgecolor='None',
                            #cmap=cmap0),
                            ##vmin=np.nanmin(data_plot), vmax=np.nanmax(data_plot))
            #else: 
                #hp=ax1.contourf(xx[:,0:depthlim],yy[:,0:depthlim],data_plot[:,0:depthlim],levels=clevel,
                            #antialiased=False,
                            #cmap=cmap0,
                            #vmin=clevel[0], vmax=clevel[-1])
                #if which_lines[0]==1:
                    #ax1.contour(xx[:,0:depthlim],yy[:,0:depthlim],data_plot[:,0:depthlim],levels=clevel[clevel<cref],
                                    #antialiased=True,
                                    #colors='k',
                                    #linewidths=0.5,
                                    #linestyles='-',
                                    #vmin=clevel[0], vmax=clevel[-1])
                #if which_lines[1]==1:
                    #ax1.contour(xx[:,0:depthlim],yy[:,0:depthlim],data_plot[:,0:depthlim],levels=clevel[clevel==cref],
                                    #antialiased=True,
                                    #colors='k',
                                    #linewidths=1.0,
                                    #linestyles='-',
                                    #vmin=clevel[0], vmax=clevel[-1])
                #if which_lines[2]==1:
                    #ax1.contour(xx[:,0:depthlim],yy[:,0:depthlim],data_plot[:,0:depthlim],levels=clevel[clevel>cref],
                                    #antialiased=True,
                                    #colors='k',
                                    #linewidths=0.5,
                                    #linestyles='--',
                                    #vmin=clevel[0], vmax=clevel[-1])
            ##hp.cmap.set_under([0.4,0.4,0.4])
            
            ##___________________________________________________________________
            ## set main axes
            #ax1.set_xlim(self.time.min(),self.time.max())
            #ax1.set_ylim(0,depth[depthlim-1])
            #ax1.invert_yaxis()
            ##ax1.set_axis_bgcolor([0.25,0.25,0.25])
            #ax1.tick_params(axis='both',which='major',direction='out',length=8,labelsize=fsize)
            #ax1.minorticks_on()
            #ax1.tick_params(axis='both',which='minor',direction='out',length=4,labelsize=fsize)
            #ax1.set_xlabel('Time [years]',fontdict=dict(fontsize=fsize))
            #ax1.set_ylabel('Depth [km]',fontdict=dict(fontsize=fsize))
            #plt.title(self.descript+' - '+self.box_define[0][2],fontdict= dict(fontsize=fsize*2),verticalalignment='bottom')
            
            ##___________________________________________________________________
            ## draw colorbar
            ##divider = make_axes_locatable(ax1)
            ##cax     = divider.append_axes("right", size="2.5%", pad=0.5)
            ##plt.clim(clevel[0],clevel[-1])
            #him = ax1.get_images()
            #for im in ax1.get_images():
                #im.set_clim(clevel[0],clevel[-1])
            
            ##cbar = plt.colorbar(hp,ax=ax1,cax=cax,ticks=clevel,drawedges=do_drawedges)
            #cbar = plt.colorbar(hp,ax=ax1,ticks=clevel,drawedges=do_drawedges)
            #cbar.set_label(self.lname+' '+self.unit+'\n'+self.str_time, size=fsize)
            
            #cl = plt.getp(cbar.ax, 'ymajorticklabels')
            #plt.setp(cl, fontsize=fsize)
            
            ## kickout some colormap labels if there are to many
            #ncbar_l=len(cbar.ax.get_yticklabels()[:])
            #idx_cref = np.where(clevel==cref)[0]
            #idx_cref = np.asscalar(idx_cref)
            #nmax_cbar_l = 10
            #nstep = ncbar_l/nmax_cbar_l
            #nstep = np.int(np.floor(nstep))
            ##plt.setp(cbar.ax.get_yticklabels()[:], visible=False)
            ##plt.setp(cbar.ax.get_yticklabels()[idx_cref::nstep], visible=True)
            ##plt.setp(cbar.ax.get_yticklabels()[idx_cref::-nstep], visible=True)
            #if nstep==0:nstep=1
            #tickl = cbar.ax.get_yticklabels()
            #idx = np.arange(0,len(tickl),1)
            #idxb = np.ones((len(tickl),), dtype=bool)                
            #idxb[idx_cref::nstep]  = False
            #idxb[idx_cref::-nstep] = False
            #idx = idx[idxb==True]
            #for ii in list(idx):
                #tickl[ii]=''
            #cbar.ax.set_yticklabels(tickl)
            
            ## reposition colorbar and axes
            #plt.tight_layout()
            #pos_ax   = ax1.get_position()
            #ax1.set_position([pos_ax.x0,pos_ax.y0,0.90-pos_ax.y0,pos_ax.height])
            #pos_ax   = ax1.get_position()
            #pos_cbar = cbar.ax.get_position()
            #cbar.ax.set_position([pos_ax.x0+pos_ax.width+0.01,pos_cbar.y0, pos_cbar.width, pos_cbar.height])
            #fig.canvas.draw()
            ##___________________________________________________________________
            ## save figure
            #if inputarray['save_fig']==True:
                #print(' --> save figure: png')
                #str_times= self.str_time.replace(' ','').replace(':','')
                #str_deps = self.str_dep.replace(' ','').replace(',','').replace(':','')
                #sfname = 'boxplot_'+self.box_define[ii][2]+'_'+self.descript+'_'+self.sname+'_'+str_times+'_'+str_deps+'.png'
                #sdname = inputarray['save_figpath']
                #if os.path.isdir(sdname)==False: os.mkdir(sdname)
                #plt.savefig(sdname+sfname, \
                            #format='png', dpi=600, \
                            #bbox_inches='tight', pad_inches=0,\
                            #transparent=True,frameon=True)
            ##___________________________________________________________________
            #plt.show(block=False)
            #print('finish')
        
        #return(fig,ax1,cbar)
        
        
    ##+___PLOT FESOM2.0 DATA IN INDEX BOX POSITION______________________________+
    ##|                                                                         |
    ##+_________________________________________________________________________+
    #def plot_index_position(self,mesh,numb=[]):
        #from set_inputarray import inputarray
        #fsize=16
        ##_______________________________________________________________________
        #if isinstance(numb,int)==True : numb = [numb]
        #if len(numb)==0 : numb=range(0,len(self.box_define))
        
        ##_______________________________________________________________________
        ## draw position of box 
        #for ii in numb:
        
            ##___________________________________________________________________
            #xmin,xmax = np.min(self.box_define[ii][0]), np.max(self.box_define[ii][0])
            #ymin,ymax = np.min(self.box_define[ii][1]), np.max(self.box_define[ii][1])
            #xmin,xmax,ymin,ymax = xmin-20.0, xmax+20.0, ymin-20.0, ymax+20.0
            #xmin,xmax,ymin,ymax = np.max([xmin,-180.0]),np.min([xmax,180.0]),np.max([ymin,-90.0]),np.min([ymax,90.0])
            
            ##___________________________________________________________________
            #figp, ax = plt.figure(figsize=(8, 8)), plt.gca()
            #map     = Basemap(projection = 'cyl',resolution = 'c',
                        #llcrnrlon = xmin, urcrnrlon = xmax, llcrnrlat = ymin, urcrnrlat = ymax)
            
            #mx,my     = map(mesh.nodes_2d_xg, mesh.nodes_2d_yg)
            
            
            #map.bluemarble()
            #fesom_plot_lmask(map,mesh,ax,'0.6')
            
            #xlabels,ylabels=[0,0,0,1],[1,0,0,0]
            #xticks,yticks = np.arange(0.,360.,10.), np.arange(-90.,90.,5.)
            #map.drawparallels(yticks,labels=ylabels,fontsize=fsize)
            #map.drawmeridians(xticks,labels=xlabels,fontsize=fsize)
            #map.drawmapboundary(linewidth=1.0)
            
            ##___________________________________________________________________
            #patch=[]
            #if len(self.box_define[ii][0])>2:
                #ax.plot(self.box_define[ii][0]    ,self.box_define[ii][1] ,linestyle='None'   ,color='w',linewidth=2.0,marker='o',mfc='w',mec='k',axes=ax)
                #patch.append(Polygon(list(zip(self.box_define[ii][0],self.box_define[ii][1])),closed=True,clip_on=True) )
            #else:
                #auxboxx = [ self.box_define[ii][0][0],\
                            #self.box_define[ii][0][1],\
                            #self.box_define[ii][0][1],\
                            #self.box_define[ii][0][0],\
                            #self.box_define[ii][0][0]]
                #auxboxy = [ self.box_define[ii][1][0],\
                            #self.box_define[ii][1][0],\
                            #self.box_define[ii][1][1],\
                            #self.box_define[ii][1][1],\
                            #self.box_define[ii][1][0],]
                #ax.plot(auxboxx    ,auxboxy ,linestyle='None'   ,color='w',linewidth=2.0,marker='o',mfc='w',mec='k',axes=ax)
                #patch.append(Polygon(list(zip(auxboxx,auxboxy)),closed=True,clip_on=True) )
            #ax.add_collection(PatchCollection(patch, alpha=1.0,facecolor='none',edgecolor='w',zorder=1,linewidth=2.0,hatch='/'))    
            
            ## plot selected mesh points
            #ax.plot(mesh.nodes_2d_xg[self.box_idx[ii]],mesh.nodes_2d_yg[self.box_idx[ii]],color='r',linestyle='None',marker='.',alpha=0.5)
            #plt.title(self.box_define[ii][2],fontdict= dict(fontsize=fsize*2),verticalalignment='bottom')
            
            ##___________________________________________________________________
            ## save figure
            #if inputarray['save_fig']==True:
                #print(' --> save figure: png')
                #sfname = 'boxplot_'+self.box_define[ii][2]+'_position'+'.png'
                #sdname = inputarray['save_figpath']
                #if os.path.isdir(sdname)==False: os.mkdir(sdname)
                #plt.savefig(sdname+sfname, \
                            #format='png', dpi=600, \
                            #bbox_inches='tight', pad_inches=0,\
                            #transparent=True,frameon=True)
            
            ##___________________________________________________________________
            #plt.show()
            #figp.canvas.draw()
