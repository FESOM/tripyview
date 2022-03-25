import pyvista as pv
import numpy as np
import vtk

from   .sub_mesh           import * 
from   .colormap_c2c       import *
R_earth  = 6371.0e3
#
#
#___CREATE PYVISTA OCEAN MESH___________________________________________________________
# make potatoefication of Earth radius 
def create_3d_ocean_mesh(mesh,potatoefac=0.5,variable='elevation', do_texture=False):
    print(' --> compute 3d ocean mesh')
    #_______________________________________________________________________________________
    # do topographic potatoefication of ocean mesh
    R_grid= R_earth
    bottom_depth_2d = -mesh.n_z;
    bottom_depth_2d[mesh.n_i==1]=0.0;
    R_grid         = R_grid-( bottom_depth_2d*100*potatoefac)
    R_grid[mesh.n_i==1]=R_earth;
    
    #_______________________________________________________________________________________
    # create sperical ocean coordinates
    xs,ys,zs = grid_cart3d(mesh.n_x, mesh.n_y, R_grid, is_deg=True)
    points = np.column_stack([xs,ys,zs])
    del xs,ys,zs
    
    #_______________________________________________________________________________________
    # Each cell in the cell array needs to include the size of the cell
    # and the points belonging to the cell.  In this example, there are 8
    # hexahedral cells that have common points between them.
    # cell_size = np.ones(mesh.n2dea, dtype=np.uint8)*3
    cell_size = np.ones(mesh.n2de, dtype=np.uint8)*3
    # cells     = np.column_stack([cell_size, mesh.elem_2d_i])
    
    cells     = np.column_stack([cell_size, mesh.e_i])
    cells     = cells.ravel()
    del cell_size

    # each cell is a VTK_TRIANGLE
    # celltypes = np.empty(mesh.n2dea, dtype=np.uint8)
    celltypes = np.empty(mesh.n2de, dtype=np.uint8)
    celltypes[:] = vtk.VTK_TRIANGLE

    # the offset array points to the start of each cell (via flat indexing)
    # offset = np.arange(0,4*mesh.n2dea,4, dtype=np.uint32 )
    # offset = np.arange(0,4*mesh.n2de,4, dtype=np.uint32 )
    
    #_______________________________________________________________________________________
    # create pyvista unstrucutred mesh object
    # meshpv_ocean = pv.UnstructuredGrid(offset, cells, celltypes, points)
    meshpv_ocean = pv.UnstructuredGrid(cells, celltypes, points)
    meshpv_ocean['elevation'] = -mesh.n_z
    #if any(x in variable for x in ['elevation','depth','topo','bathymetry']):
    
    #_______________________________________________________________________________________
    # add variables to ocean mesh
    if any(x in variable for x in ['resolution','resol']):   
        meshpv_ocean['resolution'] = mesh.n_resol/1000
    # del offset, cells, celltypes, points
    del cells, celltypes
    
    #___do texture coordinates_______________________________________________________________
    # Initialize the texture coordinates array
    if do_texture:
        meshpv_ocean.active_t_coords     = np.zeros((points.shape[0], 2))
        xs, ys, zs                       = grid_cart3d(mesh.n_x+90, mesh.n_y, R_grid/R_earth, is_deg=True)
        meshpv_ocean.active_t_coords[:,0]= 0.5 + np.arctan2(-xs,ys)/(2 * np.pi)
        meshpv_ocean.active_t_coords[:,1]= 0.5 + np.arcsin(zs)/np.pi
        del xs, ys, zs
    del points
    
    return meshpv_ocean


#
#
#___CREATE PYVISTA LAND MESH TO FILL HOLES______________________________________________
def create_3d_land_mesh(mesh, resol=1, potatoefac=1, do_topo=False, topo_path=[], 
                        topo_varname='topo', topo_dimname=['lon','lat'], do_texture=True):
    print(' --> compute 3d land mesh')
    
    from matplotlib.path import Path
    from matplotlib.tri  import Triangulation
    
    #_______________________________________________________________________________________
    # cycle over all land polygons
    for niland, lsmask in enumerate(mesh.lsmask_a):
        poly_x, poly_y = lsmask[:,0], lsmask[:,1]
        xmin, xmax     = np.floor(poly_x).min(), np.ceil(poly_x).max()
        ymin, ymax     = np.floor(poly_y).min(), np.ceil(poly_y).max()

        #resol = 1
        x_m, y_m  = np.meshgrid(np.arange(xmin, xmax, resol),np.arange(ymin, ymax, resol))
        x_m, y_m  = x_m.reshape((x_m.size, 1)), y_m.reshape((y_m.size, 1))

        #___________________________________________________________________________________
        # check if regular points are within polygon
        IN        = Path(lsmask).contains_points(np.concatenate((x_m, y_m),axis=1)) 
        x_m, y_m  = x_m[IN==True], y_m[IN==True]
        del IN

        #___________________________________________________________________________________
        # combine polygon points and regular points within polygon --> do triangulation
        outeredge = np.vstack((poly_x, poly_y)).transpose()
        points    = np.hstack((x_m, y_m))
        points    = np.vstack((outeredge,points))
        tri       = Triangulation(points[:,0], points[:,1])
        del outeredge, poly_x, poly_y

        #___________________________________________________________________________________
        # compute trinagle centroids and check if they are within polygon
        tri_cx    = np.sum(points[tri.triangles,0],axis=1)/3
        tri_cy    = np.sum(points[tri.triangles,1],axis=1)/3
        tri_cx    = np.reshape(tri_cx,(tri_cx.size,1))
        tri_cy    = np.reshape(tri_cy,(tri_cy.size,1))
        IN        = Path(lsmask).contains_points(np.concatenate((tri_cx,tri_cy),axis=1))
        tri.triangles=tri.triangles[IN==True,:]
        del tri_cx, tri_cy, IN

        #___________________________________________________________________________________
        # concatenate all land trinagles
        if niland==0:
            land_points = points
            land_elem2d = tri.triangles
        else:
            land_elem2d = np.concatenate((land_elem2d, tri.triangles+land_points.shape[0]), axis=0)
            land_points = np.concatenate((land_points, points), axis=0)
        del points    
    
    #_______________________________________________________________________________________
    # do  topographic scaling (potatoefication) for land mesh
    R_grid   = R_earth  
    if do_topo:
        from netCDF4 import Dataset
        from scipy.interpolate import griddata
        
        fid  = Dataset(topo_path,'r')
        topo = fid.variables[topo_varname][:]
        topo[topo<0]=0.0
        lon  = fid.variables[topo_dimname[0]][:]
        lat  = fid.variables[topo_dimname[1]][:]
        fid.close()
        mlon,mlat=np.meshgrid(lon,lat)
        bottom_depth_2d = griddata( np.transpose( (mlon.flatten(),mlat.flatten() ) ), topo.flatten(), land_points, method='nearest')
        R_grid         = R_grid+( bottom_depth_2d*100*potatoefac)
        del topo,lon,lat,mlon,mlat,bottom_depth_2d
        
    #_______________________________________________________________________________________    
    # create sperical ocean coordinates
    xs,ys,zs = grid_cart3d(land_points[:,0], land_points[:,1], R_grid, is_deg=True)
    points   = np.column_stack([xs,ys,zs])
    del xs,ys,zs
    
    #_______________________________________________________________________________________
    # Each cell in the cell array needs to include the size of the cell
    # and the points belonging to the cell.
    cell_size    = np.ones(land_elem2d.shape[0], dtype=np.uint8)*3
    cells        = np.column_stack([cell_size, land_elem2d])
    cells        = cells.ravel()

    # each cell is a VTK_TRIANGLE
    celltypes    = np.empty(land_elem2d.shape[0], dtype=np.uint8)
    celltypes[:] = vtk.VTK_TRIANGLE

    ## the offset array points to the start of each cell (via flat indexing)
    #offset       = np.arange(0, 4*land_elem2d.shape[0], 4, dtype=np.uint32 )
    
    #_______________________________________________________________________________________
    # create pyvista unstrucutred mesh object
    meshpv_land  = pv.UnstructuredGrid(cells, celltypes, points)
    # meshpv_land  = pv.UnstructuredGrid(offset, cells, celltypes, points)
    #del offset, cells, celltypes, points
    del cells, celltypes

    #___do texture coordinates_______________________________________________________________
    # Initialize the texture coordinates array
    if do_texture:
        meshpv_land.active_t_coords      = np.zeros((points.shape[0], 2))
        xs, ys, zs                       = grid_cart3d(land_points[:,0]+90, land_points[:,1], R_grid/R_earth, is_deg=True)
        meshpv_land.active_t_coords[:,0] = 0.5 + np.arctan2(-xs,ys)/(2 * np.pi)
        meshpv_land.active_t_coords[:,1] = 0.5 + np.arcsin(zs)/np.pi
        del xs, ys, zs
    del points
    
    #_______________________________________________________________________________________
    return meshpv_land


#
#
#___CREATE PYVISTA 3d COASTLINE_________________________________________________________
def create_3d_coastline(mesh):
    print(' --> compute 3d coastline')
    
    points_coast = list()
    for niland, lsmask in enumerate(mesh.lsmask_a):    
        xs,ys,zs = grid_cart3d(lsmask[:,0], lsmask[:,1], R_earth*1.001, is_deg=True)
        points = np.vstack((xs,ys,zs)).transpose()
        points = np.row_stack((points,points[1,:])) 

        aux_points = np.column_stack((points[:-1,:],points[1:,:]))
        aux_points = np.stack((points[:-1,:],points[1:,:]), axis=2)
        aux_points = np.moveaxis(aux_points,0,1)
        aux_points = aux_points.reshape(3,2*aux_points.shape[1]).transpose()
        
        points_coast.append(aux_points)
        del aux_points, points    
    return points_coast


#
#
#___CREATE PYVISTA 3D LONGITUDE GRID_______________________________________________________
def create_3d_lonlat_grid(dlon=30,dlat=15,potatoefac=1.0,do_topo=False):
    points_lonlat_grid = list()
    
    print(' --> compute 3d longitude grid')    
    grid_lon = np.arange(-180,180,dlon)
    dum_lat  = np.arange(-85,85+1,1)
    for nlonline in range(0,len(grid_lon)):
        if do_topo: xs,ys,zs = grid_cart3d(dum_lat*0+grid_lon[nlonline], dum_lat, R_earth+(6000*100*potatoefac), is_deg=True)
        else      : xs,ys,zs = grid_cart3d(dum_lat*0+grid_lon[nlonline], dum_lat, R_earth*1.005                , is_deg=True)
        
        points     = np.vstack((xs,ys,zs)).transpose()
        aux_points = np.column_stack((points[:-1,:],points[1:,:]))
        aux_points = np.stack((points[:-1,:],points[1:,:]), axis=2)
        aux_points = np.moveaxis(aux_points,0,1)
        aux_points = aux_points.reshape(3,2*aux_points.shape[1]).transpose()
        
        points_lonlat_grid.append(aux_points)
        del aux_points, points 

    #
    #
    #___CREATE PYVISTA 3D LONGITUDE GRID_______________________________________________________
    print(' --> compute 3d latitude grid')  
    grid_lat = np.arange(-90+dlat,90-dlat+1,dlat)
    grid_lat = np.hstack((np.array(dum_lat[0]), grid_lat, np.array(dum_lat[-1])))
    dum_lon  = np.arange(-180,180+1,1)
    for nlatline in range(0,len(grid_lat)):
        if do_topo: xs,ys,zs = grid_cart3d(dum_lon, dum_lon*0+grid_lat[nlatline], R_earth+(6000*100*potatoefac), is_deg=True)
        else      : xs,ys,zs = grid_cart3d(dum_lon, dum_lon*0+grid_lat[nlatline], R_earth*1.005, is_deg=True)
        
        points     = np.row_stack((xs,ys,zs)).transpose()
        aux_points = np.column_stack((points[:-1,:],points[1:,:]))
        aux_points = np.stack((points[:-1,:],points[1:,:]), axis=2)
        aux_points = np.moveaxis(aux_points,0,1)
        aux_points = aux_points.reshape(3,2*aux_points.shape[1]).transpose()
        
        points_lonlat_grid.append(aux_points)
        del aux_points, points 

    return points_lonlat_grid

#
#
#___CREATE PYVISTA 3D LONGITUDE GRID_______________________________________________________
def create_3d_0lon0lat_grid(dlon=30,dlat=15,potatoefac=1.0,do_topo=False):
    print(' --> compute 3d equator & 0merid line')    
    
    points_0lon0lat = list()
    
    grid_lon = [0]
    dum_lat  = np.arange(-85,85+1,1)
    for nlonline in range(0,len(grid_lon)):
        if do_topo: xs,ys,zs = grid_cart3d(dum_lat*0+grid_lon[nlonline], dum_lat, R_earth+(6000*100*potatoefac), is_deg=True)
        else      : xs,ys,zs = grid_cart3d(dum_lat*0+grid_lon[nlonline], dum_lat, R_earth*1.005                , is_deg=True)
        
        points     = np.vstack((xs,ys,zs)).transpose()
        aux_points = np.column_stack((points[:-1,:],points[1:,:]))
        aux_points = np.stack((points[:-1,:],points[1:,:]), axis=2)
        aux_points = np.moveaxis(aux_points,0,1)
        aux_points = aux_points.reshape(3,2*aux_points.shape[1]).transpose()
        
        points_0lon0lat.append(aux_points)
        del aux_points, points 
    
    grid_lat = [0]
    dum_lon  = np.arange(-180,180+1,1)
    for nlatline in range(0,len(grid_lat)):
        if do_topo: xs,ys,zs = grid_cart3d(dum_lon, dum_lon*0+grid_lat[nlatline], R_earth+(6000*100*potatoefac), is_deg=True)
        else      : xs,ys,zs = grid_cart3d(dum_lon, dum_lon*0+grid_lat[nlatline], R_earth*1.005, is_deg=True)
        
        points     = np.row_stack((xs,ys,zs)).transpose()
        aux_points = np.column_stack((points[:-1,:],points[1:,:]))
        aux_points = np.stack((points[:-1,:],points[1:,:]), axis=2)
        aux_points = np.moveaxis(aux_points,0,1)
        aux_points = aux_points.reshape(3,2*aux_points.shape[1]).transpose()
        
        points_0lon0lat.append(aux_points)
        del aux_points, points 

    return points_0lon0lat
#
#
#___________________________________________________________________________________________________________________
class widget_lon_lat_zoom():
    def __init__(self, plt, init_lon, init_lat, init_zoom):
        # default parameters
        self.kwargs = {'center_lon': init_lon, 'center_lat': init_lat, 'zoom_fac': init_zoom}
        x,y,z = grid_cart3d(self.kwargs['center_lon'], self.kwargs['center_lat'], R_earth*10/self.kwargs['zoom_fac'],is_deg=True)
        plt.camera_position = [ np.array([x,y,z]),(0,0,0),(0,0,1)]   

    def __call__(self, plt, param, value):
        self.kwargs[param] = value
        self.update(plt)

    def update(self, plt):
        x,y,z = grid_cart3d(self.kwargs['center_lon'],self.kwargs['center_lat'], R_earth*10/self.kwargs['zoom_fac'], is_deg=True)
        plt.camera_position = [ np.array([x,y,z]),(0,0,0),(0,0,1)]    
        return
