import pyvista as pv
import numpy as np
import vtk
import matplotlib.pyplot as plt
from   matplotlib.tri     import Triangulation
        
from   .sub_mesh           import * 
from   .sub_colormap       import *
R_earth  = 6371.0e3

#
#
#___CREATE PYVISTA OCEAN MESH___________________________________________________
# make potatoefication of Earth radius 
def create_3dsphere_ocean_mesh(mesh, data, potatoefac=0.5,variable='elevation', do_texture=False):
    print(' --> compute 3d ocean mesh')
    #___________________________________________________________________________
    # do topographic potatoefication of ocean mesh
    R_grid= R_earth
    bottom_depth_2d = -mesh.n_z;
    bottom_depth_2d[mesh.n_i==1]=0.0;
    R_grid         = R_grid-( bottom_depth_2d*100*potatoefac)
    R_grid[mesh.n_i==1]=R_earth;
    
    #___________________________________________________________________________
    # create sperical ocean coordinates
    xs,ys,zs = grid_cart3d(mesh.n_x, mesh.n_y, R_grid, is_deg=True)
    points = np.column_stack([xs,ys,zs])
    del xs,ys,zs
    
    #___________________________________________________________________________
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
    
    #___________________________________________________________________________
    # create pyvista unstrucutred mesh object
    # meshpv_ocean = pv.UnstructuredGrid(offset, cells, celltypes, points)
    meshpv_ocean = pv.UnstructuredGrid(cells, celltypes, points)
    
    #___________________________________________________________________________
    # add variables to ocean mesh
    vname = list(data.keys())[0]
    if not any(x in vname for x in ['ndepth','ntopo','n_depth','n_topo','zcoord','bathymetry']):    
        meshpv_ocean['topo'] = -mesh.n_z
    
    meshpv_ocean[vname] = data[vname].values
        
    del cells, celltypes
    
    #___do texture coordinates__________________________________________________
    # Initialize the texture coordinates array
    if do_texture:
        meshpv_ocean.active_t_coords     = np.zeros((points.shape[0], 2))
        xs, ys, zs                       = grid_cart3d(mesh.n_x+90, mesh.n_y, R_grid/R_earth, is_deg=True)
        meshpv_ocean.active_t_coords[:,0]= 0.5 + np.arctan2(-xs,ys)/(2 * np.pi)
        meshpv_ocean.active_t_coords[:,1]= 0.5 + np.arcsin(zs)/np.pi
        del xs, ys, zs
    del points
    
    #___________________________________________________________________________
    return meshpv_ocean

#
#
#___CREATE PYVISTA LAND MESH TO FILL HOLES______________________________________
def create_3dsphere_land_mesh(mesh, resol=1, potatoefac=1, do_topo=False, topo_path=[], 
                        topo_varname='topo', topo_dimname=['lon','lat'], do_texture=True):
    print(' --> compute 3d land mesh')
    
    from matplotlib.path import Path
    from matplotlib.tri  import Triangulation
    
    #___________________________________________________________________________
    # cycle over all land polygons
    for niland, lsmask in enumerate(mesh.lsmask_a):
        poly_x, poly_y = lsmask[:,0], lsmask[:,1]
        xmin, xmax     = np.floor(poly_x).min(), np.ceil(poly_x).max()
        ymin, ymax     = np.floor(poly_y).min(), np.ceil(poly_y).max()
        
        #resol = 1
        x_m, y_m  = np.meshgrid(np.arange(xmin, xmax, resol),np.arange(ymin, ymax, resol))
        x_m, y_m  = x_m.reshape((x_m.size, 1)), y_m.reshape((y_m.size, 1))
        
        #_______________________________________________________________________
        # check if regular points are within polygon
        IN        = Path(lsmask).contains_points(np.concatenate((x_m, y_m),axis=1)) 
        x_m, y_m  = x_m[IN==True], y_m[IN==True]
        del IN
        
        #_______________________________________________________________________
        # combine polygon points and regular points within polygon --> do triangulation
        outeredge = np.vstack((poly_x, poly_y)).transpose()
        points    = np.hstack((x_m, y_m))
        points    = np.vstack((outeredge,points))
        if np.unique(points[:,0]).size<=3 or np.unique(points[:,1]).size<=3 : continue
        tri       = Triangulation(points[:,0], points[:,1])
        del outeredge, poly_x, poly_y
        
        #_______________________________________________________________________
        # compute trinagle centroids and check if they are within polygon
        tri_cx    = np.sum(points[tri.triangles,0],axis=1)/3
        tri_cy    = np.sum(points[tri.triangles,1],axis=1)/3
        tri_cx    = np.reshape(tri_cx,(tri_cx.size,1))
        tri_cy    = np.reshape(tri_cy,(tri_cy.size,1))
        IN        = Path(lsmask).contains_points(np.concatenate((tri_cx,tri_cy),axis=1))
        tri.triangles=tri.triangles[IN==True,:]
        del tri_cx, tri_cy, IN
        
        #_______________________________________________________________________
        # concatenate all land trinagles
        if niland==0:
            land_points = points
            land_elem2d = tri.triangles
        else:
            land_elem2d = np.concatenate((land_elem2d, tri.triangles+land_points.shape[0]), axis=0)
            land_points = np.concatenate((land_points, points), axis=0)
        del points    
    
    #___________________________________________________________________________
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
        bottom_depth_2d = griddata( np.transpose( (mlon.flatten(),mlat.flatten() ) ), topo.flatten(), land_points, method='linear')
        R_grid         = R_grid+( bottom_depth_2d*100*potatoefac)
        del topo,lon,lat,mlon,mlat
        
    #___________________________________________________________________________
    # create sperical ocean coordinates
    xs,ys,zs = grid_cart3d(land_points[:,0], land_points[:,1], R_grid, is_deg=True)
    points   = np.column_stack([xs,ys,zs])
    del xs,ys,zs
    
    #___________________________________________________________________________
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
    
    #___________________________________________________________________________
    # create pyvista unstrucutred mesh object
    meshpv_land  = pv.UnstructuredGrid(cells, celltypes, points)
    # meshpv_land  = pv.UnstructuredGrid(offset, cells, celltypes, points)
    #del offset, cells, celltypes, points
    del cells, celltypes

    #___do texture coordinates__________________________________________________
    # Initialize the texture coordinates array
    if do_texture:
        meshpv_land.active_t_coords      = np.zeros((points.shape[0], 2))
        xs, ys, zs                       = grid_cart3d(land_points[:,0]+90, land_points[:,1], R_grid/R_earth, is_deg=True)
        meshpv_land.active_t_coords[:,0] = 0.5 + np.arctan2(-xs,ys)/(2 * np.pi)
        meshpv_land.active_t_coords[:,1] = 0.5 + np.arcsin(zs)/np.pi
        del xs, ys, zs
    del points
    
    #___________________________________________________________________________
    # add land topography data to pyvista mesh object
    if do_topo: 
        meshpv_land['topo'] = bottom_depth_2d
        print(bottom_depth_2d.shape)
        del bottom_depth_2d
    
    #___________________________________________________________________________
    return meshpv_land

#
#
#___CREATE PYVISTA 3d COASTLINE_________________________________________________
def create_3dsphere_coastline(mesh):
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
#___CREATE PYVISTA 3D LONGITUDE GRID____________________________________________
def create_3dsphere_lonlat_grid(dlon=30,dlat=15,potatoefac=1.0,do_topo=False):
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
    #___CREATE PYVISTA 3D LONGITUDE GRID________________________________________
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
    
    #___________________________________________________________________________
    return points_lonlat_grid

#
#
#___CREATE PYVISTA 3D LONGITUDE GRID____________________________________________
def create_3dsphere_0lon0lat_grid(dlon=30,dlat=15,potatoefac=1.0,do_topo=False):
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
        
    #___________________________________________________________________________
    return points_0lon0lat

#
#
#_______________________________________________________________________________
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



#
#
#___CREATE PYVISTA OCEAN MESH___________________________________________________
# make potatoefication of Earth radius 
def create_3dflat_ocean_bottom(xs, ys , zs, e_i, data, vname='elevation', box=None):
    print(' --> compute 3d flat ocean bottom mesh')
    
    #___________________________________________________________________________
    # create vertice array  
    if box is not None:
        points    = np.column_stack([xs+(box[1]-box[0])/2.0, ys+(box[3]-box[2])/2.0, zs])
    else:     
        points    = np.column_stack([xs,ys,zs])
    
    #___________________________________________________________________________
    # Each cell in the cell array needs to include the size of the cell
    # and the points belonging to the cell.  In this example, there are 8
    # hexahedral cells that have common points between them.
    cell_size = np.ones(e_i.shape[0], dtype=np.uint8)*3
    cells     = np.column_stack([cell_size, e_i])
    cells     = cells.ravel()
    del cell_size

    # each cell is a VTK_TRIANGLE
    celltypes = np.empty(e_i.shape[0], dtype=np.uint8)
    celltypes[:] = vtk.VTK_TRIANGLE
    #___________________________________________________________________________
    # create pyvista unstrucutred mesh object
    # meshpv_ocean = pv.UnstructuredGrid(offset, cells, celltypes, points)
    meshpv_ocean_bottom = pv.UnstructuredGrid(cells, celltypes, points)
    del cells, celltypes
    
    #___________________________________________________________________________
    # add variables to ocean mesh
    meshpv_ocean_bottom[vname] = data
    
    #___do texture coordinates__________________________________________________
    # Initialize the texture coordinates array
    # Initialize the texture coordinates array
    meshpv_ocean_bottom.active_t_coords      = np.zeros((points.shape[0], 2))
    meshpv_ocean_bottom.active_t_coords[:,0] = xs/360 + 0.5
    meshpv_ocean_bottom.active_t_coords[:,1] = ys/180 + 0.5    
    del points
    del xs, ys, zs
    
    #___________________________________________________________________________
    return meshpv_ocean_bottom



#
#
#___CREATE PYVISTA OCEAN MESH___________________________________________________
# make potatoefication of Earth radius 
def create_3dflat_ocean_surface(xs, ys, zs, e_i, bndn_xs, bndn_ys, killdist=600, box=None):
    print(' --> compute 3d flat ocean surface')
        
    #___________________________________________________________________________
    # compute distance from boundary nodes coordinates bndn_xs, bndn_ys
    npts = len(xs)
    dist = np.zeros((npts,))
    for ii in range(npts):
        # distance to coast in km 
        dphi = np.sqrt( (xs[ii]-bndn_xs)**2 + (ys[ii]-bndn_ys)**2)
        dist[ii] = np.min(dphi/180*np.pi*R_earth/1000)
        
    #___________________________________________________________________________
    tdist = dist[e_i].sum(axis=1)/2.0
    if killdist is not None:
        e_i = e_i[tdist<killdist,:]
        e_i, idx_n = reindex_regional_elem(e_i)
        xs, ys, zs, dist = xs[idx_n], ys[idx_n], zs[idx_n], dist[idx_n]
        
    #___________________________________________________________________________
    # create vertice array  
    if box is not None:
        points    = np.column_stack([xs+(box[1]-box[0])/2.0, ys+(box[3]-box[2])/2.0, zs*0])
    else:     
        points    = np.column_stack([xs,ys,zs*0])        
    #points = np.column_stack([xs,ys,zs*0])
    
    #___________________________________________________________________________
    # Each cell in the cell array needs to include the size of the cell
    # and the points belonging to the cell.  In this example, there are 8
    # hexahedral cells that have common points between them.
    cell_size = np.ones(e_i.shape[0], dtype=np.uint8)*3
    cells     = np.column_stack([cell_size, e_i])
    cells     = cells.ravel()
    del cell_size

    # each cell is a VTK_TRIANGLE
    celltypes = np.empty(e_i.shape[0], dtype=np.uint8)
    celltypes[:] = vtk.VTK_TRIANGLE

    #___________________________________________________________________________
    # create pyvista unstrucutred mesh object
    # meshpv_ocean = pv.UnstructuredGrid(offset, cells, celltypes, points)
    meshpv_ocean_surface = pv.UnstructuredGrid(cells, celltypes, points)
    meshpv_ocean_surface['dist'] = dist
    
    #___do texture coordinates__________________________________________________
    # Initialize the texture coordinates array
    meshpv_ocean_surface.active_t_coords      = np.zeros((points.shape[0], 2))
    meshpv_ocean_surface.active_t_coords[:,0] = xs/360 + 0.5
    meshpv_ocean_surface.active_t_coords[:,1] = ys/180 + 0.5    
    del points
    del xs, ys, zs
    
    #___________________________________________________________________________
    return meshpv_ocean_surface



#
#
#___CREATE PYVISTA OCEAN MESH___________________________________________________
# make potatoefication of Earth radius 
def create_3dflat_ocean_wall(xs, ys, zs, e_i, which_wall='N', nsigma=20, box=None):
    print(' --> compute 3d flat ocean {} wall'.format(which_wall))
    #___________________________________________________________________________
    # compute boundary edge of box limitet domain
    bnde = compute_boundary_edges(e_i)
    
    #___COMPUTE northern ocean wall mesh with sigma layers______________________
    ed_xs, ed_ys = xs[bnde].copy(),  ys[bnde].copy()
    if   which_wall == 'N': wall_ibnde = ((ed_ys[:,0]==ys.max()) & (ed_ys[:,1]==ys.max()))
    elif which_wall == 'S': wall_ibnde = ((ed_ys[:,0]==ys.min()) & (ed_ys[:,1]==ys.min()))
    elif which_wall == 'E': wall_ibnde = ((ed_xs[:,0]==xs.max()) & (ed_xs[:,1]==xs.max()))
    elif which_wall == 'W': wall_ibnde = ((ed_xs[:,0]==xs.min()) & (ed_xs[:,1]==xs.min()))
    wall_bnde  = bnde[wall_ibnde,:]
    
    #___________________________________________________________________________
    wall_bnde, wall_idx_n = reindex_regional_elem(wall_bnde)
    wall_xs, wall_ys, wall_zs = xs[wall_idx_n].copy(), ys[wall_idx_n].copy(), zs[wall_idx_n].copy() 
    wall_npts = len(wall_xs)
        
    #___________________________________________________________________________
    # build sigma layer distribution 
    nsigma   = 20
    dsigma   = np.logspace(0, 1, num=nsigma, endpoint=True, base=100.0)
    dsigma   = dsigma-dsigma.min()
    dsigma   = dsigma/dsigma.max()
        
    #___________________________________________________________________________
    # build sigma layer quad mesh vertices and element array
    wall_sigma_xs = wall_xs
    wall_sigma_ys = wall_ys
    wall_sigma_zs = np.zeros((wall_npts,))
    wall_sigma_vs = np.zeros((wall_npts,))
    for ii in range(nsigma-1):
        wall_sigma_xs = np.hstack([wall_sigma_xs, wall_xs])
        wall_sigma_ys = np.hstack([wall_sigma_ys, wall_ys])
        wall_sigma_zs = np.hstack([wall_sigma_zs, dsigma[ii+1]*wall_zs])
        wall_sigma_vs = np.hstack([wall_sigma_vs, wall_zs*0+dsigma[ii+1]])
        # build vertical quad elements over edges 
        # ed[0,:] o----->o ed[1,:]    quad faces: [ ed[0,:], ed[1,:], ed[1,:], ed[0,:] ]
        #         |      |
        #         |      |
        # ed[0,:] o<-----o ed[1,:] 
        aux_e_i = np.column_stack([wall_bnde[:,0]+wall_npts*ii, 
                                   wall_bnde[:,1]+wall_npts*ii, 
                                   wall_bnde[:,1]+wall_npts*(ii+1), 
                                   wall_bnde[:,0]+wall_npts*(ii+1) ])
        if ii==0: wall_sigma_e_i = aux_e_i
        else    : wall_sigma_e_i = np.vstack((wall_sigma_e_i,aux_e_i))  
        
    #___________________________________________________________________________
    #points = np.column_stack([wall_sigma_xs, wall_sigma_ys, wall_sigma_zs])
    if box is not None:
        points    = np.column_stack([wall_sigma_xs+(box[1]-box[0])/2.0, wall_sigma_ys+(box[3]-box[2])/2.0, wall_sigma_zs])
    else:     
        points    = np.column_stack([wall_sigma_xs, wall_sigma_ys, wall_sigma_zs])

    #___________________________________________________________________________
    cell_size = np.ones(wall_sigma_e_i.shape[0], dtype=np.uint8)*4
    cells     = np.column_stack([cell_size, wall_sigma_e_i])
    cells     = cells.ravel()
    del cell_size

    # each cell is a VTK_TRIANGLE
    celltypes = np.empty(wall_sigma_e_i.shape[0], dtype=np.uint8)
    celltypes[:] = vtk.VTK_QUAD

    #___________________________________________________________________________
    # create pyvista unstrucutred mesh object
    meshpv_ocean_wall = pv.UnstructuredGrid(cells, celltypes, points)
    meshpv_ocean_wall['sigma'] = wall_sigma_vs
    del cells, celltypes, points
        
    #___________________________________________________________________________
    return meshpv_ocean_wall



#
#
#___CREATE PYVISTA OCEAN MESH___________________________________________________
# make potatoefication of Earth radius 
def create_3dflat_ocean_botwall(xs, ys, zs, e_i, box, scalfac, botdepth=7000):
    print(' --> compute 3d flat ocean mesh')
    
    #___________________________________________________________________________
    # compute boundary edge of box limitet domain
    bnde = compute_boundary_edges(e_i)
    
        
    #___COMPUTE lower ocean box boundary____________________________________
    #box_bnde = bnde.copy()
    ed_xs, ed_ys = xs[bnde].copy(),  ys[bnde].copy()
    box_ibnde = (((ed_xs[:,0]==box[0]) & (ed_xs[:,1]==box[0])) |
                    ((ed_xs[:,0]==box[1]) & (ed_xs[:,1]==box[1])) |
                    ((ed_ys[:,0]==box[2]) & (ed_ys[:,1]==box[2])) |
                    ((ed_ys[:,0]==box[3]) & (ed_ys[:,1]==box[3])))
    box_bnde  = bnde[box_ibnde,:]
    #del box_ibnde, bnde
        
    #_______________________________________________________________________
    # re-index regional elem index
    box_bnde, box_idx_n = reindex_regional_elem(box_bnde)
    box_xs, box_ys, box_zs = xs[box_idx_n].copy(), ys[box_idx_n].copy(), zs[box_idx_n].copy() 
    box_npts = box_xs.size
        
    #_______________________________________________________________________
    # build vertical quad elements over edges 
    # ed[0,:] o----->o ed[1,:]    quad faces: [ ed[0,:], ed[1,:], ed[1,:], ed[0,:] ]
    #         |      |
    #         |      |
    # ed[0,:] o<-----o ed[1,:] 
    box_xs = np.hstack([box_xs, box_xs])
    box_ys = np.hstack([box_ys, box_ys])
    box_zs = np.hstack([box_zs, box_zs*0+botdepth**(1/scalfac)])
    box_e_i = np.column_stack([box_bnde[:,0], box_bnde[:,1], box_bnde[:,1]+box_npts, box_bnde[:,0]+box_npts])
    
    ocean_botwall=dict()
    ocean_botwall['xs' ]=box_xs#+(box[1]-box[0])/2.0
    ocean_botwall['ys' ]=box_ys#+(box[3]-box[2])/2.0
    ocean_botwall['zs' ]=box_zs
    ocean_botwall['e_i']=box_e_i
    
    ##_______________________________________________________________________
    #points = np.column_stack([box_xs, box_ys, box_zs])
        
    ##_______________________________________________________________________
    #cell_size = np.ones(box_e_i.shape[0], dtype=np.uint8)*4
    #cells     = np.column_stack([cell_size, box_e_i])
    #cells     = cells.ravel()
    #del cell_size

    ## each cell is a VTK_TRIANGLE
    #celltypes = np.empty(box_e_i.shape[0], dtype=np.uint8)
    #celltypes[:] = vtk.VTK_QUAD
    
    ##_______________________________________________________________________
    ## create pyvista unstrucutred mesh object
    #meshpv_ocean_botwall = pv.UnstructuredGrid(cells, celltypes, points)
    #del cells, celltypes, points
    
    #___________________________________________________________________________
    #return meshpv_ocean_botwall
    return ocean_botwall
    


#
#
#___CREATE PYVISTA LAND MESH TO FILL HOLES______________________________________
def create_3dflat_land_mesh(mesh, resol=1, box=None, do_topo=False, topo_path=[], 
                        topo_varname='topo', topo_dimname=['lon','lat'], do_texture=True, scalfac=1, botdepth = 7000):
    print(' --> compute 3d flat land mesh')
    
    from matplotlib.path import Path
    from matplotlib.tri  import Triangulation
    
    #___________________________________________________________________________
    # cycle over all land polygons
    cnt = 0
    for niland, lsmask in enumerate(mesh.lsmask_a):
        
        #_______________________________________________________________________
        poly_x, poly_y = lsmask[:,0], lsmask[:,1]
        
        inbox = ((poly_x >= box[0]) & (poly_x <= box[1]) & (poly_y >= box[2]) & (poly_y <= box[3]))
        if np.any(inbox)==False : continue
        
        xmin, xmax     = np.floor(poly_x).min(), np.ceil(poly_x).max()
        ymin, ymax     = np.floor(poly_y).min(), np.ceil(poly_y).max()
        #resol = 1
        x_m, y_m  = np.meshgrid(np.arange(xmin, xmax, resol, dtype=np.float32),np.arange(ymin, ymax, resol, dtype=np.float32))
        
        #xmin, xmax     = box[0], box[1]
        #ymin, ymax     = box[2], box[3]
        #nresol = np.int32(np.ceil((xmax-xmin)/resol))
        #x_m, y_m  = np.meshgrid(np.linspace(xmin, xmax, nresol),np.linspace(ymin, ymax, nresol))
        x_m, y_m  = x_m.reshape((x_m.size, 1)), y_m.reshape((y_m.size, 1))
        
        #_______________________________________________________________________
        # check if regular points are within polygon
        IN        = Path(lsmask).contains_points(np.concatenate((x_m, y_m),axis=1)) 
        x_m, y_m  = x_m[IN==True], y_m[IN==True]
        del IN
        
        #_______________________________________________________________________
        # combine polygon points and regular points within polygon --> do triangulation
        outeredge = np.vstack((poly_x, poly_y)).transpose()
        points    = np.hstack((x_m, y_m))
        points    = np.vstack((outeredge,points))
        tri       = Triangulation(points[:,0], points[:,1])
        del outeredge, poly_x, poly_y
        
        #_______________________________________________________________________
        # compute trinagle centroids and check if they are within polygon
        tri_cx    = np.sum(points[tri.triangles,0],axis=1)/3
        tri_cy    = np.sum(points[tri.triangles,1],axis=1)/3
        tri_cx    = np.reshape(tri_cx,(tri_cx.size,1))
        tri_cy    = np.reshape(tri_cy,(tri_cy.size,1))
        IN        = Path(lsmask).contains_points(np.concatenate((tri_cx,tri_cy),axis=1))
        tri.triangles=tri.triangles[IN==True,:]
        del tri_cx, tri_cy, IN
        
        #_______________________________________________________________________
        # concatenate all land trinagles
        if cnt==0:
            land_points = points
            land_elem2d = tri.triangles
            cnt = 1
        else:
            land_elem2d = np.concatenate((land_elem2d, tri.triangles+land_points.shape[0]), axis=0)
            land_points = np.concatenate((land_points, points), axis=0)
        del points    
    
    #___________________________________________________________________________
    # create sperical ocean coordinates
    xs, ys, zs = land_points[:,0], land_points[:,1], land_points[:,1]*0.0
    
    #___________________________________________________________________________
    # do  topographic scaling (potatoefication) for land mesh
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
        zs = griddata( np.transpose( (mlon.flatten(),mlat.flatten() ) ), topo.flatten(), land_points, method='linear')
        del topo,lon,lat,mlon,mlat
    
    zs = -zs**(1/scalfac)
    
    #___________________________________________________________________________
    # limit vertie array to box 
    if box is not None:
        idx_e = grid_cutbox_e(xs, ys, land_elem2d, box, which='soft')
        land_elem2d = land_elem2d[idx_e, :] 
        
        # re-index regional elem index
        land_elem2d, idx_n = reindex_regional_elem(land_elem2d)
        e_i = land_elem2d
        xs, ys, zs = xs[idx_n], ys[idx_n], zs[idx_n]
        
        # limit xy vertice coordinates to box walls 
        xs[xs<box[0]] = box[0]
        xs[xs>box[1]] = box[1]
        ys[ys<box[2]] = box[2]
        ys[ys>box[3]] = box[3]
        
        #___________________________________________________________________________
        # look for boundary edge points that have the same coodinates as the box
        edge    = np.concatenate((e_i[:,[0,1]], e_i[:,[0,2]], e_i[:,[1,2]]),axis=0)
        edge    = np.sort(edge,axis=1) 
            
        ## python  sortrows algorythm --> matlab equivalent
        edge    = edge.tolist()
        edge.sort()
        edge    = np.array(edge)
            
        idx     = np.diff(edge,axis=0)==0
        idx     = np.all(idx,axis=1)
        idx     = np.logical_or(np.concatenate((idx,np.array([False]))),\
                                np.concatenate((np.array([False]),idx)))
            
        # all edges that belong to boundary own jsut one triangle 
        bnde    = edge[idx==False,:]
        del edge, idx
        
        #_______________________________________________________________________
        ed_xs, ed_ys = xs[bnde].copy(),  ys[bnde].copy()
        box_ibnde = (((ed_xs[:,0]==box[0]) & (ed_xs[:,1]==box[0])) |
                     ((ed_xs[:,0]==box[1]) & (ed_xs[:,1]==box[1])) |
                     ((ed_ys[:,0]==box[2]) & (ed_ys[:,1]==box[2])) |
                     ((ed_ys[:,0]==box[3]) & (ed_ys[:,1]==box[3])))
        box_bnde  = bnde[box_ibnde,:]
        
        # make sure coastal land points have depth zero so there is no gap with 
        # ocean
        box_bnde_land = bnde[box_ibnde==False,:]
        box_bndn_land = np.unique(box_bnde_land.flatten())
        zs[box_bndn_land] = 0.0
        del box_ibnde, bnde
        
        #_______________________________________________________________________
        # re-index regional elem index
        box_bnde, box_idx_n = reindex_regional_elem(box_bnde)
        box_xs, box_ys, box_zs = xs[box_idx_n].copy(), ys[box_idx_n].copy(), zs[box_idx_n].copy() 
        box_npts = box_xs.size
        
        #_______________________________________________________________________
        box_xs = np.hstack([box_xs, box_xs])
        box_ys = np.hstack([box_ys, box_ys])
        box_zs = np.hstack([box_zs, box_zs*0+botdepth**(1/scalfac)])
        box_e_i = np.column_stack([box_bnde[:,0], box_bnde[:,1], box_bnde[:,1]+box_npts, box_bnde[:,0]+box_npts])
        
        land_botwall=dict()
        land_botwall['xs' ]=box_xs#+(box[1]-box[0])/2.0
        land_botwall['ys' ]=box_ys#+(box[3]-box[2])/2.0
        land_botwall['zs' ]=box_zs
        land_botwall['e_i']=box_e_i
        
        ##_______________________________________________________________________
        #box_points = np.column_stack([box_xs, box_ys, box_zs])
        
        ##___________________________________________________________________________
        #box_cell_size = np.ones(box_e_i.shape[0], dtype=np.uint8)*4
        #box_cells     = np.column_stack([box_cell_size, box_e_i])
        #box_cells     = box_cells.ravel()
        #del box_cell_size

        ## each cell is a VTK_TRIANGLE
        #box_celltypes = np.empty(box_e_i.shape[0], dtype=np.uint8)
        #box_celltypes[:] = vtk.VTK_QUAD

        ##___________________________________________________________________________
        ## create pyvista unstrucutred mesh object
        #meshpv_land_box = pv.UnstructuredGrid(box_cells, box_celltypes, box_points)
        #del box_cells, box_celltypes, box_points
        

    #___________________________________________________________________________
    # create vertice array  
    points   = np.column_stack([xs+(box[1]-box[0])/2.0, ys+(box[3]-box[2])/2.0, zs])
    
    #___________________________________________________________________________
    # Each cell in the cell array needs to include the size of the cell
    # and the points belonging to the cell.
    cell_size    = np.ones(land_elem2d.shape[0], dtype=np.uint8)*3
    cells        = np.column_stack([cell_size, land_elem2d])
    cells        = cells.ravel()

    # each cell is a VTK_TRIANGLE
    celltypes    = np.empty(land_elem2d.shape[0], dtype=np.uint8)
    celltypes[:] = vtk.VTK_TRIANGLE
    
    #___________________________________________________________________________
    # create pyvista unstrucutred mesh object
    meshpv_land  = pv.UnstructuredGrid(cells, celltypes, points)
    del cells, celltypes

    #___do texture coordinates__________________________________________________
    # Initialize the texture coordinates array
    if do_texture:
        meshpv_land.active_t_coords      = np.zeros((points.shape[0], 2))
        meshpv_land.active_t_coords[:,0] = xs/360 + 0.5
        meshpv_land.active_t_coords[:,1] = ys/180 + 0.5
    del points, xs, ys
    
    #___________________________________________________________________________
    # add land topography data to pyvista mesh object
    if do_topo: meshpv_land['topo'] = zs
    del zs    
    
    #___________________________________________________________________________
    return meshpv_land, land_botwall



def reindex_regional_elem(faces):
    import numpy as np
    import itertools
    
    #___________________________________________________________________________
    is_list = False
    if isinstance(faces, list):
        is_list=True
        pass
    elif isinstance(faces, np.ndarray):
        faces = faces.tolist()
    else:
        raise IOError("points should be either a list or a numpy array.")
    
    #___________________________________________________________________________
    # set() to remove repeated indices and list() to order them for later use:
    indices_to_keep = list(set(itertools.chain(*faces)))
    reindex = dict([(old_index, new_index)
                    for new_index, old_index in enumerate(indices_to_keep)])

    new_faces = [[reindex[old_index] for old_index in face] for face in faces]
    
    #___________________________________________________________________________
    original_indices = indices_to_keep
    
    #___________________________________________________________________________
    # convert new elem list to array when the original input was an array
    if not is_list: new_faces = np.asarray(new_faces)
    
    #___________________________________________________________________________
    return new_faces, original_indices 



def combine_ocean_land_botwall(ocean_botwall, land_botwall, box ):
    #___________________________________________________________________________
    ocean_npts = ocean_botwall['xs'].size
    
    xs  = np.hstack((ocean_botwall['xs' ], land_botwall['xs']))
    ys  = np.hstack((ocean_botwall['ys' ], land_botwall['ys']))
    zs  = np.hstack((ocean_botwall['zs' ], land_botwall['zs']))
    e_i = np.vstack((ocean_botwall['e_i'], land_botwall['e_i']+ocean_npts))
    
    #___________________________________________________________________________
    # need to kickout identical vertice points
    # --> https://itecnote.com/tecnote/python-checking-for-and-indexing-non-unique-duplicate-values-in-a-numpy-array/
    points   = np.column_stack([xs, ys, zs])
    unq, unq_idx, unq_cnt = np.unique(points,axis=0, return_inverse=True, return_counts=True)
    cnt_mask = unq_cnt > 1
    dup_ids  = unq[cnt_mask]
    cnt_idx, = np.nonzero(cnt_mask)
    idx_mask = np.in1d(unq_idx, cnt_idx)
    idx_idx, = np.nonzero(idx_mask)
    srt_idx  = np.argsort(unq_idx[idx_mask])
    dup_idx  = np.split(idx_idx[srt_idx], np.cumsum(unq_cnt[cnt_mask])[:-1])
    for arr in dup_idx: 
        e_i[e_i==arr[1]] = arr[0]
        
    #re-index regional elem index
    e_i, idx_n = reindex_regional_elem(e_i)
    points = points[idx_n,:]    
    
    # select northern bottom wall 
    ys =  points[:,1].copy()
    xs =  points[:,0].copy()
    
    points[:,0] = points[:,0]+(box[1]-box[0])/2.0
    points[:,1] = points[:,1]+(box[3]-box[2])/2.0
    
    
    idx_W        = np.all( ys[e_i]==np.float32(box[3]), axis=1)
    e_i_W        = e_i[idx_W,:]
    e_i_W, idx_n = reindex_regional_elem(e_i_W)
    points_W     = points[idx_n,:].copy()
    cell_size    = np.ones(e_i_W.shape[0], dtype=np.uint8)*4
    cells        = np.column_stack([cell_size, e_i_W]).ravel()
    celltypes    = np.empty(e_i_W.shape[0], dtype=np.uint8)
    celltypes[:] = vtk.VTK_QUAD
    meshpv_botwall_N = pv.UnstructuredGrid(cells, celltypes, points_W)
    
    # select southern bottom wall 
    idx_W        = np.all( ys[e_i]==np.float32(box[2]), axis=1)
    e_i_W        = e_i[idx_W,:]
    e_i_W, idx_n = reindex_regional_elem(e_i_W)
    points_W     = points[idx_n,:].copy()
    cell_size    = np.ones(e_i_W.shape[0], dtype=np.uint8)*4
    cells        = np.column_stack([cell_size, e_i_W]).ravel()
    celltypes    = np.empty(e_i_W.shape[0], dtype=np.uint8)
    celltypes[:] = vtk.VTK_QUAD
    meshpv_botwall_S = pv.UnstructuredGrid(cells, celltypes, points_W)
    
    # select western bottom wall 
    idx_W        = np.all( xs[e_i]==np.float32(box[0]), axis=1)
    e_i_W        = e_i[idx_W,:]
    e_i_W, idx_n = reindex_regional_elem(e_i_W)
    points_W     = points[idx_n,:].copy()
    cell_size    = np.ones(e_i_W.shape[0], dtype=np.uint8)*4
    cells        = np.column_stack([cell_size, e_i_W]).ravel()
    celltypes    = np.empty(e_i_W.shape[0], dtype=np.uint8)
    celltypes[:] = vtk.VTK_QUAD
    meshpv_botwall_W = pv.UnstructuredGrid(cells, celltypes, points_W)
    
    # select western bottom wall 
    idx_W        = np.all( xs[e_i]==np.float32(box[1]), axis=1)
    e_i_W        = e_i[idx_W,:]
    e_i_W, idx_n = reindex_regional_elem(e_i_W)
    points_W     = points[idx_n,:].copy()
    cell_size    = np.ones(e_i_W.shape[0], dtype=np.uint8)*4
    cells        = np.column_stack([cell_size, e_i_W]).ravel()
    celltypes    = np.empty(e_i_W.shape[0], dtype=np.uint8)
    celltypes[:] = vtk.VTK_QUAD
    meshpv_botwall_E = pv.UnstructuredGrid(cells, celltypes, points_W)
    
    #___________________________________________________________________________
    return meshpv_botwall_N, meshpv_botwall_S, meshpv_botwall_W, meshpv_botwall_E
