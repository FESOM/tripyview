import bpy
import bmesh
import os
import tripyview as tpv 
import numpy     as np
import xarray    as xr
import time      as clock
import warnings
from matplotlib.path import Path
from matplotlib.tri  import Triangulation
import sys

tpv_path  = os.path.dirname(os.path.dirname(tpv.__file__))
sys.path.append(tpv_path+'/tools/blender')
print(tpv_path+'/tools/blender')
from sub_blender  import blender_create_mesh, blender_create_txtremat

R_earth             = 6371.0e3 # meter

# real extrusion for ocean and land when !=0
potatoefac_ocean    = 0.0 
potatoefac_land     = 0.0 

# main path of data for blender project is used now the structure
# blender_fesom2/
#        |--data        can contain fesom data and mesh folder
#        |    |--mesh  
#        |--texture     contains texture and bump images
#        |--lsmask      contains land sea mask of original mesh to mask out regions 
#        |              from texture files
#        |--topo        folder with etopo1 data              
gen_path             = '/home/pscholz/Python/blender_fesom2/'

mesh_path, do_rot    = gen_path+'data/core2_srt_dep@node/', 'None'

oce_txturepath       = gen_path+'texture/Albedo.jpg'
oce_alphamap         = gen_path+'lsmask/core2_lsmask_ocean.jpg'
oce_bumpmap          = gen_path+'texture/Bump.jpg'
oce_bumpstrength     = 0.5

# load fesom2 data from path, if None no data are loaded
oce_datapath         = None # gen_path+'/data/'
oce_vname            = 'ssh'
oce_year             = [1958, 1958]
oce_mon              = None
oce_day              = None
oce_depth            = None

land_txturepath      = gen_path+'texture/Albedo.jpg'
land_alphamap        = gen_path+'lsmask/core2_lsmask_ocean.jpg'
land_bumpmap         = gen_path+'texture/Bump.jpg'
land_bumpstrength    = 0.5

# load topographic information to interpolate on land mesh part, to give it a 
# real extrusion. I used here very coarse etopo1 data. It is maybe more efficient 
# to do a fake extrusion via bump maps
topo_path, do_topo         = gen_path+'topo/topo_1deg.nc', True
topo_varname, topo_dimname = 'topo', ['lon','lat']
topo_resol                 = 1



#
#
#
#___LOAD FESOM2 MESH____________________________________________________________
mesh=tpv.load_mesh_fesom2(mesh_path, do_rot=do_rot, focus=0, do_info=True, do_pickle=False,
                          do_earea=True, do_narea=True, do_eresol=[True,'mean'], do_nresol=[True,'eresol'])



#
#
#
#===============================================================================
#=== CREATE BLENDER OCEAN MESH =================================================
#===============================================================================
n_x      = np.hstack((mesh.n_x,mesh.n_xa))
n_y      = np.hstack((mesh.n_y,mesh.n_ya))
n_z      = np.hstack((mesh.n_z,mesh.n_za))
n_i      = np.hstack((mesh.n_i,mesh.n_i[mesh.n_pbnd_a]))
n_resol  = np.hstack((mesh.n_resol,mesh.n_resol[mesh.n_pbnd_a]))
e_i      = np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia))
    
# convert sperical coordinates into cartesian coordinates
xs,ys,zs = tpv.grid_cart3d(n_x, n_y, 1.0, is_deg=True)

# create blender vertice tuple list 
# vertices = [ (V1_x, V1_y, V1_z), 
#              (V2_x, V2_y, V2_z), 
#              ...
#            ]
vertices = list(zip(xs, ys, zs))

# create blender normal are with respect to face
# normals = [ (T1x, T1y, T1z), 
#             (T2x, T2y, T2z), 
#              ...
#             ]
normals = list(zip(xs[e_i].sum(axis=1)/3, ys[e_i].sum(axis=1)/3, zs[e_i].sum(axis=1)/3 ))
del xs,ys,zs

# create blender triangle tuple list 
# triangles = [ (T1_V1, T1_V2, T1_V3), 
#               (T2_V1, T2_V2, T2_V3), 
#              ...
#             ]
triangles = list(zip(e_i[:,0], e_i[:,1], e_i[:,2] ))

# compute vertice uv mapping coordinates
#xs, ys, zs = tpv.grid_cart3d(mesh.n_x+90, mesh.n_y, 1.0, is_deg=True)
#u          = 0.5 + np.arctan2(-xs,ys)/(2 * np.pi)
xs, ys, zs = tpv.grid_cart3d(n_x, n_y, 1.0, is_deg=True)
u          = 0.5 + np.arctan2(ys,xs)/(2 * np.pi)
v          = 0.5 + np.arcsin(zs)/np.pi
uv         = list(zip(u, v))
del(xs, ys, zs, u, v)

#
#
#______________________________________________________________________________________
# add additional attribute variables that can be used in blender via an attribute node
add_meshattr           = dict()
add_meshattr['ntopo']  = -np.abs(n_z)
add_meshattr['nresol'] = n_resol

#
#
#______________________________________________________________________________________
# load specific fesom2 data with tripyview into additional attribute variables that can
# be used in blender via an attribute node
add_dataattr = dict()
if oce_datapath is not None and oce_vname is not None:
    # load data with tripyview
    data = tpv.load_data_fesom2(mesh, oce_datapath, vname=oce_vname, year=oce_year, mon=oce_mon, day=oce_day, depth=oce_depth, 
                                do_load=True, do_persist=False)
    data_plot = data[oce_vname].values
    add_dataattr[oce_vname] = np.hstack((data_plot, data_plot[mesh.n_pbnd_a]))

#
#
#______________________________________________________________________________________
# create blender mesh object
obj_ocean = blender_create_mesh('ocean', 
                                vertices, 
                                triangles, 
                                normals, 
                                uv, 
                                add_meshattr=add_meshattr, 
                                add_dataattr=add_dataattr, 
                                shade_flat=True )
                                
#
#
#______________________________________________________________________________________
# create blender texture material 
obj_ocean = blender_create_txtremat('ocean', obj_ocean, 
                                        oce_txturepath, 
                                        alphamap=oce_alphamap, alpha_invert=False, 
                                        bumpmap=oce_bumpmap, bump_strength=0.1, bump_smth=1.0)
## make mesh active
#bpy.context.view_layer.objects.active = obj_ocean
#mesh_ocean = obj_ocean.data

## Create a FloatAttribute for the vertices
#if not mesh_ocean.attributes.get(data_vname):
#    # domain='POINT' decides that these data will be attributed to the vertices , can be also attributed to the Faces
#    data_ocean = mesh_ocean.attributes.new(name=data_vname, type='FLOAT', domain='POINT')
#else:
#    data_ocean = mesh_ocean.attributes[data_vname]
#data_ocean.data.foreach_set("value",data_plot)
#data_ocean.data.update()

## put min/max value as custom properties 
#obj_ocean[f'{data_vname}_min'] = float(np.min(data_plot))
#obj_ocean[f'{data_vname}_max'] = float(np.max(data_plot))

#
#
#
#===============================================================================
#=== CREATE BLENDER LAND MESH ==================================================
#===============================================================================
# cycle over all land polygons
for niland, lsmask in enumerate(mesh.lsmask_a):
    poly_x, poly_y = lsmask[:,0], lsmask[:,1]
    xmin, xmax     = np.floor(poly_x).min(), np.ceil(poly_x).max()
    ymin, ymax     = np.floor(poly_y).min(), np.ceil(poly_y).max()
    
    #x_m, y_m  = np.meshgrid(np.arange(xmin, xmax, topo_resol),np.arange(ymin, ymax, topo_resol))
    x_m, y_m  = np.meshgrid(np.arange(xmin, xmax, topo_resol),np.arange(-90, 90+topo_resol, topo_resol))
    x_m, y_m  = x_m.reshape((x_m.size, 1)), y_m.reshape((y_m.size, 1))
        
    # check if regular points are within polygon
    IN        = Path(lsmask).contains_points(np.concatenate((x_m, y_m),axis=1)) 
    x_m, y_m  = x_m[IN==True], y_m[IN==True]
    del IN
        
    # combine polygon points and regular points within polygon --> do triangulation
    outeredge = np.vstack((poly_x, poly_y)).transpose()
    points    = np.hstack((x_m, y_m))
    pts_iscst = np.zeros(x_m.shape[0])
    points    = np.vstack((outeredge, points))
    pts_iscst = np.hstack(( np.ones(poly_x.shape), pts_iscst)) # collect if points belongs to coast
    if np.unique(points[:,0]).size<=3 or np.unique(points[:,1]).size<=3 : continue
    tri       = Triangulation(points[:,0], points[:,1])
    del outeredge, poly_x, poly_y
        
    # compute trinagle centroids and check if they are within polygon
    tri_cx    = np.sum(points[tri.triangles,0],axis=1)/3
    tri_cy    = np.sum(points[tri.triangles,1],axis=1)/3
    tri_cx    = np.reshape(tri_cx,(tri_cx.size,1))
    tri_cy    = np.reshape(tri_cy,(tri_cy.size,1))
    IN        = Path(lsmask).contains_points(np.concatenate((tri_cx,tri_cy),axis=1))
    tri.triangles=tri.triangles[IN==True,:]
    del tri_cx, tri_cy, IN
        
    # concatenate all land trinagles
    if niland==0:
        land_points = points
        land_elem2d = tri.triangles
        land_ptsiscst = pts_iscst # does points belongs to coast
    else:
        land_elem2d = np.concatenate((land_elem2d, tri.triangles+land_points.shape[0]), axis=0)
        land_points = np.concatenate((land_points, points), axis=0)
        land_ptsiscst = np.concatenate((land_ptsiscst, pts_iscst))
    del points    
    
# do  topographic scaling (potatoefication) for land mesh
R_grid   = R_earth  
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
bottom_depth_2d[land_ptsiscst==1] = 0.0
bottom_depth_2d[np.isnan(bottom_depth_2d)] = 0.0
if do_topo:
    R_grid         = R_grid+( bottom_depth_2d*100*potatoefac_land)
    del topo,lon,lat,mlon,mlat
R_grid = R_grid/R_earth
        
# create blender vertice tuple list 
# vertices = [ (V1_x, V1_y, V1_z), 
#              (V2_x, V2_y, V2_z), 
#              ...
#            ]
xs,ys,zs = tpv.grid_cart3d(land_points[:,0], land_points[:,1], R_grid, is_deg=True)
vertices_land = list(zip(xs, ys, zs))

# create blender normal are with respect to face
# normals = [ (T1x, T1y, T1z), 
#             (T2x, T2y, T2z), 
#              ...
#             ]
normals_land = list(zip(xs[land_elem2d].sum(axis=1)/3, ys[land_elem2d].sum(axis=1)/3, zs[land_elem2d].sum(axis=1)/3 ))
del xs,ys,zs

# triangles = [ (T1_V1, T1_V2, T1_V3), 
#               (T2_V1, T2_V2, T2_V3), 
#              ...
#             ]
triangles_land = list(zip(land_elem2d[:,0], land_elem2d[:,1],land_elem2d[:,2] ))
    
# compute vertice uv mapping coordinates
#xs, ys, zs = tpv.grid_cart3d(mesh.n_x+90, mesh.n_y, 1.0, is_deg=True)
#u          = 0.5 + np.arctan2(-xs,ys)/(2 * np.pi)
xs, ys, zs = tpv.grid_cart3d(land_points[:,0], land_points[:,1], 1.0, is_deg=True)
u          = 0.5 + np.arctan2(ys,xs)/(2 * np.pi)
v          = 0.5 + np.arcsin(zs)/np.pi
uv_land    = list(zip(u, v))
del(xs, ys, zs, u, v)

# add additional attribute variables that can be used in blender via an attribute node
add_meshattr           = dict()
add_meshattr['ntopo']  = np.abs(bottom_depth_2d)

#
#
#______________________________________________________________________________________
# create blender mesh object 
obj_land = blender_create_mesh('land', 
                                vertices_land, 
                                triangles_land, 
                                normals_land, 
                                uv_land, 
                                add_meshattr=add_meshattr, 
                                shade_flat=False )
#
#
#______________________________________________________________________________________
# create blender texture material 
obj_land = blender_create_txtremat('land', obj_land, 
                                        land_txturepath, 
                                        alphamap=land_alphamap, alpha_invert=True, 
                                        bumpmap=land_bumpmap, bump_strength=0.1, bump_smth=1.0)

#
#
#===============================================================================
#=== BLENDER ADD CLOUD LAYER SPHERE ============================================
#===============================================================================
#obj_cloud = bledner_create_cloud(R_grid, txture_clouds, facemultipl=2)



#
#
#
#===============================================================================
#=== COMBINE BLENDER OCEAN & LAND MESH =========================================
#===============================================================================
# Create parent object that is responsible for the rotation movement of all the 
# childs
obj_parent = bpy.data.objects.new('Parent', None)
bpy.context.collection.objects.link(obj_parent)
## obj_list = ["obj_ocean", "obj_land", "obj_cloud"]
obj_list = ["obj_ocean", "obj_land"]
for obj_name in obj_list:
    obj = bpy.data.objects.get(obj_name)
    if obj:
        obj.parent = obj_parent



##
##
##===============================================================================
##=== BLENDER ROTATION ANIMATION ================================================
##===============================================================================
#rotation_frames = 1000  # Number of frames for one complete rotation
#init_rot = 0
#deg2rad = np.pi/180
## Set the rotation mode to 'XYZ' to enable rotation around specific axes
#scene = bpy.data.scenes["Scene"]
#obj_parent.rotation_mode = 'XYZ'

#scene.frame_start = 1
#scene.frame_end = rotation_frames

## Set the initial rotation (optional)
#obj_parent.rotation_euler = (0, 0, init_rot)
#obj_parent.keyframe_insert("rotation_euler", index=2 , frame=1)  # Initial keyframe

#obj_parent.rotation_euler = (0, 0, init_rot*deg2rad + 360*deg2rad)
#obj_parent.keyframe_insert('rotation_euler', index=2 ,frame=rotation_frames)

## make planetary rotation cyclic 
## Access the animation data
#action = obj_parent.animation_data.action
#fcurve = action.fcurves.find('rotation_euler', index=2)

## Make the animation cyclic
#if fcurve:
#    # Set the extrapolation mode to 'MAKE_CYCLIC'
#    mod = fcurve.modifiers.new(type='CYCLES')
#    mod.mode_before = 'REPEAT'
#    mod.mode_after = 'REPEAT'
    

#
#
#
#===============================================================================
#=== BLENDER RENDERER_ENGINE SETTINGS ==========================================
#===============================================================================
# switch off Viewport Denoising otherwise bumpmap creates artifacts in the animation 
# Set the context to Eevee
#bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.engine = 'CYCLES'

# Enable or disable viewport denoising
bpy.context.scene.eevee.use_gtao = False  # Set to True to enable, False to disable

# Optional: You can also adjust the samples and other related settings if needed
bpy.context.scene.eevee.taa_render_samples = 64  # Set the desired number of samples
bpy.context.scene.eevee.taa_samples = 16  # Set the desired number of viewport samples
    

#scene.render.use_stamp = 1
#scene.render.stamp_background = (0,0,0,1)
#scene.render.filepath = "render/anim"
#scene.render.image_settings.file_format = "AVI_JPEG"
#bpy.ops.render.render(animation=True)
