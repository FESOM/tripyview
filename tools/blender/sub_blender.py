import bpy
import bmesh
import tripyview as tpv 
import numpy     as np
import xarray    as xr
import time      as clock
import warnings
from matplotlib.path import Path
from matplotlib.tri  import Triangulation



#
#
#_______________________________________________________________________________
def blender_create_mesh(name, vertices, triangles, normals, uv, 
                        add_meshattr=dict(), 
                        add_dataattr=dict(), 
                        shade_flat=True):
    
    # vertice  and elements dimension
    n2dn = len(vertices)
    n2de = len(triangles)
    
    #___________________________________________________________________________
    # create empty blender mesh class
    mesh = bpy.data.meshes.new("mesh_"+name)
    
    # add vertices and triangles 
    mesh.from_pydata(vertices, [], triangles, shade_flat=shade_flat)
    del(vertices, triangles)
    
    # add uv layer mapping coordinates the blender mesh
    # Create a new UV map if it doesn't already exist
    uv_layer = mesh.uv_layers.new(name="UVMap")
    
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()

    # add uv coordinates and face normals
    uv_layer = bm.loops.layers.uv.active
    for ii, face in enumerate(bm.faces):
        face.normal = normals[ii]
        for loop in face.loops:
            loop[uv_layer].uv = uv[loop.vert.index]  # Ensure this matches your UV data structure
    bm.to_mesh(mesh)
    bm.free()
    
    
    #___________________________________________________________________________
    # --> create empty blender object class connected with mesh
    obj  = bpy.data.objects.new("obj_"+name, mesh)   
    
    #___________________________________________________________________________
    # add additional attributes like for resolution and vertical discplacement
    if add_meshattr is not None:
        for attr_var in add_meshattr:
            # Create a FloatAttribute for the vertices
            if not mesh.attributes.get(attr_var):
                if    len(add_meshattr[attr_var]) == n2dn : domain='POINT'
                elif  len(add_meshattr[attr_var]) == n2de : domain='FACE'
                else: 
                    print(' mesh attribute size is not supported')
                
                # domain='POINT' decides that these data will be attributed to the vertices , can be also attributed to the Faces
                attribute = mesh.attributes.new(name=f'mesh_{attr_var}', type='FLOAT', domain='POINT')
            else:
                attribute = mesh.attributes[attr_var]
                
            attribute.data.foreach_set("value",add_meshattr[attr_var])
            attribute.data.update()

            # put min/max value as custom properties 
            obj[f'mesh_{attr_var}_min'] = float(np.nanmin(add_meshattr[attr_var]))
            obj[f'mesh_{attr_var}_max'] = float(np.nanmax(add_meshattr[attr_var]))
    
    #_________________________________________________________________________________
    # add additional attributes like for resolution and vertical discplacement
    if add_dataattr is not None:
        for attr_var in add_dataattr:
            # Create a FloatAttribute for the vertices
            if not mesh.attributes.get(attr_var):
                if    len(add_dataattr[attr_var]) == n2dn : domain='POINT'
                elif  len(add_dataattr[attr_var]) == n2de : domain='FACE'
                else: 
                    print(' data attribute size is not supported')
                
                # domain='POINT' decides that these data will be attributed to the vertices , can be also attributed to the Faces
                attribute = mesh.attributes.new(name=f'data_{attr_var}', type='FLOAT', domain='POINT')
            else:
                attribute = mesh.attributes[attr_var]
                
            attribute.data.foreach_set("value",add_dataattr[attr_var])
            attribute.data.update()

            # put min/max value as custom properties 
            obj[f'data_{attr_var}_min'] = float(np.nanmin(add_dataattr[attr_var]))
            obj[f'data_{attr_var}_max'] = float(np.nanmax(add_dataattr[attr_var]))
    
    #___________________________________________________________________________
    bpy.context.collection.objects.link(obj)
    return(obj)



#
#
#_______________________________________________________________________________
def blender_create_txtremat(name, obj, texture, 
                        alphamap=None, alpha_invert=False, 
                        bumpmap=None, bump_strength=0.1, bump_smth=1.0):
    #___________________________________________________________________________
    # create texture material
    material = bpy.data.materials.new(name="material_"+name)
    material.use_nodes = True
    
    # Remove default nodes
    nodes = material.node_tree.nodes
    for node in nodes:
        nodes.remove(node)
    
    #___________________________________________________________________________
    # Add different nodes in shader viewport that are than linked together by "wires"
    # from import to output port
    # Add an image texture node
    
    # Add a Texture Coordinate node
    texture_coord        = nodes.new(type='ShaderNodeTexCoord')
    texture_coord.location = (0, 0)
    
    # Add a Texture File node
    texture_map          = nodes.new(type='ShaderNodeTexImage')
    texture_map.image    = bpy.data.images.load(texture)
    texture_map.location = (300,0)
    
    #___________________________________________________________________________
    # Add a Alphamap File node
    alpha_map            = nodes.new(type='ShaderNodeTexImage')
    alpha_map.image      = bpy.data.images.load(alphamap)
    alpha_map.image.colorspace_settings.name = 'Non-Color'
    alpha_map.location   = (300,-300)
    if alpha_invert:
        alpha_invert            = nodes.new(type='ShaderNodeInvert')
        alpha_invert.location   = (450,-300)        
        alpha_map.location      = (150,-300)
            
    #___________________________________________________________________________
    # Add a new principled BSDF node
    bsdf_diff                 = nodes.new(type='ShaderNodeBsdfDiffuse')
    bsdf_diff.location        = (600,0)
    bsdf_mix_DG               = nodes.new(type='ShaderNodeMixShader')
    bsdf_mix_DG.location      = (800,0)
    
    bsdf_mix_G               = nodes.new(type='ShaderNodeMixShader')
    bsdf_mix_G.location      = (800,-400)
    bsdf_gloss               = nodes.new(type='ShaderNodeBsdfGlossy')
    bsdf_mix_G.location      = (600,-400)
    fresnel                  = nodes.new(type='ShaderNodeFresnel')
    
    #___________________________________________________________________________
    # Add a Bump node
    if bumpmap is not None:    
        # Add a Bump Map File node
        bump_map = nodes.new(type='ShaderNodeTexImage')
        bump_map.image = bpy.data.images.load(bumpmap)
        bump_map.image.colorspace_settings.name = 'Non-Color'
        bump_map.location = (100, -600)
        
        bump_smooth = nodes.new(type='ShaderNodeMath')
        bump_smooth.location = (250, -600)
        bump_smooth.operation = 'SMOOTH_MAX'  # Example: using the 'Multiply' operation
        bump_smooth.inputs[1].default_value = 0.5  # Second input value
        bump_smooth.inputs[2].default_value = bump_smth  # Third input value (smoothness factor)
        
        bump_node = nodes.new(type='ShaderNodeBump')
        bump_node.inputs['Strength'].default_value = bump_strength
        bump_node.location = (450, -600)
        
    #___________________________________________________________________________
    # Create a new material output node
    output               = nodes.new(type='ShaderNodeOutputMaterial')
    output.location      = (900,0)
    
    # Link the nodes
    material.node_tree.links.new(texture_map.inputs['Vector'], texture_coord.outputs[  'UV'])
    material.node_tree.links.new(bsdf_diff.inputs['Color'], texture_map.outputs[ 'Color'])
    
    # links for topographic bump mapping 
    if bumpmap is not None: 
        material.node_tree.links.new(bump_map.inputs['Vector'], texture_coord.outputs['UV'])
        material.node_tree.links.new(bump_smooth.inputs[0], bump_map.outputs['Color'])
        material.node_tree.links.new(bump_node.inputs['Height'], bump_smooth.outputs['Value'])
        material.node_tree.links.new(bsdf_diff.inputs['Normal'], bump_node.outputs['Normal'])
    
    
    material.node_tree.links.new(alpha_map.inputs[  'Vector'], texture_coord.outputs[  'UV'])
    # links for alpha land seam mask mapping     
    if alpha_invert:
        material.node_tree.links.new(alpha_invert.inputs['Color'], alpha_map.outputs['Color'])
        material.node_tree.links.new(bsdf_mix_DG.inputs['Fac'], alpha_invert.outputs['Color'])
    else:
        material.node_tree.links.new(bsdf_mix_DG.inputs[0], alpha_map.outputs['Color'])    
    material.node_tree.links.new(bsdf_mix_DG.inputs[1], bsdf_diff.outputs[ 'BSDF'])
    
    #___________________________________________________________________________
    # link fresnel glossy
    material.node_tree.links.new(bsdf_mix_G.inputs[0], fresnel.outputs[ 'Fac'])
    material.node_tree.links.new(bsdf_mix_G.inputs[2], bsdf_gloss.outputs[ 'BSDF'])
    material.node_tree.links.new(bsdf_mix_G.inputs[1], bsdf_diff.outputs[ 'BSDF'])
    material.node_tree.links.new(bsdf_mix_DG.inputs[2], bsdf_mix_G.outputs[ 'Shader'])
    material.node_tree.links.new(output.inputs[    'Surface'], bsdf_mix_DG.outputs[ 'Shader'])

    material.blend_method = 'OPAQUE'
    material.shadow_method = 'NONE'  # Example: No shadow
    material.use_backface_culling = False  # Example: Render both side    

    # Apply the material to the mesh
    mesh = obj.data
    mesh.materials.append(material)
    
    return(obj)



#
#
#_______________________________________________________________________________
def bledner_create_cloud(R_grid, txture_clouds, facemultipl=2):
   mesh = bpy.data.meshes.new('mesh_'+'cloud')
   # Create a UV sphere
   bm = bmesh.new()
   bmesh.ops.create_uvsphere(bm, u_segments=32*facemultipl, v_segments=16*facemultipl, radius=np.nanmax(R_grid)+0.01 )
   bm.to_mesh(mesh)
   bm.free()

   # create object
   obj = bpy.data.objects.new('obj_'+'cloud', mesh)

   #____________________________________________________________________________
   # create texture material
   material = bpy.data.materials.new(name='material_'+'cloud')
   material.use_nodes = True
       
   # Remove default nodes
   nodes = material.node_tree.nodes
   for node in nodes:
       nodes.remove(node)
       
   #____________________________________________________________________________
   # Add a Texture Coordinate node
   texture_coord        = nodes.new(type='ShaderNodeTexCoord')
   texture_coord.location = (0, 0)
       
   # Add a Texture File node
   texture_map          = nodes.new(type='ShaderNodeTexImage')
   texture_map.image    = bpy.data.images.load(txture_clouds)
   texture_map.projection = 'SPHERE'
   texture_map.image.colorspace_settings.name = 'Non-Color'
   texture_map.location = (300,0)
       
   bsdf                 = nodes.new(type='ShaderNodeBsdfPrincipled')
   bsdf.location        = (600,0)   

   bump = nodes.new(type='ShaderNodeBump')
   bump.inputs['Strength'].default_value = 0.5
   bump.inputs['Distance'].default_value = 0.1
   bump.location = (450, -600)         

   math = nodes.new(type='ShaderNodeMath')
   math.location = (250, -600)
   math.operation = 'SUBTRACT'  # Example: using the 'Multiply' operation
   math.inputs[1].default_value = 0.25  # Second input value

   output               = nodes.new(type='ShaderNodeOutputMaterial')
   output.location      = (900,0)
       
   # Link the nodes
   material.node_tree.links.new(texture_map.inputs['Vector'], texture_coord.outputs['Generated'])
   material.node_tree.links.new(bsdf.inputs['Base Color'], texture_map.outputs['Color'])
   material.node_tree.links.new(bump.inputs['Height'], texture_map.outputs['Color'])
   material.node_tree.links.new(bsdf.inputs['Normal'], bump.outputs[ 'Normal'])
   material.node_tree.links.new(math.inputs[0], texture_map.outputs[ 'Color'])
   material.node_tree.links.new(bsdf.inputs['Alpha'], math.outputs[ 'Value'])
   material.node_tree.links.new(output.inputs['Surface'], bsdf.outputs[ 'BSDF'])

   material.blend_method = 'BLEND'
   material.shadow_method = 'NONE'  # Example: No shadow
   material.use_backface_culling = False  # Example: Render both sides
    
   # Apply the material to the mesh
   mesh.materials.append(material)

   # add the object into the scene
   bpy.context.collection.objects.link(obj)
   return(obj)



#
#
#_______________________________________________________________________________
def blender_combine_meshes(name, list_obj):
   # Create a new empty mesh to combine both meshes into one object
   combined_mesh = bpy.data.meshes.new('mesh_'+name)
   combined_object = bpy.data.objects.new('obj_'+name, combined_mesh)
   bpy.context.collection.objects.link(combined_object)

   # Join the objects
   bpy.context.view_layer.objects.active = combined_object
   combined_object.select_set(True)
   for obj in list_obj:
       obj.select_set(True)
   bpy.ops.object.join()
   return(combined_object)
