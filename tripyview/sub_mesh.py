# Patrick Scholz, 14.12.2017

import sys
import os
import time
import numpy  as np
import pandas as pa
import joblib
import pickle5 as pickle
from   netCDF4 import Dataset
from .sub_mesh import *

# ___INITIALISE/LOAD FESOM2.0 MESH CLASS IN MAIN PROGRAMM______________________
#| IMPORTANT!!!:                                                               |                                         
#| only when mesh is initialised with this routine, the main programm is able  |
#| to recognize if mesh object already exist or not and if it exist to use it  |
#| and not to load it again !!!                                                |
#|_____________________________________________________________________________|
def load_mesh_fesom2(
                    meshpath, abg=[50,15,-90], focus=0, do_rot='None', 
                    do_augmpbnd=True, do_cavity=False, do_info=True, 
                    do_lsmask=True, do_lsmshp=True, do_pickle=True, 
                    do_earea=True, do_narea=True,
                    do_eresol=[False,'mean'], do_nresol=[False,'e_resol'] 
                    ):
    """
    ---> load FESOM2 mesh:
    ___INPUT:___________________________________________________________________
    mespath     :   str, path that leads to FESOM2.0 mesh files (*.out)
    abg         :   list, [alpha,beta,gamma] euler angles used to rotate the grids
                    within the model, default: [50,15,-90]
    focus       :   float, sets longitude center of the mesh, default: focus=0 
                    lon=[-180...180], focus=180 lon=[0...360]
    do_rot      :   str, should the grid be rotated, default: 'None' 
                    None, 'None' : no rotation is applied 
                    'r2g'        : loaded grid is rotated and transformed to geo
                    'g2r'        : loaded grid is geo and transformed to rotated
    do_augmpbnd :   bool, augment periodic boundary triangles, default: True
    do_cavity   :   bool, load also cavity files cavity_nlvls.out and 
                    cavity_elvls.out, default: False
    do_info     :   bool, print progress and mesh information, default: True
    do_lsmask   :   bool, compute land-sea mask polygon for FESOM2 mesh, 
                    see mesh.lsmask, augments its periodic boundaries see 
                    mesh.lasmask_a and computes land sea mask patch see mesh.lsmask_p,
                    default: True
    do_pickle   :   bool, store and load mesh from *.pckl binary file, default: True
    do_earea    :   bool, compute or load from fesom.mesh.diag.nc the area of elements
    do_eresol   :   list([bool,str]), compute resolution based on elements, 
                    str can be "mean": resolution based on mean element edge 
                    length, "max": resolution based on maximum edge length, 
                    "min" resolution based on minimum edge length, 
                    default: [False,'area']
    do_narea    :   bool, compute or load from fesom.mesh.diag.nc the clusterarea 
                    of vertices, default: False
    do_nresol   :   bool, compute resolution at nodes from interpolation of 
                    resolution at elements, default: False
    ___RETURNS:_________________________________________________________________
    mesh        :   object, returns fesom_mesh object
    """
    #___________________________________________________________________________
    pickleprotocol=4
    
    #___________________________________________________________________________
    # path of mesh location
    meshpath = os.path.normpath(meshpath)
    meshid   = os.path.basename(meshpath)
    
    #___________________________________________________________________________
    # build path for cach to store the pickle files, either in home directory or 
    # in path given by environment variable MYPY_MESHPATH 
    cachepath = path = os.environ.get('MESHPATH_TRIPYVIEW', os.path.join(os.path.expanduser("~"), "meshcache_tripyview"))
    cachepath = os.path.join(cachepath, meshid)
    if not os.path.isdir(cachepath):
        if do_info: print(' > create cache: {}'.format(cachepath))
        os.makedirs(cachepath)
    
    #___________________________________________________________________________
    # check if pickle file can be found somewhere either in mesh folder or in 
    # cache folder 
    #picklefname = 'mypymesh_fesom2.pckl'
    #picklefname = 'tripyview_fesom2_'+meshid+'.pckl'
    picklefname = 'tripyview_fesom2_{}_focus{}.pckl'.format(meshid,focus)
    if do_pickle:
        # check if mypy pickle meshfile is found in meshfolder
        if    ( os.path.isfile(os.path.join(meshpath, picklefname)) ):
            loadpicklepath = os.path.join(meshpath, picklefname)
            if do_info: print(' > found *.pckl file: {}'.format(os.path.dirname(loadpicklepath)))    
            
        # check if mypy pickle meshfile is found in cache folder
        elif ( os.path.isfile(os.path.join(cachepath, picklefname)) ):
            loadpicklepath = os.path.join(cachepath, picklefname)
            if do_info: print(' > found *.pckl file: {}'.format(os.path.dirname(loadpicklepath)))
            
        else:
            loadpicklepath = 'None'
            print(' > found no .pckl file in cach or mesh path')    
            
    #___________________________________________________________________________
    # load pickle file if it exists and load it from .pckl file, if it does not 
    # exist create mesh object with fesom_mesh
    # do_pickle==True and .pckl file exists
    if   do_pickle and ( os.path.isfile(loadpicklepath) ): 
        if do_info: print(' > load  *.pckl file: {}'.format(os.path.basename(loadpicklepath)))
        #_______________________________________________________________________
        fid  = open(loadpicklepath, "rb")
        mesh = pickle.load(fid)
        fid.close()
        
        do_pbndfind=False
        #_______________________________________________________________________
        # rotate mesh if its not done in .pckle file 
        if (mesh.do_rot != do_rot) : 
            do_pbndfind = True
            if   (do_rot == 'r2g'):
                #_______________________________________________________________
                mesh.do_rot          = 'r2g'
                mesh.n_xo, mesh.n_yo = mesh.n_x, mesh.n_y
                mesh.n_x , mesh.n_y  = grid_r2g(mesh.abg, mesh.n_xo, mesh.n_yo)
                
                #_______________________________________________________________
                # rotate also periodic land sea mask 
                if len(mesh.lsmask) != 0:
                    for ii in range(0,len(mesh.lsmask)):
                        auxx, auxy      = grid_r2g(mesh.abg, mesh.lsmask[ii][:,0], mesh.lsmask[ii][:,1])
                        mesh.lsmask[ii] = np.vstack((auxx,auxy)).transpose()
                        del auxx, auxy
                
            elif (do_rot == 'g2r'):
                #_______________________________________________________________
                mesh.do_rot          = 'g2r'
                mesh.n_xo, mesh.n_yo = mesh.n_x, mesh.n_y
                mesh.n_x , mesh.n_y  = grid_g2r(mesh.abg, mesh.n_xo, mesh.n_yo)
                
                #_______________________________________________________________
                # rotate also periodic land sea mask 
                if len(mesh.lsmask) != 0:
                    for ii in range(0,len(mesh.lsmask)):
                        auxx, auxy      = grid_g2r(mesh.abg, mesh.lsmask[ii][:,0], mesh.lsmask[ii][:,1])
                        mesh.lsmask[ii] = np.vstack((auxx,auxy)).transpose()
                        del auxx, auxy
                        
            elif (do_rot == None or do_rot == 'None'):    
                mesh.do_rot = 'None'
            else: 
                raise ValueError("This rotatio option in do_rot is not supported.")
            
        #_______________________________________________________________________
        # change focus 
        if (mesh.focus != focus): 
            #___________________________________________________________________
            do_pbndfind          = True
            mesh.focus           = focus
            mesh.n_xo, mesh.n_yo = mesh.n_x, mesh.n_y
            mesh.n_x,  mesh.n_y  = grid_focus(focus, mesh.n_xo, mesh.n_yo)
        
            #___________________________________________________________________
            # rotate also periodic land sea mask 
            if len(mesh.lsmask) != 0:
                for ii in range(0,len(mesh.lsmask)):
                    auxx, auxy      = grid_focus(focus, mesh.lsmask[ii][:,0], mesh.lsmask[ii][:,1])
                    mesh.lsmask[ii] = np.vstack((auxx,auxy)).transpose()
                    del auxx, auxy
                        
        #_______________________________________________________________________
        # find periodic boundary
        if do_pbndfind:
            mesh.pbnd_find()
        
        #_______________________________________________________________________
        # augment periodic boundary if it wasnot done in .pckl file
        if  ((not mesh.do_augmpbnd) and do_augmpbnd) or (do_augmpbnd and do_pbndfind):
            mesh.pbnd_augment()
        
        #_______________________________________________________________________
        # compute other properties if they are not stored in .pckl file or need 
        # to me redone anyway since focus or meshrotation has changed
        if do_pbndfind:
            if do_lsmask: 
                # if lsmask does not exist yet, compute  and augment it
                if len(mesh.lsmask)==0:
                    mesh.compute_lsmask()
                    if do_augmpbnd: mesh.augment_lsmask()
                # if lsmask exist than only augment pbnd     
                else:
                    if do_augmpbnd: mesh.augment_lsmask()
        else:
            if not mesh.do_earea and do_earea:
                mesh.compute_e_area()
            if not mesh.do_eresol[0] and do_eresol[0]:
                mesh.compute_e_resol(which=do_eresol[1])
            if not mesh.do_narea and do_narea :
                mesh.compute_n_area()
            if not mesh.do_nresol[0] and do_nresol[0]:
                mesh.compute_n_resol(which=do_nresol[1])  
            if not mesh.do_lsmask and do_lsmask:
                mesh.compute_lsmask()
                if mesh.do_augmpbnd:
                    mesh.augment_lsmask()
        
        #_______________________________________________________________________
        if do_info: print(mesh.info())
        
        #_______________________________________________________________________
        return(mesh)
    
    # (do_pickle==True and .pckl file does not exists) or (do_pickle=False)
    elif (do_pickle and not ( os.path.isfile(loadpicklepath)) ) or not do_pickle:
        if do_info: print(' > load mesh from *.out files: {}'.format(meshpath))
        #_______________________________________________________________________
        mesh = mesh_fesom2(
                    meshpath   = meshpath     , 
                    abg        = abg          , 
                    focus      = focus        ,
                    do_rot     = do_rot       ,
                    do_augmpbnd= do_augmpbnd  ,
                    do_cavity  = do_cavity    ,
                    do_info    = do_info      ,
                    do_earea   = do_earea     ,
                    do_eresol  = do_eresol    ,
                    do_narea   = do_narea     ,
                    do_nresol  = do_nresol    ,
                    do_lsmask  = do_lsmask    ,
                    do_lsmshp  = do_lsmshp)
        
        #_______________________________________________________________________
        # save mypy mesh .pckl file
        # --> try 1.st to store it in the mesh in the meshfolder, will depend of 
        #     there is permission to write
        if do_pickle:
            try: 
                savepicklepath = os.path.join(meshpath,picklefname)
                if do_info: print(' > save mesh to *.pckl in {}'.format(savepicklepath))
                fid = open(savepicklepath, "wb")
                pickle.dump(mesh, fid, protocol=pickleprotocol)
                fid.close()
            # if no permission rights for writing in meshpath folder try 
            # cachefolder   
            except PermissionError:
                try: 
                    savepicklepath = os.path.join(cachepath,picklefname)
                    mesh.cachepath = cachepath
                    if do_info: print(' > save mesh to *.pckl in {}'.format(savepicklepath))
                    fid = open(savepicklepath, "wb")
                    pickle.dump(mesh, fid, protocol=pickleprotocol)
                    fid.close()
                except PermissionError:
                    print(" > could not write *.pckl file in {} or {}".format(meshpath,cachepath))
        
        #_______________________________________________________________________
        return(mesh)



# _____________________________________________________________________________
#|                                                                             |
#|                        *** FESOM2.0 MESH CLASS ***                          |
#|                                                                             |
#|_____________________________________________________________________________|
class mesh_fesom2(object):
    """
    ---> Class that creates object that contains all information about FESOM2
         mesh. As minimum requirement the mesh path to the files nod2d.out,
         elem2d.out and aux3d.out has to be given, 
    ___PARAMETERS:______________________________________________________________
    see help(load_fesom_mesh)
     
    ___VARIABLES:_______________________________________________________________
    path          : str, path that leads to FESOM2.0 mesh files (*.out)
    id            : str, identifies mesh 
    n2dn          : int, number of 2d nodes 
    n_x,n_y       : array, lon lat position of surface nodes 
    n2de          : int, number of 2d elements
    e_i           : array, elemental array with 2d vertice indices, shape=[n2de,3] 
    nlev          : int, number of vertical full cell level
    zlev, zmid    : array, with full and mid depth levels
    n_iz,e_iz     : array, number of full depth levels at vertices and elem
    n_z           : array, bottom depth based on zlev[n_iz], 
    ___if do_cavity==True:______________________________________________________
    n_ic,e_ic     : array, full depth level index of cavity-ocean interface at 
                    vertices and elem
    n_c           : array, cavity-ocean interface depth at vertices zlev[n_ic]
    ___if do_narea, do_nresol, do_earea, do_eresol == True:_____________________
    n_area,n_resol: array, vertices area and resolution
    e_area,e_resol: array, element area and resolution
        
    e_pbnd_1      : array, elem indices of pbnd elements
    e_pbnd_0      : array, elem indices of not pbnd elements
    ___if do_augmpbnd == True:__________________________________________________
    n_xa,n_ya,n_za: array, with augmented paramters  
    n_iza,n_ca    :
    n_ica         :
    e_ia          : array, element array with augmented trinagles
    e_pbnd_a      : array, elem indices of periodic augmented elements
    n_pbnd_a      : array, vertice indices to augment pbnd
    n2dna,n2dea   : int, number of elements and nodes with periodic boundary 
                    augmentation
                    
                    --> create matplotlib triangulation with augmented periodic 
                        boundary:
                        tri = Triangulation(np.hstack((mesh.n_x,mesh.n_xa)),
                                            np.hstack((mesh.n_y,mesh.n_ya)),
                                            np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia)))
    ___if do_lsmask == True:____________________________________________________
    lsmask        : list(array1[npts,2], array2[npts,2], ...), contains all 
                    land-sea mask polygons for FESOM2 mesh, with periodic boundary
    lsmask_a      : list(array1[npts,2], array2[npts,2], ...)contains all 
                    land-sea mask polygons for FESOM2 mesh, with augmented 
                    periodic boundary
    lsmask_p      : polygon, contains polygon collection that can be plotted as 
                    closed polygon patches with ax.add_collection(PatchCollection
                    (mesh.lsmask_p,facecolor=[0.7,0.7,0.7], edgecolor='k',
                    linewidth=0.5))
    ___RETURNS:_________________________________________________________________
    mesh          : object, returns fesom_mesh object
    """
    
    #___INIT FESOM2.0 MESH OBJECT_______________________________________________
    #                                                                           
    #___________________________________________________________________________
    def __init__(self, meshpath, abg=[50,15,-90], focus=0, focus_old=0, do_rot='None', 
                 do_augmpbnd=True, do_cavity=False, do_info=True, do_earea=False,do_earea2=False, 
                 do_eresol=[False,'mean'], do_narea=False, do_nresol=[False,'n_area'], 
                 do_lsmask=True, do_lsmshp=True, do_pickle=True ):
        
        #_______________________________________________________________________
        # define meshpath and mesh id 
        self.path               = os.path.normpath(meshpath)
        self.cachepath          = 'None'
        self.id                 = os.path.basename(self.path)
        self.info_txt           = ''
        
        #_______________________________________________________________________
        # define euler angles and mesh focus 
        self.abg                = abg
        self.focus              = focus
        self.focus_old          = focus_old 
        self.cyclic             = 360
        self.do_rot             = do_rot
        self.do_augmpbnd        = do_augmpbnd
        self.do_cavity          = do_cavity
        self.do_earea           = do_earea
        self.do_eresol          = do_eresol
        self.do_narea           = do_narea
        self.do_nresol          = do_nresol
        self.do_lsmask          = do_lsmask
        self.do_lsmshp          = do_lsmshp
        
        #_______________________________________________________________________
        # define basic mesh file path
        self.fname_nod2d        = os.path.join(self.path,'nod2d.out')
        self.fname_elem2d       = os.path.join(self.path,'elem2d.out')
        self.fname_aux3d        = os.path.join(self.path,'aux3d.out')
        self.fname_nlvls        = os.path.join(self.path,'nlvls.out')
        self.fname_elvls        = os.path.join(self.path,'elvls.out')
        self.fname_cnlvls       = []  
        self.fname_celvls       = []  
        
        #_______________________________________________________________________
        # define vertices array
        self.n_x, self.n_y,     = [], []
        self.n2dn               = 0
        self.n_xo, self.n_yo    = [], []
        
        #_______________________________________________________________________
        # define elem array
        self.e_i, self.n2de     = [], 0
        
        #_______________________________________________________________________
        # define depth and topographic info for vertices and elements
        self.nlev               = 0
        self.zlev, self.zmid    = [], []
        self.n_z, self.n_iz     = [], []
        self.e_iz               = []
        
        #_______________________________________________________________________
        # define cavity vertice and element arrays
        self.n_c, self.n_ic     = [], []
        self.e_ic               = []
        
        #_______________________________________________________________________
        # define vertice and elements area
        self.n_area,self.n_resol= [], []           
        self.e_area,self.e_resol= [], []
        
        #_______________________________________________________________________
        # define element indices for periodic and non periodic elements
        self.e_pbnd_1           = [] # elem indices of pbnd elements
        self.e_pbnd_0           = [] # elem indices of not pbnd elements
        self.e_pbnd_a           = [] # elem indices to augment left/right pbnd
        
        #_______________________________________________________________________
        # define vertice and element arrays for periodic boundary augmentation
        self.n_xa, self.n_ya    = [], [] 
        self.n_za, self.n_ia    = [], []
        self.n_iza, self.n_ca   = [], []
        self.n_ica              = []
        self.e_ia               = np.empty(shape=[0, 3]) #  elem array for augmented pbnd
        self.n_pbnd_a           = [] # vertice indices to augment pbnd
        self.n2dna,self.n2dea   = 0, 0
        
        #_______________________________________________________________________
        # define lsmask
        self.lsmask             = []
        self.lsmask_a           = []
        self.lsmask_p           = []
        
        #_______________________________________________________________________
        #  ||   ||   ||   ||   ||   ||   ||   ||   ||   ||   ||   ||   ||   ||  
        # _||_ _||_ _||_ _||_ _||_ _||_ _||_ _||_ _||_ _||_ _||_ _||_ _||_ _||_ 
        # \  / \  / \  / \  / \  / \  / \  / \  / \  / \  / \  / \  / \  / \  / 
        #  \/   \/   \/   \/   \/   \/   \/   \/   \/   \/   \/   \/   \/   \/  
        #_______________________________________________________________________
        # read fesom mesh
        self.read_mesh()
        
        # read cavity depth info 
        if do_cavity:
            self.read_cavity()
        
        # rotate mesh
        if   (do_rot == 'r2g'):
            self.n_xo, self.n_yo = self.n_x, self.n_y
            self.n_x, self.n_y = grid_r2g(self.abg, self.n_xo, self.n_yo)
        elif (do_rot == 'g2r'):
            self.n_xo, self.n_yo = self.n_x, self.n_y
            self.n_x, self.n_y = grid_g2r(self.abg, self.n_xo, self.n_yo)
        elif (do_rot == None or do_rot == 'None'):    
            ...
        else: 
           raise ValueError("This rotatio option in do_rot is not supported.")
        
        # change focus 
        if (self.focus != 0): 
            self.n_xo, self.n_yo = self.n_x, self.n_y
            self.n_x, self.n_y = grid_focus(focus, self.n_xo, self.n_yo)
        
        # find periodic boundary
        self.pbnd_find()
        
        # augment periodic boundary
        if do_augmpbnd and any(self.n_x[self.e_i].max(axis=1)-self.n_x[self.e_i].min(axis=1) > 180):
            self.pbnd_augment()
        
        # compute/load element area
        if do_earea:
            self.compute_e_area()
        
        # compute element resolution
        if do_eresol[0]:
            self.compute_e_resol(which=do_eresol[1])
        
        # compute vertices area
        if do_narea:
            self.compute_n_area()
        
        # compute vertices resolution interpolate elem to vertices
        if do_nresol[0]:
            self.compute_n_resol(which=do_nresol[1])
        
        # compute lsmask
        if do_lsmask:
            self.compute_lsmask()
            
            #___________________________________________________________________
            # save land-sea mask with periodic boundnaries to shapefile
            if do_lsmshp:
                shpfname = 'tripyview_fesom2'+'_'+self.id+'_'+'pbnd'
                lsmask_2shapefile(self, lsmask=self.lsmask, fname=shpfname)
            
            #___________________________________________________________________
            # augment periodic boundaries of land sea mask
            if do_augmpbnd:
                self.augment_lsmask()
                
                #_______________________________________________________________
                # save land-sea mask with augmented  pbnd to shapefile
                if do_lsmshp:
                    lsmask_2shapefile(self,lsmask=[])
            
        if do_info:
            print(self.info())


    
    # ___READ FESOM2 MESH FROM: nod2d.out, elem2d.out,...______________________
    #| read files: nod2d.out, elem2d.out, aux3d.out, nlvls.out, elvls.out      |                                                         
    #|_________________________________________________________________________|
    def read_mesh(self):
        #____load 2d node matrix________________________________________________
        file_content = pa.read_csv(self.fname_nod2d, delim_whitespace=True, skiprows=1, \
                                      names=['node_number','x','y','flag'] )
        self.n_x     = file_content.x.values.astype('float32')
        self.n_y     = file_content.y.values.astype('float32')
        self.n_i     = file_content.flag.values.astype('uint16')   
        self.n2dn    = len(self.n_x)
        
        #____load 2d element matrix_____________________________________________
        file_content = pa.read_csv(self.fname_elem2d, delim_whitespace=True, skiprows=1, \
                                       names=['1st_node_in_elem','2nd_node_in_elem','3rd_node_in_elem'])
        self.e_i     = file_content.values.astype('uint32') - 1
        self.n2de    = np.shape(self.e_i)[0]
        # print('    : #2de={:d}'.format(self.n2de))
        
        #____load 3d nodes alligned under 2d nodes______________________________
        with open(self.fname_aux3d) as f:
            self.nlev= int(next(f))
            self.zlev= np.array([next(f).rstrip() for x in range(self.nlev)]).astype(float)
        self.zmid    = (self.zlev[:-1]+self.zlev[1:])/2.
        
        #____load number of levels at each node_________________________________
        print(self.fname_nlvls)
        if ( os.path.isfile(self.fname_nlvls) ):
            file_content = pa.read_csv(self.fname_nlvls, delim_whitespace=True, skiprows=0, \
                                           names=['numb_of_lev'])
            self.n_iz    = file_content.values.astype('uint16') - 1
            self.n_iz    = self.n_iz.squeeze()
            self.n_z     = np.float32(self.zlev[self.n_iz])
        else:
            self.n_iz    = np.zeros((self.n2dn,)) 
            self.n_z     = np.zeros((self.n2dn,)) 
        
        #____load number of levels at each elem_________________________________
        if ( os.path.isfile(self.fname_elvls) ):
            file_content = pa.read_csv(self.fname_elvls, delim_whitespace=True, skiprows=0, \
                                           names=['numb_of_lev'])
            self.e_iz    = file_content.values.astype('uint16') - 1
            self.e_iz    = self.e_iz.squeeze()
        else:
            self.e_iz    = np.zeros((self.n2de,)) 
            
        #_______________________________________________________________________
        return(self)    
    
    
    
    # ___READ FESOM2 MESH CAVITY INFO__________________________________________
    #| read files: cavity_nlvls.out, cavity_elvls.out                          |                                     
    #|_________________________________________________________________________|
    def read_cavity(self):
        
        # print(' --> read cavity files')
        self.fname_cnlvls  = os.path.join(self.path,'cavity_nlvls.out')
        self.fname_celvls  = os.path.join(self.path,'cavity_elvls.out')
        
        #____load number of levels at each node_________________________________
        # print('     > {}'.format(self.fname_cnlvls)) 
        file_content    = pa.read_csv(self.fname_cnlvls, delim_whitespace=True, skiprows=0, \
                                       names=['numb_of_lev'])
        self.n_ic= file_content.values.astype('uint16') - 1
        self.n_ic= self.n_ic.squeeze()
        self.n_c = np.float32(self.zlev[self.n_ic])
        
        #____load number of levels at each elem_________________________________
        # print('     > {}'.format(self.fname_celvls)) 
        file_content    = pa.read_csv(self.fname_celvls, delim_whitespace=True, skiprows=0, \
                                       names=['numb_of_lev'])
        self.e_ic= file_content.values.astype('uint16') - 1
        self.e_ic= self.e_ic.squeeze()
        #_______________________________________________________________________
        return(self)
    
    
    
    #___INFO ABOUT ACTUAL MESH OBJECT__________________________________________
    #| use as: print(mesh.info())                                              |
    #|_________________________________________________________________________|
    def info(self):
        self.info_txt ="""___FESOM2 MESH INFO________________________
 > path            = {}
 > id              = {}
 > do rot          = {}
 > [al,be,ga]      = {}, {}, {}
 > do augmpbnd     = {}
 > do cavity       = {}
 > do lsmask       = {}
 > do earea,eresol = {}, {}
 > do narea,nresol = {}, {}
___________________________________________
 > #node           = {}
 > #elem           = {}
 > #lvls           = {}
___________________________________________""".format(
            self.path, self.id, str(self.do_rot),str(self.abg[0]), 
            str(self.abg[1]), str(self.abg[2]), str(self.do_augmpbnd),
            str(self.do_cavity), str(self.do_lsmask),str(self.do_earea), 
            str(self.do_eresol[0]), str(self.do_narea), str(self.do_nresol[0]), 
            str(self.n2dn), str(self.n2de), str(self.nlev)
            )
        return self.info_txt

    def __repr__(self):
        return self.info()

    def __str__(self):
        return self.info()
    
    
    # ___FIND PERIODIC BOUDNARY ELEMENTS_______________________________________
    #| find elements of periodic boundary (e_pbnd_1) and elements that do not  |
    #| participate in periodic boundary (e_pbnd_0)                             |
    #|_________________________________________________________________________|
    def pbnd_find(self):
        #_______________________________________________________________________
        # find out 1st which element contribute to periodic boundary and 2nd
        # which nodes are involed in periodic boundary
        dx = self.n_x[self.e_i].max(axis=1)-self.n_x[self.e_i].min(axis=1)
        self.e_pbnd_1 = np.argwhere(dx > self.cyclic/2).ravel()
        self.e_pbnd_0 = np.argwhere(dx < self.cyclic/2).ravel()
        
        #_______________________________________________________________________
        return(self)
    
    
    
    # ___AUGMENT PERIODIC BOUDNARY ELEMENTS____________________________________
    #| add additional elements to augment the periodic boundary on the left and|
    #| right side for an even non_periodic boundary                            |
    #|_________________________________________________________________________|
    def pbnd_augment(self):
        self.do_augmpbnd = True
        #_______________________________________________________________________
        # find which nodes are involed in periodic boundary
        n_pbnd_i = np.array(self.e_i[self.e_pbnd_1,:]).flatten()
        n_pbnd_i = np.unique(n_pbnd_i)
        
        #_______________________________________________________________________
        # find out node indices that contribute to the left side of the periodic 
        # boundary (pbndn_l_2d_i) and to the right side (pbndn_r_2d_i)
        xmin, xmax = self.n_x.min(), self.n_x.max()
        aux_i      = np.argwhere(self.n_x[n_pbnd_i]>(xmin+(xmax-xmin)/2) ).ravel()
        n_pbnd_i_r = n_pbnd_i[aux_i]
        aux_i      = np.argwhere(self.n_x[n_pbnd_i]<(xmin+(xmax-xmin)/2) ).ravel()
        n_pbnd_i_l = n_pbnd_i[aux_i]
        self.n_pbnd_a   = np.hstack((n_pbnd_i_r,n_pbnd_i_l))
        nn_il,nn_ir= n_pbnd_i_l.size, n_pbnd_i_r.size
        del aux_i
        
        #_______________________________________________________________________
        # calculate augmentation positions for new left and right periodic boundaries
        aux_pos    = np.zeros(self.n2dn,dtype='uint32')
        aux_i      = np.linspace(self.n2dn,self.n2dn+nn_ir-1,nn_ir,dtype='uint32')
        aux_pos[n_pbnd_i_r] =aux_i
        aux_i      = np.linspace(self.n2dn+nn_ir,self.n2dn+nn_ir+nn_il-1,nn_il,dtype='uint32')
        aux_pos[n_pbnd_i_l]= aux_i 
        del aux_i, n_pbnd_i_l, n_pbnd_i_r
        
        #_______________________________________________________________________
        # Augment the vertices on the right and left side 
        xmin, xmax= np.floor(xmin),np.ceil(xmax)
        self.n_xa = np.concatenate((np.zeros(nn_ir)+xmin, np.zeros(nn_il)+xmax))
        self.n_ya = self.n_y[self.n_pbnd_a]
        self.n_za = self.n_z[self.n_pbnd_a]
        # self.n_ia = self.n_i[self.n_pbnd_a]
        self.n_iza= self.n_iz[self.n_pbnd_a]
        if bool(self.n_c):
            self.n_ca  = self.n_c[self.n_pbnd_a]
            self.n_ica = self.n_ica[self.n_pbnd_a]
        self.n2dna = self.n2dn + self.n_pbnd_a.size
        
        #_______________________________________________________________________
        # (ii.a) 2d elements:
        # List all triangles that touch the cyclic boundary segments
        #_______________________________________________________________________
        elem_pbnd_l = np.copy(self.e_i[self.e_pbnd_1,:])
        elem_pbnd_r = np.copy(elem_pbnd_l)
        
        for ei in range(0,self.e_pbnd_1.size):
            # node indices of periodic boundary triangle
            tri  = np.array(self.e_i[self.e_pbnd_1[ei],:]).squeeze()
            
            # which triangle points belong to left periodic bnde or right periodic
            # boundary
            idx_l = np.where(self.n_x[tri].squeeze()<xmin+(xmax-xmin)/2)[0]
            idx_r = np.where(self.n_x[tri].squeeze()>xmin+(xmax-xmin)/2)[0]
            
            # change indices to left and right augmented boundary points
            elem_pbnd_l[ei,idx_r]=aux_pos[tri[idx_r]]
            elem_pbnd_r[ei,idx_l]=aux_pos[tri[idx_l]]
        del idx_l, idx_r, tri, aux_pos
        
        #_______________________________________________________________________
        # change existing periodic boundary triangles in elem_2d_i to augmented 
        # left boundary triangles
        #self.e_i[self.e_pbnd_1,:] = elem_pbnd_l
        # add additional augmented right periodic boundary triangles
        #self.e_ia = elem_pbnd_r
        
        # add augmented left, right periodic boundary triangles
        self.e_ia     = np.vstack((elem_pbnd_r,elem_pbnd_l))
        self.e_pbnd_a = np.hstack((self.e_pbnd_1,self.e_pbnd_1))
        self.n2dea    = self.n2de + elem_pbnd_r.shape[0]
        
        #_______________________________________________________________________
        return(self)
    

    #___COMPUTE/LOAD AREA OF ELEMENTS__________________________________________
    #| either load area of elements from fesom.mesh.diag.nc if its found in    |
    #| meshpath or recompute it from scratch, [m^2]                            |
    #|_________________________________________________________________________|
    def compute_e_area(self):
        # just compute e_area if mesh.area is empty 
        if len(self.e_area)==0:
            self.do_earea=True
            if os.path.isfile(os.path.join(self.path,'fesom.mesh.diag.nc')):
                print(' > load e_area from fesom.mesh.diag.nc')
                #_______________________________________________________________
                fid = Dataset(os.path.join(self.path,'fesom.mesh.diag.nc'),'r')
                self.e_area = fid.variables['elem_area'][:]
            else: 
                print(' > comp e_area')
                #_______________________________________________________________
                # pi     = 3.14159265358979
                rad    = np.pi/180.0  
                cycl   = 360.0*rad
                Rearth = 6367500.0
                
                e_y    = self.n_y[self.e_i].sum(axis=1)/3.0
                e_y    = np.cos(e_y*rad)
                        
                n_xy   = np.vstack([self.n_x, self.n_y])*rad     
                a      = n_xy[:,self.e_i[:,1]] - n_xy[:,self.e_i[:,0]]
                b      = n_xy[:,self.e_i[:,2]] - n_xy[:,self.e_i[:,0]]  
                del(n_xy)
                
                # trim cyclic
                a[0,a[0,:]> cycl/2.0] = a[0,a[0,:]> cycl/2.0]-cycl
                a[0,a[0,:]<-cycl/2.0] = a[0,a[0,:]<-cycl/2.0]+cycl
                b[0,b[0,:]> cycl/2.0] = b[0,b[0,:]> cycl/2.0]-cycl
                b[0,b[0,:]<-cycl/2.0] = b[0,b[0,:]<-cycl/2.0]+cycl
                
                a[0,:] = a[0,:]*e_y
                b[0,:] = b[0,:]*e_y
                del(e_y)
                
                self.e_area = 0.5 * np.abs(a[0,:]*b[1,:] - b[0,:]*a[1,:])*(Rearth**2.0)
                del(a, b)
        #_______________________________________________________________________
        return(self)
    
    
    # ___COMPUTE RESOLUTION OF ELEMENTS________________________________________
    #| compute area of elements in [m], options:                               |
    #| which :   str,                                                          |       
    #|           "mean": resolution based on mean element edge legth           |
    #|           "max" : resolution based on maximum element edge length       |
    #|           "min" : resolution based on minimum element edge length       |
    #|_________________________________________________________________________|
    def compute_e_resol(self, which='mean'):
        if len(self.e_resol) == 0 :
            self.do_eresol[0]=True
            self.do_eresol[1]=which
            
            #______________::::_________________________________________________
            # compute mean length of triangle sides
            e_y  = self.n_y[self.e_i]
            e_xy = np.array([self.n_x[self.e_i], e_y])
                
            #__________::::_____________________________________________________
            # calc jacobi matrix for all triangles 
            # | dx_12 dy_12 |
            # | dx_13 dy_13 |_i , i=1....n2dea
            jacobian     = e_xy[:,:,1]-e_xy[:,:,0]
            jacobian     = np.array([jacobian,
                                        e_xy[:,:,2]-e_xy[:,:,1],
                                        e_xy[:,:,0]-e_xy[:,:,2] ])
                
            # account for triangles with periodic bounaries
            for ii in range(3):
                idx = np.where(jacobian[ii,0,:]>180); 
                jacobian[ii,0,idx] = jacobian[ii,0,idx]-360;
                idx = np.where(jacobian[ii,0,:]<-180); 
                jacobian[ii,0,idx] = jacobian[ii,0,idx]+360;
                del idx
                
            # calc from geocoord to cartesian coord
            rad        = np.pi/180
            R_earth    = 12735/2*1000;
            jacobian   = jacobian*R_earth*rad
            cos_theta  = np.cos(e_y*rad).mean(axis=1)
            del e_y
            for ii in range(3):    
                jacobian[ii,0,:] = jacobian[ii,0,:]*cos_theta;
            del cos_theta
                
            #___________________________________________________________________
            # calc vector length dr = sqrt(dx^2+dy^2)
            jacobian     = np.power(jacobian,2);
            jacobian     = np.sqrt(jacobian.sum(axis=1));
            jacobian     = jacobian.transpose()
            
            #___________________________________________________________________
            # mean resolutiuon per element
            if  which=='mean': 
                print(' > comp. e_resol from mean')
                self.e_resol = jacobian.mean(axis=1)
            elif which=='max': 
                print(' > comp. e_resol from max')
                self.e_resol = jacobian.max(axis=1)
            elif which=='min': 
                print(' > comp. e_resol from min')
                self.e_resol = jacobian.min(axis=1)    
            #___________________________________________________________________    
            else:
                raise ValueError("The option which={} in compute_e_resol is not supported.".format(str(which)))
            
        #_______________________________________________________________________
        return(self)
    

    # ___COMPUTE/LOAD CLUSTERAREA OF VERTICES__________________________________
    #| either load clusterarea of vertices from fesom.mesh.diag.nc if its found|  
    #| in meshpath or recompute it from scratch by using e_area, [m^2]         |
    #|_________________________________________________________________________|
    def compute_n_area(self):
        # just compute e_area if mesh.area is empty 
        if len(self.n_area)==0:
            self.do_narea=True
            if os.path.isfile(os.path.join(self.path,'fesom.mesh.diag.nc')):
                print(' > load n_area from fesom.mesh.diag.nc')
                #_______________________________________________________________
                fid = Dataset(os.path.join(self.path,'fesom.mesh.diag.nc'),'r')
                self.n_area = fid.variables['nod_area'][:,:]
            else: 
                print(' > comp n_area')
                #_______________________________________________________________
                # be sure that elemente area already exists
                self.compute_e_area()
                
                #_______________________________________________________________
                e_area_x3 = np.vstack((self.e_area, self.e_area, self.e_area)).transpose().flatten()
                e_iz_n    = np.vstack((self.e_iz  , self.e_iz  , self.e_iz  )).transpose().flatten()
                
                #_______________________________________________________________
                # single loop over self.e_i.flat is ~4 times faster than douple loop 
                # over for i in range(3): ,for j in range(self.n2de):
                self.n_area = np.zeros((self.nlev, self.n2dn))
                count_e = 0
                for idx in self.e_i.flat:
                    e_iz = e_iz_n[count_e]
                    self.n_area[:e_iz, idx] = self.n_area[:e_iz, idx] + e_area_x3[count_e]
                    count_e = count_e+1 # count triangle index for aux_area[count] --> aux_area =[n2de*3,]
                self.n_area = self.n_area/3.0
                del e_area_x3, e_iz_n, count_e
                
        #_______________________________________________________________________
        return(self)



    # ___COMPUTE RESOLUTION OF AT VERTICES_____________________________________
    #| compute resolution at vertices in m                                     |
    #| which :   str,                                                          |       
    #|           "n_area" : compute resolution based on vertice cluster area   |
    #|           "e_resol": compute vertice resolution by interpolating elem   |
    #|                      resolution to vertices, default                    |
    #|_________________________________________________________________________|
    def compute_n_resol(self,which='n_area'):
        # just compute e_area if mesh.area is empty 
        if len(self.n_resol)==0:
            self.do_nresol[0]=True
            self.do_nresol[1]=which
            
            #___________________________________________________________________
            # compute vertices resolution based on vertice clusterarea
            if any(x in which for x in ['n_area','narea']): 
                
                #_______________________________________________________________
                self.compute_n_area()
                print(' > comp n_resol from n_area')
                #_______________________________________________________________
                self.n_resol = np.sqrt(self.n_area[0,:]/np.pi)*2.0
            
            #___________________________________________________________________
            # compute vertices resolution based on interpolation from resolution
            # of elements    
            elif any(x in which for x in ['e_resol','eresol']):
            
                #_______________________________________________________________
                self.compute_e_area()
                self.compute_e_resol()
                self.compute_n_area()
                print(' > comp n_resol from e_resol')
                aux = np.vstack((self.e_area,
                                 self.e_area,
                                 self.e_area)).transpose().flatten()
                aux = aux * np.vstack((self.e_resol,
                                       self.e_resol,
                                       self.e_resol)).transpose().flatten()
                    
                #_______________________________________________________________
                # single loop over self.e_i.flat is ~4 times faster than douple loop 
                # over for i in range(3): ,for j in range(self.n2de):
                self.n_resol = np.zeros((self.n2dn,))
                count = 0
                for idx in self.e_i.flat:
                    self.n_resol[idx]=self.n_resol[idx] + aux[count]
                    count=count+1 # count triangle index for aux_area[count] --> aux_area =[n2de*3,]
                del aux, count
                self.n_resol=self.n_resol/self.n_area/3.0
            
            #___________________________________________________________________    
            else:
                raise ValueError("The option which={} in compute_n_resol is not supported. either 'n_area' or 'e_resol'".format(str(which)))
            
        #_______________________________________________________________________
        return(self)
    
    
    # ___COMPUTE LAND-SEA MASK CONOURLINE______________________________________
    #| compute land-sea mask contourline with periodic boundary based on edges |
    #| that contribute only to one triangle and the checking which edges are   |
    #| consequtive connected                                                   |                   
    #|_________________________________________________________________________|
    def compute_lsmask(self):
        print(' > compute lsmask')
        self.do_lsmask = True
        #_______________________________________________________________________
        # build land boundary edge matrix
        t1 = time.time()
        edge    = np.concatenate((self.e_i[:,[0,1]], \
                                  self.e_i[:,[0,2]], \
                                  self.e_i[:,[1,2]]),axis=0)
        edge    = np.sort(edge,axis=1) 
        
        # python  sortrows algorythm --> matlab equivalent
        # twice as fast as list sorting
        #sortidx = np.lexsort((edge[:,0],edge[:,1]))
        #edge    = edge[sortidx,:].squeeze()
        #edge    = np.array(edge)
        
        ## python  sortrows algorythm --> matlab equivalent
        edge    = edge.tolist()
        edge.sort()
        edge    = np.array(edge)
        
        idx     = np.diff(edge,axis=0)==0
        idx     = np.all(idx,axis=1)
        idx     = np.logical_or(np.concatenate((idx,np.array([False]))),\
                                np.concatenate((np.array([False]),idx)))
        
        # all edges that belong to boundary
        bnde    = edge[idx==False,:]
        nbnde   = bnde.shape[0];
        del edge, idx
        
        #_______________________________________________________________________
        run_cont        = np.zeros((1,nbnde))*np.nan
        run_cont[0,:2]  = bnde[0,:]  # initialise the first landmask edge
        run_bnde        = bnde[1:,:] # remaining edges that still need to be distributed
        count_init      = 1;
        init_ind        = run_cont[0,0];
        ind_lc_s        = 0;
        
        polygon_xy = []
        for ii in range(0,nbnde):
            #___________________________________________________________________
            # search for next edge that contains that contains the last node index from 
            # run_cont
            kk_rc = np.column_stack(np.where( run_bnde==np.int(run_cont[0,count_init]) ))
            #kk_rc = np.argwhere( run_bnde==np.int(run_cont[0,count_init]) ) --> slower than np.column_stack(np.where....
            kk_r  = kk_rc[:,0]
            kk_c  = kk_rc[:,1]
            count_init  = count_init+1
            
            #___________________________________________________________________
            if kk_c[0] == 0 :
                run_cont[0,count_init] = run_bnde[kk_r[0],1]
            else:
                run_cont[0,count_init] = run_bnde[kk_r[0],0]
                
            #___________________________________________________________________
            # if a land sea mask polygon is closed
            if  np.any(run_bnde[kk_r[0],:] == init_ind):
                #_______________________________________________________________
                # add points to polygon_list
                aux_xy = np.vstack((self.n_x[np.int64(run_cont[0,0:count_init+1])], 
                                    self.n_y[np.int64(run_cont[0,0:count_init+1])])).transpose()
                polygon_xy.append(aux_xy)
                del  aux_xy
                
                #_______________________________________________________________
                # delete point from list
                run_bnde   = np.delete(run_bnde,kk_r[0],0)
                
                #_______________________________________________________________
                # if no points left break the while loop
                if np.size(run_bnde)==0:
                    break
                
                #_______________________________________________________________
                # initialise new lsmask contour
                run_cont        = np.zeros((1,nbnde))*np.nan
                run_cont[0,:2]  = run_bnde[0,:]
                run_bnde        = run_bnde[1:,:]
                count_init      = 1;
                init_ind        = run_cont[0,0]
            else:
                run_bnde = np.delete(run_bnde,kk_r[0],0)
            
        #_______________________________________________________________________
        self.lsmask = polygon_xy
        #t2 = time.time()
        #print(t2-t1)
        return(self)
    
    
    
    # ___AUGMENT PERIODIC BOUNDARIES IN LAND-SEA MASK CONOURLINE_______________
    #| spilit contourlines that span over the periodic boundary into two       |
    #| separated countourlines for the left and right side of the periodic     |
    #| boundaries                                                              |
    #|_________________________________________________________________________|
    def augment_lsmask(self):
        print(' > augment lsmask')
        #self.lsmask_a = self.lsmask.copy()
        self.lsmask_a = []
        #_______________________________________________________________________
        # build land boundary edge matrix
        nlsmask = len(self.lsmask)
        
        # min/max of longitude box 
        xmin,xmax = -180+self.focus, 180+self.focus
        
        for ii in range(0,nlsmask):
            #___________________________________________________________________
            polygon_xy = self.lsmask[ii].copy()
            
            #___________________________________________________________________
            # idx compute how many periodic boudnaries are in the polygon 
            idx        = np.argwhere(np.abs(self.lsmask[ii][1:,0]-self.lsmask[ii][:-1,0])>self.cyclic/2).ravel()
            
            #___________________________________________________________________
            if len(idx)!=0:
                # unify starting point of polygon, the total polygon should start
                # at the left periodic boudnary at the most northward periodic point
                aux_i      = np.hstack((idx,idx+1))
                aux_x      = polygon_xy[aux_i,0]
                aux_il     = np.sort(aux_i[np.argwhere(aux_x < self.focus).ravel()])
                isort_y    = np.flip(np.argsort(polygon_xy[aux_il,1]))
                aux_il     = aux_il[isort_y]
                del isort_y, aux_x, aux_i
                
                # shift polygon indices so that new starting point is at index 0
                polygon_xy = np.vstack(( (polygon_xy[aux_il[0]:,:]),(polygon_xy[:aux_il[0],:]) ))
                del aux_il
                
                # ensure that total polygon is closed
                if np.any(np.diff(polygon_xy[[0,-1],:])!=0): polygon_xy = np.vstack(( (polygon_xy,polygon_xy[0,:]) ))
            
            else: 
                self.lsmask_a.append(polygon_xy)   
                continue
            
            #___________________________________________________________________
            # recompute indices of periodic boundaries 
            # check if lsmask contour contains 0 (no periodic boundary), 1 (polar
            # contour) or >2 (non polar contour) pbnd edges. 
            # idx = np.argwhere(np.abs(self.lsmask[ii][1:,0]-self.lsmask[ii][:-1,0])>self.cyclic/2).ravel()
            idx = np.argwhere(np.abs(polygon_xy[1:,0]-polygon_xy[:-1,0])>self.cyclic/2).ravel()
            
            #___________________________________________________________________
            # none polar lsmask contour with pbnd boundary that needs to be cutted
            # in two or more polygons
            if   len(idx) >= 2:
                aux_i      = np.hstack((idx,idx+1))
                aux_x      = polygon_xy[aux_i,0]
                
                # compure index location of left and right pbnd points
                aux_il     = np.sort(aux_i[np.argwhere(aux_x < self.focus).ravel()])
                aux_ir     = np.sort(aux_i[np.argwhere(aux_x > self.focus).ravel()])
                del aux_x, aux_i
                
                #_______________________________________________________________
                # do polygon on left periodic boundary
                polygon_xyl = polygon_xy.copy()
                for jj in range(0,len(aux_il),2):
                    polygon_xyl[[aux_il[jj],aux_il[jj+1]],0]=xmin
                    polygon_xyl[aux_ir[jj]:aux_ir[jj+1]+1,:]=np.nan
                    
                # eliminate nab points from right boundary    
                polygon_xyl = np.delete(polygon_xyl,
                                        np.argwhere(np.isnan(polygon_xyl[:,0])).ravel(),axis=0)    
                
                # close polygon
                if np.any(np.diff(polygon_xyl[[0,-1],:])!=0): 
                    polygon_xyl = np.vstack(( (polygon_xyl,polygon_xyl[0,:]) ))
                
                ## polygon must have at last 3 points
                #if polygon_xyl.shape[0]==2: 
                    #polygon_xyl = np.vstack(( polygon_xyl, polygon_xyl[0,:] ))
                    
                self.lsmask_a.append(polygon_xyl)
                
                #_______________________________________________________________
                # do polygon on right periodic boundary
                polygon_xyr = polygon_xy.copy()
                polygon_xyr[aux_ir[0],0]   = xmax   
                polygon_xyr[:aux_il[0]+1,:]= np.nan   
                for jj in range(1,len(aux_ir)-1,2):
                    polygon_xyr[[aux_ir[jj],aux_ir[jj+1]],0] = xmax
                    polygon_xyr[aux_il[jj]:aux_il[jj+1]+1,:] = np.nan
                polygon_xyr[aux_ir[-1],0]  = xmax
                polygon_xyr[aux_il[-1]:,:] = np.nan  
                
                # eliminate nan points from left boudnary
                polygon_xyr = np.delete(polygon_xyr,
                                        np.argwhere(np.isnan(polygon_xyr[:,0])).ravel(),axis=0)    
                
                # close polygon
                if np.any(np.diff(polygon_xyr[[0,-1],:])!=0): 
                    polygon_xyr = np.vstack(( (polygon_xyr,polygon_xyr[0,:]) ))
                
                ## polygon must have at last 3 points
                #if polygon_xyr.shape[0]==2: 
                    #polygon_xyr = np.vstack(( polygon_xyr, polygon_xyr[0,:] ))
                    
                self.lsmask_a.append(polygon_xyr)
                
                del polygon_xy, aux_il, aux_ir
                
            #polar lsmask contour with pbnd boundary
            elif len(idx) == 1:
                #_______________________________________________________________
                # create single  polar polygon
                aux_i      = np.hstack((idx,idx+1))
                #aux_x,aux_y= self.lsmask_a[ii][aux_i,0], self.lsmask_a[ii][aux_i,1]
                aux_x,aux_y= polygon_xy[aux_i,0], polygon_xy[aux_i,1]
                
                # indeces for left and right pbnd points
                aux_il     = np.sort(aux_i[np.argwhere(aux_x < self.focus).ravel()])[0]
                aux_ir     = np.sort(aux_i[np.argwhere(aux_x > self.focus).ravel()])[0]
                #polygon_xy = self.lsmask_a[ii]
                
                # set corner points for polar polygon
                pbndl   ,pbndr    = polygon_xy[aux_ir,:], polygon_xy[aux_il,:]
                pbndl[0],pbndr[0] = xmin, xmax
                if np.all(aux_y<0): 
                    pcrnl, pcrnr = np.array([xmin, -90],ndmin=2), np.array([xmax,-90],ndmin=2)
                else:
                    pcrnl, pcrnr = np.array([xmin, 90],ndmin=2), np.array([xmax,90],ndmin=2)
                
                # augment and close polygon wit corner points
                if aux_ir<aux_il:
                    polygon_xy = np.vstack((  polygon_xy[:aux_ir,:], 
                                           pbndr, pcrnr, pcrnl ,pbndl, 
                                           polygon_xy[aux_il+1:,:]  ))
                    
                else:
                    polygon_xy = np.vstack((  polygon_xy[:aux_il,:], 
                                           pbndl, pcrnl, pcrnr ,pbndr, 
                                           polygon_xy[aux_ir+1:,:]  ))
                
                ## polygon must have at least 3 points
                #if polygon_xy.shape[0]==2: 
                    #polygon_xy = np.vstack(( polygon_xy,polygon_xy[0,:] ))
                    
                self.lsmask_a.append(polygon_xy)
                del polygon_xy, pbndr, pcrnr, pcrnl ,pbndl, 
                del aux_il, aux_ir, aux_i, aux_x, aux_y
                
        #_______________________________________________________________________
        # create lsmask patch to plot 
        self.lsmask_p = lsmask_patch(self.lsmask_a)
        
        #_______________________________________________________________________
        return(self)


    
# ___COMPUTE POLYGON PATCH FROM LAND-SEA MASK CONTOUR__________________________
#| computes polygon collection that can be plotted as closed polygon patches   |
#| with ax.add_collection(PatchCollection(mesh.lsmask_p,                       |
#| facecolor=[0.7,0.7,0.7], edgecolor='k',linewidth=0.5))                      |
#| ___INPUT_________________________________________________________________   |
#| lsmask   :   list([array1[npts,2], array2[npts,2]], ...)                    |
#|                    array1=np.array([ [x1,y1];                               |
#|                                      [x2,y2];                               |
#|                                        ...  ])                              |
#| ___RETURNS_______________________________________________________________   |
#| lsmask_p :   shapely Multipolygon object                                    |
#| ___INFO__________________________________________________________________   |
#| how to plot in matplotlib:                                                  |
#|          from descartes import PolygonPatch                                 |
#|          ax.add_patch(PolygonPatch(mesh.lsmask_p,facecolor=[0.7,0.7,0.7],   |
#|                                    edgecolor='k',linewidth=0.5))            |
#| how to plot in cartopy:                                                     |
#|          import cartopy.crs as ccrs                                         |
#|          ax.add_geometries(mesh.lsmask_p, crs=ccrs.PlateCarree(),           |
#|                            facecolor=[0.6,0.6,0.6], edgecolor='k',          |
#|                            linewidth=0.5)                                   |
#|_____________________________________________________________________________|
def lsmask_patch(lsmask):
    from shapely.geometry import Polygon, MultiPolygon
    #___________________________________________________________________________
    polygonlist=[]
    for xycoord in lsmask: polygonlist.append(Polygon(xycoord))
    lsmask_p = MultiPolygon(polygonlist)
    
    #___________________________________________________________________________
    return(lsmask_p)



# ___SAVE POLYGON LAND-SEA MASK CONTOUR TO SHAPEFILE___________________________
#| save FESOM2 grid land-sea mask polygons to shapefile                        |
#| ___INPUT_________________________________________________________________   |
#| mesh     :   fesom2 mesh object, contains periodic augmented land-sea mask  |
#|              polygonss in mesh.lsmask_a                                     |
#| lsmask   :   list, if empty mesh.lsmask_a is stored in shapefile, if not    |
#|              empty lsmask=lsmaskin than lsmaskin will be stored in          |
#|              shapefile                                                      |
#| path     :   str if empty mesh.path (or cache path depending on writing     |
#|              permission) is used as location to store the shapefile, if     |
#|              path=pathin than this string serves as location to store       |
#|              .shp file                                                      |
#| fname    :   str if empty fixed filename is used mypymesh_fesom2_ID_        |
#|              focus=X.shp, if not empty than fname=fnamein is used as        |
#|              filename for shape file                                        |
#| do_info  :   bool, print info where .shp file is saved, default = True      |       
#| ___RETURNS_______________________________________________________________   |
#| nothing                                                                     |
#| ___INFO__________________________________________________________________   |
#| --> to load and plot shapefile patches                                      |
#|      import shapefile as shp                                                |
#|      from matplotlib.patches import Polygon                                 |
#|      from matplotlib.collections import PatchCollection                     |
#|                                                                             |
#|      shpfname = 'tripyview_fesom2'+'_'+mesh.id+'_'+                          |
#|                  '{}={}'.format('focus',mesh.focus)+'.shp'                  |
#|      shppath  = os.path.join(mesh.cachepath,shpfname)                       |
#|                                                                             |
#|      sf = shp.Reader(shppath)                                               |
#|      patches = []                                                           |
#|      for shape in sf.shapes(): patches.append(Polygon(shape.points))        |
#|                                                                             |
#|      plt.figure()                                                           |
#|      ax = plt.gca()                                                         |
#|      ax.add_collection(PatchCollection(patches,                             |
#|                        facecolor=[0.7,0.7,0.7],                             |
#|                        edgecolor='k', linewidths=1.))                       |
#|      ax.set_xlim([-180,180])                                                |
#|      ax.set_ylim([-90,90])                                                  |
#|      plt.show()                                                             |
#|_____________________________________________________________________________|
def lsmask_2shapefile(mesh, lsmask=[], path=[], fname=[], do_info=True):
    import geopandas as gpd
    from shapely.geometry import Polygon
    
    #___________________________________________________________________________
    # Create an empty geopandas GeoDataFrame
    newdata = gpd.GeoDataFrame()

    #___________________________________________________________________________
    # Add every polygon to GeoDataFrame
    
    # do not use polygon that is stored in mesh object use polygon that is given 
    # by lsmask=lsmask
    if not lsmask:
        for ii in range(0,len(mesh.lsmask_a)):
            coordinates = mesh.lsmask_a[ii]
            if coordinates.shape[0]==2: continue
            
            # Create a Shapely polygon from the coordinate-tuple list
            poly = Polygon(coordinates)
            
            # Insert the polygon into 'geometry' -column at index 0
            newdata.loc[ii, 'geometry'] = poly
            newdata.loc[ii, 'location'] = "{} {}".format('polygon',str(ii))
            
    # use polygon storred in mesh object
    else:
        for ii in range(0,len(lsmask)):
            coordinates = lsmask[ii]
            if coordinates.shape[0]==2: continue
            
            # Create a Shapely polygon from the coordinate-tuple list
            poly = Polygon(coordinates)
            
            # Insert the polygon into 'geometry' -column at index 0
            newdata.loc[ii, 'geometry'] = poly
            newdata.loc[ii, 'location'] = "{} {}".format('polygon',str(ii))
    
    #___________________________________________________________________________
    # write shapefile either in meshpath when there is writing permission 
    # otherwise write in cachepath
    if not path:
        if os.access(mesh.path, os.W_OK): 
            shppath = mesh.path
        else:
#             cachepath = path = os.environ.get('MYPY_MESHPATH', os.path.join(os.path.expanduser("~"), "mesh_mypycache"))
            cachepath = path = os.environ.get('MESHPATH_TRIPYVIEW', os.path.join(os.path.expanduser("~"), "meshcache_tripyview"))
            cachepath = os.path.join(cachepath, mesh.id)
            if not os.path.isdir(cachepath):
                if do_info: print(' > create cache directory: {}'.format(cachepath))
                os.makedirs(cachepath)
            shppath = cachepath  
    # if an extra path is given us this to store the shapefile         
    else:
        shppath = path
        
    #___________________________________________________________________________
    # write lsmask to shapefile 
    if not fname:
        shpfname = 'tripyview_fesom2'+'_'+mesh.id+'_'+'{}={}'.format('focus',mesh.focus)+'.shp'
    else:
        shpfname = fname+'.shp'
    shppath = os.path.join(shppath,shpfname)
    if do_info: print(' > save *.shp to {}'.format(shppath))
    newdata.to_file(shppath)
    
    #___________________________________________________________________________
    return



# ___COMPUTE EULER ROTATION MATRIX_____________________________________________
#| compute euler rotation matrix based on alpha, beta and gamma angle          |
#| ___INPUT_________________________________________________________________   |
#| abg      :   list, with euler angles [alpha, beta, gamma]                   |
#| ___RETURNS_______________________________________________________________   |
#| rmat     :   array, [3 x 3] rotation matrix to transform from geo to rot    |
#|_____________________________________________________________________________|
def grid_rotmat(abg):
    #___________________________________________________________________________
    rad = np.pi/180
    al  = abg[0] * rad 
    be  = abg[1] * rad
    ga  = abg[2] * rad
        
    #___________________________________________________________________________
    rmat= np.zeros((3,3))
    rmat[0,0] =( np.cos(ga)*np.cos(al) - np.sin(ga)*np.cos(be)*np.sin(al) )
    rmat[0,1] =( np.cos(ga)*np.sin(al) + np.sin(ga)*np.cos(be)*np.cos(al) )
    rmat[0,2] =( np.sin(ga)*np.sin(be) )
        
    rmat[1,0] =(-np.sin(ga)*np.cos(al) - np.cos(ga)*np.cos(be)*np.sin(al) )
    rmat[1,1] =(-np.sin(ga)*np.sin(al) + np.cos(ga)*np.cos(be)*np.cos(al) )
    rmat[1,2] =( np.cos(ga)*np.sin(be) )
        
    rmat[2,0] =( np.sin(be)*np.sin(al) )
    rmat[2,1] =(-np.sin(be)*np.cos(al) )        
    rmat[2,2] =( np.cos(be) )
        
    #___________________________________________________________________________
    return(rmat)



# ___COMPUTE3D CARTESIAN COORDINAT_____________________________________________
#| compute 3d cartesian coordinates from spherical geo coordinates             |
#| (lon, lat, R=1.0)                                                           |
#| ___INPUT_________________________________________________________________   |
#| lon, lat :   array, longitude and latitude coordinates in radians           |
#| ___RETURNS_______________________________________________________________   |
#| x, y, z  :   array, x y z cartesian coordinates                             |
#|_____________________________________________________________________________|    
def grid_cart3d(lon,lat,R=1.0, is_deg=False):
    if is_deg: 
        rad = np.pi/180
        lat = lat * rad
        lon = lon * rad
    
    x = R*np.cos(lat) * np.cos(lon)
    y = R*np.cos(lat) * np.sin(lon)
    z = R*np.sin(lat)
    #___________________________________________________________________________
    return(x,y,z)



# ___ROTATE GRID FROM: ROT-->GEO_______________________________________________
#| compute grid rotation from sperical rotated frame back towards normal geo   |
#| frame using the euler angles alpha, beta, gamma                             | 
#| ___INPUT_________________________________________________________________   |
#| abg      :   list, with euler angles [alpha, beta, gamma]                   |
#| rlon,rlat:   array, longitude and latitude coordinates of sperical rotated  |
#|              frame in degree                                                |
#| ___RETURNS_______________________________________________________________   |
#| lon,lat  :   array, longitude and latitude coordinates in normal geo        |
#|              frame in degree                                                |
#|_____________________________________________________________________________|    
def grid_r2g(abg, rlon, rlat):
    
    #___________________________________________________________________________
    # build rotation matrix
    rmat = grid_rotmat(abg)
    rmat = np.linalg.pinv(rmat)

    #___________________________________________________________________________
    # compute 3d cartesian coordinates
    rad      = np.pi/180
    rlat     = rlat * rad
    rlon     = rlon * rad
    xr,yr,zr = grid_cart3d(rlon,rlat)
    
    #___________________________________________________________________________
    # rotate to geographical cartesian coordinates:
    xg=rmat[0,0]*xr + rmat[0,1]*yr + rmat[0,2]*zr;
    yg=rmat[1,0]*xr + rmat[1,1]*yr + rmat[1,2]*zr;
    zg=rmat[2,0]*xr + rmat[2,1]*yr + rmat[2,2]*zr;
    
    #___________________________________________________________________________
    # compute to geographical coordinates:
    lon, lat = np.arctan2(yg,xg), np.arcsin(zg)        
    lon, lat = lon/rad, lat/rad
    
    #___________________________________________________________________________
    return(lon,lat)



# ___ROTATE GRID FROM: GEO-->ROT_______________________________________________
#| compute grid rotation from normal geo frame towards sperical rotated        |
#| frame using the euler angles alpha, beta, gamma                             | 
#| ___INPUT_________________________________________________________________   |
#| abg      :   list, with euler angles [alpha, beta, gamma]                   |
#| lon, lat :   array, longitude and latitude coordinates of normal geo        |
#|              frame in degree                                                |
#| ___RETURNS_______________________________________________________________   |
#| rlon,rlat:   array, longitude and latitude coordinates in sperical rotated  |
#|              frame in degree                                                |
#|_____________________________________________________________________________|    
def grid_g2r(abg, lon, lat):

    #___________________________________________________________________________
    # build rotation matrix
    rmat = grid_rotmat(abg)
    
    #___________________________________________________________________________
    # compute 3d cartesian coordinates
    rad      = np.pi/180
    lat      = lat * rad
    lon      = lon * rad
    xg,yg,zg = grid_cart3d(lon,lat)

    #___________________________________________________________________________
    # rotate to geographical cartesian coordinates:
    xr=rmat[0,0]*xg + rmat[0,1]*yg + rmat[0,2]*zg;
    yr=rmat[1,0]*xg + rmat[1,1]*yg + rmat[1,2]*zg;
    zr=rmat[2,0]*xg + rmat[2,1]*yg + rmat[2,2]*zg;

    #___________________________________________________________________________
    # compute to geographical coordinates:
    rlon = np.arctan2(yr,xr)     
    rlat = np.arcsin(zr)        
    rlon = rlon/rad
    rlat = rlat/rad
    
    #___________________________________________________________________________
    return(rlon,rlat)



# ___ROTATE GRID FOCUS: 0-->FOCUS______________________________________________
#| compute grid rotation around z-axis to change the focus center of the lon,  |
#| lat grid, by default focus=0-->lon=[-180...180], if focus=180-->lon[0..360] | 
#| ___INPUT_________________________________________________________________   |
#| focus    :   float, longitude of grid center                                |
#| rlon,rlat:   array, longitude and latitude in focus=0-->lon=[-180...180]    |
#|              in degree                                                      |
#| ___RETURNS_______________________________________________________________   |
#| lon,lat  :   array, longitude and latitude in lon=[-180+focus...180+focus]  |
#|              frame in degree                                                |
#|_____________________________________________________________________________|    
def grid_focus(focus, rlon, rlat):

    #___________________________________________________________________________
    # build rotation matrix
    abg = [-focus, 0, 0]
    rmat = grid_rotmat(abg)
    rmat = np.linalg.pinv(rmat)

    #___________________________________________________________________________
    # compute 3d cartesian coordinates
    rad      = np.pi/180
    rlat     = rlat * rad
    rlon     = rlon * rad
    xr,yr,zr = grid_cart3d(rlon,rlat)

    #___________________________________________________________________________
    # rotate to geographical cartesian coordinates:
    xg=rmat[0,0]*xr + rmat[0,1]*yr + rmat[0,2]*zr;
    yg=rmat[1,0]*xr + rmat[1,1]*yr + rmat[1,2]*zr;
    zg=rmat[2,0]*xr + rmat[2,1]*yr + rmat[2,2]*zr;

    #___________________________________________________________________________
    # compute to geographical coordinates:
    lon, lat = np.arctan2(yg,xg), np.arcsin(zg)        
    lon, lat = lon/rad, lat/rad
    lon      = lon + focus 
    
    #___________________________________________________________________________
    return(lon,lat)
    
    

# ___ROTATE VECTOR FROM: ROT-->GEO_____________________________________________
#| in FESOM2 the vector variables are always given in the rotated coordiante   | 
#| frame in which the model works and need to be rotated into normal geo       |
#| coordinates                                                                 |
#| ___INPUT_________________________________________________________________   |
#| abg      :   list, with euler angles [alpha, beta, gamma]                   |
#| lon, lat :   array, longitude and latitude                                  |
#| urot,vrot:   array, zonal and meridional velocities in rotated frame        |
#| gridis   :   str, in which coordinate frame are given lon, lat              |
#|              'geo','g','geographical': lon,lat is given in geo coordinates  |
#|              'rot','r','rotated'     : lon,lat is given in rot coordinates  |
#| ___RETURNS_______________________________________________________________   |
#| ugeo,vgeo|   array, zonal and meridional velocities in normal geo frame     |
#|_____________________________________________________________________________|    
def vec_r2g(abg, lon, lat, urot, vrot, gridis='geo', do_info=False ):
    
    #___________________________________________________________________________
    # create grid coorinates for geo and rotated frame
    if any(x in gridis for x in ['geo','g','geographical']): 
        rlon, rlat = grid_g2r(abg, lon, lat)        
    elif any(x in gridis for x in ['rot','r','rotated']):     
        rlon, rlat = lon, lat 
        lon,  lat  = grid_g2r(abg, rlon, rlat)
    else:
        raise ValueError("The option gridis={} in vec_r2g is not supported.\n (only: 'geo','g','geographical', 'rot','r','rotated') ".format(str(gridis)))
    
    #___________________________________________________________________________
    # compute rotation matrix
    rmat = grid_rotmat(abg)
    rmat = np.linalg.pinv(rmat)
    rad  = np.pi/180 

    #___________________________________________________________________________
    # degree --> radian  
    lon , lat  = lon*rad , lat*rad
    rlon, rlat = rlon*rad, rlat*rad
    
    #___________________________________________________________________________
    # rotation of one dimensional vector data
    if vrot.ndim==1 or urot.ndim==1: 
        #_______________________________________________________________________
        # compute vector in rotated cartesian coordinates
        vxr = -vrot*np.sin(rlat)*np.cos(rlon) - urot*np.sin(rlon)
        vyr = -vrot*np.sin(rlat)*np.sin(rlon) + urot*np.cos(rlon)
        vzr =  vrot*np.cos(rlat)
        
        #_______________________________________________________________________
        # compute vector in geo cartesian coordinates
        vxg = rmat[0,0]*vxr + rmat[0,1]*vyr + rmat[0,2]*vzr
        vyg = rmat[1,0]*vxr + rmat[1,1]*vyr + rmat[1,2]*vzr
        vzg = rmat[2,0]*vxr + rmat[2,1]*vyr + rmat[2,2]*vzr
        
        #_______________________________________________________________________
        # compute vector in geo coordinates 
        vgeo= np.array(vxg*-np.sin(lat)*np.cos(lon) - 
                       vyg* np.sin(lat)*np.sin(lon) + 
                       vzg* np.cos(lat))
        ugeo= np.array(vxg*-np.sin(lon) + vyg*np.cos(lon))
        
    #___________________________________________________________________________
    # rotation of two dimensional vector data    
    elif vrot.ndim==2 or urot.ndim==2: 
        nd1,nd2=urot.shape
        ugeo, vgeo = urot.copy(), vrot.copy()
        if do_info: print('nlev:{:d}'.format(nd2))
        for nd2i in range(0,nd2):
            #___________________________________________________________________
            if do_info: 
                print('{:02d}|'.format(nd2i), end='')
                if np.mod(nd2i+1,10)==0: print('')
            #___________________________________________________________________
            aux_urot, aux_vrot = urot[:,nd2i], vrot[:,nd2i]
            #___________________________________________________________________
            # compute vector in rotated cartesian coordinates
            vxr = -aux_vrot*np.sin(rlat)*np.cos(rlon) - aux_urot*np.sin(rlon)
            vyr = -aux_vrot*np.sin(rlat)*np.sin(rlon) + aux_urot*np.cos(rlon)
            vzr =  aux_vrot*np.cos(rlat)
            
            #___________________________________________________________________
            # compute vector in geo cartesian coordinates
            vxg = rmat[0,0]*vxr + rmat[0,1]*vyr + rmat[0,2]*vzr
            vyg = rmat[1,0]*vxr + rmat[1,1]*vyr + rmat[1,2]*vzr
            vzg = rmat[2,0]*vxr + rmat[2,1]*vyr + rmat[2,2]*vzr
            
            #___________________________________________________________________
            # compute vector in geo coordinates 
            vgeo[:,nd2i]= np.array(vxg*-np.sin(lat)*np.cos(lon) - 
                                   vyg* np.sin(lat)*np.sin(lon) + 
                                   vzg* np.cos(lat))
            ugeo[:,nd2i]= np.array(vxg*-np.sin(lon) + vyg*np.cos(lon))
        
    #___________________________________________________________________________    
    else: raise ValueError('This number of dimensions is in moment not supported for vector rotation')    
    #___________________________________________________________________________
    return(ugeo, vgeo)



# ___CUTOUT REGION BASED ON BOX________________________________________________
#| cutout region based on box and return mesh elements indices that are        |
#| within the box                                                              |
#| ___INPUT_________________________________________________________________   |
#| nx       :   longitude vertice coordinates                                  |
#| ny       :   latitude  vertice coordinates                                  |
#| e_i      :   element array                                                  |
#| box      :   list, [lonmin, lonmax, latmin, latmax]                         |
#| which    :   str, how limiting should be the selection                      |
#|              'soft' : elem with at least 1 vertices in box are selected     |
#|              'mid'  : elem with at least 2 vertices in box are selected     |
#|              'hard' : elem with at all vertices in box are selected         |
#| ___RETURNS_______________________________________________________________   |
#| e_inbox :   array, boolian array with 1 in box, 0 outside box               |
#|_____________________________________________________________________________|    
def grid_cutbox_e(n_x, n_y, e_i, box, which='mid'):# , do_outTF=False):
    
    #___________________________________________________________________________
    n_inbox = grid_cutbox_n(n_x, n_y, box)
    
    #___________________________________________________________________________
    e_inbox = n_inbox[e_i]
    
    # considers triangles where at least one node is in box
    if   which == 'soft': e_inbox =  np.any(e_inbox,axis=1)
    elif which == 'mid' : e_inbox = (np.sum(e_inbox,axis=1)>1)
    # considers triangles where all node must be in box (serated edge)
    elif which == 'hard': e_inbox =  np.all(e_inbox,axis=1) 
    else: raise ValueError("The option which={} in grid_cutbox is not supported. \n(only: 'hard', 'soft')".format(str(which)))
    
    #___________________________________________________________________________
    return(e_inbox)


# ___CUTOUT REGION BASED ON BOX________________________________________________
#| cutout region based on box and return mesh elements indices that are        |
#| within the box                                                              |
#| ___INPUT_________________________________________________________________   |
#| nx       :   longitude vertice coordinates                                  |
#| ny       :   latitude  vertice coordinates                                  |
#| e_i      :   element array                                                  |
#| box      :   list, [lonmin, lonmax, latmin, latmax]                         |
#| ___RETURNS_______________________________________________________________   |
#| n_inbox :   array, boolian array with 1 in box, 0 outside box               |
#|_____________________________________________________________________________|    
def grid_cutbox_n(n_x, n_y, box):# , do_outTF=False):

    #___________________________________________________________________________
    n_inbox = ((n_x >= box[0]) & (n_x <= box[1]) & 
               (n_y >= box[2]) & (n_y <= box[3]))
    
    #___________________________________________________________________________
    return(n_inbox)



# ___INTERPOLATE FROM ELEMENTS TO VERTICES_____________________________________
#|                                                                             |
#|_____________________________________________________________________________|
def grid_interp_e2n(mesh,data_e):
    
    #___________________________________________________________________________
    mesh = mesh.compute_e_area()
    mesh = mesh.compute_n_area()
    if data_e.ndim==1:
        # print('~~ >-)))°> .°oO A')
        aux  = np.vstack((mesh.e_area,mesh.e_area,mesh.e_area)).transpose().flatten()
        aux  = aux * np.vstack((data_e,data_e,data_e)).transpose().flatten()
        
        #___________________________________________________________________________
        # single loop over self.e_i.flat is ~4 times faster than douple loop 
        # over for i in range(3): ,for j in range(self.n2de):
        data_n = np.zeros((mesh.n2dn,))
        count = 0
        for idx in mesh.e_i.flat:
            data_n[idx]=data_n[idx] + aux[count]
            count=count+1 # count triangle index for aux_area[count] --> aux_area =[n2de*3,]
        del aux, count
        #with np.errstate(divide='ignore',invalid='ignore'):
        data_n=data_n/mesh.n_area[0,:]/3.0
        
    elif data_e.ndim==2:
        # print('~~ >-)))°> .°oO B')
        nd     = data_e.shape[1]
        data_n = np.zeros((mesh.n2dn, nd))
        aux1   = np.vstack((mesh.e_area,mesh.e_area,mesh.e_area)).transpose().flatten()
        for ndi in range(0,nd):
            aux  = aux1 * np.vstack((data_e[:,ndi],data_e[:,ndi],data_e[:,ndi])).transpose().flatten()
            #___________________________________________________________________________
            # single loop over self.e_i.flat is ~4 times faster than douple loop 
            # over for i in range(3): ,for j in range(self.n2de):
            count = 0
            for idx in mesh.e_i.flat:
                data_n[idx, ndi]=data_n[idx, ndi] + aux[count]
                count=count+1 # count triangle index for aux_area[count] --> aux_area =[n2de*3,]
            del aux, count
            #with np.errstate(divide='ignore',invalid='ignore'):
            data_n[:, ndi]=data_n[:, ndi]/mesh.n_area[ndi, :]/3.0
        
    #___________________________________________________________________________
    return(data_n)

    
# ___EQUIVALENT OF MATLAB ISMEMBER FUNCTION____________________________________
#|                                                                             |
#|_____________________________________________________________________________|
def ismember_rows(a, b):
    return np.flatnonzero(np.in1d(b[:,0], a[:,0]) & np.in1d(b[:,1], a[:,1]))
