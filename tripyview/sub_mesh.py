# Patrick Scholz, 14.12.2017

import sys
import os
import time as clock
import numpy  as np
import dask.array as da
import pandas as pa
import joblib
import warnings
try:
    import pickle5 as pickle
    foundmodpickle=True
except ModuleNotFoundError:
    foundmodpickle=False
    pass
from   netCDF4 import Dataset
from .sub_mesh import *

from numba import jit, njit, float32, int32, prange, types
from numba.typed import Dict
import numba

from shapely.geometry import Polygon

rad     = np.pi/180
R_earth = 12735/2*1000;

# ___INITIALISE/LOAD FESOM2.0 MESH CLASS IN MAIN PROGRAMM______________________
#| IMPORTANT!!!:                                                               |                                         
#| only when mesh is initialised with this routine, the main programm is able  |
#| to recognize if mesh object already exist or not and if it exist to use it  |
#| and not to load it again !!!                                                |
#|_____________________________________________________________________________|
def load_mesh_fesom2(
                    meshpath, 
                    abg         = [50 , 15, -90]        , 
                    focus       = 0                     , 
                    cyclic      = 360                   , 
                    do_rot      = 'None'                , 
                    do_augmpbnd = True                  , 
                    do_cavity   = False                 , 
                    do_lsmask   = True                  , 
                    do_lsmshp   = False                 , 
                    do_earea    = True                  , 
                    do_narea    = True                  , 
                    do_eresol   = [False,'mean']        , 
                    do_nresol   = [False,'e_resol']     ,
                    do_loadraw  = False                 , 
                    do_pickle   = False                 , 
                    do_joblib   = True                  , 
                    do_redosave = True                  ,
                    do_f14cmip6 = False                 ,
                    do_info     = True                  , 
                    ):
    """
    --> load FESOM2 mesh
    
    Parameters:

        :mespath:       str, path that leads to FESOM2.0 mesh files (.out)

        :abg:           list, [alpha,beta,gamma], (default=[50,15,-90]) euler angles used to rotate the grids within the model

        :focus:         float, (default=0) sets longitude center of the mesh, lon=[-180...180], focus=180 lon=[0...360]

        :cyclic:        float, (default=360.0), length of cyclic domain in lon degree can be different channel configuration

        :do_rot:        str, (default='None') should the grid be rotated, default: 'None' 
                        - None, 'None' ... no rotation is applied 
                        - 'r2g'        ... loaded grid is rotated and transformed to geo
                        - 'g2r'        ... loaded grid is geo and transformed to rotated

        :do_augmpbnd:   bool, (default=True) augment periodic boundary triangles, default: True

        :do_cavity:     bool, (default=False) load also cavity files cavity_nlvls.out and cavity_elvls.out

        :do_lsmask:     bool, (default=True) 
                        Compute land-sea mask polygon for FESOM2 mesh see mesh.lsmask, augments 
                        its periodic boundaries see mesh.lasmask_a and computes land sea mask 
                        patch see mesh.lsmask_p

        :do_lsmshp:     bool, (default=True) save land-sea mask with periodic boundnaries to shapefile         

        :do_earea:      bool, (default=True) compute or load from fesom.mesh.diag.nc the area of elements

        :do_narea:      bool, (default=True) compute or load from fesom.mesh.diag.nc the clusterarea 
                        of vertices

        :do_eresol:     list([bool,str]), (default: [False,'mean']) compute 
                        resolution based on elements, str can be...
                        - "mean": resolution based on mean element edge length, 
                        - "max": resolution based on maximum edge length, 
                        - "min" resolution based on minimum edge length, 

        :do_nresol:     list([bool,str]), (default: [False,'e_resol']), compute resolution at nodes from interpolation of 
                        resolution at elements

        :do_loadraw:    bool, (default=False) also load the raw vertical level information
                        for elements. Its the vertical level information before the exclusion
                        of elements that have three boundary nodes in the topography

        :do_pickle:     bool, (default=True) store and load mesh from .pckl binary file, 
                        pickle5 is just supported until < python3.9. If pickel library 
                        cant be  found it switches automatic to joblib

        :do_joblib:     bool, (default=False) store and load mesh from .joblib binary file
        :do_load:       bool, (default=True) loading infrastructureshould be used

        :do_f14cmip6:   bool, (default=False) load FESOM1.4 mesh information and squeeze it into
                        the framework of FESOM2. Needed here to compute AMOC on fesom1.4 
                        cmorized CMIP6 data.

        :do_info:       bool, (default=True) print progress and mesh information

    Returns:
    
        :mesh:          object, returns fesom_mesh object

    """
    
    pickleprotocol=4
    #___________________________________________________________________________
    if foundmodpickle==False and do_pickle==True:
        print(' > warning: pickle5 module could not be found, no do_pickle \n is possible! Therefor switch to joblib saving/loading')
        do_pickle, do_joblib = False, True
    if do_joblib==True and do_pickle==True:   
        raise ValueError(' error: both do_joblib and do_pickle are set to True, select only one!')
    
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
    picklefname = 'tripyview_mesh_{:s}_focus{:d}.pckl'.format(meshid,focus)
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
    
    joblibfname = 'tripyview_mesh_{:s}_focus{:d}.jlib'.format(meshid,focus)
    if do_joblib:
        # check if mypy pickle meshfile is found in meshfolder
        if    ( os.path.isfile(os.path.join(meshpath, joblibfname)) ):
            loadjoblibpath = os.path.join(meshpath, joblibfname)
            if do_info: print(' > found *.jlib file: {}'.format(os.path.dirname(loadjoblibpath)))    
            
        # check if mypy pickle meshfile is found in cache folder
        elif ( os.path.isfile(os.path.join(cachepath, joblibfname)) ):
            loadjoblibpath = os.path.join(cachepath, joblibfname)
            if do_info: print(' > found *.jlib file: {}'.format(os.path.dirname(loadjoblibpath)))
            
        else:
            loadjoblibpath = 'None'
            print(' > found no .jlib file in cach or mesh path')            
            
    #___________________________________________________________________________
    # load pickle file if it exists and load it from .pckl file, if it does not 
    # exist create mesh object with fesom_mesh
    # do_pickle==True and .pckl file exists
    if  (( do_pickle and ( os.path.isfile(loadpicklepath) )) or \
        ( do_joblib and ( os.path.isfile(loadjoblibpath) ))) and \
          (do_redosave==False): 
            
        
        #_______________________________________________________________________
        if   ( do_pickle and ( os.path.isfile(loadpicklepath) )):
            if do_info: print(' > load  *.pckl file: {}'.format(os.path.basename(loadpicklepath)))
            fid  = open(loadpicklepath, "rb")
            mesh = pickle.load(fid)
            fid.close()
        elif ( do_joblib and ( os.path.isfile(loadjoblibpath) )):
            if do_info: print(' > load  *.jlib file: {}'.format(os.path.basename(loadjoblibpath)))
            mesh = joblib.load(loadjoblibpath)
            
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
    elif ((do_pickle and not ( os.path.isfile(loadpicklepath)) ) or not do_pickle) or \
         ((do_joblib and not ( os.path.isfile(loadjoblibpath)) ) or not do_joblib) or \
           do_redosave:
             
        if do_info: print(' > load mesh from *.out files: {}'.format(meshpath))
        #_______________________________________________________________________
        mesh = mesh_fesom2(
                        meshpath   = meshpath     , 
                        abg        = abg          , 
                        focus      = focus        ,
                        cyclic     = cyclic       ,
                        do_rot     = do_rot       ,
                        do_augmpbnd= do_augmpbnd  ,
                        do_cavity  = do_cavity    ,
                        do_info    = do_info      ,
                        do_earea   = do_earea     ,
                        do_eresol  = do_eresol    ,
                        do_narea   = do_narea     ,
                        do_nresol  = do_nresol    ,
                        do_lsmask  = do_lsmask    ,
                        do_lsmshp  = do_lsmshp    ,
                        do_loadraw = do_loadraw   ,
                        do_f14cmip6 = do_f14cmip6   ,
                        )
        
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
        # save mypy mesh .jlib file
        # --> try 1.st to store it in the mesh in the meshfolder, will depend of 
        #     there is permission to write
        if do_joblib:
            try: 
                savejoblibpath = os.path.join(meshpath, joblibfname)
                if do_info: print(' > save mesh to *.jlib in {}'.format(savejoblibpath))
                fid = open(savejoblibpath, "wb")
                joblib.dump(mesh, fid, protocol=pickleprotocol)
                fid.close()
            # if no permission rights for writing in meshpath folder try 
            # cachefolder   
            except PermissionError:
                try: 
                    savejoblibpath = os.path.join(cachepath,joblibfname)
                    mesh.cachepath = cachepath
                    if do_info: print(' > save mesh to *.pckl in {}'.format(savejoblibpath))
                    fid = open(savejoblibpath, "wb")
                    joblib.dump(mesh, fid, protocol=pickleprotocol)
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
    --> Class that creates object that contains all information about FESOM2
        mesh. As minimum requirement the mesh path to the files nod2d.out,
        elem2d.out and aux3d.out has to be given, 
    
    __________________________________________________
    
    Parameters:    
        
        see help(load_fesom_mesh)
    
    __________________________________________________
    
    Variables:    
    
        path:           str, path that leads to FESOM2.0 mesh files (.out)
        
        id:             str, identifies mesh 
        
        n2dn:           int, number of 2d nodes 
        
        n2de:           int, number of 2d elements
        
        n_x:            array, lon position of surface nodes 
        
        n_y:            array, lat position of surface nodes 
        
        e_i:            array, elemental array with 2d vertice indices, shape=[n2de,3] 
        
        ___vertical info____________________________________
        
        nlev:           int, number of vertical full cell level
        
        zlev:           array, with full depth levels
        
        zmid:           array, with mid depth levels
        
        n_z:            array, bottom depth based on zlev[n_iz], 
        
        n_iz:           array, number of full depth levels at vertices
        
        e_iz:           array, number of full depth levels at elem
        
        ___cavity info (if do_cavity==True)_________________
        
        n_ic:           array, full depth level index of cavity-ocean interface at vertices
                        
        e_ic:           array, full depth level index of cavity-ocean interface at elem
                        
        n_c:            array, cavity-ocean interface depth at vertices zlev[n_ic]
        
        ___area and resoltion info___________________________
        
        n_area:         array, area at vertices
        
        n_resol:        array, resolutionat vertices
        
        e_area:         array, area at elements
        
        e_resol:        array, resolution at elements
        
        ___periodic boundary augmentation____________________
        
        n_xa:           array, with augmented vertice paramters
        
        n_ya:           ...
        
        n_za:           ...
        
        n_iza:          ...
        
        n_ca:           ... 
        
        n_ica:          ...
        
        e_ia:           array, element array with augmented triangles --> np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia))
        
        e_pbnd_1:       array, elem indices of pbnd elements
        
        e_pbnd_0:       array, elem indices of not pbnd elements
        
        e_pbnd_a:       array, elem indices of periodic augmented elements --> data_plot = np.hstack((data_plot[mesh.e_pbnd_0],data_plot[mesh.e_pbnd_a]))
        
        n_pbnd_a:       array, vertice indices to augment pbnd --> data_plot = np.hstack((data_plot,data_plot[mesh.n_pbnd_a]))
        
        n2dna:          int, number of vertices with periodic boundary augmentation
        
        n2dea:          int, number of elements with periodic boundary augmentation
                        
                        
        ___land sea mask (if do_lsmask == True)______________
        
        lsmask: list(array1[npts,2], array2[npts,2], ...), contains all land-sea mask polygons for FESOM2 mesh, with periodic boundary

        lsmask_a: list(array1[npts,2], array2[npts,2], ...)contains all land-sea mask polygons for FESOM2 mesh, with augmented 
        periodic boundary

        lsmask_p: polygon, contains polygon collection that can be plotted as 
        closed polygon patches with ax.add_collection(PatchCollection
        (mesh.lsmask_p,facecolor=[0.7,0.7,0.7], edgecolor='k', linewidth=0.5))
    
    __________________________________________________
    
    Returns:

        mesh:       object, returns fesom_mesh object

    __________________________________________________
    
    Info:
    
    create matplotlib triangulation with augmented periodic boundary
    
    ::
        
        tri = Triangulation(np.hstack((mesh.n_x,mesh.n_xa)), 
                            np.hstack((mesh.n_y,mesh.n_ya)),
                            np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia)))

    """
    
    #___INIT FESOM2.0 MESH OBJECT_______________________________________________
    #                                                                           
    #___________________________________________________________________________
    def __init__(self, meshpath, abg=[50,15,-90], focus=0, cyclic=360, focus_old=0, do_rot='None', 
                 do_augmpbnd=True, do_cavity=False, do_info=True, do_earea=False,do_earea2=False, 
                 do_eresol=[False,'mean'], do_narea=False, do_nresol=[False,'n_area'], 
                 do_lsmask=True, do_lsmshp=True, do_pickle=True, do_loadraw=True,
                 do_f14cmip6=False):
        
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
        self.cyclic             = cyclic
        self.do_rot             = do_rot
        self.do_augmpbnd        = do_augmpbnd
        self.do_cavity          = do_cavity
        self.do_earea           = do_earea
        self.do_eresol          = do_eresol
        self.do_narea           = do_narea
        self.do_nresol          = do_nresol
        self.do_lsmask          = do_lsmask
        self.do_lsmshp          = do_lsmshp
        self.do_loadraw         = do_loadraw
        self.do_f14cmip6         = do_f14cmip6
        
        #_______________________________________________________________________
        # define basic mesh file path
        self.fname_nod2d        = os.path.join(self.path,'nod2d.out')
        if self.do_f14cmip6: self.fname_nod3d = os.path.join(self.path,'nod3d.out')
        self.fname_elem2d       = os.path.join(self.path,'elem2d.out')
        self.fname_aux3d        = os.path.join(self.path,'aux3d.out')
        self.fname_nlvls        = os.path.join(self.path,'nlvls.out')
        self.fname_elvls        = os.path.join(self.path,'elvls.out')
        self.fname_elvls_raw    = os.path.join(self.path,'elvls_raw.out')
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
        
        
        # we assume that the center focus of the mesh is at 0deg (-180°...180°) 
        # check if this right change focus 
        src_focus = (np.ceil(self.n_x.max()) + np.floor(self.n_x.min()))*0.5
        if (src_focus != 0):
            self.n_xo, self.n_yo = self.n_x.copy(), self.n_y.copy()
            self.n_x, self.n_y = grid_focus(0, self.n_xo, self.n_yo)
        
        # now change focus to destined longitude
        # cartopy always assumes that the input grid goes from -180...180 it than can internally
        # rotate the grid to a different focus
        if (self.focus != 0): 
            self.n_xo, self.n_yo = self.n_x.copy(), self.n_y.copy()
            self.n_x, self.n_y = grid_focus(self.focus, self.n_xo, self.n_yo)
        
        # find periodic boundary
        self.pbnd_find()
        
        # augment periodic boundary
        if do_augmpbnd and any(self.n_x[self.e_i].max(axis=1)-self.n_x[self.e_i].min(axis=1) > self.cyclic/2):
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
                shpfname = 'tripyview_lsmask_{:s}_pbnd.shp'.format(self.id)
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
        """
        --> part of fesom mesh class, read mesh files nod2d.out, elem2d.out, 
        aux3d.out, nlvls.out and elvls.out   
        
        """
        #____load 2d node matrix________________________________________________
        #file_content = pa.read_csv(self.fname_nod2d, delim_whitespace=True, skiprows=1, \
        file_content = pa.read_csv(self.fname_nod2d, sep='\\s+', skiprows=1, \
                                      names=['node_number','x','y','flag'] )
        self.n_x     = np.ascontiguousarray(file_content.x.values.astype('float32'))
        self.n_y     = np.ascontiguousarray(file_content.y.values.astype('float32'))
        self.n_i     = np.ascontiguousarray(file_content.flag.values.astype('int16'))   
        self.n2dn    = len(self.n_x)
        
        #____load 2d element matrix_____________________________________________
        #file_content = pa.read_csv(self.fname_elem2d, delim_whitespace=True, skiprows=1, \
        file_content = pa.read_csv(self.fname_elem2d, sep='\\s+', skiprows=1, \
                                    names=['1st_node_in_elem','2nd_node_in_elem','3rd_node_in_elem'])
        self.e_i     = file_content.values.astype('int32') - 1
        
        # ensure C-contiguous (row-major) format, will significantly speed up all 
        # numba operations 
        self.e_i     = np.ascontiguousarray(self.e_i)
        
        self.n2de    = np.shape(self.e_i)[0]
        # print('    : #2de={:d}'.format(self.n2de))
        
        #____load 3d nodes alligned under 2d nodes______________________________
        if not self.do_f14cmip6:
            with open(self.fname_aux3d) as f:
                self.nlev= int(next(f))
                self.zlev= np.array([next(f).rstrip() for x in range(self.nlev)]).astype(float)
                self.zlev= -np.abs(self.zlev)
            self.zmid    = (self.zlev[:-1]+self.zlev[1:])/2.
            
        else:
            t1=clock.time()
            # number of vertical levels
            with open(self.fname_aux3d) as f: self.nlev= int(next(f))
            
            # 3d vertice index below surface vertices index
            file_content = pa.read_csv(self.fname_aux3d, skiprows=0, nrows=self.nlev*self.n2dn)
            self.n32     = file_content.values.astype('int32') - 1
            self.n32     = self.n32.reshape((self.n2dn,self.nlev)).transpose()
            self.n32     = np.ascontiguousarray(self.n32)
            
            # Lick out bufferlayer in fesom1.4 mesh
            self.n32     = self.n32[:-1,:]
            self.nlev    = self.nlev-1
            
            # identify the vertical levels
            with open(self.fname_nod3d) as f: n3dn= int(next(f))
            #file_content = pa.read_csv(self.fname_nod3d, delim_whitespace=True, usecols=[3])
            file_content = pa.read_csv(self.fname_nod3d, sep='\\s+', usecols=[3])
            aux_n3z      = file_content.values.astype('int16') 
            self.zlev    = np.unique(aux_n3z)[::-1]
            #self.zlev    = np.hstack((self.zlev, self.zlev[-1]+(self.zlev[-1]-self.zlev[-2])))
            self.zmid    = (self.zlev[:-1]+self.zlev[1:])/2.
            
            # compute bottom topography at vertice
            self.n_z     = np.ascontiguousarray(aux_n3z[self.n32.max(axis=0),0])
            del(aux_n3z)
            
            # compute bottom index at vertice
            aux_n32      = np.zeros(self.n32.shape)
            aux_n32[self.n32>=0] = 1
            self.n_iz    = np.ascontiguousarray((aux_n32.sum(axis=0).astype('int16')-1))

        
        #____load number of levels at each node_________________________________
        if ( os.path.isfile(self.fname_nlvls) ):
            #file_content = pa.read_csv(self.fname_nlvls, delim_whitespace=True, skiprows=0, \
            file_content = pa.read_csv(self.fname_nlvls, sep='\\s+', skiprows=0, \
                                           names=['numb_of_lev'])
            self.n_iz    = file_content.values.astype('int16') - 1
            self.n_iz    = np.ascontiguousarray(self.n_iz.squeeze())
            self.n_z     = np.ascontiguousarray(np.float32(self.zlev[self.n_iz]))
            
        elif self.do_f14cmip6: print(f' --> you are in fesom1.4 mode, no nlvls information!')    
        else                : raise ValueError(f' --> could not find file {self.fname_nlvls} !')
            #self.n_iz    = np.zeros((self.n2dn,)) 
            #self.n_z     = np.zeros((self.n2dn,)) 
        
        #____load number of levels at each elem_________________________________
        if ( os.path.isfile(self.fname_elvls) ):
            #file_content = pa.read_csv(self.fname_elvls, delim_whitespace=True, skiprows=0, \
            file_content = pa.read_csv(self.fname_elvls, sep='\\s+', skiprows=0, \
                                           names=['numb_of_lev'])
            self.e_iz    = file_content.values.astype('int16') - 1
            self.e_iz    = np.ascontiguousarray(self.e_iz.squeeze())
            
        elif self.do_f14cmip6: print(f' --> you are in fesom1.4 mode, no elvls information!')        
        else                : raise ValueError(f' --> could not find file {self.fname_elvls} !')
            #self.e_iz    = np.zeros((self.n2de,)) 
        
        #____load number of raw levels at each elem_____________________________
        if (self.do_loadraw and  not self.do_f14cmip6):
            if ( os.path.isfile(self.fname_elvls_raw) ):
                #file_content = pa.read_csv(self.fname_elvls_raw, delim_whitespace=True, skiprows=0, \
                file_content = pa.read_csv(self.fname_elvls_raw, sep='\\s+', skiprows=0, \
                                            names=['numb_of_lev'])
                self.e_iz_raw    = file_content.values.astype('int16') - 1
                self.e_iz_raw    = np.ascontiguousarray(self.e_iz_raw.squeeze())
            else:
                raise ValueError(f' --> could not find file {self.fname_elvls_raw} !')
        
        #_______________________________________________________________________
        # vertical level information of fesom1.4 mesh
        #if self.do_f14cmip6:

        #_______________________________________________________________________
        return(self)    
    
    
    
    # ___READ FESOM2 MESH CAVITY INFO__________________________________________
    #| read files: cavity_nlvls.out, cavity_elvls.out                          |                                     
    #|_________________________________________________________________________|
    def read_cavity(self):
        """
        --> part of fesom mesh class, read cavity mesh files, cavity_nlvls.out
        cavity_elvls.out and cavity_elvls_raw.out (if do_loadraw=True)
        
        """
        
        #____load number of cavity levels at each node__________________________
        self.fname_cnlvls = os.path.join(self.path,'cavity_nlvls.out')
        if ( os.path.isfile(self.fname_cnlvls) ):
            file_content      = pa.read_csv(self.fname_cnlvls, delim_whitespace=True, skiprows=0, names=['numb_of_lev'])
            self.n_ic= file_content.values.astype('int16') - 1
            self.n_ic= np.ascontiguousarray(self.n_ic.squeeze())
        else:
            raise ValueError(f' --> could not find file {self.fname_cnlvls} !')
        
        #____load number of cavity levels at each elem__________________________
        self.fname_celvls = os.path.join(self.path,'cavity_elvls.out')
        if ( os.path.isfile(self.fname_cnlvls) ):
            file_content      = pa.read_csv(self.fname_celvls, delim_whitespace=True, skiprows=0, names=['numb_of_lev'])
            self.e_ic= file_content.values.astype('int16') - 1
            self.e_ic= np.ascontiguousarray(self.e_ic.squeeze())
        else:
            raise ValueError(f' --> could not find file {self.fname_celvls} !')
        
        #____load number of raw cavity levels at each elem______________________
        # number of cavity levels before it becomes iteratively optimse to avoid 
        # isolated cells 
        if self.do_loadraw:
            self.fname_celvls_raw = os.path.join(self.path,'cavity_elvls_raw.out')
            if ( os.path.isfile(self.fname_celvls_raw) ): 
                file_content    = pa.read_csv(self.fname_celvls_raw, delim_whitespace=True, skiprows=0, names=['numb_of_lev'])
                self.e_ic_raw   = file_content.values.astype('int16') - 1
                self.e_ic_raw   = np.ascontiguousarray(self.e_ic_raw.squeeze())
            else:
                raise ValueError(f' --> could not find file {self.fname_celvls_raw} !')
        
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
        """
        --> part of fesom mesh class, find elements that cross over the periodic
        boundary
        
        """         
        e0  = self.e_i[:, 0]  
        e1  = self.e_i[:, 1]
        e2  = self.e_i[:, 2]
        
        # compute per-element min/max in one pass without creating big (nelem,3) array
        maxlon = np.maximum(np.maximum(self.n_x[e0], self.n_x[e1]), self.n_x[e2])
        minlon = np.minimum(np.minimum(self.n_x[e0], self.n_x[e1]), self.n_x[e2])
        dx     = maxlon - minlon

        # threshold same as before
        thresh = self.cyclic * (2.0/3.0)

        idx_pbnd = dx > thresh
        self.e_pbnd_1 = np.nonzero( idx_pbnd)[0]
        self.e_pbnd_0 = np.nonzero(~idx_pbnd)[0]
        return self

    
    
    # ___AUGMENT PERIODIC BOUDNARY ELEMENTS____________________________________
    #| add additional elements to augment the periodic boundary on the left and|
    #| right side for an even non_periodic boundary                            |
    #|_________________________________________________________________________|
    def pbnd_augment(self):
        """
        --> part of fesom mesh class, adds additional elements to augment the 
        periodic boundary on the left and right side so that an even non_periodic 
        boundary is created left and right [-180, 180] of the domain.
        
        Vectorized version, semantics matching original loop implementation.
        """

        self.do_augmpbnd = True

        #_______________________________________________________________________
        # this are all the periodic boundary element
        e_i_pbnd = self.e_i[self.e_pbnd_1]      
        lon_tri  = self.n_x[e_i_pbnd]     
    
        # number of triangles that participate in periodic boudnary
        ntri = e_i_pbnd.shape[0]
        rows = np.arange(ntri)

        #_______________________________________________________________________
        # sort vertices by longitude per triangle: left, middle, right (by lon)
        idx_sort = np.argsort(lon_tri, axis=1)  # (ntri, 3)
        idx_l = idx_sort[:, 0]                  # position of left vertex in tri (0,1,2)
        idx_m = idx_sort[:, 1]                  # middle
        idx_r = idx_sort[:, 2]                  # right

        # node indices of left/mid/right vertices
        n_l = e_i_pbnd[rows, idx_l]
        n_m = e_i_pbnd[rows, idx_m]
        n_r = e_i_pbnd[rows, idx_r]

        #_______________________________________________________________________
        # decide whether middle vertex belongs to left or right boundary
        # is_pbnd_i_m_lr == True  => middle closer to right
        #                         => belongs to right boundary
        dist_m_to_r = np.abs(self.n_x[n_r] - self.n_x[n_m])
        dist_m_to_l = np.abs(self.n_x[n_l] - self.n_x[n_m])
        m_to_right  = dist_m_to_r < dist_m_to_l   # same criterion as original code

        # build sets of boundary nodes (same as original hstack + unique)
        n_pbnd_r = np.concatenate((n_r, n_m[ m_to_right]))
        n_pbnd_l = np.concatenate((n_l, n_m[~m_to_right]))
        n_pbnd_r = np.unique(n_pbnd_r)
        n_pbnd_l = np.unique(n_pbnd_l)

        # total periodic boundary nodes (augmented)
        self.n_pbnd_a = np.concatenate((n_pbnd_r, n_pbnd_l))

        #_______________________________________________________________________
        # mapping from original node index -> augmented node index
        nn_r = n_pbnd_r.size
        nn_l = n_pbnd_l.size
        
        aux_pos = np.full(self.n2dn, -1, dtype=np.int64)
        aux_pos[n_pbnd_r] = np.arange(self.n2dn,        self.n2dn + nn_r       , dtype=np.int64)
        aux_pos[n_pbnd_l] = np.arange(self.n2dn + nn_r, self.n2dn + nn_r + nn_l, dtype=np.int64)

        #_______________________________________________________________________
        # new augmented node coordinates (right boundary first at xmin, left at xmax)
        if self.cyclic == 360:
            xmin = -self.cyclic / 2.0 + self.focus
            xmax =  self.cyclic / 2.0 + self.focus
        else:
            xmin, xmax = 0.0, float(self.cyclic)

        self.n_xa  = np.concatenate((np.full(nn_r, xmin), np.full(nn_l, xmax)))
        self.n_ya  = self.n_y[self.n_pbnd_a]
        self.n_za  = self.n_z[self.n_pbnd_a]
        self.n_iza = self.n_iz[self.n_pbnd_a]

        if isinstance(self.n_c, np.ndarray):
            self.n_ca = self.n_c[self.n_pbnd_a]
        if isinstance(self.n_ic, np.ndarray):
            self.n_ica = self.n_ic[self.n_pbnd_a]

        # new total node count (old + augmented)
        self.n2dna = self.n2dn + nn_r + nn_l

        #_______________________________________________________________________
        # augment triangles: create left and right augmented copies
        elem_L = e_i_pbnd.copy()
        elem_R = e_i_pbnd.copy()

        # right vertex in left triangles gets augmented copy (right boundary → xmin)
        elem_L[rows, idx_r] = aux_pos[n_r]
        # left vertex in right triangles gets augmented copy (left boundary → xmax)
        elem_R[rows, idx_l] = aux_pos[n_l]

        # middle vertex augmentation:
        # original code:
        #   if is_pbnd_i_m_lr[ei] (middle closer to right):   elem_pbnd_l[...] = aux_pos[tri[idx_m]]
        #   else:                                             elem_pbnd_r[...] = aux_pos[tri[idx_m]]
        mid_aug = aux_pos[n_m]

        # middle belongs to RIGHT boundary (m_to_right == True) → augment in LEFT element
        elem_L[rows[m_to_right],  idx_m[m_to_right]]  = mid_aug[m_to_right]
        # middle belongs to LEFT boundary (m_to_right == False) → augment in RIGHT element
        elem_R[rows[~m_to_right], idx_m[~m_to_right]] = mid_aug[~m_to_right]

        #_______________________________________________________________________
        # stack augmented triangles: first right, then left (like your original)
        self.e_ia     = np.vstack((elem_R, elem_L))
        self.e_pbnd_a = np.hstack((self.e_pbnd_1, self.e_pbnd_1))
        self.n2dea    = self.n2de + elem_R.shape[0]
    
        #_______________________________________________________________________
        return self




    #___COMPUTE/LOAD AREA OF ELEMENTS__________________________________________
    #| either load area of elements from fesom.mesh.diag.nc if its found in    |
    #| meshpath or recompute it from scratch, [m^2]                            |
    #|_________________________________________________________________________|
    def compute_e_area(self):
        """
        --> part of fesom mesh class, either load area of elements from 
        fesom.mesh.diag.nc if its found in meshpath or recompute it from 
        scratch, [m^2]             

        """
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
                cycl   = self.cyclic*rad
                
                # convert lon/lat to radians
                n_x = self.n_x * rad
                n_y = self.n_y * rad

                # coordinate differences
                dx1 = n_x[self.e_i[:,1]] - n_x[self.e_i[:,0]]
                dy1 = n_y[self.e_i[:,1]] - n_y[self.e_i[:,0]]

                dx2 = n_x[self.e_i[:,2]] - n_x[self.e_i[:,0]]
                dy2 = n_y[self.e_i[:,2]] - n_y[self.e_i[:,0]]
                del(n_x, n_y)
                
                # sphere angle wrap dx back into -180...180 using modulo trick
                # dx > 360: dx = dx - 360
                dx1 = (dx1 + cycl/2) % cycl - cycl/2
                dx2 = (dx2 + cycl/2) % cycl - cycl/2
                
                # mean latitude per element (3 nodes)
                e_y = (self.n_y[self.e_i[:,0]] +
                       self.n_y[self.e_i[:,1]] +
                       self.n_y[self.e_i[:,2]]) / 3.0
                e_y = np.cos(e_y * rad)
                
                # scale dx by cos(lat)
                dx1 *= e_y
                dx2 *= e_y
                del(e_y)

                # area of spherical triangles (planar approx)
                self.e_area = 0.5 * np.abs(dx1*dy2 - dx2*dy1) * (R_earth*R_earth)
            
            self.e_area = np.ascontiguousarray(self.e_area)
        #_______________________________________________________________________
        return(self)
    
    
    # ___COMPUTE RESOLUTION OF ELEMENTS________________________________________
    #| compute area of elements in [m], options:                               |
    #| which :   str,                                                          |       
    #|           "mean": resolution based on mean element edge legth           |
    #|           "max" : resolution based on maximum element edge length       |
    #|           "min" : resolution based on minimum element edge length       |
    #|_________________________________________________________________________|
    def compute_e_resol(self, which='height'):
        """
        --> part of fesom mesh class, compute area of elements in [m], options:

            Parameter:

                which: str,
                        - "mean"  ... resolution based on mean element edge legth
                        - "max"   ... resolution based on maximum element edge length
                        - "min"   ... resolution based on minimum element edge length
                        - "height"... resolution based on height of element 

        """
        if len(self.e_resol) == 0 :
            self.do_eresol[0]=True
            self.do_eresol[1]=which
            
            #___________________________________________________________________
            # compute mean length of triangle sides
            e_x = self.n_x[self.e_i]   # longitudes
            e_y = self.n_y[self.e_i]   # latitudes
            
            # Mean latitude per edge (needed for dx scaling)
            cos_lat = np.cos(e_y * rad)
            
            #___________________________________________________________________
            # Longitude/Latitude differences 
            dx = (e_x[:, [1,2,0]] - e_x[:, [0,1,2]])
            dy = (e_y[:, [1,2,0]] - e_y[:, [0,1,2]])
            del(e_x, e_y)
            
            # cyclic wrap using modulo
            dx = (dx + 180.0) % 360.0 - 180.0  # wrapped into [-180,180]

            # compute cartesian dx, dy in [m]
            dx *= cos_lat.mean(axis=1)[:, None] * rad * R_earth
            dy *= rad * R_earth
            
            #___________________________________________________________________
            #  compute triangle edge edge lengths
            #  sqrt( dx^2 + dy^2 )
            edge_len = np.sqrt(dx*dx + dy*dy)
            
            #___________________________________________________________________
            # mean resolutiuon per element
            if  which=='mean': 
                print(' > comp. e_resol from edge mean')
                self.e_resol = edge_len.mean(axis=1)
                
            elif which=='max': 
                print(' > comp. e_resol from edge max')
                self.e_resol = edge_len.max(axis=1)
                
            elif which=='min': 
                print(' > comp. e_resol from edge min')
                self.e_resol = edge_len.min(axis=1)   
                
            elif which == 'height':
                print(" > comp. e_resol from triangle height")
                # semi-perimeter
                s = edge_len.sum(axis=1) * 0.5

                # Heron's formula area
                area = np.sqrt(s * (s - edge_len[:,0]) * (s - edge_len[:,1]) * (s - edge_len[:,2]))

                # height = 2*area / longest edge
                self.e_resol = 2.0 * area / edge_len.max(axis=1)
                
            elif which == 'area':
                print(" > comp. e_resol from sqrt(2*area)")
                # semi-perimeter
                s = edge_len.sum(axis=1) * 0.5

                # Heron’s area
                area = np.sqrt(s * (s - edge_len[:,0]) * (s - edge_len[:,1]) * (s - edge_len[:,2]))

                # sqrt(2 * area)
                self.e_resol = np.sqrt(2.0 * area)
                
            #___________________________________________________________________    
            else:
                raise ValueError("The option which={} in compute_e_resol is not supported.".format(str(which)))
            
            self.e_resol = np.ascontiguousarray(self.e_resol)
        #_______________________________________________________________________
        return(self)
    

    # ___COMPUTE/LOAD CLUSTERAREA OF VERTICES__________________________________
    #| either load clusterarea of vertices from fesom.mesh.diag.nc if its found|  
    #| in meshpath or recompute it from scratch by using e_area, [m^2]         |
    #|_________________________________________________________________________|
    def compute_n_area(self):
        """
        --> part of fesom mesh class, either load clusterarea of vertices from 
        fesom.mesh.diag.nc if its found in meshpath or recompute it from scratch 
        by using e_area, [m^2]     
        
        """
        # just compute e_area if mesh.area is empty 
        if len(self.n_area)==0:
            self.do_narea=True
            
            #___________________________________________________________________
            # load FESOM2 mesh
            if not self.do_f14cmip6:
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
                    z = np.arange(self.nlev, dtype=np.int32)[:, None]  
                    # mask contains bottom topography information on elements
                    mask = (z <= self.e_iz[None, :]).astype(np.float32)
                    self.n_area = njit_ie2n_accum_2d(self.nlev, self.n2dn, self.n2de, 
                                                     self.e_i, self.e_area, mask)
                    del(mask, z)
                self.n_area = np.ascontiguousarray(self.n_area)
                    
            
            #___________________________________________________________________
            # load FESOM1.4 mesh
            else:
                if os.path.isfile(os.path.join(self.path,'griddes.nc')):
                    print(' > load n_area from griddes.nc')
                    #_______________________________________________________________
                    fid = Dataset(os.path.join(self.path,'griddes.nc'),'r')
                    self.n_area = fid.variables['cell_area'][:]
                else: 
                    print(' > comp n_area')
                    # be sure that elemente area already exists
                    self.compute_e_area()
                    
                    #_______________________________________________________________
                    e_area_x3 = np.vstack((self.e_area, self.e_area, self.e_area)).transpose().flatten()
                    
                    #_______________________________________________________________
                    # single loop over self.e_i.flat is ~4 times faster than douple loop 
                    # over for i in range(3): ,for j in range(self.n2de):
                    self.n_area = np.zeros((self.n2dn))
                    count_e = 0
                    for idx in self.e_i.flat:
                        self.n_area[idx] = self.n_area[idx] + e_area_x3[count_e]
                        count_e = count_e+1 # count triangle index for aux_area[count] --> aux_area =[n2de*3,]
                        self.n_area = self.n_area/3.0
                    del e_area_x3, count_e
                self.n_area = np.ascontiguousarray(self.n_area)    
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
        """
        --> part of fesom mesh class, compute resolution at vertices in m, options:

            Parameter:

                which: str,
                        - "n_area" ... compute resolution based on vertice cluster area
                        - "e_resol"  ... compute vertice resolution by interpolating elem resolution to vertices, default      

        """
        # just compute e_area if mesh.area is empty 
        if len(self.n_resol)==0:
            self.do_nresol[0]=True
            self.do_nresol[1]=which
            
            #___________________________________________________________________
            # compute vertices resolution based on vertice clusterarea
            if any(x in which for x in ['n_area','narea']): 
                
                #_______________________________________________________________
                self.compute_n_area()
                print(' > comp n_resol from 2*sqrt(n_area/pi)')
                #_______________________________________________________________
                # You assign a single horizontal resolution length 
                # L such that:
                #   A=pi * (L/2)^2
                #   
                # Solving:
                #   L=2*np.sqrt(A/pi)
                #   
                # L is the diameter of a circle with the same area as the vertex 
                # polygon. This gives the correct effective grid spacing for scalar 
                # and vector operators that assume an isotropic control volume.
                self.n_resol = np.sqrt(self.n_area[0,:]/np.pi)*2.0
            
            #___________________________________________________________________
            # compute vertices resolution based on interpolation from resolution
            # of elements    
            elif any(x in which for x in ['e_resol','eresol']):
                print(' > comp n_resol from e2n interpolation of e_resol')
                #_______________________________________________________________
                self.compute_e_area()
                self.compute_e_resol()
                self.n_resol, _ = njit_ie2n_1d(self.n2dn, self.n2de, self.e_i, self.e_area, self.e_resol)
                
            #___________________________________________________________________    
            else:
                raise ValueError("The option which={} in compute_n_resol is not supported. either 'n_area' or 'e_resol'".format(str(which)))
            
            self.n_resol = np.ascontiguousarray(self.n_resol)
        #_______________________________________________________________________
        return(self)
    
    
    
    # ___COMPUTE LAND-SEA MASK CONOURLINE______________________________________
    #| compute land-sea mask contourline with periodic boundary based on edges |
    #| that contribute only to one triangle and the checking which edges are   |
    #| consequtive connected                                                   |                   
    #|_________________________________________________________________________|
    def compute_lsmask(self):
        """
        --> part of fesom mesh class, compute land-sea mask contourline with 
        periodic boundary based on boundary edges that contribute only to one triangle 
        and then checking which edges can be consequtive connected                                                   |
        """
        print(" > compute lsmask")
        self.do_lsmask = True

        #_______________________________________________________________________
        # build land boundary edge matrix
        bnde = njit_compute_boundary_edges(self.e_i)   # (nbnde, 2)
        nbnde = bnde.shape[0]

        #_______________________________________________________________________
        # build adjacency list, which vertice points are adjacent to each other 
        from collections import defaultdict
        adj = defaultdict(list)
        for ii, jj in bnde:
            adj[int(ii)].append(int(jj))
            adj[int(jj)].append(int(ii))

        #_______________________________________________________________________
        # 3. Find connected loops (coastlines)
        visited = set()
        polygons = []

        # Loop over 
        for start in adj.keys():
            if start in visited: continue
            
            loop = []
            current = start
            prev = None
            
            while True:
                loop.append(current)
                visited.add(current)
                
                neighbors = adj[current]
                
                # we skip extremely tiny polygons later anyway
                if len(neighbors) != 2: break
                
                # pick neighbor not equal to previous
                nxt = neighbors[0] if neighbors[0] != prev else neighbors[1]
                
                prev, current = current, nxt
                
                # jump out of loop when starting point is reached again 
                if current == start: break
                
            # convert to XY polygon
            if len(loop) > 4:   # avoid tiny invalid polygons
                xy = np.column_stack((self.n_x[loop], self.n_y[loop]))
                polygons.append(xy)
        
        self.lsmask = polygons
        return self
    
    
    
    # ___AUGMENT PERIODIC BOUNDARIES IN LAND-SEA MASK CONTOURLINE______________
    #| spilit contourlines that span over the periodic boundary into two       |
    #| separated countourlines for the left and right side of the periodic     |
    #| boundaries                                                              |
    #|_________________________________________________________________________|
    def augment_lsmask(self):
        """
        --> part of fesom mesh class, split contourlines that span over the 
        periodic boundary into two separated countourlines for the left and 
        right side of the periodic boundaries
        """
        print(' > augment lsmask')
        #self.lsmask_a = self.lsmask.copy()
        self.lsmask_a = []
        #_______________________________________________________________________
        # build land boundary edge matrix
        nlsmask = len(self.lsmask)
        
        # min/max of longitude box 
        # xmin,xmax = -180+self.focus, 180+self.focus
        xmin, xmax = np.floor(self.n_x.min()), np.ceil(self.n_x.max())
        
        for ii in range(0,nlsmask):
            #___________________________________________________________________
            polygon_xy = self.lsmask[ii].copy()
            
            #___________________________________________________________________
            #import matplotlib.pyplot as plt 
            #plt.figure()
            #plt.plot(polygon_xy[:,0], polygon_xy[:,1])
            #plt.show()
            
            #___________________________________________________________________
            # idx compute how many periodic boudnaries are in the polygon 
            idx        = np.argwhere(np.abs(self.lsmask[ii][1:,0]-self.lsmask[ii][:-1,0])>self.cyclic/2).ravel()
            
            #___________________________________________________________________
            if len(idx)!=0:
                # unify starting point of polygon, the total polygon should start
                # at the left periodic boudnary at the most northward periodic point
                aux_i      = np.hstack((idx,idx+1))
                aux_x      = polygon_xy[aux_i,0]
                #aux_il     = np.sort(aux_i[np.argwhere(aux_x < self.focus).ravel()])
                aux_il     = np.sort(aux_i[np.argwhere(aux_x < (xmin+xmax)*0.5).ravel()])
                isort_y    = np.flip(np.argsort(polygon_xy[aux_il,1]))
                aux_il     = aux_il[isort_y]
                del isort_y, aux_x, aux_i
                
                # shift polygon indices so that new starting point is at index 0
                polygon_xy = np.vstack(( (polygon_xy[aux_il[0]:,:]),(polygon_xy[:aux_il[0],:]) ))
                del aux_il
                
                # ensure that total polygon is closed
                if np.any(np.diff(polygon_xy[[0,-1],:])!=0): polygon_xy = np.vstack(( (polygon_xy,polygon_xy[0,:]) ))
            
            else: 
                if polygon_xy.shape[0]<=3: 
                    ...
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
                #aux_il     = np.sort(aux_i[np.argwhere(aux_x < self.focus).ravel()])
                #aux_ir     = np.sort(aux_i[np.argwhere(aux_x > self.focus).ravel()])
                aux_il     = np.sort(aux_i[np.argwhere(aux_x < (xmin+xmax)*0.5).ravel()])
                aux_ir     = np.sort(aux_i[np.argwhere(aux_x > (xmin+xmax)*0.5).ravel()])
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
                if polygon_xyl.shape[0]<=3: 
                    ...
                else:    
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
                if polygon_xyr.shape[0]<=3: 
                    ...
                else:    
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
                #aux_il     = np.sort(aux_i[np.argwhere(aux_x < self.focus).ravel()])[0]
                #aux_ir     = np.sort(aux_i[np.argwhere(aux_x > self.focus).ravel()])[0]
                aux_il     = np.sort(aux_i[np.argwhere(aux_x < (xmin+xmax)*0.5).ravel()])[0]
                aux_ir     = np.sort(aux_i[np.argwhere(aux_x > (xmin+xmax)*0.5).ravel()])[0]
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
                if polygon_xy.shape[0]<=3: 
                    ...
                else:
                    self.lsmask_a.append(polygon_xy)
                    
                    
                del polygon_xy, pbndr, pcrnr, pcrnl ,pbndl, 
                del aux_il, aux_ir, aux_i, aux_x, aux_y
                
        #_______________________________________________________________________
        # create lsmask patch to plot 
        self.lsmask_p = lsmask_patch(self.lsmask_a)
        
        #_______________________________________________________________________
        return(self)
    
    

"""
    def augment_lsmask_unfinished(self):
       
        print(" > augment lsmask")
        self.lsmask_a = []

        cyclic = float(self.cyclic)
        xmin   = np.floor(self.n_x.min())
        xmax   = xmin + cyclic
        xmid   = 0.5 * (xmin + xmax)

        # Identify polar candidates
        southmost_idx = np.argmin([poly[:, 1].min() for poly in self.lsmask])
        northmost_idx = np.argmax([poly[:, 1].max() for poly in self.lsmask])

        # ------------------------------------------------------------------
        # helper functions
        # ------------------------------------------------------------------
        def close_poly(P):
            #Ensure polygon is closed.
            if not (P[0, 0] == P[-1, 0] and P[0, 1] == P[-1, 1]):
                return np.vstack((P, P[0]))
            return P

        def split_at_seams_unwrapped(Pw):
            # split unwrapped polygon Pw into seam-free segments.
            x = Pw[:, 0]
            wrap_idx = np.floor((x - xmin) / cyclic).astype(int)
            cuts = np.where(np.diff(wrap_idx) != 0)[0] + 1
            parts = np.split(Pw, cuts)
            return parts

        def rewrap(seg):
            # Rewrap segment to [xmin, xmax] and close.
            Q = seg.copy()
            Q[:, 0] = (Q[:, 0] - xmin) % cyclic + xmin
            return close_poly(Q)

        def add_polar_cap(P, seam_idx, pole_lat):
            # Build polar trapezoid polygon (Option A) for a ring with
            # exactly one seam crossing.
            # P: closed polygon (N x 2)
            # seam_idx: index of seam edge (between i and i+1)
            # pole_lat: -90 (Antarctic) or +90 (Arctic)
            Pw = P[:-1].copy()  # drop duplicate last point
            N  = Pw.shape[0]
            s  = seam_idx
            i0, i1 = s, (s + 1) % N
            p0, p1 = Pw[i0], Pw[i1]

            # figure left/right ends w.r.t. xmid
            if p0[0] < xmid:
                left_idx, right_idx = i0, i1
            else:
                left_idx, right_idx = i1, i0

            # coast path from right_cut → ... → left_cut
            coast_idx = np.concatenate((np.arange(right_idx, N),
                                        np.arange(0, left_idx + 1)))
            coast = Pw[coast_idx]

            # ensure coast runs left→right in longitude (after wrapping)
            coast_lons = (coast[:, 0] - xmin) % cyclic + xmin
            if coast_lons[0] > coast_lons[-1]:
                coast      = coast[::-1]
                coast_lons = coast_lons[::-1]

            # snap endpoints to boundaries
            coast_aug = coast.copy()
            coast_aug[0, 0]  = xmin
            coast_aug[-1, 0] = xmax

            # build trapezoid: coast + [xmax,pole_lat] + [xmin,pole_lat]
            poly_aug = np.vstack([
                coast_aug,
                [xmax, pole_lat],
                [xmin, pole_lat],
            ])
            poly_aug = close_poly(poly_aug)
            return poly_aug

        def split_single_seam_nonpolar(P, seam_idx):
            # Handle non-polar polygon with exactly one seam crossing.
            # Split into two polygons: one on the 'left' side and one on the 'right'.
            # P: closed polygon (N x 2)
            # seam_idx: index s where edge P[s] -> P[s+1] crosses the seam
            Pw = P[:-1].copy()  # length N
            N  = Pw.shape[0]
            s  = seam_idx

            i0, i1 = s, (s + 1) % N
            p0, p1 = Pw[i0], Pw[i1]

            # Decide which endpoint is "left" vs "right" using xmid
            if p0[0] < xmid:
                left_idx, right_idx = i0, i1
            else:
                left_idx, right_idx = i1, i0

            # Build two open polylines:
            #   L: from left_idx -> ... -> right_idx (around one way)
            #   R: from right_idx -> ... -> left_idx (other way)
            if left_idx < right_idx:
                L = Pw[left_idx:right_idx + 1]
                R = np.vstack((Pw[right_idx:left_idx - 1:-1],))  # reversed other side
            else:
                L = np.vstack((Pw[left_idx:], Pw[:right_idx + 1]))
                R = np.vstack((Pw[right_idx:left_idx + 1],))

            # Wrap both so that L is on the left side, R on the right side
            # --- Left polygon: longitudes near xmin
            Lw = L.copy()
            Lw[:, 0] = (Lw[:, 0] - xmin) % cyclic + xmin
            # snap endpoints to xmin
            Lw[0, 0]  = xmin
            Lw[-1, 0] = xmin
            Lw = close_poly(Lw)

            # --- Right polygon: longitudes near xmax
            Rw = R.copy()
            Rw[:, 0] = (Rw[:, 0] - xmin) % cyclic + xmin
            # shift to be near xmax
            # if mean lon is closer to xmin, add +cyclic
            if np.abs(Rw[:, 0].mean() - xmin) < np.abs(Rw[:, 0].mean() - xmax):
                Rw[:, 0] += cyclic
            Rw[:, 0] = (Rw[:, 0] - xmin) % cyclic + xmin
            # snap endpoints to xmax
            Rw[0, 0]  = xmax
            Rw[-1, 0] = xmax
            Rw = close_poly(Rw)

            out = []
            if Lw.shape[0] > 3:
                out.append(Lw)
            if Rw.shape[0] > 3:
                out.append(Rw)
            return out

        # ------------------------------------------------------------------
        # main loop
        # ------------------------------------------------------------------
        for ii, poly in enumerate(self.lsmask):

            P = close_poly(poly.copy())

            # seam detection
            dlon      = np.diff(P[:, 0])
            seam_idx  = np.where(np.abs(dlon) > cyclic / 2.0)[0]
            seam_count = seam_idx.size

            # CASE 0: no seam crossings
            if seam_count == 0:
                if P.shape[0] > 3:
                    self.lsmask_a.append(P)
                continue

            is_antarctica = (ii == southmost_idx)
            is_arctic     = (ii == northmost_idx)

            # CASE 1: polar rings with exactly 1 seam crossing
            if seam_count == 1 and (is_antarctica or is_arctic):
                pole_lat = -90.0 if is_antarctica else 90.0
                poly_aug = add_polar_cap(P, seam_idx[0], pole_lat)
                if poly_aug.shape[0] > 3:
                    self.lsmask_a.append(poly_aug)
                continue

            # CASE 2: non-polar polygon with exactly 1 seam crossing
            if seam_count == 1:
                pieces = split_single_seam_nonpolar(P, seam_idx[0])
                self.lsmask_a.extend(pieces)
                continue

            # CASE 3: non-polar polygons with >= 2 seams
            # general unwrap/split/rewrap
            x = P[:, 0].copy()
            x_unwrap = x.copy()
            for k in seam_idx:
                if x_unwrap[k + 1] < x_unwrap[k]:
                    x_unwrap[k + 1:] += cyclic
                else:
                    x_unwrap[k + 1:] -= cyclic

            Pw = P.copy()
            Pw[:, 0] = x_unwrap

            for seg in split_at_seams_unwrapped(Pw):
                if seg.shape[0] <= 3:
                    continue
                Q = rewrap(seg)
                if Q.shape[0] > 3:
                    self.lsmask_a.append(Q)

        # ------------------------------------------------------------------
        # build patch for plotting
        self.lsmask_p = lsmask_patch(self.lsmask_a)
        return self
"""


#
#
#____COMPUTE POLYGON PATCH FROM LAND-SEA MASK CONTOUR___________________________
def lsmask_patch(lsmask):
    """    
    --> computes polygon collection that can be plotted as closed polygon patches
    with ax.add_collection(PatchCollection(mesh.lsmask_p,
    facecolor=[0.7,0.7,0.7], edgecolor='k',linewidth=0.5))

    Parameters:
    
        lsmask: list()
            list([array1[npts,2], array2[npts,2]], ...) \n
            array1=np.array([ [x1,y1]; [x2,y2]; ...  ])
    

    
    Returns:
    
        :lsmask_p: shapely Multipolygon object
        
    Info:
    
        - how to plot in matplotlib:
          from descartes import PolygonPatch
          ax.add_patch(PolygonPatch(mesh.lsmask_p,facecolor=[0.7,0.7,0.7],
        
        - how to plot in cartopy:
          import cartopy.crs as ccrs
          ax.add_geometries(mesh.lsmask_p, crs=ccrs.PlateCarree(), facecolor=[0.6,0.6,0.6], edgecolor='k',
          linewidth=0.5)

    """
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
    from shapely.validation import make_valid
    
    #import matplotlib.pyplot as plt 
    #hfig = plt.figure()
    #ax = plt.gca()
    #___________________________________________________________________________
    polygonlist=[]
    #for xycoord in lsmask: polygonlist.append(Polygon(xycoord))
    for ii, xycoord in enumerate(lsmask):
        #print(ii, xycoord.shape)
        poly = Polygon(xycoord)
                    
        # Ensure polygons are counterclockwise
        if not poly.exterior.is_ccw:
            poly = Polygon(list(poly.exterior.coords)[::-1])

        # Ensure polygon is valid
        if not poly.is_valid:
            poly = make_valid(poly)  # Attempt to fix the geometry
        
        #npoly = len(polygonlist)
        # Check if make_valid() returned a MultiPolygon
        if   isinstance(poly, MultiPolygon):
            polygonlist.extend(poly.geoms)  # Unpack MultiPolygon into list
            
            #for auxpoly in polygonlist[npoly:]:
                #coords = np.array(auxpoly.exterior.coords)
                #ax.plot(coords[:,0], coords[:,1], 'c*')
            
        # Check if make_valid() returned a GeometryCollection
        elif isinstance(poly, GeometryCollection):             
             # Extract only Polygon or MultiPolygon from the GeometryCollection
            for geom in poly.geoms:
                if isinstance(geom, Polygon):
                    polygonlist.append(geom)
                elif isinstance(geom, MultiPolygon):
                    polygonlist.extend(geom.geoms)
            
            #for auxpoly in polygonlist[npoly:]:
                #coords = np.array(auxpoly.exterior.coords)
                #ax.plot(coords[:,0], coords[:,1], 'r*')
        
        elif isinstance(poly, Polygon):
        #elif poly.is_valid:
            polygonlist.append(poly)
            
            #for auxpoly in polygonlist[npoly:]:
                #coords = np.array(auxpoly.exterior.coords)
                #ax.plot(coords[:,0], coords[:,1], 'k*')
        
    
    lsmask_p = MultiPolygon(polygonlist)
    #plt.show()
    #___________________________________________________________________________
    return(lsmask_p)



#
#
#___SAVE POLYGON LAND-SEA MASK CONTOUR TO SHAPEFILE_____________________________
def lsmask_2shapefile(mesh, lsmask=[], path=[], fname=[], do_info=True):
    """
    --> save FESOM2 grid land-sea mask polygons to shapefile

    Parameters:
    
        :mesh:      fesom2 mesh object, contains periodic augmented land-sea mask
                    polygonss in mesh.lsmask_a
                    
        :lsmask:    list, if empty mesh.lsmask_a is stored in shapefile, if not
                    empty lsmask=lsmaskin than lsmaskin will be stored in
                    
        :path:      strm, if empty mesh.path (or cache path depending on writing
                    permission) is used as location to store the shapefile, if
                    path=pathin than this string serves as location to store
                    .shp file
                    
        :fname:     str, if empty fixed filename is used mypymesh_fesom2_ID_focus=X.shp, 
                    if not empty than fname=fnamein is used as filename for shape file
                    
        :do_info:   bool, print info where .shp file is saved, default = True
    
    Returns:
    
        :return: nothing
        
    Info:
    
    --> to load and plot shapefile patches
    
    ::
    
        import shapefile as shp
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        
        shpfname = 'tripyview_fesom2'+'_'+mesh.id+'_'+
                '{}={}'.format('focus',mesh.focus)+'.shp'
        shppath  = os.path.join(mesh.cachepath,shpfname)
        
        sf = shp.Reader(shppath)
        patches = []
        for shape in sf.shapes(): patches.append(Polygon(shape.points))
        
        plt.figure()
        ax = plt.gca()
        ax.add_collection(PatchCollection(patches,
                        facecolor=[0.7,0.7,0.7],
                        edgecolor='k', linewidths=1.))
        ax.set_xlim([-180,180]) 
        ax.set_ylim([-90,90])
        plt.show()
        
    """
    import geopandas as gpd
    from shapely.geometry import Polygon
    
    #___________________________________________________________________________
    # Create an empty geopandas GeoDataFrame
    newdata = gpd.GeoDataFrame(geometry=gpd.GeoSeries())

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
    # Ensure the GeoDataFrame has a CRS
    newdata.set_crs("EPSG:4326", inplace=True)  # Set to the correct CRS

    #___________________________________________________________________________
    # write lsmask to shapefile 
    if not fname:
        shpfname = 'tripyview_lsmask_{:s}_focus{:d}.shp'.format(mesh.id, mesh.focus)
    else: 
        shpfname = fname
    shppath = os.path.join(shppath,shpfname)
    if do_info: print(' > save *.shp to {}'.format(shppath))
    newdata.to_file(str(shppath))
    
    #___________________________________________________________________________
    return



#
#
# ___COMPUTE EULER ROTATION MATRIX_____________________________________________
def grid_rotmat(abg):
    """
    --> compute euler rotation matrix based on alpha, beta and gamma angle

    Parameters: 
    
        :abg:   list, with euler angles [alpha, beta, gamma]

    Returns: 

        :rmat:  array, [3 x 3] rotation matrix to transform from geo to rot

    """
    al, be, ga = float(abg[0]), float(abg[1]), float(abg[2])
    #___________________________________________________________________________
    return njit_grid_rotmat(al, be, ga)

@njit(cache=True, fastmath=True)
def njit_grid_rotmat(alpha_deg, beta_deg, gamma_deg):
    #___________________________________________________________________________
    #rad = np.pi/180
    al  = alpha_deg * rad 
    be  = beta_deg  * rad
    ga  = gamma_deg * rad
        
    #___________________________________________________________________________
    rmat      = np.zeros((3,3), dtype=np.float64)
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



#
#
# ___COMPUTE3D CARTESIAN COORDINAT_____________________________________________
def grid_cart3d(lon, lat, R=1.0, is_deg=False):
    """
    --> compute 3d cartesian coordinates from spherical geo coordinates (lon, lat, R=1.0)                                                           |
        
    Parameters: 
    
        :lon:       array, longitude coordinates in radians
        :lat:       array, latitude coordinates in radians
        :R:         float, (default=1.0), Radius of sphere
        :is_deg:    bool, (default=False) is lon,lat in degree (True) otherwise 
                    otherwise (False) assumed its in radians
    
    Returns:

        :x:         array, x y z cartesian coordinates 
        :y:         ...
        :z:         ...

    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    if is_deg:
        lon = lon * rad
        lat = lat * rad
    #___________________________________________________________________________
    return njit_grid_cart3d(lon, lat, R)

@njit(cache=True, fastmath=True)
def njit_grid_cart3d(lon_rad, lat_rad, R=1.0):
    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    z = R * np.sin(lat_rad)
    #___________________________________________________________________________
    return(x, y, z)



#
#
# ___ROTATE GRID FROM: ROT-->GEO_______________________________________________
def grid_r2g(abg, rlon, rlat):
    """
    --> compute grid rotation from sperical rotated frame back towards normal geo
    frame using the euler angles alpha, beta, gamma
    
    Parameters:
    
        :abg:    list, with euler angles [alpha, beta, gamma]   
        :rlon:   array, longitude coordinates of sperical rotated frame in degree
        :rlat:   array,  latitude coordinates of sperical rotated frame in degree

    Returns: 

        :lon:    array, longitude coordinates in normal geo frame in degree
        :lat:    array, latitude coordinates in normal geo frame in degree

    """
    rlon = np.asarray(rlon, dtype=np.float64)
    rlat = np.asarray(rlat, dtype=np.float64)
    al, be, ga = float(abg[0]), float(abg[1]), float(abg[2])
    #___________________________________________________________________________
    return njit_grid_r2g(al, be, ga, rlon, rlat)

@njit(cache=True, fastmath=True)
def njit_grid_r2g(alpha_deg, beta_deg, gamma_deg, rlon_deg, rlat_deg):

    #___________________________________________________________________________
    # build inverse rotation matrix
    rmat0= njit_grid_rotmat(alpha_deg, beta_deg, gamma_deg)
    rmat = rmat0.T # --> inverse

    #___________________________________________________________________________
    # compute 3d cartesian coordinates
    rlat_rad   = rlat_deg * rad
    rlon_rad   = rlon_deg * rad
    xr, yr, zr = njit_grid_cart3d(rlon_rad, rlat_rad, 1.0)
    
    #___________________________________________________________________________
    # rotate to geographical cartesian coordinates:
    xg=rmat[0,0]*xr + rmat[0,1]*yr + rmat[0,2]*zr;
    yg=rmat[1,0]*xr + rmat[1,1]*yr + rmat[1,2]*zr;
    zg=rmat[2,0]*xr + rmat[2,1]*yr + rmat[2,2]*zr;
    
    #___________________________________________________________________________
    # compute to geographical coordinates:
    lon_rad = np.arctan2(yg, xg)
    lat_rad = np.arcsin(zg)   
    lon_deg = lon_rad/rad 
    lat_deg = lat_rad/rad
    
    #___________________________________________________________________________
    return(lon_deg, lat_deg)



#
#
# ___ROTATE GRID FROM: GEO-->ROT________________________________________________
def grid_g2r(abg, lon, lat):
    """
    --> compute grid rotation from normal geo frame towards sperical rotated
    frame using the euler angles alpha, beta, gamma
    
    Parameters:
    
        :abg:   list, with euler angles [alpha, beta, gamma]
        
        :lon:   array, longitude coordinates of normal geo frame in degree 
        
        :lat:   array, latitude coordinates of normal geo frame in degree
        
    Returns:    
    
        :rlon:   array, longitude coordinates in sperical rotated frame in degree
        
        :rlat:   array, latitude coordinates in sperical rotated frame in degree
    
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    al, be, ga = float(abg[0]), float(abg[1]), float(abg[2])
    #___________________________________________________________________________
    return njit_grid_g2r(al, be, ga, lon, lat)

@njit(cache=True, fastmath=True)
def njit_grid_g2r(alpha_deg, beta_deg, gamma_deg, lon_deg, lat_deg):
    #___________________________________________________________________________
    # build rotation matrix
    rmat = njit_grid_rotmat(alpha_deg, beta_deg, gamma_deg)
    
    #___________________________________________________________________________
    # compute 3d cartesian coordinates
    lat_rad    = lat_deg * rad
    lon_rad    = lon_deg * rad
    xg, yg, zg = njit_grid_cart3d(lon_rad, lat_rad, 1.0)

    #___________________________________________________________________________
    # rotate to geographical cartesian coordinates:
    xr=rmat[0,0]*xg + rmat[0,1]*yg + rmat[0,2]*zg;
    yr=rmat[1,0]*xg + rmat[1,1]*yg + rmat[1,2]*zg;
    zr=rmat[2,0]*xg + rmat[2,1]*yg + rmat[2,2]*zg;

    #___________________________________________________________________________
    # compute to geographical coordinates:
    rlon_rad = np.arctan2(yr,xr)     
    rlat_rad = np.arcsin(zr)        
    rlon_deg = rlon_rad/rad
    rlat_deg = rlat_rad/rad
    
    #___________________________________________________________________________
    return(rlon_deg,rlat_deg)



#
#
# ___ROTATE GRID FOCUS: 0-->FOCUS______________________________________________
def grid_focus(focus, rlon, rlat):
    """
    --> compute grid rotation around z-axis to change the focus center of the lon,
    lat grid, by default focus=0-->lon=[-180...180], if focus=180-->lon[0..360] 
    
    Parameters: 
    
        :focus:  float, longitude of grid center

        :rlon:   array, longitude in focus=0-->lon=[-180...180] in degree

        :rlat:   array, latitude in focus=0-->lon=[-180...180] in degree

    Returns:    

        :lon:   array, longitude in lon=[-180+focus...180+focus] frame in degree
  
        :lat:   array, latitude in lon=[-180+focus...180+focus] frame in degree

    """
    rlon = np.asarray(rlon, dtype=np.float64)
    rlat = np.asarray(rlat, dtype=np.float64)
    al = -float(focus)
    be = 0.0
    ga = 0.0
    
    #___________________________________________________________________________
    lon_deg, lat_deg = njit_grid_r2g(al, be, ga, rlon, rlat)
    lon_deg = lon_deg + focus
    
    #___________________________________________________________________________
    return lon_deg, lat_deg
    
    

#
#
# ___ROTATE VECTOR FROM: ROT-->GEO_____________________________________________
def vec_r2g(abg, lon, lat, urot, vrot, gridis='geo', do_info=False ):
    """
    --> In FESOM2 the vector variables are usually given in the rotated coordinate 
    frame in which the model works and need to be rotated into normal geo    
    coordinates, however in the latest FESOM2 version there is also the option that
    they are rotated in the model via a flaf. So make sure what applies to you                                                           
    
    Parameters:
        
        :abg:       list, with euler angles [alpha, beta, gamma]
        
        :lon:       array, longitude
        
        :lat:       array, latitude
        
        :urot:      array, zonal velocities in rotated frame
        
        :vrot:      array, meridional velocities in rotated frame
        
        :gridis:    str, in which coordinate frame are given lon, lat
                    'geo','g','geographical': lon,lat is given in geo coordinates
                    'rot','r','rotated'     : lon,lat is given in rot coordinates
                    
    Returns:
    
        :ugeo:      array, zonal velocities in normal geo frame
        
        :vgeo:      array, meridional velocities in normal geo frame
    
    """
    #___________________________________________________________________________
    # create grid coorinates for geo and rotated frame
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    urot = np.asarray(urot, dtype=np.float64)
    vrot = np.asarray(vrot, dtype=np.float64)
    
    if any(x in gridis for x in ['geo','g','geographical']): 
        rlon, rlat = grid_g2r(abg, lon, lat)        
    elif any(x in gridis for x in ['rot','r','rotated']):     
        rlon, rlat = lon, lat 
        lon,  lat  = grid_r2g(abg, rlon, rlat)
    else:
        raise ValueError("The option gridis={} in vec_r2g is not supported.\n (only: 'geo','g','geographical', 'rot','r','rotated') ".format(str(gridis)))
    
    #___________________________________________________________________________
    # compute rotation matrix
    al, be, ga = float(abg[0]), float(abg[1]), float(abg[2])
    rmat0 = njit_grid_rotmat(al, be, ga)
    rmat  = rmat0.T
    
    #___________________________________________________________________________
    # degree --> radian  
    lon_rad  = lon  * rad
    lat_rad  = lat  * rad
    rlon_rad = rlon * rad
    rlat_rad = rlat * rad
    
    #___________________________________________________________________________
    # rotation of one dimensional vector data
    if   vrot.ndim==1 and urot.ndim==1: 
        if do_info: print('     > 1D rotation')
    elif vrot.ndim==2 and urot.ndim==2: 
        if do_info: print('     > 2D rotation')
    elif vrot.ndim==3 and urot.ndim==3: 
        if do_info: print('     > 3D rotation')
    else: raise ValueError('This number of dimensions is in not supported for vector rotation \n' +
                           ' or the number of dimensions between u and v does not match' )    
    ugeo, vgeo = njit_vec_r2g_123d(rmat, lon_rad, lat_rad, rlon_rad, rlat_rad, urot, vrot)
    
    #___________________________________________________________________________
    return(ugeo, vgeo)


def dask_vec_r2g(abg, lon, lat, urot, vrot, gridis='geo', do_info=False):
    """
    Rotate vector data from rotated coordinates into geographical coordinates.
    This version uses Dask for parallelization.
    """
    # --- FAST PATH: if vector data is 1D, do NOT use Dask ---
    if urot.ndim == 1:
        #if do_info:
        print("vec_r2g_dask: using fast direct numba kernel for 1D vector rotation")

        # Convert all to numpy arrays
        lon  = np.asarray(lon)
        lat  = np.asarray(lat)
        urot = np.asarray(urot)
        vrot = np.asarray(vrot)
        
        # Compute rotated coordinate frame
        if any(x in gridis for x in ['geo', 'g', 'geographical']):
            rlon, rlat = grid_g2r(abg, lon, lat)
        else:
            rlon, rlat = lon, lat
            lon, lat = grid_r2g(abg, rlon, rlat)

        rad = np.pi / 180
        lon_rad  = lon  * rad
        lat_rad  = lat  * rad
        rlon_rad = rlon * rad
        rlat_rad = rlat * rad

        rmat = grid_rotmat(abg).T  # inv rotation
        
        #t1 = clock.time()
        
        # DIRECT numba call → super fast
        ugeo, vgeo = njit_vec_r2g_123d(rmat, lon_rad, lat_rad, rlon_rad, rlat_rad,
                                       urot, vrot)
            
        #print(' elapsed time 1d fast', clock.time()-t1)
        return ugeo, vgeo
    
    #___________________________________________________________________________
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError("vec_r2g_dask: lon and lat must be 1D arrays (node coords).")
    if lon.shape != lat.shape:
        raise ValueError("vec_r2g_dask: lon and lat must have the same shape.")
    npts = lon.size
    
    #___________________________________________________________________________
    # create grid coorinates for geo and rotated frame
    if any(x in gridis for x in ['geo', 'g', 'geographical']):
        rlon, rlat = grid_g2r(abg, lon, lat)
    elif any(x in gridis for x in ['rot', 'r', 'rotated']):
        rlon, rlat = lon, lat
        lon , lat  = grid_r2g(abg, rlon, rlat)
    else:
        raise ValueError(f"Unsupported gridis={gridis}, expected 'geo' or 'rot'.")

    #___________________________________________________________________________
    # compute rotation matrix
    rmat0 = grid_rotmat(abg)
    rmat  = rmat0.T
    
    #___________________________________________________________________________
    # degree --> radian 
    rad        = np.pi / 180  # Degree to radian conversion
    lon_rad  = lon  * rad
    lat_rad  = lat  * rad
    rlon_rad = rlon * rad
    rlat_rad = rlat * rad
    
    #___________________________________________________________________________
    # 5. Ensure urot, vrot are dask arrays and chunked only in non-node dims
    u_da = da.asarray(urot)
    v_da = da.asarray(vrot)

    if u_da.shape != v_da.shape:
        raise ValueError("vec_r2g_dask: urot and vrot must have the same shape.")

    if u_da.shape[-1] != npts:
        raise ValueError("vec_r2g_dask: last dimension of urot/vrot ({}) "
                         "must match len(lon)=={} as node dimension."
                        .format(u_da.shape[-1], npts))

    ## Rechunk so that the last axis (nodes) is a single chunk of length npts
    ## and we only split along time/depth axes
    #chunks = list(u_da.chunks)
    ## Replace last-dim chunks with a single chunk spanning all nodes
    #chunks[-1] = (npts,)
    #u_da = u_da.rechunk(chunks)
    #v_da = v_da.rechunk(chunks)

    if do_info:
        print("vec_r2g_dask:")
        print("  u/v shape :", u_da.shape)
        print("  chunks    :", u_da.chunks)
        print("  npts      :", npts)

    
    
    #___________________________________________________________________________
    # Block function: calls the numba kernel on each Dask block
    # NOTE: lon/lat/rmat are closed over as NumPy objects
    def _block(u_block, v_block):
        ugeo_block, vgeo_block = njit_vec_r2g_123d(
            rmat, lon_rad, lat_rad, rlon_rad, rlat_rad,
            u_block, v_block
        )
        return np.stack((ugeo_block, vgeo_block), axis=0)  # (2, ...)

    #___________________________________________________________________________
    # Use map_blocks: output has an extra leading axis (2, ...)
    #t1 = clock.time()
    stacked = da.map_blocks(_block, u_da, v_da, dtype=u_da.dtype,
                            chunks=( (2,), ) + u_da.chunks,
                            new_axis=0,
                            )

    #___________________________________________________________________________
    ugeo_da = stacked[0]
    vgeo_da = stacked[1]
    #print(' elapsed time 123d', clock.time()-t1)
    #___________________________________________________________________________
    return ugeo_da, vgeo_da


@njit(cache=True, fastmath=True)
def njit_vec_r2g_0d(rmat, 
                    sin_lat , cos_lat , sin_lon , cos_lon ,
                    sin_rlat, cos_rlat, sin_rlon, cos_rlon, 
                    urot, vrot):
    
    # rotated cartesian components
    vxr = -vrot*sin_rlat*cos_rlon - urot*sin_rlon
    vyr = -vrot*sin_rlat*sin_rlon + urot*cos_rlon
    vzr =  vrot*cos_rlat

    # geo cartesian
    vxg = rmat[0,0]*vxr + rmat[0,1]*vyr + rmat[0,2]*vzr
    vyg = rmat[1,0]*vxr + rmat[1,1]*vyr + rmat[1,2]*vzr
    vzg = rmat[2,0]*vxr + rmat[2,1]*vyr + rmat[2,2]*vzr

    # back to lon/lat components
    vgeo = (- vxg*sin_lat*cos_lon 
            - vyg*sin_lat*sin_lon 
            + vzg*cos_lat)
    ugeo = (- vxg*sin_lon
            + vyg*cos_lon)
    
    #___________________________________________________________________________
    return ugeo, vgeo

@njit(cache=True, fastmath=True)
def njit_vec_r2g_123d(rmat, lon_rad, lat_rad,
                      rlon_rad, rlat_rad,
                      urot, vrot):

    ndim = urot.ndim

    # __________________________________________________________________________
    # precompute trig values (1D arrays)
    sin_lat  = np.sin(lat_rad);  cos_lat  = np.cos(lat_rad)
    sin_lon  = np.sin(lon_rad);  cos_lon  = np.cos(lon_rad)
    sin_rlat = np.sin(rlat_rad); cos_rlat = np.cos(rlat_rad)
    sin_rlon = np.sin(rlon_rad); cos_rlon = np.cos(rlon_rad)

    #___________________________________________________________________________
    # 1 Dimensional urot, vrot = [npts] 
    if ndim == 1:
        ugeo, vgeo = njit_vec_r2g_0d(rmat,
                                     sin_lat , cos_lat , sin_lon , cos_lon ,
                                     sin_rlat, cos_rlat, sin_rlon, cos_rlon,
                                     urot, vrot)
        return ugeo, vgeo
    #___________________________________________________________________________
    # 2 Dimensional urot, vrot = [nlev, npts] 
    elif ndim == 2:
        nd, npts = urot.shape
        ugeo = np.empty_like(urot)
        vgeo = np.empty_like(vrot)
        for di in range(nd):
            ugeo[di,:], vgeo[di,:] = njit_vec_r2g_0d(
                                                    rmat,
                                                    sin_lat , cos_lat , sin_lon , cos_lon ,
                                                    sin_rlat, cos_rlat, sin_rlon, cos_rlon,
                                                    urot[di,:], vrot[di,:])
        return ugeo, vgeo
    #___________________________________________________________________________
    # 3 Dimensional urot, vrot = [ntime, nlev, npts] 
    elif ndim == 3:
        nt, nd, npts = urot.shape
        ugeo = np.empty_like(urot)
        vgeo = np.empty_like(vrot)
        for ti in range(nt):
            for di in range(nd):
                ugeo[ti, di, :], vgeo[ti, di, :] = njit_vec_r2g_0d(
                                                    rmat,
                                                    sin_lat , cos_lat , sin_lon , cos_lon ,
                                                    sin_rlat, cos_rlat, sin_rlon, cos_rlon,
                                                    urot[ti, di, :], vrot[ti, di, :])
        return ugeo, vgeo
    #___________________________________________________________________________
    else: raise ValueError("invalid ndim")
   
        
        
#
#
# ___CUTOUT REGION BASED ON BOX________________________________________________
def grid_cutbox_e(n_x, n_y, e_i, box, which='mid'):# , do_outTF=False):
    """
    --> cutout region based on box and return mesh elements indices that are
    within the box                                                      
    
    Parameters:
    
        :nx:        longitude vertice coordinates

        :ny:        latitude  vertice coordinates

        :e_i:       element array               

        :box:       list, [lonmin, lonmax, latmin, latmax]

        :which:     str, how limiting should be the selection
                    - 'soft' : elem with at least 1 vertices in box are selected
                    - 'mid'  : elem with at least 2 vertices in box are selected
                    - 'hard' : elem with at all vertices in box are selected

    Returns:
    
        :e_inbox:   array, boolian array with 1 in box, 0 outside box

    """
    
    
    # if the global selection box center is not at 0 (-180...180), also need to offset
    # cutting box
    offset = 0.0
    if (box[1]-box[0]) >= 359 and (box[0]+box[1])*0.5!=0.0: offset = (box[0]+box[1])*0.5
    
    #___________________________________________________________________________
    n_inbox = grid_cutbox_n(n_x+offset, n_y, box)
    
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



#
#
# ___CUTOUT REGION BASED ON BOX________________________________________________
def grid_cutbox_n(n_x, n_y, box):# , do_outTF=False):
    """
    --> cutout region based on box and return mesh elements indices that are
    within the box
    
    Parameters:
    
        :nx:    longitude vertice coordinates

        :ny:    latitude  vertice coordinates

        :e_i:   element array                

        :box:   list, [lonmin, lonmax, latmin, latmax]

    Returns:    

        :n_inbox:   array, boolian array with 1 in box, 0 outside box

    """
    #___________________________________________________________________________
    n_inbox = ((n_x >= box[0]) & (n_x <= box[1]) & 
               (n_y >= box[2]) & (n_y <= box[3]))
    
    #___________________________________________________________________________
    return(n_inbox)



# ___INTERPOLATE FROM ELEMENTS TO VERTICES_____________________________________
#|                                                                             |
#|_____________________________________________________________________________|
def grid_interp_e2n(mesh, data_e, data_e2=None, client=None):
    """
    --> interpolate data from elements to vertices e.g velocity from elements to 
    velocity on nodes
    
    Parameter:
    
        :mesh: fesom2 mesh object
        
        :data_e: np.array with datas on elements either 2d or 3d 
        
    Returns:
    
        :data_n: np.array with datas on vertices 2d or 3d
    
    """
    #___________________________________________________________________________
    # compute area weights if not already exist    
    mesh = mesh.compute_e_area()
    mesh = mesh.compute_n_area()
    e_i    = np.ascontiguousarray(mesh.e_i, dtype=np.int32)
    e_area = np.ascontiguousarray(mesh.e_area, dtype=np.float32)
    data_e = np.ascontiguousarray(data_e, dtype=np.float32)
        
    #___________________________________________________________________________   
    # do ie2n for 1d data [nelem]
    t0= clock.time()
    if data_e.ndim==1:
        if data_e2 is None:
            data_n, _       = njit_ie2n_1d(mesh.n2dn, mesh.n2de, mesh.e_i, e_area, data_e)
            print(' > jit 1d ie2n sca elapsed time: {:2.3f} sec.'.format( clock.time()-t0) )
        else:
            data_e2 = np.ascontiguousarray(data_e2, dtype=np.float32)
            data_n, data_n2 = njit_ie2n_1d(mesh.n2dn, mesh.n2de, mesh.e_i, e_area, data_e, data_e2)
            print(' > jit 1d ie2n vec elapsed time: {:2.3f} sec.'.format( clock.time()-t0) )
    
    #___________________________________________________________________________  
    # do ie2n for 2d data [ndi, nelem]
    elif data_e.ndim==2:
        nd       = data_e.shape[0]
        blocksize= np.int32(12) # --> fixed vertical block size for dask 
        if data_e2 is None:
            data_n, _       = dask_njit_ie2n_2d(nd, mesh.n2dn, mesh.n2de, mesh.e_i, e_area, data_e, 
                                               client=client, blocksize=blocksize)
            print(' 2d ie2n sca elapsed time: {:2.3f} sec.'.format( clock.time()-t0) )      
        else:
            data_e2 = np.ascontiguousarray(data_e2, dtype=np.float32)
            data_n, data_n2 = dask_njit_ie2n_2d(nd, mesh.n2dn, mesh.n2de, mesh.e_i, e_area, data_e, data_e2, 
                                               client=client, blocksize=blocksize)
            print(' 2d ie2n vec elapsed time: {:2.3f} sec.'.format( clock.time()-t0) )      
            
    #___________________________________________________________________________  
    # do ie2n for 3d data [nti, ndi, nelem]
    elif data_e.ndim==3:
        nt       = data_e.shape[0]
        nd       = data_e.shape[1]
        blocksize= np.int32(12) # --> fixed vertical block size for dask 
        if data_e2 is None:
            data_n, _       = dask_njit_ie2n_3d(nt, nd, mesh.n2dn, mesh.n2de, mesh.e_i, e_area, data_e, 
                                               client=client, blocksize=blocksize)
            print(' 3d ie2n sca elapsed time: {:2.3f} sec.'.format( clock.time()-t0) )      
        else:
            data_e2 = np.ascontiguousarray(data_e2, dtype=np.float32)
            data_n, data_n2 = dask_njit_ie2n_3d(nt, nd, mesh.n2dn, mesh.n2de, mesh.e_i, e_area, data_e, data_e2, 
                                               client=client, blocksize=blocksize)
            print(' 3d ie2n vec elapsed time: {:2.3f} sec.'.format( clock.time()-t0) )      
            
    #___________________________________________________________________________
    if data_e2 is None: return(data_n)
    else              : return(data_n, data_n2)



#
#
#_______________________________________________________________________________
# inline numba protype caller routine to acumulate data from elem --> nodes
@njit(inline='always', cache=True, fastmath=True)
def njit_ie2n_accum(n2dn, n2de, e_i, e_area, data_e):
    """
    Accumulate area muliplied data from elements to nodes

    n2dn   : #nodes
    n2de   : #elements
    e_i    : (n2de, 3) int32  element->node connectivity
    e_area : (n2de,) float32  element areas
    data_e : (n2de,) float32  primary variable
    """
    
    # this is for element blockwise treatment
    if n2de==0: n2de = e_i.shape[0] 
    data_n  = np.zeros(n2dn, dtype=np.float32)
    for ii in range(n2de): 
        v  = data_e[ii]
        # check for land sea mask it can be either 0.0 or NaN,  v == v --> fastest NaN check
        if (v == v) and (v != 0.0):          
            # elem --> vertices indices
            i0 = e_i[ii, 0]
            i1 = e_i[ii, 1]
            i2 = e_i[ii, 2]
            
            # compute elem area weighted data
            a  = e_area[ii]
            v  *= a 
            
            # data_e*e_area on vertice
            data_n[ i0] += v 
            data_n[ i1] += v
            data_n[ i2] += v
    return(data_n)       



#
#
#_______________________________________________________________________________
# inline numba protype caller routine to acumulate data from elem --> nodes
@njit(inline='always', cache=True, fastmath=True)
def njit_ie2n_accum_2d(nd, n2dn, n2de, e_i, e_area, data_e):
    """
    Accumulate area muliplied data from elements to nodes

    n2dn   : #nodes
    n2de   : #elements
    e_i    : (n2de, 3) int32  element->node connectivity
    e_area : (n2de,) float32  element areas
    data_e : (n2de,) float32  primary variable
    """
    
    # this is for element blockwise treatment
    if n2de==0: n2de = e_i.shape[0] 
    data_n  = np.zeros((nd, n2dn), dtype=np.float32)
        
    # depth loop
    for di in range(nd):
        
        for ii in range(n2de): 
            v  = data_e[di, ii]
            # check for land sea mask it can be either 0.0 or NaN,  v == v --> fastest NaN check
            if (v == v) and (v != 0.0):          
                # elem --> vertices indices
                i0 = e_i[ii, 0]
                i1 = e_i[ii, 1]
                i2 = e_i[ii, 2]
                
                # compute elem area weighted data
                a  = e_area[ii]
                v  *= a 
                
                # data_e*e_area on vertice
                data_n[di, i0] += v 
                data_n[di, i1] += v
                data_n[di, i2] += v
    return(data_n)      


#
#
#_______________________________________________________________________________
# inline numba protype caller routine to acumulate data from elem --> nodes
@njit(inline='always', cache=True, fastmath=True)
def njit_ie2n_accum_inline(n2dn, n2de, e_i, e_area, data_e, data_e2, out_n, out_n2, out_a, use2var):
    """
    Accumulate area muliplied data from elements to nodes

    n2dn   : #nodes
    n2de   : #elements
    e_i    : (n2de, 3) int32  element->node connectivity
    e_area : (n2de,) float32  element areas
    data_e : (n2de,) float32  primary variable
    data_e2: (n2de,) float32  secondary variable (ignored if use2var=False)
    out_n  : (n2dn,) float32  accumulated data_e*area
    out_n2 : (n2dn,) float32  accumulated data_e2*area (or dummy)
    out_a  : (n2dn,) float32  accumulated area per node
    use2var: bool to use 1 or 2 variables as input 
    """
    
    # this is for element blockwise treatment
    if n2de==0: n2de = e_i.shape[0] 
    
    for ii in range(n2de): 
        v  = data_e[ii]
        # check for land sea mask it can be either 0.0 or NaN,  v == v --> fastest NaN check
        if (v == v) and (v != 0.0):          
            # elem --> vertices indices
            i0 = e_i[ii, 0]
            i1 = e_i[ii, 1]
            i2 = e_i[ii, 2]
            
            # compute elem area weighted data
            a  = e_area[ii]
            v  *= a 
            
            # data_e*e_area on vertice
            out_n[ i0] += v 
            out_n[ i1] += v
            out_n[ i2] += v
            
            # accumulated elem area  per vertice
            out_a[ i0] += a
            out_a[ i1] += a
            out_a[ i2] += a
            
            # data2*e_area on vertice
            if use2var:
                v2 = data_e2[ii]*a
                out_n2[i0] += v2
                out_n2[i1] += v2
                out_n2[i2] += v2



#
#
#_______________________________________________________________________________
# 1 Dimensional numba optimized elem --> node interpolation for single variable 
# and vector variable 
@njit(cache=True, fastmath=True)
def njit_ie2n_1d(n2dn, n2de, e_i, e_area, data_e, data_e2=None):
    """
    compute area weighted mean from elements to nodes in 1Dimension

    n2dn   : #nodes
    n2de   : #elements
    e_i    : (n2de, 3) int32  element->node connectivity
    e_area : (n2de,) float32  element areas
    data_e : (n2de,) float32  primary variable
    data_e2: (n2de,) float32  secondary variable (ignored if use2var=False)
    """
    
    # Determine if we have 2nd variable and # Create a dummy array so worker 
    # always receives arrays
    use2var = data_e2 is not None
    if not use2var: data_e2 = np.zeros(2, dtype=np.float32) # just a dummy filler array if data_e2=None
      
    data_n  = np.zeros(n2dn, dtype=np.float32)
    data_a  = np.zeros(n2dn, dtype=np.float32)
    if not use2var: data_n2 = np.zeros(2   , dtype=np.float32) # just a dummy filler array if data_e2=None
    else          : data_n2 = np.zeros(n2dn, dtype=np.float32)
    
    # compute data * area  per node
    njit_ie2n_accum_inline(n2dn, n2de, e_i, e_area,
                           data_e, data_e2,
                           data_n, data_n2, data_a, use2var)
    
    # compute area weighted mean on nodes 
    for ii in range(n2dn):
        a = data_a[ii]
        if a > 0.0:
            data_n[ ii] /= a
            if use2var: data_n2[ii] /= a
        else:
            data_n[ ii] = np.nan
            if use2var: data_n2[ii] = np.nan
    return(data_n, data_n2)



#
#
#_______________________________________________________________________________
# 2 Dimensional numba optimized elem --> node interpolation for single variable 
# and vector variable 
@njit(cache=True, fastmath=True)
def njit_ie2n_2d(nd, n2dn, n2de, e_i, e_area, data_e, data_e2=None):
    """
    compute area weighted mean from elements to nodes in 2Dimension
    
    nd     : #depth levels
    n2dn   : #nodes
    n2de   : #elements
    e_i    : (n2de, 3) int32  element->node connectivity
    e_area : (n2de,) float32  element areas
    data_e : (nd, n2de,) float32  primary variable
    data_e2: (nd, n2de,) float32  secondary variable (ignored if use2var=False)
    """
    
    # Determine if we have 2nd variable and # Create a dummy array so worker 
    # always receives arrays
    use2var = data_e2 is not None
    if not use2var: data_e2 = np.zeros((nd, 2), dtype=np.float32) # just a dummy filler array if data_e2=None
    
    data_n  = np.zeros((nd, n2dn), dtype=np.float32)
    if not use2var: data_n2 = np.zeros((nd, 2   ), dtype=np.float32) # just a dummy filler array if data_e2=None
    else          : data_n2 = np.zeros((nd, n2dn), dtype=np.float32)
    
    # auxilary arrays for each depth 
    di_n  = np.zeros(n2dn, dtype=np.float32)
    di_a  = np.zeros(n2dn, dtype=np.float32)
    if not use2var: di_n2 = np.zeros(2   , dtype=np.float32) # just a dummy filler array if data_e2=None
    else          : di_n2 = np.zeros(n2dn, dtype=np.float32)
        
    # depth loop
    for di in range(nd):
        
        # faster than np.zeros within the loop 
        for ni in range(n2dn):
            di_n[ ni] = 0.0
            di_a[ ni] = 0.0
            if use2var: di_n2[ ni] = 0.0
        
        # compute data * area  per node and depth
        njit_ie2n_accum_inline(n2dn, n2de, e_i, e_area,
                               data_e[di,:], data_e2[di,:],
                               di_n, di_n2, di_a, use2var)
        
        # compute area weighted mean on nodes and depth
        for ii in range(n2dn):
            a = di_a[ii]
            if a > 0.0:
                data_n[ di, ii] = di_n[ ii] / a
                if use2var: data_n2[di, ii] = di_n2[ii] / a
            else:
                data_n[ di, ii] = np.nan
                if use2var: data_n2[di, ii] = np.nan
    return data_n, data_n2
#
#
#_______________________________________________________________________________
# 3 Dimensional numba optimized elem --> node interpolation for single variable 
# and vector variable 
@njit(cache=True, fastmath=True)
def njit_ie2n_3d(nt, nd, n2dn, n2de, e_i, e_area, data_e, data_e2=None):
    """
    compute area weighted mean from elements to nodes in 2Dimension
    
    nt     : #time slices
    nd     : #depth levels
    n2dn   : #nodes
    n2de   : #elements
    e_i    : (n2de, 3) int32  element->node connectivity
    e_area : (n2de,) float32  element areas
    data_e : (nt, nd, n2de,) float32  primary variable
    data_e2: (nt, nd, n2de,) float32  secondary variable (ignored if use2var=False)
    """
    
    # Determine if we have 2nd variable and # Create a dummy array so worker 
    # always receives arrays
    use2var = data_e2 is not None
    if not use2var: data_e2 = np.zeros((nt, nd, 2), dtype=np.float32) # just a dummy filler array if data_e2=None
    
    data_n  = np.zeros((nt, nd, n2dn), dtype=np.float32)
    if not use2var: data_n2 = np.zeros((nt, nd, 2   ), dtype=np.float32) # just a dummy filler array if data_e2=None
    else          : data_n2 = np.zeros((nt, nd, n2dn), dtype=np.float32)
    
    # auxilary arrays for each depth 
    di_n  = np.zeros(n2dn, dtype=np.float32)
    di_a  = np.zeros(n2dn, dtype=np.float32)
    if not use2var: di_n2 = np.zeros(2   , dtype=np.float32) # just a dummy filler array if data_e2=None
    else          : di_n2 = np.zeros(n2dn, dtype=np.float32)
    
    # time loop
    for ti in range(nt):
        
        # depth loop
        for di in range(nd):
            
            # faster than np.zeros within the loop 
            for ni in range(n2dn):
                di_n[ ni] = 0.0
                di_a[ ni] = 0.0
                if use2var: di_n2[ ni] = 0.0
            
            # compute data * area  per node and depth
            njit_ie2n_accum_inline(n2dn, n2de, e_i, e_area,
                                   data_e[ti, di,:], data_e2[ti, di,:],
                                   di_n, di_n2, di_a, use2var)
            
            # compute area weighted mean on nodes and depth
            for ii in range(n2dn):
                a = di_a[ii]
                if a > 0.0:
                    data_n[ti, di, ii] = di_n[ ii] / a
                    if use2var: data_n2[ti, di, ii] = di_n2[ii] / a
                else:
                    data_n[ti, di, ii] = np.nan
                    if use2var: data_n2[ti, di, ii] = np.nan
    return data_n, data_n2



#
#
#_______________________________________________________________________________
# 2 Dimensional wrapped numba/dask optimized elem --> node interpolation for single variable 
# and vector variable. The ie2n interpolation is parallelized with dask 
# over the vertical dimension if a dask client is present
def dask_njit_ie2n_2d(nd, n2dn, n2de, e_i, e_area, data_e, data_e2=None, 
                     client=None, blocksize=16):
    """
    compute area weighted mean from elements to nodes in 2Dimension using dask 
    wrapper to parallelize the vertical dimension
    
    nd        : #depth levels
    n2dn      : #nodes
    n2de      : #elements
    e_i       : (n2de, 3) int32  element->node connectivity
    e_area    : (n2de,) float32  element areas
    data_e    : (nd, n2de,) float32  primary variable
    data_e2   : (nd, n2de,) float32  secondary variable (ignored if use2var=False)
    client    : None or dask.client
    blocksize : 16 vertical parallel blocksize
    """
    
    # No Dask client → fallback to single-core numba version
    if client is None:
        print(' > use jit', end='')
        return njit_ie2n_2d(nd, n2dn, n2de, e_i, e_area, data_e, data_e2)

    # If nd small, Dask overhead not worth it
    if nd <= blocksize:
        print(' > use jit', end='')
        return njit_ie2n_2d(nd, n2dn, n2de, e_i, e_area, data_e, data_e2)

    print(' > use dask/jit', end='')
    # ---- Submit blocks ----
    futures = []
    for start in range(0, nd, blocksize):
        end = min(start + blocksize, nd)
        nd_block = end-start    
        block_e  = data_e[start:end, :].copy()
        block_e2 = None if data_e2 is None else data_e2[start:end, :].copy()

        fut = client.submit(njit_ie2n_2d ,
                            nd_block, n2dn, n2de, e_i, e_area, block_e, block_e2,
                            pure=False
                            )
        futures.append((start, end, fut))

    # ---- Assemble output ----
    data_n  = np.zeros((nd, n2dn), dtype=np.float32)
    data_n2 = None
    use2var = data_e2 is not None
    if use2var: data_n2 = np.zeros((nd, n2dn), dtype=np.float32)

    for start, end, fut in futures:
        dn_block, dn2_block = fut.result()
        data_n[start:end, :] = dn_block
        if use2var: data_n2[start:end, :] = dn2_block

    return data_n, data_n2
#
#
#_______________________________________________________________________________
# 3 Dimensional wrapped numba/dask optimized elem --> node interpolation for single variable 
# and vector variable. The ie2n interpolation is parallelized with dask 
# over the vertical dimension if a dask client is present
def dask_njit_ie2n_3d(nt, nd, n2dn, n2de, e_i, e_area, data_e, data_e2=None, 
                     client=None, blocksize=16):
    """
    compute area weighted mean from elements to nodes in 2Dimension using dask 
    wrapper to parallelize the vertical dimension
    
    nt        : #time slices
    nd        : #depth levels
    n2dn      : #nodes
    n2de      : #elements
    e_i       : (n2de, 3) int32  element->node connectivity
    e_area    : (n2de,) float32  element areas
    data_e    : (nt, nd, n2de,) float32  primary variable
    data_e2   : (nt, nd, n2de,) float32  secondary variable (ignored if use2var=False)
    client    : None or dask.client
    blocksize : 16 vertical parallel blocksize
    """
    
    # No Dask client → fallback to single-core numba version
    if client is None:
        print(' > use jit', end='')
        return njit_ie2n_3d(nt, nd, n2dn, n2de, e_i, e_area, data_e, data_e2)

    # If nd small, Dask overhead not worth it
    if nd <= blocksize:
        print(' > use jit', end='')
        return njit_ie2n_3d(nt, nd, n2dn, n2de, e_i, e_area, data_e, data_e2)

    print(' > use dask/jit', end='')
    
    # allocate 
    data_n  = np.zeros((nt, nd, n2dn), dtype=np.float32)
    use2var = data_e2 is not None
    data_n2 = np.zeros((nt, nd, n2dn), dtype=np.float32) if use2var else None
        
    # parallelized loop over depth 
    futures = []
    for start in range(0, nd, blocksize):
        end = min(start + blocksize, nd)
        nd_block = end-start    
        block_e  = data_e[:, start:end, :].copy()
        block_e2 = None if data_e2 is None else data_e2[:, start:end, :].copy()
            
        fut = client.submit(njit_ie2n_3d,
                            nt, nd_block, n2dn, n2de, e_i, e_area, block_e, block_e2,
                            pure=False
                            )
        futures.append((start, end, fut))
        
    # gather block together´
    for start, end, fut in futures:
        dn_block, dn2_block = fut.result()
        data_n[:, start:end, :] = dn_block
        if use2var: data_n2[:, start:end, :] = dn2_block

    return data_n, data_n2



#
#
# ___COMPUTE BOUNDARY EDGES____________________________________________________
def compute_boundary_edges(e_i):
    """
    --> compute edges that have only one adjacenbt triangle
    
    Parameters:
    
        :e_i:   np.array([n2de x 3]), elemental array
    
    Returns:
    
        :bnde:
    
    """
    # set boundary depth to zero
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
    return(bnde)



#
#
# ___COMPUTE BOUNDARY EDGES____________________________________________________
@njit(cache=True)
def njit_compute_boundary_edges(e_i):
    """
    --> compute edges that have only one adjacenbt triangle
    
    Parameters:
    
        :e_i:   np.array([n2de x 3]), elemental array
    
    Returns:
    
        :bnde:
    
    """
    
    # create dict: link int64 keys -> int64 values
    edge_count = Dict.empty(key_type=types.int64, value_type=types.int64)

    #___________________________________________________________________________
    # build edges and count occurrences
    n2de = e_i.shape[0]
    for ii in range(n2de):
        
        # vertice indices of triangle 
        a = e_i[ii, 0]
        b = e_i[ii, 1]
        c = e_i[ii, 2]
        
        # form edges --> smallest vertice index first
        if a < b: edp11, edp12 = a, b
        else    : edp11, edp12 = b, a
        
        if a < c: edp21, edp22 = a, c
        else    : edp21, edp22 = c, a
        
        if b < c: edp31, edp32 = b, c
        else    : edp31, edp32 = c, b
        
        # trick here: 64-bit packed keys:
        # you originally needed to store an edge as: (low, high) tuple But Numba 
        # has trouble with tuple keys. The workaround is to encode the pair into 
        # a single integer.
        # Given: 
        #   low0  = the smaller node index
        #   high0 = the larger node index
        #
        # We create a single 64-bit value: key0 = (low0 << 32) | high0
        #  low0 << 32       This shifts all bits of low0 32 bits to the left
        #  | high0          is bitwise OR. This fills the lower 32 bits with high0
        #
        # So the packed format becomes:  
        #   key = (low << 32) + high
        # This is reversible and unique as long as low and high are ≤ 2^32−1 
        # (which they are for all FESOM nodes).   
        #
        # We reverse the process:
        #   low  = key >> 32            shifts everything down → gets low
        #   high = key & 0xffffffff     key & 0xffffffff masks the lower 32 bits → gets high
        # 
        key0 = (edp11 << 32) | edp12
        key1 = (edp21 << 32) | edp22
        key2 = (edp31 << 32) | edp32

        # count edge occurence. If its inner edge, edge_count==2 if its boundary 
        # edge edge count should be 1
        edge_count[key0] = edge_count.get(key0, 0) + 1
        edge_count[key1] = edge_count.get(key1, 0) + 1
        edge_count[key2] = edge_count.get(key2, 0) + 1

    #___________________________________________________________________________
    # compute exact number of boundary edges to allocate boundary edge array 
    # properly
    nbnde = 0
    for _, cnt in edge_count.items():
        if cnt == 1: nbnde += 1
    bnde = np.empty((nbnde, 2), np.int64)
    
    #___________________________________________________________________________
    # fill up boundary edge array
    k = 0
    for key, cnt in edge_count.items():
        # whereever edge_count==1 must be boundary edge
        if cnt == 1:
            
            # extract vertice indices from 64-bit packed keys
            edp1 = key >> 32
            edp2 = key & 0xffffffff
            
            # write boundary edge vertice indices into boundary edge array
            bnde[k, 0] = edp1
            bnde[k, 1] = edp2
            k += 1

    return bnde



#
#
# ___COMPUTE NODE NEIGHBORHOOD WITH RESPECT TO ELEMENTS_________________________
def compute_nINe(n2dn, e_i, do_arr=False):
    """
    --> compute element indices list that contribute to cetain vertice
    
    Parameters:
    
        :n2dn:      int, number of vertices
        :e_i:       np.array([n2de x 3]), elemental array
        :do_arr:    bool (default=False) shut output be list or numpy array
    
    Returns:
    
        :nod_in_elem2D:
    
    """
    t1=clock.time()
    # allocate list of size n2dn, create independent empty lists for each entry, 
    nod_in_elem2D     = [[] for _ in range(n2dn)]
    nod_in_elem2D_num = [0]*n2dn
    
    # append element index to each vertice list entry where it appears
    for e_i, n_i in enumerate(e_i.flatten()): 
        nod_in_elem2D[    n_i].append(np.int32(e_i/3))
        nod_in_elem2D_num[n_i] += 1
    
    # convert list into numpy array
    if do_arr:
        nod_in_elem2D_arr = np.full((n2dn, max(nod_in_elem2D_num)), -1, dtype=np.int32)
        for ii, lst_ii in enumerate(nod_in_elem2D):
            nod_in_elem2D_arr[ii, :len(lst_ii)] = lst_ii
        nod_in_elem2D = nod_in_elem2D_arr    
        print(' --> elapsed time for nod_in_elem2D:', clock.time()-t1)
        return(nod_in_elem2D, nod_in_elem2D_num)
    else:   
        print(' --> elapsed time for nod_in_elem2D:', clock.time()-t1)
        return(nod_in_elem2D)



#
#
# ___COMPUTE NODE NEIGHBORHOOD WITH RESPECT TO ELEMENTS_________________________
@njit(cache=True)
def njit_compute_nINe(e_i):
    """
    --> compute element indices list that contribute to cetain vertice, determine
        vertice neighborhood with respect to elements
        
    Parameters:
    
        :e_i:       np.array([n2de x 3]), elemental array
    
    Returns:
    
        :nINe:
        :nINe_num:
    
    """
    
    n2de = e_i.shape[0]
    n2dn = np.max(e_i) + 1
    
    #___________________________________________________________________________
    # compute maximum number of elements that contribute to node
    nINe_num = np.zeros(n2dn, dtype=np.int32)
    for ii in range(n2de):
        a = e_i[ii, 0]
        b = e_i[ii, 1]
        c = e_i[ii, 2]
        
        nINe_num[a] += 1
        nINe_num[b] += 1
        nINe_num[c] += 1

    # max number of elements connected to any node
    max_nINe_num = np.max(nINe_num)

    #___________________________________________________________________________
    # allocate node in elem output array, initilaise with -1
    nINe = -1 * np.ones((n2dn, max_nINe_num), dtype=np.int32)

    # pointer array to track next insertion pos per node
    pos_fill = np.zeros(n2dn, dtype=np.int32)

    #___________________________________________________________________________
    # fill in node in elem array, use filling pointer to keep track of fill in 
    # position
    for ii in range(n2de):
        a = e_i[ii, 0]
        b = e_i[ii, 1]
        c = e_i[ii, 2]
        
        # take out fill position 
        pa = pos_fill[a]
        pb = pos_fill[b]
        pc = pos_fill[c]
        
        # fill node in elem position with elem index
        nINe[a, pa] = ii
        nINe[b, pb] = ii
        nINe[c, pc] = ii
        
        # count up fill position
        pos_fill[a] = pa + 1
        pos_fill[b] = pb + 1
        pos_fill[c] = pc + 1

    return nINe, nINe_num



#
#
# ___COMPUTE ELEM NEIGHBORHOOD WITH RESPECT TO ELEMENTS_________________________
@njit(cache=True)
def njit_compute_eINe(e_i):
    """
    --> compute element indices list that contribute to cetain element, determine
        elem neighborhood with respect to elements
        
    Parameters:
    
        :e_i:       np.array([n2de x 3]), elemental array
    
    Returns:
    
        :eINe:
        :eINe_num:
    
    """
    
    #___________________________________________________________________________
    # allocate
    n2de = e_i.shape[0]   
    nedg = 3 * n2de
    key  = np.empty(nedg, dtype=np.int64)
    elem = np.empty(nedg, dtype=np.int32)
    
    #___________________________________________________________________________
    # loop over elements
    kk = 0
    for ii in range(n2de):
        a = e_i[ii, 0]
        b = e_i[ii, 1]
        c = e_i[ii, 2]
        
        # trick here: 64-bit packed keys:
        # you originally needed to store an edge as: (low, high) tuple But Numba 
        # has trouble with tuple keys. The workaround is to encode the pair into 
        # a single integer.
        # Given: 
        #   low0  = the smaller node index
        #   high0 = the larger node index
        #
        # We create a single 64-bit value: key0 = (low0 << 32) | high0
        #  low0 << 32       This shifts all bits of low0 32 bits to the left
        #  | high0          is bitwise OR. This fills the lower 32 bits with high0
        #
        # So the packed format becomes:  
        #   key = (low << 32) + high
        # This is reversible and unique as long as low and high are ≤ 2^32−1 
        # (which they are for all FESOM nodes).   
        #
        # We reverse the process:
        #   low  = key >> 32            shifts everything down → gets low
        #   high = key & 0xffffffff     key & 0xffffffff masks the lower 32 bits → gets high
        # 
        # sorted edges a<b<c, create unique edge key based on node indexes that 
        # build up the edge, i have three edges --> create three keys k1,k2,k3
        # edge 0–1
        
        # edge a-b
        if a < b: key[kk] = (a << 32) | b
        else:     key[kk] = (b << 32) | a
        elem[kk] = ii
        kk += 1
        
        # edge a-c
        if a < c: key[kk] = (a << 32) | c
        else:     key[kk] = (c << 32) | a
        elem[kk] = ii
        kk += 1
        
        # edge b-c
        if b < c: key[kk] = (b << 32) | c
        else:     key[kk] = (c << 32) | b
        elem[kk] = ii
        kk += 1
    
    #___________________________________________________________________________
    # correct lexicographic sorting of keys that means elem that share the same 
    # edge have the same edge key and are thus near togehther after sorting
    idx  = np.argsort(key)
    key  = key[idx]
    elem = elem[idx]

    #___________________________________________________________________________
    # allocate adjacency arrays
    eINe = -1 * np.ones((n2de, 3), dtype=np.int32)
    eINe_num = np.zeros(n2de, dtype=np.int32)

    #___________________________________________________________________________
    # loop over all the eges 
    ii = 0
    while ii < nedg - 1:
        if key[ii] == key[ii+1]:
            a = elem[ii]
            b = elem[ii+1]
            
            na = eINe_num[a]
            nb = eINe_num[b]
            
            eINe[a, na] = b
            eINe[b, nb] = a
            
            eINe_num[a] = na+1
            eINe_num[b] = nb+1
            
            ii += 2
        else:
            ii += 1
    return eINe, eINe_num



#
#
# ___COMPUTE NODE NEIGHBORHOOD WITH RESPECT TO VERTICES_________________________
@njit(cache=True)
def njit_compute_nINn(e_i):
    """
    --> compute nodes indices list that contribute to cetain vertice, determine
        vertice neighborhood with respect to vertices
        
    Parameters:
    
        :e_i:       np.array([n2de x 3]), elemental array
    
    Returns:
    
        :nINn:
        :nINn_num:
    
    """
    n2de = e_i.shape[0]
    n2dn = np.max(e_i) + 1

    #___________________________________________________________________________
    # count node degrees
    nINn_num_est = np.zeros(n2dn, dtype=np.int32)
    for ii in range(n2de):
        a = e_i[ii, 0]
        b = e_i[ii, 1]
        c = e_i[ii, 2]
        
        # Each triangle connects a,b,c pairwise
        #            o a --> a has neighbor b and c, thats why count up +2
        #           / \ 
        #          /   \
        #         /     \  
        #        /       \
        #     c o---------o b 
        nINn_num_est[a] += 2       # b,c
        nINn_num_est[b] += 2       # a,c
        nINn_num_est[c] += 2       # a,b

    # BUT this counts duplicates via multiple triangles, so we will fix it
    # by filling adjacency and removing duplicates afterwards.
    max_nINn_num_est = np.max(nINn_num_est)
    nINn = -1 * np.ones((n2dn, max_nINn_num_est), dtype=np.int32)
    pos = np.zeros(n2dn, dtype=np.int32)

    #___________________________________________________________________________
    # Insert neighbors (may contain duplicates)
    for ii in range(n2de):
        a = e_i[ii, 0]
        b = e_i[ii, 1]
        c = e_i[ii, 2]

        # a neighbors: b,c
        pa = pos[a]
        nINn[a, pa] = b
        nINn[a, pa+1] = c
        pos[a] = pa + 2

        # b neighbors: a,c
        pb = pos[b]
        nINn[b, pb] = a
        nINn[b, pb+1] = c
        pos[b] = pb + 2

        # c neighbors: a,b
        pc = pos[c]
        nINn[c, pc] = a
        nINn[c, pc+1] = b
        pos[c] = pc + 2

    #___________________________________________________________________________
    # Deduplicate neighbors per node
    nINn_num = np.zeros(n2dn, dtype=np.int32)
    mask     = np.zeros(n2dn, dtype=np.uint8)
    for node in range(n2dn):
        row_end = pos[node]
        write = 0

        for i in range(row_end):
            nb = nINn[node][i]
            if mask[nb] == 0:
                mask[nb] = 1
                nINn[node][write] = nb
                write += 1

        # clear mask
        for i in range(write):
            mask[nINn[node][i]] = 0

        # mark remainder as unused
        for i in range(write, row_end):
            nINn[node][i] = -1

        nINn_num[node] = write

    #___________________________________________________________________________
    # Build compact adjacency array
    real_max = 0
    for i in range(n2dn):
        if nINn_num[i] > real_max:
            real_max = nINn_num[i]

    nINn_compact = -1 * np.ones((n2dn, real_max), dtype=np.int32)
    for node in range(n2dn):
        deg = nINn_num[node]
        for j in range(deg):
            nINn_compact[node, j] = nINn[node, j]

    return nINn_compact, nINn_num
