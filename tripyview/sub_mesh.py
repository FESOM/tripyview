# Patrick Scholz, 14.12.2017

import sys
import os
import time as clock
import numpy  as np
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
                    do_lsmshp   = True                  , 
                    do_earea    = True                  , 
                    do_narea    = True                  , 
                    do_eresol   = [False,'mean']        , 
                    do_nresol   = [False,'e_resol']     ,
                    do_loadraw  = False                 , 
                    do_pickle   = True                  , 
                    do_joblib   = False                 , 
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
    
    joblibfname = 'tripyview_fesom2_{}_focus{}.jlib'.format(meshid,focus)
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
    if  ( do_pickle and ( os.path.isfile(loadpicklepath) )) or \
        ( do_joblib and ( os.path.isfile(loadjoblibpath) )): 
            
        
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
         ((do_joblib and not ( os.path.isfile(loadjoblibpath)) ) or not do_joblib):
             
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
        """
        --> part of fesom mesh class, read mesh files nod2d.out, elem2d.out, 
        aux3d.out, nlvls.out and elvls.out   
        
        """
        #____load 2d node matrix________________________________________________
        #file_content = pa.read_csv(self.fname_nod2d, delim_whitespace=True, skiprows=1, \
        file_content = pa.read_csv(self.fname_nod2d, sep='\\s+', skiprows=1, \
                                      names=['node_number','x','y','flag'] )
        self.n_x     = file_content.x.values.astype('float32')
        self.n_y     = file_content.y.values.astype('float32')
        self.n_i     = file_content.flag.values.astype('int16')   
        self.n2dn    = len(self.n_x)
        
        #____load 2d element matrix_____________________________________________
        #file_content = pa.read_csv(self.fname_elem2d, delim_whitespace=True, skiprows=1, \
        file_content = pa.read_csv(self.fname_elem2d, sep='\\s+', skiprows=1, \
                                    names=['1st_node_in_elem','2nd_node_in_elem','3rd_node_in_elem'])
        self.e_i     = file_content.values.astype('int32') - 1
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
            self.n_z     = aux_n3z[self.n32.max(axis=0),0]
            del(aux_n3z)
            
            # compute bottom index at vertice
            aux_n32      = np.zeros(self.n32.shape)
            aux_n32[self.n32>=0] = 1
            self.n_iz    = aux_n32.sum(axis=0).astype('int16')-1

        
        #____load number of levels at each node_________________________________
        if ( os.path.isfile(self.fname_nlvls) ):
            #file_content = pa.read_csv(self.fname_nlvls, delim_whitespace=True, skiprows=0, \
            file_content = pa.read_csv(self.fname_nlvls, sep='\\s+', skiprows=0, \
                                           names=['numb_of_lev'])
            self.n_iz    = file_content.values.astype('int16') - 1
            self.n_iz    = self.n_iz.squeeze()
            self.n_z     = np.float32(self.zlev[self.n_iz])
            
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
            self.e_iz    = self.e_iz.squeeze()
            
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
                self.e_iz_raw    = self.e_iz_raw.squeeze()
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
            self.n_ic= self.n_ic.squeeze()
        else:
            raise ValueError(f' --> could not find file {self.fname_cnlvls} !')
        
        #____load number of cavity levels at each elem__________________________
        self.fname_celvls = os.path.join(self.path,'cavity_elvls.out')
        if ( os.path.isfile(self.fname_cnlvls) ):
            file_content      = pa.read_csv(self.fname_celvls, delim_whitespace=True, skiprows=0, names=['numb_of_lev'])
            self.e_ic= file_content.values.astype('int16') - 1
            self.e_ic= self.e_ic.squeeze()
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
                self.e_ic_raw   = self.e_ic_raw.squeeze()
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
        # identify the polar triangle
        #self.e_pbnd_p = np.argmin(90.0-self.n_y[self.e_i].sum(axis=1)/3.0)
        #self.n_x  = np.hstack((self.n_x , self.n_x[self.e_i[self.e_pbnd_p, :]].sum()/3.0))
        #self.n_y  = np.hstack((self.n_y , 90.0))
        #self.n_i  = np.hstack((self.n_i , 0))
        #self.n_iz = np.hstack((self.n_iz, self.n_iz[self.e_i[self.e_pbnd_p, :]].max() ))
        #self.n_z  = np.hstack((self.n_z , self.n_z[self.e_i[self.e_pbnd_p, :]].min() ))
        #if isinstance(self.n_ic, np.ndarray):
            #self.n_ic = np.hstack((self.n_ic , self.n_ic[self.e_i[self.e_pbnd_p, :]].min() ))
        #if isinstance(self.n_c , np.ndarray):
            #self.n_c  = np.hstack((self.n_c , self.n_c[self.e_i[self.e_pbnd_p, :]].min() ))
        
        ## replace with 3 new augmeted polar triangles
        #e_i_p = np.vstack( [self.e_i[self.e_pbnd_p, :]]*3 )
        #e_i_p[0,0], e_i_p[1,1], e_i_p[2,2] = self.n2dn, self.n2dn, self.n2dn
        #self.e_i  = np.vstack((self.e_i, e_i_p))
        #self.e_iz = np.hstack((self.e_iz, [self.e_iz[self.e_pbnd_p]]*3 ))
        #if isinstance(self.e_ic , np.ndarray): 
            #self.e_ic = np.hstack((self.e_ic, [self.e_ic[self.e_pbnd_p]]*3 ))
            
        ## delete orignal polar triangle elemental row 
        #self.e_i  = np.delete(self.e_i , self.e_pbnd_p, axis=0)
        #self.e_iz = np.delete(self.e_iz, self.e_pbnd_p, axis=0)
        #if isinstance(self.e_ic , np.ndarray): 
            #self.e_ic = np.delete(self.e_ic, self.e_pbnd_p, axis=0)
        
        #self.e_pbnd_p = np.arange(self.n2de-1, self.n2de+2,1)
        #self.n2dn +=1
        #self.n2de +=2
        
        #_______________________________________________________________________
        # find out 1st which element contribute to periodic boundary and 2nd
        # which nodes are involed in periodic boundary
        dx = self.n_x[self.e_i].max(axis=1)-self.n_x[self.e_i].min(axis=1)
        self.e_pbnd_1 = np.argwhere(dx > self.cyclic*2/3).ravel()
        self.e_pbnd_0 = np.argwhere(dx < self.cyclic*2/3).ravel()
        #self.e_pbnd_1 = np.unique(np.hstack((self.e_pbnd_1, self.e_pbnd_p)))
        
        #_______________________________________________________________________
        return(self)
    
    
    
    # ___AUGMENT PERIODIC BOUDNARY ELEMENTS____________________________________
    #| add additional elements to augment the periodic boundary on the left and|
    #| right side for an even non_periodic boundary                            |
    #|_________________________________________________________________________|
    def pbnd_augment(self):
        """
        --> part of fesom mesh class, adds additional elements to augment the 
        periodic boundary on the left and right side for an even non_periodic 
        boundary is created left and right [-180, 180] of the domain         

        """
        self.do_augmpbnd = True
        #_______________________________________________________________________
        # this are all the periodic boundary element
        e_i_pbnd   = self.e_i[self.e_pbnd_1,:]
        e_i_pbnd_x = self.n_x[e_i_pbnd]
        
        # in each periodic boundary element look what is the vertices indices with 
        # max lon value for the right boundary
        nidxine_l, nidxine_m, nidxine_r = np.squeeze(np.split(np.argsort(e_i_pbnd_x, axis=1), 3, axis=1))
        
        eidx       = np.arange(e_i_pbnd.shape[0])
        n_pbnd_i_l = e_i_pbnd[eidx, nidxine_l.squeeze()]
        n_pbnd_i_m = e_i_pbnd[eidx, nidxine_m.squeeze()]
        n_pbnd_i_r = e_i_pbnd[eidx, nidxine_r.squeeze()]
        del(e_i_pbnd_x, eidx)
        
        # now decide if the remaining "middle" point should be attributet to left 
        # or right periodic boundary by its distance to the already known min left and 
        # max right vertic points
        is_pbnd_i_m_lr = np.abs(self.n_x[n_pbnd_i_r]-self.n_x[n_pbnd_i_m]) < np.abs(self.n_x[n_pbnd_i_l]-self.n_x[n_pbnd_i_m])
        
        # remaining "middle" point becomes part of right boundary
        n_pbnd_i_r = np.hstack((n_pbnd_i_r, n_pbnd_i_m[is_pbnd_i_m_lr==True]))
        n_pbnd_i_r = np.unique(n_pbnd_i_r)
        
        # remaining "middle" point becomes part of left boundary
        n_pbnd_i_l = np.hstack((n_pbnd_i_l, n_pbnd_i_m[is_pbnd_i_m_lr==False]))
        n_pbnd_i_l = np.unique(n_pbnd_i_l)        
        
        # total array of left and augmented periodic boundary nodes         
        self.n_pbnd_a   = np.hstack((n_pbnd_i_r,n_pbnd_i_l))
        nn_il,nn_ir= n_pbnd_i_l.size, n_pbnd_i_r.size
        
        #_______________________________________________________________________
        # calculate augmentation positions for new left and right periodic boundaries
        aux_pos    = np.zeros(self.n2dn,dtype='uint32')
        aux_i      = np.linspace(self.n2dn,self.n2dn+nn_ir-1,nn_ir,dtype='uint32')
        aux_pos[n_pbnd_i_r] =aux_i
        aux_i      = np.linspace(self.n2dn+nn_ir,self.n2dn+nn_ir+nn_il-1,nn_il,dtype='uint32')
        aux_pos[n_pbnd_i_l]= aux_i 
        del(aux_i, n_pbnd_i_l, n_pbnd_i_r, n_pbnd_i_m)
        
        #_______________________________________________________________________
        # Augment the vertices on the right and left side 
        if self.cyclic == 360:
            #xmin, xmax= np.floor(xmin),np.ceil(xmax)
            xmin, xmax= -self.cyclic/2+self.focus, self.cyclic/2+self.focus
        else:    
            xmin, xmax = 0, self.cyclic
        self.n_xa  = np.concatenate((np.zeros(nn_ir)+xmin, np.zeros(nn_il)+xmax))
        self.n_ya  = self.n_y[self.n_pbnd_a]
        self.n_za  = self.n_z[self.n_pbnd_a]
        # self.n_ia = self.n_i[self.n_pbnd_a]
        self.n_iza = self.n_iz[self.n_pbnd_a]
        
        # if there is cavity information
        if isinstance(self.n_c , np.ndarray): self.n_ca  = self.n_c[self.n_pbnd_a]
        if isinstance(self.n_ic, np.ndarray): self.n_ica = self.n_ic[self.n_pbnd_a]
        
        # length of augmented vertice array
        self.n2dna = self.n2dn + self.n_pbnd_a.size
        
        #_______________________________________________________________________
        # (ii.a) 2d elements:
        # List all triangles that touch the cyclic boundary segments
        #_______________________________________________________________________
        elem_pbnd_l = np.copy(e_i_pbnd)
        elem_pbnd_r = np.copy(e_i_pbnd)
        for ei in range(0,self.e_pbnd_1.size):
            # node indices of periodic boundary triangle
            tri  = e_i_pbnd[ei,:]
            
            # which triangle points belong to left periodic bnde or right periodic
            # boundary
            idx_l, idx_r, idx_m = nidxine_l[ei], nidxine_r[ei], nidxine_m[ei]
            
            # change indices to left and right augmented boundary points
            elem_pbnd_l[ei,idx_r]=aux_pos[tri[idx_r]]
            elem_pbnd_r[ei,idx_l]=aux_pos[tri[idx_l]]
            
            # change indices of point beteen left and right vertices depending 
            # if it will be attributet to left or the right boundary
            if is_pbnd_i_m_lr[ei]: elem_pbnd_l[ei,idx_m]=aux_pos[tri[idx_m]]
            else                 : elem_pbnd_r[ei,idx_m]=aux_pos[tri[idx_m]]
                
        del idx_l, idx_r, idx_m, tri, aux_pos
        del nidxine_l, nidxine_m, nidxine_r, is_pbnd_i_m_lr
        
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
                # pi     = 3.14159265358979
                rad    = np.pi/180.0  
                cycl   = self.cyclic*rad
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
        """
        --> part of fesom mesh class, compute area of elements in [m], options:

            Parameter:

                which: str,
                        - "mean" ... resolution based on mean element edge legth
                        - "max"  ... resolution based on maximum element edge length
                        - "min"  ... resolution based on minimum element edge length

        """
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
                warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")
                self.n_resol=self.n_resol/self.n_area[0,:]/3.0
                warnings.resetwarnings()
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
        """
        --> part of fesom mesh class, compute land-sea mask contourline with 
        periodic boundary based on boundary edges that contribute only to one triangle 
        and then checking which edges can be consequtive connected                                                   |
        """
        print(' > compute lsmask')
        self.do_lsmask = True
        #_______________________________________________________________________
        # build land boundary edge matrix
        t1 = clock.time()
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
            kk_rc = np.column_stack(np.where( run_bnde==np.int32(run_cont[0,count_init]) ))
            #kk_rc = np.argwhere( run_bnde==np.int32(run_cont[0,count_init]) ) --> slower than np.column_stack(np.where....
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
        #t2 = clock.time()
        #print(t2-t1)
        return(self)
    
    
    
    # ___AUGMENT PERIODIC BOUNDARIES IN LAND-SEA MASK CONOURLINE_______________
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
    
    # import matplotlib.pyplot as plt 
    # hfig = plt.figure()
    # ax = plt.gca()
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

        # npoly = len(polygonlist)
        # Check if make_valid() returned a MultiPolygon
        if   isinstance(poly, MultiPolygon):
            polygonlist.extend(poly.geoms)  # Unpack MultiPolygon into list
            
            # for auxpoly in polygonlist[npoly:]:
            #     coords = np.array(auxpoly.exterior.coords)
            #     ax.plot(coords[:,0], coords[:,1], 'c*')

        # Check if make_valid() returned a GeometryCollection
        elif isinstance(poly, GeometryCollection):             
             # Extract only Polygon or MultiPolygon from the GeometryCollection
            for geom in poly.geoms:
                if isinstance(geom, Polygon):
                    polygonlist.append(geom)
                elif isinstance(geom, MultiPolygon):
                    polygonlist.extend(geom.geoms)
            
            # for auxpoly in polygonlist[npoly:]:
            #     coords = np.array(auxpoly.exterior.coords)
            #     ax.plot(coords[:,0], coords[:,1], 'r*')
        
        elif isinstance(poly, Polygon):
        #elif poly.is_valid:
            polygonlist.append(poly)
            
            # for auxpoly in polygonlist[npoly:]:
            #     coords = np.array(auxpoly.exterior.coords)
            #     ax.plot(coords[:,0], coords[:,1], 'k*')
        
    
    lsmask_p = MultiPolygon(polygonlist)
    # plt.show()
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



#
#
# ___COMPUTE3D CARTESIAN COORDINAT_____________________________________________
def grid_cart3d(lon,lat,R=1.0, is_deg=False):
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
    if is_deg: 
        rad = np.pi/180
        lat = lat * rad
        lon = lon * rad
    
    x = R*np.cos(lat) * np.cos(lon)
    y = R*np.cos(lat) * np.sin(lon)
    z = R*np.sin(lat)
    #___________________________________________________________________________
    return(x,y,z)



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


#
#
# ___ROTATE GRID FROM: GEO-->ROT_______________________________________________
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
        if do_info: print('     > 1D rotation')
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
        vgeo= vxg*-np.sin(lat)*np.cos(lon) - vyg* np.sin(lat)*np.sin(lon) + vzg* np.cos(lat)
        ugeo= vxg*-np.sin(lon) + vyg*np.cos(lon)
        
    #___________________________________________________________________________
    # rotation of two dimensional vector data    
    elif vrot.ndim==2 or urot.ndim==2: 
        if do_info: print('     > 2D rotation')
        nd1,nd2=urot.shape
        ugeo, vgeo = urot.copy(), vrot.copy()
        if do_info: print('nlev:{:d}'.format(nd2))
        for nd2i in range(0,nd2):
            #___________________________________________________________________
            if do_info: 
                print('{:02d}|'.format(nd2i), end='')
                if np.mod(nd2i+1,10)==0: print('')
            
            #___________________________________________________________________
            t1=clock.time()
            aux_urot, aux_vrot = urot[:,nd2i], vrot[:,nd2i]
            
            #___________________________________________________________________
            # compute vector in rotated cartesian coordinates
            t1=clock.time()
            vxr = -aux_vrot*np.sin(rlat)*np.cos(rlon) - aux_urot*np.sin(rlon)
            vyr = -aux_vrot*np.sin(rlat)*np.sin(rlon) + aux_urot*np.cos(rlon)
            vzr =  aux_vrot*np.cos(rlat)
            
            #___________________________________________________________________
            # compute vector in geo cartesian coordinates
            t1=clock.time()
            vxg = rmat[0,0]*vxr + rmat[0,1]*vyr + rmat[0,2]*vzr
            vyg = rmat[1,0]*vxr + rmat[1,1]*vyr + rmat[1,2]*vzr
            vzg = rmat[2,0]*vxr + rmat[2,1]*vyr + rmat[2,2]*vzr
            
            #___________________________________________________________________
            # compute vector in geo coordinates 
            t1=clock.time()
            vgeo[:,nd2i]= vxg*-np.sin(lat)*np.cos(lon) - vyg* np.sin(lat)*np.sin(lon) + vzg* np.cos(lat)
            ugeo[:,nd2i]= vxg*-np.sin(lon) + vyg*np.cos(lon)
            
    #___________________________________________________________________________    
    else: raise ValueError('This number of dimensions is in moment not supported for vector rotation')    
    #___________________________________________________________________________
    return(ugeo, vgeo)



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
def grid_interp_e2n(mesh,data_e):
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
        
    #___________________________________________________________________________    
    if data_e.ndim==1:
        
        # compute data on elements times area of elements
        data_exa = np.vstack((mesh.e_area,mesh.e_area,mesh.e_area)) * data_e
        data_exa = data_exa.transpose().flatten()
        
        # single loop over self.e_i.flat is ~4 times faster than douple loop 
        # over for i in range(3): ,for j in range(self.n2de):
        data_n = np.zeros(mesh.n2dn)
        for e_i, n_i in enumerate(mesh.e_i.flat): data_n[n_i] = data_n[n_i] + data_exa[e_i]
        data_n=data_n/mesh.n_area[0,:]/3.0        
        del data_exa
        
    #___________________________________________________________________________        
    elif data_e.ndim==2:
        
        nd        = data_e.shape[1]
        data_n    = np.zeros((mesh.n2dn, nd))
        data_area = np.vstack((mesh.e_area,mesh.e_area,mesh.e_area)).transpose().flatten()
        
        #_______________________________________________________________________
        def e2n_di(di, data_e, area_e, e_i_flat, n_iz):
            data_exa  = area_e * np.vstack((data_e[:,di],data_e[:,di],data_e[:,di])).transpose().flatten()
            data_n_di = np.zeros(n_iz.shape)
            data_a_di = np.zeros(n_iz.shape)
        
            # single loop over self.e_i.flat is ~4 times faster than douple loop 
            # over for i in range(3): ,for j in range(self.n2de):
            for e_i, n_i in enumerate(e_i_flat):
                if n_iz[n_i]<di: continue
                data_n_di[n_i] = data_n_di[n_i] + data_exa[ e_i]
                data_a_di[n_i] = data_a_di[n_i] + area_e[   e_i]
            with np.errstate(divide='ignore',invalid='ignore'):    
                data_n_di = data_n_di/data_a_di
            return(data_n_di)
        
        #_______________________________________________________________________
        #t1 = clock.time()
        #for di in range(0,nd):
            #data_n[:, di] = e2n_di(di, data_e, aux1, mesh.e_i.flatten(), mesh.n_iz) 
        #print(clock.time()-t1)
        
        t1 = clock.time()
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=20)(delayed(e2n_di)(di, data_e, data_area, mesh.e_i.flatten(), mesh.n_iz) for di in range(0,nd))
        data_n = np.vstack(results).transpose()
        print(' --> elapsed time:', clock.time()-t1)
        
    #___________________________________________________________________________
    return(data_n)



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
# ___COMPUTE LIST/ARRAY NODE_IN_ELEM____________________________________________
def compute_nod_in_elem2D(n2dn, e_i, do_arr=False):
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
