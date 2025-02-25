# Patrick Scholz, 23.01.2018
import sys
import os
import numpy                    as np
import time                     as clock
from   scipy.signal             import convolve2d
import matplotlib.pylab         as plt
from   matplotlib.ticker        import MultipleLocator, AutoMinorLocator, ScalarFormatter
from   matplotlib.colors        import ListedColormap
import matplotlib.patheffects   as path_effects
import fnmatch
from   .sub_mesh                import * 
from   .sub_data                import *
from   .sub_plot                import *
from   .sub_utility             import *
from   .sub_colormap            import *
import pandas as pd
import warnings

import dask.array               as da
#xr.set_options(enable_cftimeindex=True)

#
#
#_______________________________________________________________________________
# --> pre-analyse defined tranects
def do_analyse_transects(input_transect     , 
                        mesh                , 
                        edge                , 
                        edge_tri            , 
                        edge_dxdy_l         , 
                        edge_dxdy_r         , 
                        do_rot      = False , 
                        do_info     = False , 
                        Pdxy        = 10.0  
                        ):
    """
    --> pre-analyse defined transects, with respect to which triangles and edges 
        are crossed by the transect line. Create transport path edge to compute 
        modell accurate volume transports.
    
    Parameters:   
    
        :input_transect:    list, to define transects, transects= [[[lon pts], 
                            [lat pts], name, inverse_flowdir], [...]]. With inverse_flowdir
                            = True you can inverse the flowdirection if it doesnt 
                            agree with your in/out flow direction
        
        :mesh:              fesom2 mesh object, with all mesh information
        
        :edge:              np.array([2, nedges]), edge array with indices of participating vertice 
                            points. Best load info from fesom.mesh.diag.nc
        
        :edge_tri:          np.array([2, nedges]), edge triangle array of indices of triangles that 
                            participate in the edge  [2 x nedges]. If edge is boundary
                            edge edge_tri[1, idx] = -1. Best load info from fesom.mesh.diag.nc
                            
        :edge_dxdy_l:       np.array([2, nedges]), array with dx, dy cartesian length of distance from 
                            edge mid points to centroid of left side triangle from 
                            edge. Best load info from fesom.mesh.diag.nc
                            
        :edge_dxdy_r:       np.array([2, nedges]), array with dx, dy cartesian length of distance from 
                            edge mid points to centroid of right side triangle from 
                            edge. Best load info from fesom.mesh.diag.nc  
                            
        :do_rot:            bool, (default=True) assume that the the edge_dxdy_l, 
                            edge_dxdy_r arrays are in the rotated coordinate frame
                            and needed to be rotated into geo coordinates
                            
        :do_info:           bool, (default=False) print info of transect dictionary
        
        :Pdxy:              float, (default=15) buffer lon/lat width of lon/lat box around transect
                            for preselection of cutted edges within this box. On 
                            very coarse meshes if it looks like that certain edges 
                            are not cutted increase this number 

    Return:
    
        :transect_list:     list of transect dictionary 
        
    transect dictionary keys:    
    
    ::
    
        #_______________________________________________________________________
        # arrays that define cross-section 
        transect['Name'         ] = [] # Name of transect
        transect['lon'          ] = [] # transect defining longitude list
        transect['lat'          ] = [] # transect defining latitude list
        transect['ncsi'         ] = [] # running index of number of defined transects
        transect['ncs'          ] = [] # number transect defining points
        transect['Px'           ] = [] # lon points  that define the transect edges 
        transect['Py'           ] = [] # lat points  that define the transect edges 
        transect['e_vec'        ] = [] # unit vector of transect edges
        transect['e_norm'       ] = [] # length of unit vector (length of edge)
        transect['n_vec'        ] = [] # normal vector of transect edges
        transect['alpha'        ] = [] # bearing of transect edge
        
        #_______________________________________________________________________
        # arrays that define the intersection between cross-section and edges
        transect['edge_cut_i'   ] = [] # indice of edges that are cutted by transect
        transect['edge_cut_evec'] = [] # unit vector of those cutted edges
        transect['edge_cut_P'   ] = [] # lon, lat point where transect cutted with edge
        transect['edge_cut_midP'] = [] # mid points of cutted edge
        transect['edge_cut_lint'] = [] # interpolator for cutting points on edge
        transect['edge_cut_ni'  ] = [] # node indices of intersectted edges
        transect['edge_cut_dist'] = [] # distance of cutted edge midpoint from start point of transect
        
        #_______________________________________________________________________
        # arrays to define transport path
        transect['path_xy'      ] = [] # lon/lat coordinates, edge midpoints --> elem centroid --> edge mid points ...
        transect['path_ei'      ] = [] # elem indices
        transect['path_ni'      ] = [] # node indices of elems
        transect['path_cut_ni'  ] = [] # node indices of edges  
        transect['path_dx'      ] = [] # dx of path sections
        transect['path_dy'      ] = [] # dy of path sections
        transect['path_dist'    ] = [] # dy of path sections
        transect['path_nvec_cs' ] = [] # normal vector of transection segment
    
    ____________________________________________________________________________

    """
    transect_list = []
    # loop over transects in list
    for transec_ii in input_transect:
        #_______________________________________________________________________
        if len(transec_ii)==3: 
            transec_lon, transec_lat, transec_name = transec_ii
            transec_flowdir = False
        elif len(transec_ii)==4: 
            transec_lon, transec_lat, transec_name, transec_flowdir = transec_ii
        else:
            raise ValueError('--> trasect definition must be: [ [lon], [lat], transect-name, inverse_flowdir(optional) ] ')
        
        #_______________________________________________________________________
        # allocate dictionary for total cross-section 
        sub_transect = _do_init_transect()
        sub_transect['Name'] = transec_name
        
        #_______________________________________________________________________
        # fix orientation of section 
        auxx, auxy = transec_lon[-1]-transec_lon[0], transec_lat[-1]-transec_lat[0]
        auxn       = np.sqrt(auxx**2+auxy**2)
        auxx, auxy = auxx/auxn, auxy/auxn
        sub_transect['e_vec_tot'    ] = [ auxx, auxy]
        sub_transect['n_vec_tot'    ] = [-auxy, auxx]
        
        alpha      = np.arctan2(auxy,auxx)*180/np.pi
        sub_transect['alpha_tot'    ] = alpha
        if alpha>=-180 and alpha<=-90:
            transec_lon = np.flip(transec_lon)
            transec_lat = np.flip(transec_lat)
            sub_transect['e_vec_tot'] = -sub_transect['e_vec_tot']
            sub_transect['n_vec_tot'] = -sub_transect['n_vec_tot']
        del(auxx, auxy, auxn, alpha)
        
        #_______________________________________________________________________
        sub_transect['lon']  = transec_lon
        sub_transect['lat']  = transec_lat
        sub_transect['ncs']  = len(transec_lon)-1
        
        #_______________________________________________________________________
        # loop over transect points
        for ii in range(0,len(transec_lon)-1):
            #print(' --> subtransect_ii', ii)
            sub_transect['ncsi'].append(ii)
            
            #___________________________________________________________________
            # points defining cross-section line
            sub_transect['Px'].append([transec_lon[ii], transec_lon[ii+1]])
            sub_transect['Py'].append([transec_lat[ii], transec_lat[ii+1]])
            
            #___________________________________________________________________
            # compute unit and normal vector of cross-section line
            sub_transect = _do_calc_csect_vec(sub_transect)
            
            #___________________________________________________________________
            # pre-limit edges based on min/max lon and lat points that form 
            # cruise-line
            #Pdxy = 10.0
            Pxmin, Pxmax = min(sub_transect['Px'][-1]), max(sub_transect['Px'][-1])
            Pymin, Pymax = min(sub_transect['Py'][-1]), max(sub_transect['Py'][-1])
            
            Pdx ,Pdy = Pdxy, Pdxy
            # if section reaches into polar areas than increase the longitudinal width 
            # of the buffer box around the transect
            if Pymin<-70.0 or Pymax>80.0: Pdx=180.0 
            
            Pxmin, Pxmax = Pxmin-Pdx, Pxmax+Pdx
            Pymin, Pymax = Pymin-Pdy, Pymax+Pdy
            idx_edlimit  = np.where( ( mesh.n_x[edge].min(axis=0)>=Pxmin ) &  
                                     ( mesh.n_x[edge].max(axis=0)<=Pxmax ) & 
                                     ( mesh.n_y[edge].min(axis=0)>=Pymin ) &  
                                     ( mesh.n_y[edge].max(axis=0)<=Pymax ) )[0]
            del(Pxmin, Pxmax, Pymin, Pymax, Pdx, Pdy)
            
            #___________________________________________________________________
            # compute which edges are intersected by cross-section line 
            #sub_transect = _do_find_intersected_edges(mesh, sub_transect, edge, idx_edlimit)
            sub_transect = _do_find_intersected_edges_fast(mesh, sub_transect, edge, idx_edlimit)
            
            #___________________________________________________________________
            # sort intersected edges after distance from cross-section start point
            sub_transect = _do_sort_intersected_edges(sub_transect)
            
            #___________________________________________________________________
            # build transport path
            sub_transect = _do_build_path(mesh, sub_transect, edge_tri, edge_dxdy_l, edge_dxdy_r)
            
        #_______________________________________________________________________
        # combine all cross-section sub segments into a single cross-section line
        transect = _do_concat_subtransects(sub_transect)
        del(sub_transect)
        
        #_______________________________________________________________________    
        # insert land pts (nan) when there are 2 onsecutive cutted boundary 
        transect = _do_insert_landpts(transect, edge_tri)
        
        #_______________________________________________________________________    
        # rotate path_dx and path_dy from rot --> geo coordinates
        if do_rot:
            #transect['path_dx'], transect['path_dy'] = vec_r2g(mesh.abg, 
                                                        #mesh.n_x[transect['path_ni']].sum(axis=1)/3.0, 
                                                        #mesh.n_y[transect['path_ni']].sum(axis=1)/3.0,
                                                        #transect['path_dx'], transect['path_dy'])
            
            transect['path_dx'], transect['path_dy'] = vec_r2g(mesh.abg, 
                                                        mesh.n_x[transect['path_cut_ni']].sum(axis=1)/2.0, 
                                                        mesh.n_y[transect['path_cut_ni']].sum(axis=1)/2.0,
                                                        transect['path_dx'], transect['path_dy'])
            
        #_______________________________________________________________________    
        # rotate path_dx and path_dy from rot --> geo coordinates
        if transec_flowdir:
             transect['path_dx'], transect['path_dy'] =  -transect['path_dx'], -transect['path_dy']
             transect['path_nvec_cs'] =  -transect['path_nvec_cs']
             
        #_______________________________________________________________________ 
        # buld distance from start point array [km]
        transect = _do_compute_distance_from_startpoint(transect)
        
        #_______________________________________________________________________
        if do_info:
            print(' --> Name:', transect['Name'])
            print('edge_cut_i    =', transect['edge_cut_i'   ].shape)
            print('edge_cut_evec =', transect['edge_cut_evec'].shape)
            print('edge_cut_P    =', transect['edge_cut_P'   ].shape)
            print('edge_cut_midP =', transect['edge_cut_midP'].shape)
            print('edge_cut_lint =', transect['edge_cut_lint'].shape)
            print('edge_cut_ni   =', transect['edge_cut_ni'  ].shape)
            print('path_ei       =', transect['path_ei'      ].shape)
            print('path_ni       =', transect['path_ni'      ].shape)
            print('path_cut_ni   =', transect['path_cut_ni'  ].shape)
            print('path_dx       =', transect['path_dx'      ].shape)
            print('path_dy       =', transect['path_dy'      ].shape)
            print('path_nvec_cs  =', transect['path_nvec_cs' ].shape)
            print('path_xy       =', transect['path_xy'      ].shape)
            print('path_dist     =', transect['path_dist'    ].shape) 
            print('')
        #_______________________________________________________________________  
        #if len(input_transect)>1:
        transect_list.append(transect)
        #else:
            #transect_list = transect
        
    #___________________________________________________________________________
    return(transect_list)



#
#
#_______________________________________________________________________________
def _do_init_transect():
    transect = dict()
    #___________________________________________________________________________
    # arrays that define cross-section 
    transect['Name'         ] = []
    transect['lon'          ] = []
    transect['lat'          ] = []
    transect['ncsi'         ] = []
    transect['ncs'          ] = []
    transect['Px'           ] = []
    transect['Py'           ] = []
    transect['e_vec'        ] = []
    transect['e_norm'       ] = []
    transect['n_vec'        ] = []
    transect['alpha'        ] = []
    transect['e_vec_tot'    ] = []
    transect['n_vec_tot'    ] = []
    transect['alpha_tot'    ] = []
    
    #___________________________________________________________________________
    # arrays that define the intersection between cross-section and edges
    transect['edge_cut_i'   ] = []
    transect['edge_cut_evec'] = []
    transect['edge_cut_P'   ] = []
    transect['edge_cut_midP'] = []
    transect['edge_cut_lint'] = []
    transect['edge_cut_ni'  ] = []
    transect['edge_cut_dist'] = []
    
    #___________________________________________________________________________
    # arrays to define transport path
    transect['path_xy'      ] = [] # lon/lat coordinates, edge midpoints --> elem centroid --> edge mid points ...
    transect['path_ei'      ] = [] # elem indices
    transect['path_ni'      ] = [] # node indices of elems
    transect['path_cut_ni'  ] = [] # node indices of elems 
    transect['path_dx'      ] = [] # dx of path sections
    transect['path_dy'      ] = [] # dy of path sections
    transect['path_dist'    ] = [] # dy of path sections
    transect['path_evec_cs' ] = [] # normal vector of transection segment
    transect['path_nvec_cs' ] = [] # normal vector of transection segment
    
    #___________________________________________________________________________
    return(transect)



#
#
#_______________________________________________________________________________
def _do_concat_subtransects(sub_transect):
    #___________________________________________________________________________
    # combine all cross-ection sub segments into a single cross-section
    transect = sub_transect.copy()
    
    #___________________________________________________________________________
    transect['edge_cut_i'   ] = sub_transect['edge_cut_i'   ][0]
    transect['edge_cut_evec'] = sub_transect['edge_cut_evec'][0]
    transect['edge_cut_P'   ] = sub_transect['edge_cut_P'   ][0]
    transect['edge_cut_midP'] = sub_transect['edge_cut_midP'][0]
    transect['edge_cut_lint'] = sub_transect['edge_cut_lint'][0]
    transect['edge_cut_ni'  ] = sub_transect['edge_cut_ni'  ][0]
    transect['path_xy'      ] = sub_transect['path_xy'      ][0]
    transect['path_ei'      ] = sub_transect['path_ei'      ][0]
    transect['path_ni'      ] = sub_transect['path_ni'      ][0]
    transect['path_cut_ni'  ] = sub_transect['path_cut_ni'  ][0]
    transect['path_dx'      ] = sub_transect['path_dx'      ][0]
    transect['path_dy'      ] = sub_transect['path_dy'      ][0]
    transect['path_evec_cs' ] = sub_transect['path_evec_cs' ][0]
    transect['path_nvec_cs' ] = sub_transect['path_nvec_cs' ][0]
    
    #___________________________________________________________________________
    if sub_transect['ncs'] > 1:    
        for ii in range(1,sub_transect['ncs']):
            transect['edge_cut_i'   ] = np.hstack((transect['edge_cut_i'   ], sub_transect['edge_cut_i'   ][ii]))
            transect['edge_cut_evec'] = np.vstack((transect['edge_cut_evec'], sub_transect['edge_cut_evec'][ii]))
            transect['edge_cut_P'   ] = np.vstack((transect['edge_cut_P'   ], sub_transect['edge_cut_P'   ][ii]))
            transect['edge_cut_midP'] = np.vstack((transect['edge_cut_midP'], sub_transect['edge_cut_midP'][ii]))
            transect['edge_cut_lint'] = np.hstack((transect['edge_cut_lint'], sub_transect['edge_cut_lint'][ii]))
            transect['edge_cut_ni'  ] = np.vstack((transect['edge_cut_ni'  ], sub_transect['edge_cut_ni'  ][ii]))
            transect['path_ei'      ] = np.hstack((transect['path_ei'      ], sub_transect['path_ei'      ][ii]))
            transect['path_ni'      ] = np.vstack((transect['path_ni'      ], sub_transect['path_ni'      ][ii]))
            transect['path_cut_ni'  ] = np.vstack((transect['path_cut_ni'  ], sub_transect['path_cut_ni'  ][ii]))
            transect['path_dx'      ] = np.hstack((transect['path_dx'      ], sub_transect['path_dx'      ][ii]))
            transect['path_dy'      ] = np.hstack((transect['path_dy'      ], sub_transect['path_dy'      ][ii]))
            transect['path_nvec_cs' ] = np.vstack((transect['path_nvec_cs' ], sub_transect['path_nvec_cs' ][ii]))
            transect['path_evec_cs' ] = np.vstack((transect['path_evec_cs' ], sub_transect['path_evec_cs' ][ii]))
            
            # if there are multiple subsection to one transect than the last point
            # of the previous subsection and the first point of the actual subsection 
            # are identical so when they concatenated they are dublicated, therefor 
            # the first point of a subsecuent subsection must be kicked out 
            transect['path_xy'      ] = np.vstack((transect['path_xy'      ], sub_transect['path_xy'      ][ii][1:]))
            #transect['path_xy'      ] = np.vstack((transect['path_xy'      ], sub_transect['path_xy'      ][ii]))
            
    del(sub_transect)
    
    #___________________________________________________________________________
    return(transect)



#
#
#_______________________________________________________________________________
def _do_calc_csect_vec(transect):
    # unit and norm vector of cross-section line
    evec  = np.array([transect['Px'][-1][1]-transect['Px'][-1][0], 
                      transect['Py'][-1][1]-transect['Py'][-1][0]])
    enorm = np.sqrt((evec[0]**2+evec[1]**2))
    transect['e_vec' ].append(evec/enorm)
    transect['e_norm'].append(enorm)
    
    # normal vector --> norm vector definition (-dy,dx) --> shows to the 
    # left of the e_vec
    # make sure the normal direction of section is fixed from:
    # NW --> SE
    #  W --> E
    # SW --> NE´
    #  S --> N 
    if -evec[1]<0 or evec[0]<0:
        transect['n_vec' ].append(np.array([ evec[1], -evec[0]])/enorm)  
    else:     
        transect['n_vec' ].append(np.array([-evec[1], evec[0]])/enorm)  
    del(evec, enorm)
    
    #___________________________________________________________________________
    return(transect)



#
#
#_______________________________________________________________________________
#                            3
#                            o
#                           /^
#                          /  \ vec_c
#             C1          /    \           C2
#              o---------/------\--------->>o  obj vec_C
#                  vec_b/        \
#                      v          \
#                     o----------->o
#                    1    vec_a    2
#
#          CALC: vec_C : (C1x,C1y) + rc (C2x-C1x,C2y-C1y)
#                vec_a : (P1x,P1y) + ra (P2x-P1x,P2y-P1y)
#                vec_b : ...
#                vec_c : ...
#               - cross-section of vec_C and vec_a --> Vectoralgebra
#
#                (C1x) + rc (C2x-C1x) =  (P1x) + ra (P2x-P1x)
#                (C1y) + rc (C2y-C1y) =  (P1y) + ra (P2y-P1y)
#
#                / (P2x-P1x)  -(C2x-C1x) \   / ra \   / C1x-P1x \
#                \ (P2y-P1y)  -(C2y-C1y) / * \ rc / = \ C1y-P1y /
#
#                            A             *   X    =      P    
#
#               - same for cross-section of vec_C and vec_b & vec_C
#                 and vec_c
#                 A[2x2] * X[2x1] = P[2x1] --> solve for X = inv(A)*P
#                 X[0,0] --> norm of vectors until the points where its 
#                 intersected. If |X[0,0]| > |vec_a| than edge is not intersected
def _do_find_intersected_edges(mesh, transect, edge, idx_ed):
    #___________________________________________________________________________
    transect['edge_cut_i'   ].append(list())
    transect['edge_cut_evec'].append(list())
    transect['edge_cut_P'   ].append(list())
    transect['edge_cut_midP'].append(list())
    transect['edge_cut_lint'].append(list())
    transect['edge_cut_ni'  ].append(list())
    
    #___________________________________________________________________________
    A, P    = np.zeros((2,2)), np.zeros((2,))
    A[0, 1] = -transect['e_vec'][-1][0]
    A[1, 1] = -transect['e_vec'][-1][1]
    
    #___________________________________________________________________________
    # loop over indices of edges-->use already lon,lat limited edge indices 
    #plt.figure()
        
    for edi in idx_ed:
        
        A[0, 0] = mesh.n_x[edge[1, edi]]-mesh.n_x[edge[0, edi]]
        A[1, 0] = mesh.n_y[edge[1, edi]]-mesh.n_y[edge[0, edi]]
        normA0  = np.sqrt(A[0, 0]**2 + A[1, 0]**2)
        A[:, 0] = A[:, 0]/normA0
        P[0]    = transect['Px'][-1][0]-mesh.n_x[edge[0, edi]]
        P[1]    = transect['Py'][-1][0]-mesh.n_y[edge[0, edi]]
        
        # solve linear equation system: A * X = P --> solve for X[0]
        div     = (A[0,0]*A[1,1]-A[0,1]*A[1,0])
        X0      = (P[0]*A[1,1]-P[1]*A[0,1])/ div
        X1      = (P[0]*A[1,0]-P[1]*A[0,0])/-div
        
        # determine if edge is intersected by crossection line
        if ((X0>=0.0-np.finfo(np.float32).eps) & (X0<=normA0                +np.finfo(np.float32).eps) &
            (X1>=0.0-np.finfo(np.float32).eps) & (X1<=transect['e_norm'][-1]+np.finfo(np.float32).eps) ):
            # indice of cutted edge
            transect['edge_cut_i'   ][-1].append(edi)
            
            # evec of cutted edge 
            transect['edge_cut_evec'][-1].append([A[0, 0], A[1, 0]])
            
            # cutting point on edge
            transect['edge_cut_P'   ][-1].append([mesh.n_x[edge[0,edi]]+A[0,0]*X0, 
                                                  mesh.n_y[edge[0,edi]]+A[1,0]*X0])
            
            # mid point on edge
            transect['edge_cut_midP'][-1].append([mesh.n_x[edge[0,edi]]+A[0,0]*normA0/2.0, 
                                                  mesh.n_y[edge[0,edi]]+A[1,0]*normA0/2.0])
            
            # interpolator for cutting points on edge Vcut= V1+(V2-V1)*(rc-r1)/(r2-r1)=V1+(V2-V1)*lint
            transect['edge_cut_lint'][-1].append(X0/normA0)
            
            # node indices of intersectted edges 
            transect['edge_cut_ni'  ][-1].append(edge[:,edi])
    
    #___________________________________________________________________________
    # transform from list --> np.array
    transect['edge_cut_i'   ][-1] = np.asarray(transect['edge_cut_i'   ][-1])
    transect['edge_cut_evec'][-1] = np.asarray(transect['edge_cut_evec'][-1])
    transect['edge_cut_P'   ][-1] = np.asarray(transect['edge_cut_P'   ][-1])
    transect['edge_cut_midP'][-1] = np.asarray(transect['edge_cut_midP'][-1])
    transect['edge_cut_lint'][-1] = np.asarray(transect['edge_cut_lint'][-1])
    transect['edge_cut_ni'  ][-1] = np.asarray(transect['edge_cut_ni'  ][-1])
    del(A, P, X0, X1, normA0)
    
    #___________________________________________________________________________
    return(transect)    

def _do_find_intersected_edges_fast(mesh, transect, edge, idx_ed):
    # Initialize lists
    for key in ['edge_cut_i', 'edge_cut_evec', 'edge_cut_P', 'edge_cut_midP', 'edge_cut_lint', 'edge_cut_ni']:
        transect[key].append([])

    # Precompute values
    eps     = np.finfo(np.float32).eps
    e_vec   = transect['e_vec'][-1]
    e_norm  = transect['e_norm'][-1]

    # Compute edge vectors in one go
    edge_x  = mesh.n_x[edge[1, idx_ed]] - mesh.n_x[edge[0, idx_ed]]
    edge_y  = mesh.n_y[edge[1, idx_ed]] - mesh.n_y[edge[0, idx_ed]]
    normA0  = np.sqrt(edge_x**2 + edge_y**2)
    normA0_inv = 1 / normA0

    # Normalize edge vectors
    edge_x *= normA0_inv
    edge_y *= normA0_inv

    # Compute P for all edges
    P_x     = transect['Px'][-1][0] - mesh.n_x[edge[0, idx_ed]]
    P_y     = transect['Py'][-1][0] - mesh.n_y[edge[0, idx_ed]]

    # Solve linear equation system
    div     = edge_x * -e_vec[1] - edge_y * -e_vec[0]
    # avoid division by zero div
    div = np.where(np.abs(div) > np.finfo(np.float32).eps, div, np.nan)
    
    
    X0      = (P_x * -e_vec[1] - P_y * -e_vec[0]) /  div
    X1      = (P_x *  edge_y   - P_y *  edge_x  ) / -div
    del(e_vec)

    # Boolean mask for valid intersections
    mask    = (~np.isnan(X0) & (X0 >= -eps) & (X0 <= normA0 + eps) &
                               (X1 >= -eps) & (X1 <= e_norm + eps))

    # Select only valid intersections
    valid_idx    = idx_ed[mask]
    X0_valid     = X0[mask]
    normA0_valid = normA0[mask]
    edge_x_valid = edge_x[mask]
    edge_y_valid = edge_y[mask]
    del(P_x, P_y, div, X0, X1, edge_x, edge_y, normA0, e_norm)

    # Store results
    transect['edge_cut_i'   ][-1] = valid_idx
    transect['edge_cut_evec'][-1] = np.column_stack((edge_x_valid, edge_y_valid))
    transect['edge_cut_P'   ][-1] = np.column_stack((
        mesh.n_x[edge[0, valid_idx]] + edge_x_valid * X0_valid,
        mesh.n_y[edge[0, valid_idx]] + edge_y_valid * X0_valid
    ))
    transect['edge_cut_midP'][-1] = np.column_stack((
        mesh.n_x[edge[0, valid_idx]] + edge_x_valid * normA0_valid / 2.0,
        mesh.n_y[edge[0, valid_idx]] + edge_y_valid * normA0_valid / 2.0
    ))
    transect['edge_cut_lint'][-1] = X0_valid / normA0_valid
    transect['edge_cut_ni'  ][-1] = edge[:, valid_idx].T
    
    #___________________________________________________________________________
    return transect



#
#
#_______________________________________________________________________________
# compute distance of cutting points from cross-section start point --> than 
# sort cutted edges after this distance
def _do_sort_intersected_edges(transect):
    #___________________________________________________________________________
    idx_sort = np.argsort((transect['edge_cut_P'][-1][:,0]-transect['Px'][-1][0])**2 +
                          (transect['edge_cut_P'][-1][:,1]-transect['Py'][-1][0])**2)
    
    #___________________________________________________________________________
    transect['edge_cut_i'   ][-1] = transect['edge_cut_i'   ][-1][idx_sort  ]
    transect['edge_cut_evec'][-1] = transect['edge_cut_evec'][-1][idx_sort,:]
    transect['edge_cut_P'   ][-1] = transect['edge_cut_P'   ][-1][idx_sort,:]
    transect['edge_cut_midP'][-1] = transect['edge_cut_midP'][-1][idx_sort,:]
    transect['edge_cut_lint'][-1] = transect['edge_cut_lint'][-1][idx_sort  ]
    transect['edge_cut_ni'  ][-1] = transect['edge_cut_ni'  ][-1][idx_sort,:]
    del(idx_sort)
    
    #___________________________________________________________________________
    return(transect)
    


#
#
#_______________________________________________________________________________
def _do_build_path(mesh, transect, edge_tri, edge_dxdy_l, edge_dxdy_r):

    #___________________________________________________________________________
    # allocate path arrays
    path_xy = list()
    path_ei = list()
    path_ni = list()
    path_dx = list()
    path_dy = list()
    path_cut_ni = list()
    
    #_______________________________________________________________________
    # compute angle bearing of cross-section line: 
    #                            N
    #                            ^  
    #          alpha=[90...180]  |   alpha=[0...90]  
    #                            |
    #                    W<------|------> E
    #                            |
    #        alpha=[-90...-180]  |   alpha=[0...-90]  
    #                            v
    #                            S
    #alpha = -np.arctan2(transect['e_vec'][-1][1], transect['e_vec'][-1][0])
    alpha = np.arctan2(transect['e_vec'][-1][1], transect['e_vec'][-1][0])
    transect['alpha'].append(alpha*180/np.pi)
    #print('> alpha:', alpha*180/np.pi)
    #___________________________________________________________________________
    # loop over intersected edges 
    nced = transect['edge_cut_i'][-1].size
    for edi in range(0,nced):
        #print(' --> edgecut_ii', edi)
        
        #_______________________________________________________________________
        # --> rotate edge with bearing angle -alpha
        # determine if edge shows to the left or to the right with 
        # respect to cross-section line, if theta<0, edge shows to the right
        # in this case use [L]eft triangle, if theta>0 edge shows to the left
        # in this case use right triangle (its that the "downstream" triangle
        # with respect to cross-section direction)
        # --> rotate edge by minus alpha angle
        # use Rotationmatrixs R_alpha = | cos(-alpha) -sin(-alpha) |
        #                               | sin(-alpha)  cos(-alpha) |
        #   
        # R_alpha * |ed_x | = | ed_x*cos(-alpha)-ed_y*sin(-alpha) | 
        #           |ed_y |   | ed_x*sin(-alpha)+ed_y*cos(-alpha) | 
        auxx = transect['edge_cut_evec'][-1][edi,0]*np.cos(-alpha)-transect['edge_cut_evec'][-1][edi,1]*np.sin(-alpha)
        auxy = transect['edge_cut_evec'][-1][edi,0]*np.sin(-alpha)+transect['edge_cut_evec'][-1][edi,1]*np.cos(-alpha)
        theta= np.arctan2(auxy,auxx)
        #print('--> theta:', theta*180/np.pi)
        del(auxx, auxy)
                
        # indices of [L]eft and [R]ight triangle with respect to the edge
        edge_elem  = edge_tri[:, transect['edge_cut_i'][-1][edi]]
        
        #_______________________________________________________________________
        # add upsection element to path if it exist --> path_xy coodinate points 
        # for the element always come from downsection triangle 
        path_xy, path_ei, path_ni, path_dx, path_dy, path_cut_ni = __add_upsection_elem2path(mesh, transect, edi, nced, theta, 
                                                  edge_elem, edge_dxdy_l, edge_dxdy_r,
                                                  path_xy, path_ei, path_ni, path_dx, path_dy, path_cut_ni)
        
        # add edge mid point of cutted edge
        path_xy.append(transect['edge_cut_midP'][-1][edi])
        
        # add downsection element to path if it exist
        path_xy, path_ei, path_ni, path_dx, path_dy, path_cut_ni = __add_downsection_elem2path(mesh, transect, edi, nced, theta, 
                                                  edge_elem, edge_dxdy_l, edge_dxdy_r,
                                                  path_xy, path_ei, path_ni, path_dx, path_dy, path_cut_ni)
        
    #___________________________________________________________________________    
    # reformulate from list --> np.array
    transect['path_xy'].append(np.asarray(path_xy))
    transect['path_ei'].append(np.asarray(path_ei))
    transect['path_ni'].append(np.asarray(path_ni))
    transect['path_dx'].append(np.asarray(path_dx))
    transect['path_dy'].append(np.asarray(path_dy))
    transect['path_cut_ni'].append(np.asarray(path_cut_ni))
    
    aux = np.ones((transect['path_dx'][-1].size,2))
    aux[:,0], aux[:,1] = transect['n_vec'][-1][0], transect['n_vec'][-1][1]
    transect['path_nvec_cs'].append(aux)
    
    aux = np.ones((transect['path_dx'][-1].size,2))
    aux[:,0], aux[:,1] = transect['e_vec'][-1][0], transect['e_vec'][-1][1]
    transect['path_evec_cs'].append(aux)
    del(aux)
    
    # !!! Make sure positive Transport is defined S-->N and W-->E
    # --> Preliminary --> not 100% sure its universal
    #rad = np.pi/180
    #if (-alpha/rad>=-180 and -alpha/rad<=-90 ) or (-alpha/rad>90 and -alpha/rad<=180 ):
    
    # make sure the normal direction of section is fixed from:
    # NW --> SE
    #  W --> E
    # SW --> NE´
    #  S --> N 
    #if transect['n_vec'][-1][0]<0 or transect['n_vec'][-1][1]<0:
    if transect['n_vec_tot'][0]<0 or transect['n_vec_tot'][1]<0:
        transect['path_dx'][-1] = -transect['path_dx'][-1]
        transect['path_dy'][-1] = -transect['path_dy'][-1]
        
    del(path_xy, path_ei, path_ni, path_dx, path_dy, edge_elem)

    #___________________________________________________________________________
    return(transect)

#
#
#_______________________________________________________________________________
def __add_downsection_elem2path(mesh, transect, edi, nced, theta, edge_elem, edge_dxdy_l, edge_dxdy_r,
                                path_xy, path_ei, path_ni, path_dx, path_dy, path_cut_ni):
    
    #___________________________________________________________________________
    # if theta>0 edge shows to the left
    #                          ^ Section
    #                  [R]ight | Triangle downsection triangle
    #                          |
    #           2o<------------+--------------o1   edge
    #                          | 
    #                   [L]eft | Triangle 
    if (theta>=0):
        # right triangle
        if edge_elem[1]>=0:
            # take care if there are periodic boundaries
            e_xR = mesh.n_x[mesh.e_i[edge_elem[1],:]]
            if np.max(e_xR)-np.min(e_xR)>180:
                if np.sum(e_xR>0) > np.sum(e_xR<0): e_xR[e_xL<0] = e_xR[e_xL<0]+360.0
                else                              : e_xR[e_xL>0] = e_xR[e_xL>0]-360.0
            e_xR, e_yR = np.sum(e_xR)/3.0, np.sum(mesh.n_y[mesh.e_i[edge_elem[1],:]])/3.0
            path_xy.append(np.array([e_xR, e_yR]))
            del(e_xR, e_yR)
            
            path_ei.append(edge_elem[1])
            path_ni.append(mesh.e_i[edge_elem[1], :])
            path_dx.append(edge_dxdy_r[0, transect['edge_cut_i'][-1][edi]])
            path_dy.append(edge_dxdy_r[1, transect['edge_cut_i'][-1][edi]])
            
            # needed to interpolate temperature to edge mid point for heat flux 
            # computation
            path_cut_ni.append(transect['edge_cut_ni'][-1][edi,:])
        
        # edge is boundary edge right triangle does not exist--> put dummy values
        # instead of centroid position (transect['edge_cut_midP'][-1][edi])  
        else:
            #print(' >-)))°> .°oO: downsection', edi)
            path_xy.append(transect['edge_cut_midP'][-1][edi])  #(***)
            path_ei.append(-1)                     #--> -1 is here dummy index
            path_ni.append(np.array([-1, -1, -1])) #--> -1 is here dummy index
            path_dx.append(np.nan)
            path_dy.append(np.nan)
            
            # needed to interpolate temperature to edge mid point for heat flux 
            # computation
            path_cut_ni.append(np.array([-1, -1]))
    
    #___________________________________________________________________________
    # if theta<0 edge shows to the right
    #                          ^ Section
    #                   [L]eft | Triangle downsection triangle 
    #                          |
    #           1o-------------+------------->o2   edge
    #                          | 
    #                 [R]right | Triangle 
    else:
        # left triangle
        e_xL = mesh.n_x[mesh.e_i[edge_elem[0],:]]
        if np.max(e_xL)-np.min(e_xL)>180:
            if np.sum(e_xL>0) > np.sum(e_xL<0): e_xL[e_xL<0] = e_xL[e_xL<0]+360.0
            else                              : e_xL[e_xL>0] = e_xL[e_xL>0]-360.0
        e_xL, e_yL = np.sum(e_xL)/3.0, np.sum(mesh.n_y[mesh.e_i[edge_elem[0],:]])/3.0
        path_xy.append(np.array([e_xL, e_yL]))
        del(e_xL, e_yL)
        
        path_ei.append(edge_elem[0])
        path_ni.append(mesh.e_i[edge_elem[0], :])
        path_dx.append(edge_dxdy_l[0, transect['edge_cut_i'][-1][edi]])
        path_dy.append(edge_dxdy_l[1, transect['edge_cut_i'][-1][edi]])
        
        # needed to interpolate temperature to edge mid point for heat flux 
        # computation
        path_cut_ni.append(transect['edge_cut_ni'][-1][edi,:])
        
    #___________________________________________________________________________
    return(path_xy, path_ei, path_ni, path_dx, path_dy, path_cut_ni)
#
#
#_______________________________________________________________________________
def __add_upsection_elem2path(mesh, transect, edi, nced, theta, edge_elem, edge_dxdy_l, edge_dxdy_r,
                            path_xy, path_ei, path_ni, path_dx, path_dy, path_cut_ni):
    #___________________________________________________________________________
    # if theta>0 edge shows to the left
    #                          ^ Section
    #                  [R]ight | Triangle 
    #                          |
    #           2o<------------+--------------o1   edge
    #                          | 
    #                   [L]eft | Triangle upsection triangle
    if (theta>=0):
        if edi==0:
            # take care if there are periodic boundaries
            e_xL = mesh.n_x[mesh.e_i[edge_elem[0],:]]
            if np.max(e_xL)-np.min(e_xL)>180:
                if np.sum(e_xL>0) > np.sum(e_xL<0): e_xL[e_xL<0] = e_xL[e_xL<0]+360.0
                else                              : e_xL[e_xL>0] = e_xL[e_xL>0]-360.0
            e_xL, e_yL = np.sum(e_xL)/3.0, np.sum(mesh.n_y[mesh.e_i[edge_elem[0],:]])/3.0
            path_xy.append(np.array([e_xL, e_yL]))
            del(e_xL, e_yL)
            
        # left triangle 
        path_ei.append(edge_elem[0])
        path_ni.append(mesh.e_i[edge_elem[0], :])
        
        # need to flip edge_dxdy vector through minus sign since its opposite 
        # directed to the transect direction --> upsection
        path_dx.append(-edge_dxdy_l[0, transect['edge_cut_i'][-1][edi]])
        path_dy.append(-edge_dxdy_l[1, transect['edge_cut_i'][-1][edi]])
        
        # needed to interpolate temperature to edge mid point for heat flux 
        # computation
        path_cut_ni.append(transect['edge_cut_ni'][-1][edi,:])
        
    #___________________________________________________________________________
    # if theta<0 edge shows to the right
    #                          ^ Section
    #                   [L]eft | Triangle
    #                          |
    #           1o-------------+------------->o2   edge
    #                          | 
    #                 [R]right | Triangle upsection triangle
    else:
        # right triangle
        if edge_elem[1]>=0:
            if edi==0:
                # take care if there are periodic boundaries
                e_xR = mesh.n_x[mesh.e_i[edge_elem[1],:]]
                if np.max(e_xR)-np.min(e_xR)>180:
                    if np.sum(e_xR>0) > np.sum(e_xR<0): e_xR[e_xL<0] = e_xR[e_xL<0]+360.0
                    else                              : e_xR[e_xL>0] = e_xR[e_xL>0]-360.0
                e_xR, e_yR = np.sum(e_xR)/3.0, np.sum(mesh.n_y[mesh.e_i[edge_elem[1],:]])/3.0
                path_xy.append(np.array([e_xR, e_yR]))
                del(e_xR, e_yR)
                
            path_ei.append(edge_elem[1])
            path_ni.append(mesh.e_i[edge_elem[1], :])
            
            # need to flip edge_dxdy vector through minus sign since its opposite 
            # directed to the transect direction --> upsection
            path_dx.append(-edge_dxdy_r[0, transect['edge_cut_i'][-1][edi]])
            path_dy.append(-edge_dxdy_r[1, transect['edge_cut_i'][-1][edi]])
            
            # needed to interpolate temperature to edge mid point for heat flux 
            # computation
            path_cut_ni.append(transect['edge_cut_ni'][-1][edi,:])
            
        # edge is boundary edge right triangle does not exist--> put dummy values
        # instead of centroid position (transect['edge_cut_midP'][-1][edi])  
        else:
            # enter additional point close to the boundary with additional 
            # ei, ni, dx and dy --> ensures that transect is porperly plotted
            # when it cutts over land 
            #print(' >-))))°> .°oO: upsection', edi)
            path_xy.append(transect['edge_cut_midP'][-1][edi])  #(***)
            path_ei.append(-1)                                  #(***) 
            path_ni.append(np.array([-1, -1, -1]))              #(***)
            path_dx.append(np.nan)
            path_dy.append(np.nan)
            
            # needed to interpolate temperature to edge mid point for heat flux 
            # computation
            path_cut_ni.append(np.array([-1, -1]))
            
            if edi!=0 and edi!=nced-1:
                # in case of a single transect
                path_ei.append(-1)                     
                path_ni.append(np.array([-1, -1, -1])) 
                path_dx.append(np.nan)
                path_dy.append(np.nan)
                path_cut_ni.append(np.array([-1, -1]))
            elif transect['ncsi'][-1]!=0 and edi==0:        
                # in case of transect consist of subtransects
                path_xy.append(transect['edge_cut_midP'][-1][edi])  #(***)
                path_ei.append(-1)                     
                path_ni.append(np.array([-1, -1, -1])) 
                path_dx.append(np.nan)
                path_dy.append(np.nan)
                path_cut_ni.append(np.array([-1, -1]))
            elif transect['ncsi'][-1]!=transect['ncs'] and edi==nced-1:   
                # in case of transect consist of subtransects
                path_xy.append(transect['edge_cut_midP'][-1][edi])  #(***)
                path_ei.append(-1)                     
                path_ni.append(np.array([-1, -1, -1])) 
                path_dx.append(np.nan)
                path_dy.append(np.nan)
                path_cut_ni.append(np.array([-1, -1]))
                
    #___________________________________________________________________________
    return(path_xy, path_ei, path_ni, path_dx, path_dy, path_cut_ni)



#
#
#_______________________________________________________________________________   
# insert land pts (nan) when there are 2 consecutive cutted boundary 
def _do_insert_landpts(transect, edge_tri):
    # edges edge_tri[1,:] = -1
    edge_cut_ei = edge_tri[1, transect['edge_cut_i']]
    
    # search if there are two consecutive boundary edges 
    idx_bdedge = np.array(edge_cut_ei==-1).astype(np.int8())
    idx_bdedge = idx_bdedge[:-1]+idx_bdedge[1:]
    if np.any(idx_bdedge==2):
        transect_bnd = _do_init_transect()
        list2update  = list(transect.keys())
        for entry in fnmatch.filter(list2update,'edge_cut_*'): list2update.remove(entry)
        # copy everything except edges_cut_* --> here still must insert boundary cut
        for entry in list2update: transect_bnd[entry] = transect[entry]
        
        found_bnd=False
        ni_s,ni_e = 0, 0
        for ni in range(edge_cut_ei.size-1):
            # found transect that cuts over land 
            if edge_cut_ei[ni]==-1 and edge_cut_ei[ni+1]==-1:
                ni_e = ni+1
                if not found_bnd:
                    transect_bnd['edge_cut_i'   ] = np.hstack((transect['edge_cut_i'   ][ni_s:ni_e  ], np.ones((2,), dtype=np.int32)*-1 ))
                    transect_bnd['edge_cut_evec'] = np.vstack((transect['edge_cut_evec'][ni_s:ni_e,:], np.ones((2,2))*np.nan))
                    transect_bnd['edge_cut_P'   ] = np.vstack((transect['edge_cut_P'   ][ni_s:ni_e,:], transect['edge_cut_P'   ][[ni,ni+1],:]))
                    transect_bnd['edge_cut_midP'] = np.vstack((transect['edge_cut_midP'][ni_s:ni_e,:], transect['edge_cut_midP'][[ni,ni+1],:]))
                    transect_bnd['edge_cut_lint'] = np.hstack((transect['edge_cut_lint'][ni_s:ni_e  ], np.ones((2,))*np.nan ))
                    transect_bnd['edge_cut_ni'  ] = np.vstack((transect['edge_cut_ni'  ][ni_s:ni_e,:], np.ones((2,2), dtype=np.int32)*-1))
                else:
                    transect_bnd['edge_cut_i'   ] = np.hstack((transect_bnd['edge_cut_i'   ], transect['edge_cut_i'   ][ni_s:ni_e  ],np.ones((2,), dtype=np.int32)*-1 ))
                    transect_bnd['edge_cut_evec'] = np.vstack((transect_bnd['edge_cut_evec'], transect['edge_cut_evec'][ni_s:ni_e,:],np.ones((2,2))*np.nan))
                    transect_bnd['edge_cut_P'   ] = np.vstack((transect_bnd['edge_cut_P'   ], transect['edge_cut_P'   ][ni_s:ni_e,:],transect['edge_cut_P'   ][[ni,ni+1],:]))
                    transect_bnd['edge_cut_midP'] = np.vstack((transect_bnd['edge_cut_midP'], transect['edge_cut_midP'][ni_s:ni_e,:],transect['edge_cut_midP'][[ni,ni+1],:]))
                    transect_bnd['edge_cut_lint'] = np.hstack((transect_bnd['edge_cut_lint'], transect['edge_cut_lint'][ni_s:ni_e  ],np.ones((2,))*np.nan ))
                    transect_bnd['edge_cut_ni'  ] = np.vstack((transect_bnd['edge_cut_ni'  ], transect['edge_cut_ni'  ][ni_s:ni_e,:],np.ones((2,2), dtype=np.int32)*-1))
                ni_s = ni+1
                found_bnd=True
        transect_bnd['edge_cut_i'   ] = np.hstack((transect_bnd['edge_cut_i'   ], transect['edge_cut_i'   ][ni_s:  ]))
        transect_bnd['edge_cut_evec'] = np.vstack((transect_bnd['edge_cut_evec'], transect['edge_cut_evec'][ni_s:,:]))
        transect_bnd['edge_cut_P'   ] = np.vstack((transect_bnd['edge_cut_P'   ], transect['edge_cut_P'   ][ni_s:,:]))
        transect_bnd['edge_cut_midP'] = np.vstack((transect_bnd['edge_cut_midP'], transect['edge_cut_midP'][ni_s:,:]))
        transect_bnd['edge_cut_lint'] = np.hstack((transect_bnd['edge_cut_lint'], transect['edge_cut_lint'][ni_s:  ]))
        transect_bnd['edge_cut_ni'  ] = np.vstack((transect_bnd['edge_cut_ni'  ], transect['edge_cut_ni'  ][ni_s:,:]))
        transect = transect_bnd.copy()
    
    #___________________________________________________________________________
    return(transect)



#
#
#_______________________________________________________________________________    
def _do_compute_distance_from_startpoint(transect):
    # build distance from start point for transport path [km]
    Rearth = 6367.5  # [km]
    x,y,z  = grid_cart3d(transect['path_xy'][:,0], transect['path_xy'][:,1], is_deg=True)
    dist   = x[:-1]*x[1:] + y[:-1]*y[1:] + z[:-1]*z[1:]
    # avoid nan's in arccos from numerics
    dist[dist>=1.0] = 1.0
    dist   = np.arccos(dist)*Rearth
    transect['path_dist'] = dist.cumsum()   
    transect['path_dist'] = transect['path_dist']-transect['path_dist'][0]
    del(dist, x, y, z)
    
    # build distance from start point for edge cut mid points [km]
    x,y,z  = grid_cart3d(transect['edge_cut_midP'][:,0], transect['edge_cut_midP'][:,1], is_deg=True)
    dist   = x[:-1]*x[1:] + y[:-1]*y[1:] + z[:-1]*z[1:]
    # avoid nan's in arccos from numerics
    dist[dist>=1.0] = 1.0
    dist   = np.arccos(dist)*Rearth
    transect['edge_cut_dist'] = np.hstack([0,dist.cumsum()])
    del(dist, x, y, z)
    
    #___________________________________________________________________________
    return(transect)
    

#
#
#___COMPUTE VOLUME TRANSPORT THROUGH TRANSECT___________________________________
def calc_transect_Xtransp(mesh, data, transects, dataX=None, data_Xref=0.0,
                          do_transectattr=False, do_rot=False, do_info=True, do_nveccs=True):
    """
    --> Compute fesom2 modell accurate transport through defined transect
    
    Parameters:
    
        :mesh:          fesom2 tripyview mesh object,  with all mesh information 
        
        :data:          object, with xarray dataset object with 3d element zonal and meridional
                        velocities
        
        :transects:     list with analysed transect dictionary information computed by 
                        do_analyse _trasnsects
                        
        :dataX:         object (default=None), with xarray dataset object with 3d 
                        vertice temperature or salinity data to compute heatflux/
                        saltflux through section   
                        
        :data_Xref:     float (default=0.0) reference temperature or salinity to 
                        compute flux
        
        :do_transectattr: bool, (default=True) write full transect info into return
                        xarray dataset attribute
        
        :do_rot:        bool, (default=True), do rotation of velocities from rotated
                        to geo coordinates
        
        :do_info:       bool, (default=True), write info
    
    Return:
    
        :data:          list with returned xarray dataset object that contain volume 
                        transport through every section of the transport path
    
    ____________________________________________________________________________
    """
    #___________________________________________________________________________
    if isinstance(transects, list): 
        #if len(transects)>1:
        is_list = True
        transp  = list([])
        #else:
        #    is_list = False
    else:
        is_list = False
        transects=list([transects])
    
    #___________________________________________________________________________
    # variable name of zonal meridional velocity component
    vnameU, vnameV = list(data.keys())
    
    # variable name of Xtransp component (temp or salt)
    vnameX = None
    if dataX is not None: vnameX = list(dataX.keys())[0]
    
    #___________________________________________________________________________
    # loop over various transects
    for transect in transects:
        #_______________________________________________________________________
        # select only necessary elements 
        if 'elem' in data.dims:
            data_uv = data.isel(elem=transect['path_ei']).load()
            
        elif 'nod2' in data.dims:    
            warnings.warn("\n"+
                          "--> It should be mentioned that a transport computed based on \n"
                          "    vertice velocity data is less accurate since the model operates \n"+
                          "    on element velocities. So there can be quite a big amplitude offset \n"+
                          "    between the transport based on u+v or unod+vnod. Although the variability \n"+
                          "    seems to be unaffected. So if you have the data try to compute the \n"+
                          "    transport based on velocities on elements. \n")
            data_uv = data.isel(nod2=xr.DataArray(transect['path_cut_ni'], dims=['nod2', 'n2'])).load()
            
        else:
            raise ValueError("--> Could not find proper dimension in uv velocity data")
        vel_u   = data_uv[vnameU ].values
        vel_v   = data_uv[vnameV].values
        del(data_uv)
        
        #_______________________________________________________________________
        # select only necessary elements from temp data
        var_X = None
        if dataX is not None:
            var_X = dataX.isel(nod2=xr.DataArray(transect['path_cut_ni'], dims=['nod2', 'n2'])).load()
            var_X = var_X[vnameX].values
        
        #_______________________________________________________________________
        # rotate vectors insert topography as nan
        if 'time' in list(data.dims):
            #___________________________________________________________________
            if 'elem' in data.dims:
                # Here i introduce the vertical topography, we replace the zeros that
                # describe the topography with nan
                for ii, ei in enumerate(transect['path_ei']):
                    vel_u[:,ii,mesh.e_iz[ei]:], vel_v[:,ii,mesh.e_iz[ei]:] = np.nan, np.nan
                
                # here rotate the velocities from roted frame to geo frame 
                if do_rot:
                    for nti in range(data.sizes['time']):
                        vel_u[nti,:,:], vel_v[nti,:,:] = vec_r2g(mesh.abg, 
                                            mesh.n_x[transect['path_ni']].sum(axis=1)/3.0, 
                                            mesh.n_y[transect['path_ni']].sum(axis=1)/3.0,
                                            vel_u[nti,:,:], vel_v[nti,:,:])
            
            #___________________________________________________________________        
            elif 'nod2' in data.dims:
                for ii, (ni0, ni1) in enumerate(transect['path_cut_ni']):
                    vel_u[:, ii, 0, mesh.n_iz[ni0]:], vel_v[:, ii, 0, mesh.n_iz[ni0]:] = np.nan, np.nan
                    vel_u[:, ii, 1, mesh.n_iz[ni1]:], vel_v[:, ii, 0, mesh.n_iz[ni1]:] = np.nan, np.nan
                    
                # average vertice velocities to the edge centers
                vel_u, vel_v = vel_u.sum(axis=2)*0.5, vel_v.sum(axis=2)*0.5
                
                # here rotate the velocities from roted frame to geo frame 
                if do_rot:
                    for nti in range(data.sizes['time']):
                        vel_u[nti,:,:], vel_v[nti,:,:] = vec_r2g(mesh.abg, 
                                            mesh.n_x[transect['path_cut_ni']].sum(axis=1)*0.5, 
                                            mesh.n_y[transect['path_cut_ni']].sum(axis=1)*0.5,
                                            vel_u[nti,:,:], vel_v[nti,:,:])
            
            #___________________________________________________________________        
            if dataX is not None:
                # Here i introduce the vertical topography, we replace the zeros that
                # describe the topography with nan's
                for ii, (ni0, ni1) in enumerate(transect['path_cut_ni']):
                    var_X[:, ii, 0, mesh.n_iz[ni0]:] = np.nan
                    var_X[:, ii, 1, mesh.n_iz[ni1]:] = np.nan
            
                # average vertice temperature to the edge centers
                var_X = var_X.sum(axis=2)*0.5
                
        else: 
            #___________________________________________________________________
            if 'elem' in data.dims:
                # Here i introduce the vertical topography, we replace the zeros that
                # describe the topography with nan's
                for ii, ei in enumerate(transect['path_ei']):
                    vel_u[ii, mesh.e_iz[ei]:], vel_v[ii,mesh.e_iz[ei]:] = np.nan, np.nan
                
                # here rotate the velocities from roted frame to geo frame 
                if do_rot:
                    vel_u, vel_v = vec_r2g(mesh.abg, 
                                        mesh.n_x[transect['path_ni']].sum(axis=1)/3.0, 
                                        mesh.n_y[transect['path_ni']].sum(axis=1)/3.0,
                                        vel_u, vel_v)
                
            #___________________________________________________________________        
            elif 'nod2' in data.dims:
                for ii, (ni0, ni1) in enumerate(transect['path_cut_ni']):
                    vel_u[ii, 0, mesh.n_iz[ni0]:], vel_v[ii, 0, mesh.n_iz[ni0]:] = np.nan, np.nan
                    vel_u[ii, 1, mesh.n_iz[ni1]:], vel_v[ii, 0, mesh.n_iz[ni1]:] = np.nan, np.nan
                    
                # average vertice velocities to the edge centers
                vel_u, vel_v = vel_u.sum(axis=1)*0.5, vel_v.sum(axis=1)*0.5
                
                # here rotate the velocities from roted frame to geo frame 
                if do_rot:
                    vel_u, vel_v = vec_r2g(mesh.abg, 
                                        mesh.n_x[transect['path_cut_ni']].sum(axis=1)*0.5, 
                                        mesh.n_y[transect['path_cut_ni']].sum(axis=1)*0.5,
                                        vel_u, vel_v)
            
            #___________________________________________________________________        
            if dataX is not None:
                # Here i introduce the vertical topography, we replace the zeros that
                # describe the topography with nan's
                for ii, (ni0, ni1) in enumerate(transect['path_cut_ni']):
                    var_X[ii, 0, mesh.n_iz[ni0]:] = np.nan
                    var_X[ii, 1, mesh.n_iz[ni1]:] = np.nan
                    
                # average vertice temperature to the edge centers
                var_X = var_X.sum(axis=1)*0.5
        
        #_______________________________________________________________________
        # multiply with dz
        if   'nz1'  in data.dims: dz = -np.diff(mesh.zlev)
        elif 'nz_1' in data.dims: dz = -np.diff(mesh.zlev)    
        elif 'nz'   in data.dims: dz = np.hstack(((mesh.zlev[0]-mesh.zlev[1])/2.0, mesh.zmid[:-1]-mesh_zmid[1:], (mesh.zlev[-2]-mesh.zlev[-1])/2.0))
        vel_u, vel_v = vel_u*dz, vel_v*dz
        
        #_______________________________________________________________________
        # multiple u*dy and v*(-dx) --> vec_v * vec_n
        if do_nveccs: 
            nvecx, nvecy= transect['path_nvec_cs'][:,0], transect['path_nvec_cs'][:,1]
            dx, dy      = np.abs(transect['path_dx'])*nvecy, np.abs(transect['path_dy'])*nvecx
        else:
            dx, dy      = transect['path_dx'], -transect['path_dy']    
        
        # proper transport sign with respect to normal vector of section 
        # normal vector definition --> norm vector definition (-dy,dx) --> shows 
        # to the left of the e_vec
        if 'elem' in data.dims:
            # use velocities located at left and rights handside elements with 
            # respect to the edge
            if 'time' in list(data.dims):
                aux_transp = (vel_u.transpose((0,2,1))*dy + vel_v.transpose((0,2,1))*dx)
            else:
                aux_transp = (vel_u.T*(dy) + vel_v.T*dx)
                
        elif 'nod2' in data.dims:
            # use edge centered averaged velocities, thats why this velocity needs 
            # to be dublicated with np.repeat along the horizontal dimension
            if 'time' in list(data.dims):
                #vel_u, vel_v = np.repeat(vel_u, 2, axis=1), np.repeat(vel_v, 2, axis=1)
                aux_transp = (vel_u.transpose((0,2,1))*dy + vel_v.transpose((0,2,1))*dx)
            else:    
                #vel_u, vel_v = np.repeat(vel_u, 2, axis=0), np.repeat(vel_v, 2, axis=0)
                aux_transp = (vel_u.T*dy + vel_v.T*dx)
        
        #_______________________________________________________________________
        # multiply transp_uv with temp 
        if dataX is not None:
            if 'time' in list(data.dims):
                #aux_transp = aux_transp * (np.repeat(var_X, 2, axis=1).transpose((0,2,1)) - data_Xref)
                aux_transp = aux_transp * (var_X.transpose((0,2,1)) - data_Xref)
            else:
                #aux_transp = aux_transp * (np.repeat(var_X, 2, axis=0).T - data_Xref)
                aux_transp = aux_transp * (var_X.T - data_Xref)
            
        #_______________________________________________________________________
        data_vars = dict()
        aux_attr  = data[vnameU].attrs

        if (vnameX=='temp') or (vnameU=='tu' and vnameV=='tv'):
            rho0 = 1030 # kg/m^3
            cp   = 3850 # J/kg/K
            inPW = 1.0e-15
            aux_transp = aux_transp * rho0 * cp * inPW
            aux_attr['long_name'], aux_attr['units'] = 'Heat Transport', 'PW'
            vnameFLX = 'Hflx'
        
        elif (vnameX=='salt') or (vnameU=='su' and vnameV=='sv'):
            rho0 = 1030 # kg/m^3
            # 1psu = 1g(NaCl)/1kg(H2O)=1/1000--> Umrechnung von psu*kg/s --> kg/s -->.../1000
            aux_transp = aux_transp * rho0 / 1000
            aux_attr['long_name'], aux_attr['units'] = 'Salinity Transport', 'kg/s'
            vnameFLX = 'Sflx'    
            
        else:    
            inSv = 1.0e-6
            aux_transp = aux_transp * inSv
            aux_attr['long_name'], aux_attr['units'] = 'Volume Transport', 'Sv'
            vnameFLX = 'Vflx'
            
        #_______________________________________________________________________
        lon = transect['path_xy'][:-1,0]
        lat = transect['path_xy'][:-1,1]
        #lon = transect['path_xy'][:-1,0] + (transect['path_xy'][1:,0]-transect['path_xy'][:-1,0])/2
        #lat = transect['path_xy'][:-1,1] + (transect['path_xy'][1:,1]-transect['path_xy'][:-1,1])/2
        dst = transect['path_dist']
        
        #_______________________________________________________________________
        # define dimensions
        list_dimname, list_dimsize, list_dimval = dict(), dict(), dict()
        if   'time' in data.dims: 
            list_dimname['time'] , list_dimsize['time'] , list_dimval['time']  = 'time', data.sizes['time'], data.time.data #pd.to_datetime(data.time)            
        if   'nz1'  in data.dims: 
            list_dimname['depth'], list_dimsize['depth'], list_dimval['depth'] = 'nz1' , mesh.zmid.size   , mesh.zmid
        elif 'nz'   in data.dims: 
            list_dimname['depth'], list_dimsize['depth'], list_dimval['depth'] = 'nz'  , mesh.zlev.size   , mesh.zlev
        list_dimname['horiz'], list_dimsize['horiz'] = 'npts', transect['path_dx'].size    
        
        #_______________________________________________________________________
        # define variable 
        gattrs = data.attrs
        gattrs['proj']          = 'index+depth+xy'
        aux_attr['transect_name'] = transect['Name']
        aux_attr['transect_lon']  = transect['lon']
        aux_attr['transect_lat']  = transect['lat']
        if do_transectattr: aux_attr['transect'] = transect
        data_vars[vnameFLX] = (list(list_dimname.values()), aux_transp, aux_attr) 
        
        #_______________________________________________________________________
        # define coordinates
        if 'time' in data.dims:
            coords = {
                      'time'  : (list_dimname['time' ], list_dimval['time']),
                      'depth' : (list_dimname['depth'], list_dimval['depth']),
                      'lon'   : (list_dimname['horiz'], lon),
                      'lat'   : (list_dimname['horiz'], lat),
                      'dst'   : (list_dimname['horiz'], dst)}
        else:    
            coords = {'depth' : (list_dimname['depth'], list_dimval['depth']),
                      'lon'   : (list_dimname['horiz'], lon),
                      'lat'   : (list_dimname['horiz'], lat),
                      'dst'   : (list_dimname['horiz'], dst)}
        
        # add more information about the transport path
        coords['path_xy'] =  ( ['npts1','np2'], transect['path_xy'])
        coords['path_dx'] =  (list_dimname['horiz' ], transect['path_dx'])
        coords['path_dy'] =  (list_dimname['horiz' ], transect['path_dy'])
        coords['path_ei'] =  (list_dimname['horiz' ], transect['path_ei'])
        
        #_______________________________________________________________________
        # create dataset
        if is_list:
            transp.append(xr.Dataset(data_vars=data_vars, coords=coords, attrs=gattrs))
            # we have to set the time here with assign_coords otherwise if its 
            # setted in xr.Dataset(..., coords=dict(...),...)xarray does not 
            # recognize the cfttime format and things like data['time.year']
            # are not possible
            if 'time' in data.dims: transp[-1] = transp[-1].assign_coords(time=data.time)  
            if do_info:
                print(' --------> Name:', transp[-1][vnameFLX].attrs['transect_name'])
                if 'time' in data.dims:
                    print(' mean neto transport:', '{:6.4f} / {:s}'.format(transp[-1][vnameFLX].sum(dim=('npts','nz1'), skipna=True).mean(dim=('time'), skipna=True).data, aux_attr['units']))
                    print(' mean (+) transport :', '{:6.4f} / {:s}'.format(transp[-1][vnameFLX].where(transp[-1][vnameFLX]>0).sum(dim=('npts','nz1'), skipna=True).mean(dim=('time'), skipna=True).data, aux_attr['units']))
                    print(' mean (-) transport :', '{:6.4f} / {:s}'.format(transp[-1][vnameFLX].where(transp[-1][vnameFLX]<0).sum(dim=('npts','nz1'), skipna=True).mean(dim=('time'), skipna=True).data, aux_attr['units']))
                else:
                    print(' neto transport:', '{:6.4f} / {:s}'.format(transp[-1][vnameFLX].sum(dim=('npts','nz1'), skipna=True).data, aux_attr['units']))
                    print(' (+) transport :', '{:6.4f} / {:s}'.format(transp[-1][vnameFLX].where(transp[-1][vnameFLX]>0).sum(dim=('npts','nz1'), skipna=True).data, aux_attr['units']))
                    print(' (-) transport :', '{:6.4f} / {:s}'.format(transp[-1][vnameFLX].where(transp[-1][vnameFLX]<0).sum(dim=('npts','nz1'), skipna=True).data, aux_attr['units']))
                print('')
                
        else:
            transp = xr.Dataset(data_vars=data_vars, coords=coords, attrs=gattrs )
            # we have to set the time here with assign_coords otherwise if its 
            # setted in xr.Dataset(..., coords=dict(...),...)xarray does not 
            # recognize the cfttime format and things like data['time.year']
            # are not possible
            if 'time' in data.dims: transp = transp.assign_coords(time=data.time)  
            if do_info:
                print(' --------> Name:', transp[vnameFLX].attrs['transect_name'])
                if 'time' in data.dims:
                    print(' mean neto transport:', '{:6.4f} / {:s}'.format(transp[vnameFLX].sum(dim=('npts','nz1'), skipna=True).mean(dim=('time'), skipna=True).data, aux_attr['units']))
                    print(' mean (+) transport :', '{:6.4f} / {:s}'.format(transp[vnameFLX].where(transp[vnameFLX]>0).sum(dim=('npts','nz1'), skipna=True).mean(dim=('time'), skipna=True).data, aux_attr['units']))
                    print(' mean (-) transport :', '{:6.4f} / {:s}'.format(transp[vnameFLX].where(transp[vnameFLX]<0).sum(dim=('npts','nz1'), skipna=True).mean(dim=('time'), skipna=True).data, aux_attr['units']))
                else:
                    print(' neto transport:', '{:6.4f} / {:s}'.format(transp[vnameFLX].sum(dim=('npts','nz1'), skipna=True).data, aux_attr['units']))
                    print(' (+) transport :', '{:6.4f} / {:s}'.format(transp[vnameFLX].where(transp[vnameFLX]>0).sum(dim=('npts','nz1'), skipna=True).data, aux_attr['units']))
                    print(' (-) transport :', '{:6.4f} / {:s}'.format(transp[vnameFLX].where(transp[vnameFLX]<0).sum(dim=('npts','nz1'), skipna=True).data, aux_attr['units']))
                print('')
    #___________________________________________________________________________
    return(transp)


#
#
#___COMPUTE TRANSECT OF SCALAR VERTICE/ELEMENTAL VARIABLE_______________________
def calc_transect_scalar(mesh, data, transects, nodeinelem=None, 
                         do_transectattr=False, do_info=False):
    """
    --> interpolate scalar vertice values onto cutting point position where
        cross-section intersects with edge
    
    Parameters: 
        
        :mesh:          fesom2 tripyview mesh object,  with all mesh information 
        
        :data:          object, with xarray dataset object containing 3d vertice values. 
                        Can also be done with scalar datas on elements but than nodeinelem is needed
        
        transects:     list with analysed transect dictionary information computed by 
                        do_analyse _trasnsects
                        
        nodeinelem:     np.array with elem indices that contribute to a vertice 
                        point (default=None)              
                        
        :do_transectattr: bool, (default=True) write full transect info into return
                        xarray dataset attribute
                        
        :do_info:       bool, (default=True), write info     
        
    Return:
    
        :data:          list with returned xarray dataset object that contains to
                        the cutting point interpolated scalar values of transect
    
    ____________________________________________________________________________
    """
    t1=clock.time()
    #___________________________________________________________________________
    if isinstance(transects, list): 
        #if len(transects)>1:
        is_list = True
        csects  = list([])
        #else:
        #    is_list = False
    else:
        is_list = False
        transects=list([transects])
    
    #___________________________________________________________________________
    vname = list(data.keys())[0]
    for transect in transects:
        #_______________________________________________________________________
        if 'nod2' in list(data.dims):
            # select only necessary vertices
            scalarP1 = data[vname].isel(nod2=transect['edge_cut_ni'][:,0]).load().values
            scalarP2 = data[vname].isel(nod2=transect['edge_cut_ni'][:,1]).load().values
            
            # put back the nans 
            if 'nz1' in data.dims or 'nz_1' in data.dims:
                for ii, ni in enumerate(transect['edge_cut_ni'][:,0]): scalarP1[ii,mesh.n_iz[ni]:] = np.nan
                for ii, ni in enumerate(transect['edge_cut_ni'][:,1]): scalarP2[ii,mesh.n_iz[ni]:] = np.nan
            elif 'nz' in data.dims:
                for ii, ni in enumerate(transect['edge_cut_ni'][:,0]): scalarP1[ii,mesh.n_iz[ni]+1:] = np.nan
                for ii, ni in enumerate(transect['edge_cut_ni'][:,1]): scalarP2[ii,mesh.n_iz[ni]+1:] = np.nan
                
            # interpolate scalar vertice values onto cutting point position where
            # cross-section intersects with edge
            scalarPcut = scalarP1.T + (scalarP2.T-scalarP1.T)*transect['edge_cut_lint']
            #scalarPcut = scalarP1.T 
            #scalarPcut = scalarP2.T 
            del(scalarP1, scalarP2)
            
            # define lon, lat , distance arrays
            lon = transect['edge_cut_P'][:,0]
            lat = transect['edge_cut_P'][:,1]
            dst = transect['edge_cut_dist']
        
        #_______________________________________________________________________
        elif 'elem' in list(data.dims):
            if nodeinelem is None:
                # select only necessary elements 
                scalarPcut = data[vname].isel(elem=transect['path_ei'][::2]).load().values
                
                # put back the nans 
                if 'nz1' in data.dims or 'nz_1' in data.dims:
                    for ii, ei in enumerate(transect['path_ei'][::2]): scalarPcut[ii,mesh.e_iz[ei]:  ] = np.nan
                elif 'nz' in data.dims:
                    for ii, ei in enumerate(transect['path_ei'][::2]): scalarPcut[ii,mesh.e_iz[ei]+1:] = np.nan
                
                scalarPcut = scalarPcut.T
                
                if scalarPcut.shape[1] == transect['edge_cut_P'][:,0].shape[0]: 
                    # define lon, lat , distance arrays
                    lon = transect['edge_cut_P'][:,0]
                    lat = transect['edge_cut_P'][:,1]
                    dst = transect['edge_cut_dist'][:]
                else:
                    # define lon, lat , distance arrays
                    lon = transect['edge_cut_P'][:-1,0]
                    lat = transect['edge_cut_P'][:-1,1]
                    dst = transect['edge_cut_dist'][:-1]
                
            else:
                # average over all elemental values that contribute to edge node 1
                #        o-----o
                #       / \   / \
                #      /   \ / x-\-----elem_i_p
                #     o-----O-----o 
                #      \   /1\   /
                #       \ /   \ /  
                #        o-----o  
                #
                elem_i_P = nodeinelem[:,transect['edge_cut_ni'][:,0]]
                elem_A_P = mesh.e_area[elem_i_P]
                elem_A_P[elem_i_P<0]=0.0
                scalarP1 = data[vname].isel(elem=elem_i_P.flatten()).load().values.T #.reshape(elem_i_P.shape)
                
                # set nan's from the land sea mask to zeros
                scalarP1[np.isnan(scalarP1)]=0
                
                dim1,dum = scalarP1.shape # --> dim1 depth dimension 
                dim2,dim3= elem_i_P.shape # --> dim2 total numbers of neighbouring elem, dim3 dimension edge nodes 
                del(elem_i_P)
                
                # weighted mean over elements that belong to node
                scalarP1 = scalarP1*elem_A_P.flatten()
                scalarP1 = scalarP1.reshape((dim1, dim2, dim3))
                scalarP1 = scalarP1.sum(axis=1)/elem_A_P.sum(axis=0)
                del(elem_A_P)
                
                # put back the nans 
                if 'nz1' in data.dims or 'nz_1' in data.dims:
                    for ii, ni in enumerate(transect['edge_cut_ni'][:,0]): scalarP1[mesh.n_iz[ni]:  ,ii] = np.nan
                elif 'nz' in data.dims:
                    for ii, ni in enumerate(transect['edge_cut_ni'][:,0]): scalarP1[mesh.n_iz[ni]+1:,ii] = np.nan
                    
                # average elemental values to edge node 2
                elem_i_P = nodeinelem[:,transect['edge_cut_ni'][:,1]]
                elem_A_P = mesh.e_area[elem_i_P]
                elem_A_P[elem_i_P<0]=0.0
                scalarP2 = data[vname].isel(elem=elem_i_P.flatten()).load().values.T #.reshape(elem_i_P.shape)
                
                # set nan's from the land sea mask to zeros
                scalarP2[np.isnan(scalarP2)]=0
                
                dim1,dum = scalarP2.shape # --> dim1 depth dimension 
                dim2,dim3= elem_i_P.shape # --> dim2 total numbers of neighbouring elem, dim3 dimension edge nodes 
                del(elem_i_P)
                
                # weighted mean over elements that belong to node
                scalarP2 = scalarP2*elem_A_P.flatten()
                scalarP2 = scalarP2.reshape((dim1, dim2, dim3))
                scalarP2 = scalarP2.sum(axis=1)/elem_A_P.sum(axis=0)
                del(elem_A_P)
                
                # put back the nans based on node depth
                if 'nz1' in data.dims or 'nz_1' in data.dims:
                    for ii, ni in enumerate(transect['edge_cut_ni'][:,1]): scalarP2[mesh.n_iz[ni]:  ,ii] = np.nan
                elif 'nz' in data.dims:
                    for ii, ni in enumerate(transect['edge_cut_ni'][:,1]): scalarP2[mesh.n_iz[ni]+1:,ii] = np.nan
                
                # interpolate scalar vertice values onto cutting point position where
                # cross-section intersects with edge
                scalarPcut = scalarP1 + (scalarP2-scalarP1)*transect['edge_cut_lint']
                del(scalarP1, scalarP2)
                
                # define lon, lat , distance arrays
                lon = transect['edge_cut_midP'][:,0]
                lat = transect['edge_cut_midP'][:,1]
                dst = transect['edge_cut_dist']
        
        #_______________________________________________________________________
        else: raise ValueError('cant find nod2 or elem dimensions in data')   
        
        #_______________________________________________________________________
        # define dimensions
        list_dimname, list_dimsize, list_dimval = list(), list(), list()
        if   'time' in data.dims: list_dimname.append('time'), list_dimsize.append(data.dims['time']), list_dimval.append(pd.to_datetime(data.time))             
        if   'nz1'  in data.dims: list_dimname.append('nz1' ), list_dimsize.append(mesh.zmid.size), list_dimval.append(mesh.zmid) 
        elif 'nz_1' in data.dims: list_dimname.append('nz_1'), list_dimsize.append(mesh.zmid.size), list_dimval.append(mesh.zmid)  
        elif 'nz'   in data.dims: list_dimname.append('nz'  ), list_dimsize.append(mesh.zlev.size), list_dimval.append(mesh.zlev)  
        list_dimname.append('npts'), list_dimsize.append(dst.size)
        
        
        gattrs = data.attrs
        gattrs['proj']          = 'index+depth+xy'
        
        # define variable 
        data_vars = dict()
        lattr  = data[vname].attrs
        lattr['transect_name'] = transect['Name']
        lattr['transect_lon']  = transect['lon']
        lattr['transect_lat']  = transect['lat']
        if do_transectattr: lattr['transect'] = transect
        
        #data_vars['transp'] = (list_dimname, scalarPcut, aux_attr) 
        data_vars[vname] = (list_dimname, scalarPcut, lattr) 
        del(lattr)
        
        # define coordinates
        coords = dict()
        for ii, dim in enumerate(list_dimname): 
            if 'npts' in dim:
                coords['lon'] = (dim, lon)
                coords['lat'] = (dim, lat)
                coords['dst'] = (dim, dst)
            if 'nz' in dim:
                coords['depth'] = (dim, list_dimval[ii])
                                  
        #coords    = {'depth' : (list_dimname[0], list_dimval[0]),
                     #'lon'   : (list_dimname[1], lon),
                     #'lat'   : (list_dimname[1], lat),
                     #'dst'   : (list_dimname[1], dst),
                    #}

        # create dataset
        if is_list:
            csects.append(xr.Dataset(data_vars=data_vars, coords=coords, attrs=gattrs))
            # we have to set the time here with assign_coords otherwise if its 
            # setted in xr.Dataset(..., coords=dict(...),...)xarray does not 
            # recognize the cfttime format and things like data['time.year']
            # are not possible
            if 'time' in data.dims: csects[-1] = csects[-1].assign_coords(time=data.time)  
        else:
            csects = xr.Dataset(data_vars=data_vars, coords=coords, attrs=gattrs)
            # we have to set the time here with assign_coords otherwise if its 
            # setted in xr.Dataset(..., coords=dict(...),...)xarray does not 
            # recognize the cfttime format and things like data['time.year']
            # are not possible
            if 'time' in data.dims: csects = csects.assign_coords(time=data.time)  
            
    #___________________________________________________________________________
    if do_info: print('        elapsed time: {:3.2f}min.'.format((clock.time()-t1)/60.0))
    return(csects)


#
#
#___PLOT TRANSECT POSITION______________________________________________________
def plot_transect_position(mesh, transect, edge=None, zoom=None, fig=None,  figsize=[10,10], 
                           proj='nears', box = [-180, 180,-90, 90], 
                           resolution='low', do_path=True, do_labels=True, do_title=True,
                           do_nvec=False, do_evec_cs=False, do_nvec_cs=True, 
                           do_grid=False, ax_pos=[0.90, 0.05, 0.45, 0.45]):
    """
    --> plot transect positions
    
    Parameters:
    
        :mesh:          fesom2 tripyview mesh object,  with all mesh information 
        
        :transects:     list with analysed transect dictionary information computed by 
                        do_analyse _trasnsects
        
        :edge:          provide np.array with node indices of edge (default=None)
        
        :zoom:          float, (default=None), zzom factor for nearside projection
        
        :fig:           matplotlib figure handle, (default=None)
        
        :figsize:       list, (default=[10,10]) width and height of figure
        
        :proj:          str, (default='nears'), can be any other projections string
                        
                        - pc     ... PlateCarree         (box=[lonmin, lonmax, latmin, latmax])
                        - merc   ... Mercator            (box=[lonmin, lonmax, latmin, latmax])
                        - rob    ... Robinson            (box=[lonmin, lonmax, latmin, latmax])
                        - eqearth... EqualEarth          (box=[lonmin, lonmax, latmin, latmax])
                        - mol    ... Mollweide           (box=[lonmin, lonmax, latmin, latmax])
                        - nps    ... NorthPolarStereo    (box=[-180, 180, >0, latmax])
                        - sps    ... SouthPolarStereo    (box=[-180, 180, latmin, <0])
                        - ortho  ... Orthographic        (box=[loncenter, latcenter]) 
                        - nears  ... NearsidePerspective (box=[loncenter, latcenter, zoom]) 
                        - channel... PlateCaree
                        
        :box:           None, list (default:  [-180, 180,-90, 90]) 
                        regional limitation of plot. For ortho...
                        box=[lonc, latc], nears...box=[lonc, latc, zoom], for all others box = 
                        [lonmin, lonmax, latmin, latmax]. For nears box is computed based
                        on transect definition.
                        
        :do_path:       bool, (default=True) plot entire transport path       
        
        :do_labels:     bool, (default=True) draw lon, lat axes do_labels
        
        :do_title:      bool, (default=True) draw title with name of transect
        
        :do_grid:       bool, (default=False) plot fesom mesh in background
        
    Returns:
    
        :hfig: returns figure handle 

        :hax: returns axes handle 
    
    ____________________________________________________________________________
    """
    #___________________________________________________________________________
    # compute zoom factor based on the length of the transect
    if zoom is None:
        if np.abs(np.diff(transect['path_xy'][[0,-1],0]))<=180: 
            Rearth = 6367.5  # [km]
            x,y,z  = grid_cart3d(transect['path_xy'][[0,-1],0], transect['path_xy'][[0,-1],1], is_deg=True)
            dist   = np.arccos(x[0]*x[1] + y[0]*y[1] + z[0]*z[1])*Rearth
            #zoom = (np.pi*6367.5)/transect['edge_cut_dist'].max()
            zoom = (np.pi*Rearth)/dist
            del(dist, x, y, z)
        else:
            if proj == 'nears': proj = 'rob'
    
    #___________________________________________________________________________
    proj_from = ccrs.PlateCarree()
    if proj == 'nears': box = [transect['edge_cut_P'][:,0].mean(), transect['edge_cut_P'][:,1].mean(), zoom]
    proj_to, box = do_projection(mesh, proj, box)
    
    #___________________________________________________________________________
    if fig is None: 
        fig = plt.figure(figsize=figsize)
        ax  = plt.axes(projection=proj_to)
    else:
        ax = fig.add_axes(ax_pos, projection=proj_to)
        
    pkg_path = os.path.dirname(__file__)
    bckgrndir = os.path.normpath(pkg_path+'/backgrounds/')
    os.environ["CARTOPY_USER_BACKGROUNDS"] = bckgrndir
    ax.background_img(name='bluemarble', resolution=resolution)
    
    #___________________________________________________________________________
    if do_grid==True:
        tri = Triangulation(np.hstack((mesh.n_x,mesh.n_xa)),
                            np.hstack((mesh.n_y,mesh.n_ya)),
                            np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia)))
        
        tri.x, tri.y = proj_to.transform_points(proj_from, tri.x, tri.y)[:,0:2].T
        
        xpts, ypts = tri.x[tri.triangles].sum(axis=1)/3, tri.y[tri.triangles].sum(axis=1)/3
        
        crs_pts = list(zip(xpts,ypts))
        fig_pts = ax.transData.transform(crs_pts)
        ax_pts  = ax.transAxes.inverted().transform(fig_pts)
        x, y    =  ax_pts[:,0], ax_pts[:,1]
        e_idxbox= (x>=-0.05) & (x<=1.05) & (y>=-0.05) & (y<=1.05)
        #ax.triplot(tri.x, tri.y, tri.triangles[e_idxbox,:], color='w', linewidth=0.25, alpha=0.35, transform=proj_from)
        ax.triplot(tri.x, tri.y, tri.triangles[e_idxbox,:], color='w', linewidth=0.25, alpha=0.35,)
    
    #___________________________________________________________________________
    #intersected edges 
    if edge is not None:
        ax.plot(mesh.n_x[edge[:,transect['edge_cut_i']]], mesh.n_y[edge[:,transect['edge_cut_i']]],'-', color=[0.75,0.75,0.75], linewidth=1.5, transform=proj_from)

    # intersection points
    ax.plot(transect['edge_cut_P'][ :, 0], transect['edge_cut_P'][ :, 1],'limegreen', marker='None', linestyle='-', markersize=7, markerfacecolor='w', linewidth=3.0, transform=proj_from)

    #transport path
    if do_path:
        ax.plot(transect['path_xy'   ][ :, 0], transect['path_xy'   ][ :, 1],'m', marker='None', linestyle='-', markersize=7, markerfacecolor='w', linewidth=1.0, transform=proj_from)

    # end start point [green] and end point [red]
    ax.plot(transect['lon'], transect['lat'],'k', marker='o', linestyle='None', markersize=5, markerfacecolor='w', transform=proj_from)
    ax.plot(transect['edge_cut_P'][ 0, 0], transect['edge_cut_P'][ 0, 1],'k', marker='o', linestyle='None', markersize=10, markerfacecolor='g', transform=proj_from)
    ax.plot(transect['edge_cut_P'][-1, 0], transect['edge_cut_P'][-1, 1],'k', marker='o', linestyle='None', markersize=10, markerfacecolor='r', transform=proj_from)
    
    # plot norm vector 
    if do_nvec:
        xx   , yy    = mesh.n_x[transect['path_ni'][1:-1]].sum(axis=1)/3.0, mesh.n_y[transect['path_ni'][1:-1]].sum(axis=1)/3.0
        vecxx, vecyy = -transect['path_dy'][1:-1], transect['path_dx'][1:-1]
        vecxx, vecyy = proj_to.transform_vectors(ccrs.PlateCarree(), xx, yy, vecxx, vecyy)
        ax.quiver(xx, yy, vecxx, vecyy, color='c', transform=proj_from)
    
    if do_nvec_cs:
        xx           = np.array(transect['Px']).sum(axis=1)/2.0
        yy           = np.array(transect['Py']).sum(axis=1)/2.0
        vecxx, vecyy = np.array(transect['n_vec'])[:,0], np.array(transect['n_vec'])[:,1]
        ax.quiver(xx, yy, 
                  vecxx, vecyy,
                  facecolor=np.array([0, 192, 255])/255, transform=proj_from,
                  edgecolor='w', linewidth=0.5, scale=10)
        
    if do_evec_cs:
        xx           = np.array(transect['Px']).sum(axis=1)/2.0
        yy           = np.array(transect['Py']).sum(axis=1)/2.0
        vecxx, vecyy = np.array(transect['e_vec'])[:,0], np.array(transect['e_vec'])[:,1]
        ax.quiver(xx, yy, 
                  vecxx, vecyy,
                  facecolor=np.array([255, 192, 0])/255, transform=proj_from,
                  edgecolor='w', linewidth=0.5, scale=10)    
        
    ## plot cross-section 
    #for ii, (Px, Py) in enumerate(zip(transect['Px'],transect['Py'])):
        ## plot evec
        #ax.plot(Px, Py,'k', marker='o', linestyle='None', markersize=3, markerfacecolor='w', transform=orig)
        ##ax.quiver(Px[0]+np.diff(Px)/2.0, Py[0]+np.diff(Py)/2.0,transect['n_vec'][ii][0], transect['n_vec'][ii][1],
                ##units='xy', scale_units='xy', scale=1/3, color='teal', width=1/3, edgecolor='k', linestyle='-', linewidth=0.5)

    #___________________________________________________________________________
    # ax.coastlines()
    ax.add_geometries(mesh.lsmask_p, crs=ccrs.PlateCarree(), facecolor='None', edgecolor='k', linewidth=1.0)
    ax.gridlines(color='w', linestyle='-', linewidth=1.0, draw_labels=do_labels, 
                 alpha=0.25,)
    
    #___________________________________________________________________________
    if do_title: ax.set_title(transect['Name'], fontsize=16, fontweight='bold')
    
    #ax.set_extent([transect['edge_cut_P'][:,0].min(),
                   #transect['edge_cut_P'][:,0].max(),
                   #transect['edge_cut_P'][:,1].min(),
                   #transect['edge_cut_P'][:,1].max()])
    
    #___________________________________________________________________________
    plt.show()
    fig.canvas.draw()
    
    #___________________________________________________________________________
    return(fig, ax)


#
#
#___COMPUTE ZONAL MEAN SECTION BASED ON BINNING_________________________________
def load_zmeantransect_fesom2(mesh                  , 
                              data                  , 
                              box_list              , 
                              dlat          =0.5    , 
                              boxname       =None   ,
                              diagpath      =None   , 
                              do_checkbasin =False  , 
                              do_compute    =False  , 
                              do_load       =True   , 
                              do_persist    =False  , 
                              do_info       =False  , 
                              **kwargs,):
    """
    --> compute zonal means transect, defined by regional box_list
    
    Parameters:
        
        :mesh:      fesom2 mesh object, with all mesh information

        :data:      xarray dataset object, or list of xarray dataset object with 3d vertice
                    data

        :box_list:  None, list (default: None)  list with regional box limitation for index computation box can be: 

                    - ['global']   ... compute global index 
                    - [shp.Reader] ... index region defined by shapefile 
                    - [ [lonmin,lonmax,latmin, latmax], boxname] index region defined by rect box 
                    - [ [ [px1,px2...], [py1,py2,...]], boxname] index region defined by polygon
                    - [ np.array(2 x npts), boxname] index region defined by polygon

        :dlat:      float, (default=0.5) resolution of latitudinal bins
        
        :diagpath:  str, (default=None), path to diagnostic file only needed when 
                    w_A weights for area average are not given in the dataset, than 
                    he will search for the diagnostic file_loader
                    
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
                      
    Returns:
    
        :index_list:    list with xarray dataset of zonal mean array
    
    ____________________________________________________________________________
    
    """
    #___________________________________________________________________________
    # str_anod    = ''
    index_list  = []
    cnt         = 0
    
    #___________________________________________________________________________
    # loop over box_list
    vname = list(data.keys())[0]
    which_ddim = None
    if   'nz'  in list(data[vname].dims): which_ddim, ndi, depth = 'nz' , mesh.nlev  , mesh.zlev     
    elif 'nz1' in list(data[vname].dims) or 'nz_1' in list(data[vname].dims): 
        which_ddim, ndi, depth = 'nz1', mesh.nlev-1, mesh.zmid
    
    
    for box in box_list:
        if not isinstance(box, shp.Reader) and not box =='global' and not box==None :
            if len(box)==2: boxname, box = box[1], box[0]
        
        #_______________________________________________________________________
        # compute box mask index for nodes
        if   'nod2' in data.dims:
            idxin = xr.DataArray(do_boxmask(mesh, box, do_elem=False), dims='nod2')
        elif 'elem' in data.dims:     
            idxin = xr.DataArray(do_boxmask(mesh, box, do_elem=True), dims='elem')
            
        #___________________________________________________________________________
        if do_checkbasin:
            from matplotlib.tri import Triangulation
            tri = Triangulation(np.hstack((mesh.n_x,mesh.n_xa)), np.hstack((mesh.n_y,mesh.n_ya)), np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia)))
            plt.figure()
            plt.triplot(tri, color='k')
            if 'elem' in data.dims:
                plt.triplot(tri.x, tri.y, tri.triangles[ np.hstack((idxin[mesh.e_pbnd_0], idxin[mesh.e_pbnd_a])) ,:], color='r')
            else:
                plt.plot(mesh.n_x[idxin], mesh.n_y[idxin], 'or', linestyle='None', markersize=1)
            plt.title('Basin selection')
            plt.show()
        
        #___________________________________________________________________________
        # do zonal mean calculation either on nodes or on elements        
        # keep in mind that node area info is changing over depth--> therefor load from file 
        fname = data[vname].attrs['runid']+'.mesh.diag.nc'            
        if diagpath is None:
            if   os.path.isfile( os.path.join(data[vname].attrs['datapath'], fname) ): 
                dname = data[vname].attrs['datapath']
            elif os.path.isfile( os.path.join( os.path.join(os.path.dirname(os.path.normpath(data[vname].attrs['datapath'])),'1/'), fname) ): 
                dname = os.path.join(os.path.dirname(os.path.normpath(data[vname].attrs['datapath'])),'1/')
            elif os.path.isfile( os.path.join(mesh.path,fname) ): 
                dname = mesh.path
            else:
                raise ValueError('could not find directory with...mesh.diag.nc file')
            
            diagpath = os.path.join(dname,fname)
            if do_info: print(' --> found diag in directory:{}', diagpath)
        else:
            if os.path.isfile(os.path.join(diagpath,fname)):
                diagpath = os.path.join(diagpath,fname)
            elif os.path.isfile(os.path.join(os.path.join(os.path.dirname(os.path.normpath(diagpath)),'1/'),fname)) :
                diagpath = os.path.join(os.path.join(os.path.dirname(os.path.normpath(diagpath)),'1/'),fname)
                
        #___________________________________________________________________________
        # compute area weighted vertical velocities on elements
        if 'elem' in data.dims:        
            #_______________________________________________________________________
            # load elem area from diag file
            if 'w_A' not in list(data.coords):
                if ( os.path.isfile(diagpath)):
                    nz_w_A = xr.open_mfdataset(diagpath, parallel=True, **kwargs)['elem_area']#.chunk({'elem':1e4})
                    if 'elem_n' in list(nz_w_A.dims): nz_w_A = nz_w_A.rename({'elem_n':'elem'})
                    if 'nl'     in list(nz_w_A.dims): nz_w_A = nz_w_A.rename({'nl'    :'nz'  })
                    if 'nl1'    in list(nz_w_A.dims): nz_w_A = nz_w_A.rename({'nl1'   :'nz1' })
                else: 
                    raise ValueError('could not find ...mesh.diag.nc file')
                data = data.assign_coords(w_A=nz_w_A)
                
            data = data.assign_coords(new_w_A = data['w_A'].expand_dims({which_ddim: data[which_ddim]}).transpose())
            data = data.drop_vars('w_A').rename({'new_w_A':'w_A'})
            
            #___________________________________________________________________
            # select MOC basin
            data_zm = data.isel(elem=idxin).load()
            
            #___________________________________________________________________
            # replace bottom topo with zeros
            # The notnull() function, often used in the context of pandas and xarray, 
            # is used to check for non-null (non-NaN) values within a Series or DataFrame. 
            # It returns a Boolean mask with the same shape as the input data, where 
            # each element is True if the original element is not null, and False if 
            # the original element is null (NaN).
            #mat_area = mat_area.where(mat_mean.notnull(), other=0)
            data_zm['w_A'] = data_zm['w_A'].where(data_zm[vname].notnull(), other=0)
            
            #___________________________________________________________________
            # calculate area weighted mean
            data_zm[vname] = data_zm[vname]*data_zm['w_A']
            
            #___________________________________________________________________
            # create meridional bins --> this trick is from Nils Brückemann (ICON)
            lat_bin = xr.DataArray(data=np.round(data_zm.lat/dlat)*dlat, dims='elem', name='lat')
        
        # compute area weighted vertical velocities on vertices
        else:     
            #___________________________________________________________________
            # load vertice cluster area from diag file
            if 'w_A' not in list(data.coords):
                if ( os.path.isfile(diagpath)):
                    nz_w_A = xr.open_mfdataset(diagpath, parallel=True, **kwargs)['nod_area']
                    if 'nod_n' in list(nz_w_A.dims): nz_w_A = nz_w_A.rename(  {'nod_n':'nod2'})
                    if 'nl'    in list(nz_w_A.dims): nz_w_A = nz_w_A.rename(  {'nl'   :'nz'  })
                    if 'nl1'   in list(nz_w_A.dims): nz_w_A = nz_w_A.rename(  {'nl1'  :'nz1' })    
                    nz_w_A = nz_w_A.drop_vars(['nz'])
                    #nz_w_A = nz_w_A.isel(nod2=n_idxin).load()     
                else: 
                    raise ValueError('could not find ...mesh.diag.nc file')
                
                # data are on mid depth levels
                if which_ddim=='nz1': 
                    nz_w_A = nz_w_A.isel(nz=slice(None, -1)).rename({'nz':'nz1'})
                
                data = data.assign_coords(w_A=nz_w_A)
                
            #___________________________________________________________________
            # select MOC basin
            data_zm = data.isel(nod2=idxin).load()
            
            #___________________________________________________________________
            # replace bottom topo with zeros
            # The notnull() function, often used in the context of pandas and xarray, 
            # is used to check for non-null (non-NaN) values within a Series or DataFrame. 
            # It returns a Boolean mask with the same shape as the input data, where 
            # each element is True if the original element is not null, and False if 
            # the original element is null (NaN).
            #mat_area = mat_area.where(mat_mean.notnull(), other=0)
            data_zm['w_A'] = data_zm['w_A'].where(data_zm[vname].notnull(), other=0)
            
            #___________________________________________________________________
            # calculate area weighted mean
            data_zm[vname] = data_zm[vname]*data_zm['w_A']
            
            #___________________________________________________________________
            # create meridional bins
            lat_bin = xr.DataArray(data=np.round(data_zm.lat/dlat)*dlat, dims='nod2', name='lat')

        #___________________________________________________________________________
        # change area weight from coordinates to variable, create new variable for topography
        data_zm['var_w_A'] = data_zm['w_A']
        data_zm = data_zm.drop_vars('w_A').rename({'var_w_A':'w_A'})
        
        if which_ddim is not None:
            if 'elem' in data.dims:
                data_zm['bottom'] = xr.DataArray(-mesh.zmid[data_zm['elemiz']]*data_zm['w_A'].isel({which_ddim:0}), dims='elem')
            else:    
                data_zm['bottom'] = xr.DataArray(-mesh.zmid[data_zm['nodiz']]*data_zm['w_A'].isel({which_ddim:0}), dims='nod2')
            
            if 'depth' not in list(data.coords):
                data_zm    = data_zm.rename_vars({which_ddim:'depth'})
        
        #___________________________________________________________________________
        # group data by bins
        if do_info==True: print(' --> do binning of latitudes')
        #data_zm    = data_zm.persist()
        data_zm    = data_zm.groupby(lat_bin)
        
        # zonal sumation of var*weight and weight over bins
        if do_info==True: print(' --> do sumation/integration over bins')
        data_zm    = data_zm.sum(skipna=True)
        
        # compute weighted mean 
        data_zm[vname] = data_zm[vname]/data_zm['w_A']
        if which_ddim is not None:
            if 'elem' in data.dims:
                data_zm['bottom'] = data_zm['bottom']/data_zm['w_A'].isel({which_ddim:0})
            else:
                data_zm['bottom'] = data_zm['bottom']/data_zm['w_A'].isel({which_ddim:0})
            data_zm = data_zm.set_coords('bottom')
        data_zm = data_zm.drop_vars('w_A')
        
        # transpose data from [lat x nz] --> [nz x lat]
        dtime, dhz, dnz = 'None', 'lat', which_ddim
        if 'time' in list(data_zm.dims): dtime = 'time'
        data_zm = data_zm.transpose(dtime, dnz, dhz, missing_dims='ignore')
        
        #_______________________________________________________________________
        # change attributes
        local_attr  = data[vname].attrs
        if 'long_name' in data[vname].attrs:
            data_zm[vname].attrs['long_name'] = " zonal mean {}".format(data[vname].attrs['long_name']) 
        else:
            data_zm[vname].attrs['long_name'] = " zonal mean {}".format(vname) 
        
        if 'short_name' in data[vname].attrs:
            data_zm[vname].attrs['short_name'] = " zonal mean {}".format(data[vname].attrs['short_name']) 
        else:
            data_zm[vname].attrs['short_name'] = " zonal mean {}".format(vname) 
        
        #_______________________________________________________________________
        if box is None or box == 'global': 
            data_zm[vname].attrs['transect_name'] = 'global zonal mean'
        elif isinstance(box, shp.Reader):
            str_name = box.shapeName.split('/')[-1].replace('_',' ')
            data_zm[vname].attrs['transect_name'] = '{} zonal mean'.format(str_name.lower())
        
        # for the choice of vertical plotting mode
        data_zm.attrs['proj'] = 'index+depth+xy'
        #_______________________________________________________________________
        if do_compute : data_zm = data_zm.compute() 
        if do_load    : data_zm = data_zm.load()
        if do_persist : data_zm = data_zm.persist()    
        
        #_______________________________________________________________________
        # append index to list
        index_list.append(data_zm)
        
        #_______________________________________________________________________
        cnt = cnt + 1
        
    #___________________________________________________________________________
    return(index_list)



#
#
#___COMPUTE ZONAL MEAN SECTION BASED ON BINNING_________________________________
def calc_transect_zm_mean_dask(mesh                  , 
                               data                  , 
                               box_list              , 
                               do_parallel           , 
                               parallel_nprc         ,
                               do_lonlat     ='lat'  , 
                               dlonlat       =0.5    , 
                               boxname       =None   ,
                               diagpath      =None   , 
                               do_checkbasin =False  , 
                               do_info       =False  , 
                               **kwargs,):
    """
    --> compute zonal or meridional means transect, defined by regional box_list
        using dask parallel approach on chunks
    
    Parameters:
        
        :mesh:      fesom2 mesh object, with all mesh information

        :data:      xarray dataset object, or list of xarray dataset object with 3d vertice
                    data

        :box_list:  None, list (default: None)  list with regional box limitation for index computation box can be: 

                    - ['global']   ... compute global index 
                    - [shp.Reader] ... index region defined by shapefile 
                    - [ [lonmin,lonmax,latmin, latmax], boxname] index region defined by rect box 
                    - [ [ [px1,px2...], [py1,py2,...]], boxname] index region defined by polygon
                    - [ np.array(2 x npts), boxname] index region defined by polygon
        
        :do_parallel: bool, (default=False) if dask cluster is running or not True/False
        
        :parallel_nprc: int, (default=48) how many dask workers are used if dask client is running
        
        :do_lonlat: str, (default='lat') is mean binning approach is done for latitudinal bins
                         (zonal mean) or done for longitudinal bins (merdional mean)
        
        :dlonlat:      float, (default=0.5) resolution of latitudinal/longitudinal bins
        
        :diagpath:  str, (default=None), path to diagnostic file only needed when 
                    w_A weights for area average are not given in the dataset, than 
                    he will search for the diagnostic file_loader
                    
        :do_checkbasin: bool, (default=False) additional plot with selected region/
                        basin information
                        
        :do_info:       bool (defalt=False), print variable info at the end 
                      
    Returns:
    
        :index_list:    list with xarray dataset of zonal/meridional mean array
    
    ____________________________________________________________________________
    
    """
    #___________________________________________________________________________
    # str_anod    = ''
    index_list  = []
    cnt         = 0
    
    #___________________________________________________________________________
    # loop over box_list
    vname = list(data.keys())[0]
    dimn_v = None
    dimn_h, dimn_v = 'dum', 'dum'
    if   ('nod2' in data.dims): dimn_h = 'nod2'
    elif ('elem' in data.dims): dimn_h = 'elem'
    if   'nz'  in list(data[vname].dims): 
        dimn_v  = 'nz'    
    elif 'nz1' in list(data[vname].dims) or 'nz_1' in list(data[vname].dims): 
        dimn_v  = 'nz1'
    
    for box in box_list:
        if not isinstance(box, shp.Reader) and not box =='global' and not box==None :
            if   len(box)==2: boxname, box = box[1], box[0]
            elif len(box)==4 and boxname==None: boxname = '[{:03.2f}...{:03.2f}°E, {:03.2f}...{:03.2f}°N]'.format(box[0],box[1],box[2],box[3])
            
        #_______________________________________________________________________
        # compute box mask index for nodes
        if   'nod2' in data.dims:
            idxin = xr.DataArray(do_boxmask(mesh, box, do_elem=False), dims='nod2')
        elif 'elem' in data.dims:     
            idxin = xr.DataArray(do_boxmask(mesh, box, do_elem=True), dims='elem')
            
        #___________________________________________________________________________
        if do_checkbasin:
            from matplotlib.tri import Triangulation
            tri = Triangulation(np.hstack((mesh.n_x,mesh.n_xa)), np.hstack((mesh.n_y,mesh.n_ya)), np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia)))
            plt.figure()
            plt.triplot(tri, color='k', linewidth=0.1)
            if 'elem' in data.dims:
                plt.triplot(tri.x, tri.y, tri.triangles[ np.hstack((idxin[mesh.e_pbnd_0], idxin[mesh.e_pbnd_a])) ,:], color='r')
            else:
                plt.plot(mesh.n_x[idxin], mesh.n_y[idxin], 'or', linestyle='None', markersize=1)
            plt.title('Basin selection')
            plt.show()
        
        #___________________________________________________________________________
        # do zonal mean calculation either on nodes or on elements        
        # keep in mind that node area info is changing over depth--> therefor load from file 
        fname = data[vname].attrs['runid']+'.mesh.diag.nc'            
        if diagpath is None:
            if   os.path.isfile( os.path.join(data[vname].attrs['datapath'], fname) ): 
                dname = data[vname].attrs['datapath']
            elif os.path.isfile( os.path.join( os.path.join(os.path.dirname(os.path.normpath(data[vname].attrs['datapath'])),'1/'), fname) ): 
                dname = os.path.join(os.path.dirname(os.path.normpath(data[vname].attrs['datapath'])),'1/')
            elif os.path.isfile( os.path.join(mesh.path,fname) ): 
                dname = mesh.path
            else:
                raise ValueError('could not find directory with...mesh.diag.nc file')
            
            diagpath = os.path.join(dname,fname)
            if do_info: print(' --> found diag in directory:{}', diagpath)
        else:
            if os.path.isfile(os.path.join(diagpath,fname)):
                diagpath = os.path.join(diagpath,fname)
            elif os.path.isfile(os.path.join(os.path.join(os.path.dirname(os.path.normpath(diagpath)),'1/'),fname)) :
                diagpath = os.path.join(os.path.join(os.path.dirname(os.path.normpath(diagpath)),'1/'),fname)
                
        #___________________________________________________________________________
        # compute area weighted vertical velocities on elements
        if 'elem' in data.dims:        
            #_______________________________________________________________________
            # load elem area from diag file
            if 'w_A' not in list(data.coords):
                if ( os.path.isfile(diagpath)):
                    nz_w_A = xr.open_mfdataset(diagpath, parallel=True, **kwargs)['elem_area']#.chunk({'elem':1e4})
                    if 'elem_n' in list(nz_w_A.dims): nz_w_A = nz_w_A.rename({'elem_n':'elem'})
                    if 'nl'     in list(nz_w_A.dims): nz_w_A = nz_w_A.rename({'nl'    :'nz'  })
                    if 'nl1'    in list(nz_w_A.dims): nz_w_A = nz_w_A.rename({'nl1'   :'nz1' })
                else: 
                    raise ValueError('could not find ...mesh.diag.nc file')
                data = data.assign_coords(w_A=nz_w_A)
            
            #___________________________________________________________________
            # select basin to compute mean over
            data_zm = data.isel(elem=idxin)
        
        # compute area weighted vertical velocities on vertices
        else:     
            #___________________________________________________________________
            # load vertice cluster area from diag file
            if 'w_A' not in list(data.coords):
                if ( os.path.isfile(diagpath)):
                    nz_w_A = xr.open_mfdataset(diagpath, parallel=True, **kwargs)['nod_area']
                    if 'nod_n' in list(nz_w_A.dims): nz_w_A = nz_w_A.rename(  {'nod_n':'nod2'})
                    if 'nl'    in list(nz_w_A.dims): nz_w_A = nz_w_A.rename(  {'nl'   :'nz'  })
                    if 'nl1'   in list(nz_w_A.dims): nz_w_A = nz_w_A.rename(  {'nl1'  :'nz1' })    
                    nz_w_A = nz_w_A.drop_vars(['nz'])
                    
                else: 
                    raise ValueError('could not find ...mesh.diag.nc file')
                
                # data are on mid depth levels
                if dimn_v=='nz1': 
                    nz_w_A = nz_w_A.isel(nz=slice(None, -1)).rename({'nz':'nz1'})
                
                data = data.assign_coords(w_A=nz_w_A)
                
            #___________________________________________________________________
            # select basin to compute mean over
            data_zm = data.isel(nod2=idxin)
            
        #_______________________________________________________________________
        # determine/adapt actual chunksize
        nchunk = 1
        if do_parallel and isinstance(data_zm[vname].data, da.Array)==True :
            
            if   'elem' in data_zm.dims: nchunk = len(data_zm.chunks['elem'])
            elif 'nod2' in data_zm.dims: nchunk = len(data_zm.chunks['nod2'])
            print(' --> nchunk=', nchunk)   
            
            # after all the time and depth operation after the loading there will 
            # be worker who have no chunk piece to work on  --> therfore we need
            # to rechunk make sure the workload is distributed between all 
            # availabel worker equally         
            if nchunk<parallel_nprc*0.75:
                print(' --> rechunk array size', end='')
                if   'elem' in data_zm.dims: 
                    data_zm = data_zm.chunk({'elem': np.ceil(data_zm.dims['elem']/(parallel_nprc)).astype('int'), dimn_v:-1})
                    nchunk = len(data_zm.chunks['elem'])
                elif 'nod2' in data_zm.dims: 
                    data_zm = data_zm.chunk({'nod2': np.ceil(data_zm.dims['nod2']/(parallel_nprc)).astype('int'), dimn_v:-1})
                    nchunk = len(data_zm.chunks['nod2'])
                print(' --> nchunk_new=', nchunk)    
        
        # in case of climatology data because there i need to make compute() after 
        # interpolation which destroys the chunking so i try to rechunk it
        elif do_parallel and isinstance(data_zm[vname].data, da.Array)==False: 
            if   'elem' in data_zm.dims: 
                data_zm = data_zm.chunk({'elem': np.ceil(data_zm.dims['elem']/(parallel_nprc)).astype('int'), dimn_v:-1}).unify_chunks().persist()
                nchunk = len(data_zm.chunks['elem'])
            elif 'nod2' in data_zm.dims: 
                data_zm = data_zm.chunk({'nod2': np.ceil(data_zm.dims['nod2']/(parallel_nprc)).astype('int'), dimn_v:-1}).unify_chunks().persist()
                nchunk = len(data_zm.chunks['nod2'])
            print(' --> nchunk_new=', nchunk)        
        
        #_______________________________________________________________________
        # create zonal/meridional bins
        lonlat_min    = np.floor(data_zm[ do_lonlat ].min().compute())
        lonlat_max    = np.ceil( data_zm[ do_lonlat ].max().compute())
        lonlat_bins   = np.arange(lonlat_min, lonlat_max+dlonlat/2, dlonlat)
        nlonlat, nlev = len(lonlat_bins)-1, data_zm.dims[dimn_v]
        # print('nlonlat, nlev = ', nlonlat, nlev)
        
        #_______________________________________________________________________
        # The centroid position of the periodic boundary triangle causes problems 
        # when determining in which bin they should be --> therefor we kick them out 
        # with this index
        if 'elem_pbnd' not in data_zm.coords: 
            data_zm = data_zm.assign_coords(elem_pbnd=xr.DataArray(np.zeros(data_zm[do_lonlat].shape, dtype=bool), dims=data_zm[do_lonlat].dims))
            if isinstance(data_zm[do_lonlat].data, da.Array)==True: 
                data_zm['elem_pbnd'] = data_zm['elem_pbnd'].chunk(data_zm[do_lonlat].chunks)
        
        #_______________________________________________________________________
        # Apply zonal mean over chunk
        # its important to use: 
        # data_zm[do_lonlat  ].data[:,None]
        # data_zm['elem_pbnd'].data[:,None]
        # all the input to da.map_blocks(calc_transect_zm_mean_chnk... must have the
        # same dimensionality otherwise it wont work. I also hat to use drop_axis = [1] 
        # which drops the chunking along the second dimension only leaving  the 
        # chunking of the first dimension. I also only managed to return the results
        # as a flattened array the attempt to return as a more dimensional matrix
        # failed. THats why i need to use shape afterwards 
        chnk_lonlat = data_zm[do_lonlat  ].data[:, None] 
        chnk_epbnd  = data_zm['elem_pbnd'].data[:, None]
        # when we are on elements w_A is a 1D field, whereas when we are on 
        # vertices w_A is 2D, that why we need to add a dimension for the elem case
        if np.ndim(data_zm['w_A'].data)==1: chnk_wA = data_zm['w_A'].data[:, None] 
        else                              : chnk_wA = data_zm['w_A'].data
        bin_zm = da.map_blocks(calc_transect_zm_mean_chnk    ,
                                  lonlat_bins                ,   # mean bin definitions
                                  chnk_lonlat                ,   # lon/lat nod2 coordinates
                                  chnk_wA                    ,   # area weight
                                  chnk_epbnd                 ,   # if elem is pbnd element
                                  data_zm[vname      ].data  ,   # data chunk piece
                                  dtype     = np.float32     ,   # Tuple dtype
                                  drop_axis = [1]            ,   # drop dim nz1
                                  chunks    = (2*nlonlat*nlev, ) # Output shape
                                )
        
        #_______________________________________________________________________
        # reshape axis over chunks 
        bin_zm = bin_zm.reshape((nchunk, 2, nlonlat, nlev))
        
        #_______________________________________________________________________
        # do dask axis reduction across chunks dimension
        bin_zm = da.reduction(bin_zm,                   
                                 chunk     = lambda x, axis=None, keepdims=None: x,  # Updated to handle axis and keepdims
                                 aggregate = np.sum,
                                 dtype     = np.float32,
                                 axis      = 0,
                                 ).compute()
        
        #_______________________________________________________________________
        # compute mean velocities ber bin --> avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            bin_zm = np.where(bin_zm[1]>0, bin_zm[0]/bin_zm[1], np.nan)
        
        #________________________________________________________________________________________________    
        data_bin_zm = xr.Dataset(data_vars = {vname : ((dimn_v,do_lonlat), bin_zm.T, data_zm[vname].attrs)
                                              }, 
                                 coords    = {do_lonlat       : ((do_lonlat      ), (lonlat_bins[:-1]+lonlat_bins[1:])*0.5)              , 
                                              do_lonlat+'_bnd': ((do_lonlat,'n2' ), np.column_stack((lonlat_bins[:-1], lonlat_bins[1:]))), 
                                              'depth'         : (('nz1'          ), data_zm[dimn_v].values) 
                                              },
                                 attrs     = data_zm.attrs)
        
        #_______________________________________________________________________
        # change attributes
        if   do_lonlat == 'lat': whichmean = 'zonal'
        elif do_lonlat == 'lon': whichmean = 'meridional'
    
        if 'long_name' in data[vname].attrs:
            data_bin_zm[vname].attrs['long_name' ] = "{} mean {}".format(whichmean, data[vname].attrs['long_name']) 
        else:
            data_bin_zm[vname].attrs['long_name' ] = "{} mean {}".format(whichmean, vname) 
        
        if 'short_name' in data[vname].attrs:
            data_bin_zm[vname].attrs['short_name'] = "{} mean {}".format(whichmean, data[vname].attrs['short_name']) 
        else:
            data_bin_zm[vname].attrs['short_name'] = "{} mean {}".format(whichmean, vname) 
        
        #_______________________________________________________________________
        if box is None or box == 'global': 
            data_bin_zm[vname].attrs['transect_name'] = 'global {} mean'.format(whichmean)
        elif isinstance(box, shp.Reader):
            str_name = box.shapeName.split('/')[-1].replace('_',' ')
            data_bin_zm[vname].attrs['transect_name'] = '{} {} mean'.format(str_name.lower(), whichmean)
        else:
            str_name = boxname.replace('_',' ')
            data_bin_zm[vname].attrs['transect_name'] = '{} {} mean'.format(boxname, whichmean)
        
        # for the choice of vertical plotting mode
        data_bin_zm.attrs['proj'] = 'index+depth+xy'
        ##_______________________________________________________________________
        #if do_compute : data_zm = data_zm.compute() 
        #if do_load    : data_zm = data_zm.load()
        #if do_persist : data_zm = data_zm.persist()    
        
        #_______________________________________________________________________
        # append index to list
        index_list.append(data_bin_zm)
        
        #_______________________________________________________________________
        cnt = cnt + 1
        
    #___________________________________________________________________________
    return(index_list)



#
#
#_______________________________________________________________________________  
def calc_transect_zm_mean_chnk(lonlat_bins, chnk_lonlat, chnk_wA, chnk_pnbd, chnk_d):
    """

    """
    #___________________________________________________________________________
    # only needthe additional dimension at the point where the function is started
    if   np.ndim(chnk_lonlat) == 2: chnk_lonlat = chnk_lonlat[:, 0]
    elif np.ndim(chnk_lonlat) == 3: chnk_lonlat = chnk_lonlat[:, 0, 0]
    if   np.ndim(chnk_pnbd)   == 2: chnk_pnbd   = chnk_pnbd[  :, 0]
    elif np.ndim(chnk_pnbd)   == 3: chnk_pnbd   = chnk_pnbd[  :, 0, 0]
    
    # Replace NaNs with 0 value to summation issues
    chnk_wA     = np.where(np.isnan(chnk_d), 0, chnk_wA)
    chnk_d      = np.where(np.isnan(chnk_d), 0, chnk_d)
    nnod, nlev  = chnk_d.shape
    
    # Use np.digitize to find bin indices for longitudes and latitudes
    idx_lonlat  = np.digitize(chnk_lonlat, lonlat_bins)-1  # Adjust to get 0-based index
    nlonlat     = len(lonlat_bins)-1
    
    # Initialize binned data storage 
    binned_d    = np.zeros((2, nlonlat, nlev), dtype=np.float32)  
    # binned_d[0,...] - data
    # binned_d[1,...] - area weight counter
    
    # Precompute mask outside the loop
    idx_valid   = (idx_lonlat >= 0) & (idx_lonlat < nlonlat) & (~chnk_pnbd)
    del(chnk_pnbd)

    # Apply mask before looping
    idx_lonlat  = idx_lonlat[idx_valid]
    chnk_d      = chnk_d[    idx_valid, :]
    chnk_wA     = chnk_wA[   idx_valid, :]
    nnod        = len(idx_lonlat)
    
    # Sum data based on binned indices
    for nod_i in range(0,nnod):
        jj = idx_lonlat[nod_i]
        binned_d[0, jj, :] = binned_d[0, jj, :] + chnk_d[ nod_i, :] * chnk_wA[nod_i, :]
        binned_d[1, jj, :] = binned_d[1, jj, :] + chnk_wA[nod_i, :]
    
    #___________________________________________________________________________
    return binned_d.flatten()


#
#
#____DO ANOMALY_________________________________________________________________
def do_transect_anomaly(index1,index2):
    """
    --> compute anomaly between two transect list xarray Datasets
    
    Parameters:
    
        :index1:   list with transect xarray dataset object

        :index2:   list with transect xarray dataset object

    Returns:
    
        :anom:   list with  transect xarray dataset object, data1-data2

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
                    if key in ['long_name']:
                        anom_idx[vname].attrs[key] = 'anom. '+anom_idx[vname].attrs[key].capitalize() 
                    elif key in ['short_name']:
                        anom_idx[vname].attrs[key] = 'anom. '+anom_idx[vname].attrs[key]        
                    elif key in ['units',]: 
                        continue
                    elif key in ['descript']: 
                        if len(idx1[vname].attrs[key])+len(idx2[vname2].attrs[key])>30:
                            anom_idx[vname].attrs[key]  = idx1[vname].attrs[key]+'\n - '+idx2[vname2].attrs[key]
                        else:     
                            anom_idx[vname].attrs[key]  = idx1[vname].attrs[key]+' - '+idx2[vname2].attrs[key]
                        
                    elif key in ['do_rescale']: 
                        anom_idx[vname].attrs[key]  = idx1[vname].attrs[key]    
                
                    elif idx1[vname].attrs[key] != idx2[vname2].attrs[key]:
                        anom_idx[vname].attrs[key]  = idx1[vname].attrs[key]+' - '+idx2[vname2].attrs[key]
                    
        #___________________________________________________________________________
        anom_index.append(anom_idx)
    #___________________________________________________________________________
    return(anom_index)
