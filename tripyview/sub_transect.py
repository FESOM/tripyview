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
                        do_rot      = True  , 
                        ):
    """
    --> pre-analyse defined transects, with respect to which triangles and edges 
        are crossed by the transect line. Create transport path edge to compute 
        modell accurate volume transports.
    
    Parameters:   
    
        :input_transect:    list, to define transects, transects= [[[lon pts], 
                            [lat pts], name], [...]]
        
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
        transect['path_dx'      ] = [] # dx of path sections
        transect['path_dy'      ] = [] # dy of path sections
        transect['path_dist'    ] = [] # dy of path sections
        transect['path_nvec_cs' ] = [] # normal vector of transection segment
    
    ____________________________________________________________________________

    """
    transect_list = []
    # loop over transects in list
    for transec_lon, transec_lat, transec_name in input_transect:
        #_______________________________________________________________________
        # allocate dictionary for total cross-section 
        sub_transect = _do_init_transect()
        sub_transect['Name'] = transec_name
        sub_transect['lon']  = transec_lon
        sub_transect['lat']  = transec_lat
        sub_transect['ncs']  = len(transec_lon)-1
        
        #_______________________________________________________________________
        # loop over transect points
        for ii in range(0,len(transec_lon)-1):
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
            Pdxy = 2.0
            Pxmin, Pxmax = min(sub_transect['Px'][-1]), max(sub_transect['Px'][-1])
            Pymin, Pymax = min(sub_transect['Py'][-1]), max(sub_transect['Py'][-1])
            Pxmin, Pxmax = Pxmin-Pdxy, Pxmax+Pdxy
            Pymin, Pymax = Pymin-Pdxy, Pymax+Pdxy
            idx_edlimit  = np.where( ( mesh.n_x[edge].min(axis=0)>=Pxmin ) &  
                                     ( mesh.n_x[edge].max(axis=0)<=Pxmax ) & 
                                     ( mesh.n_y[edge].min(axis=0)>=Pymin ) &  
                                     ( mesh.n_y[edge].max(axis=0)<=Pymax ) )[0]
            del(Pxmin, Pxmax, Pymin, Pymax, Pdxy)
            
            #___________________________________________________________________
            # compute which edges are intersected by cross-section line 
            sub_transect = _do_find_intersected_edges(mesh, sub_transect, edge, idx_edlimit)
            
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
            transect['path_dx'], transect['path_dy'] = vec_r2g(mesh.abg, 
                                                        mesh.n_x[transect['path_ni']].sum(axis=1)/3.0, 
                                                        mesh.n_y[transect['path_ni']].sum(axis=1)/3.0,
                                                        transect['path_dx'], transect['path_dy'])
            
        #_______________________________________________________________________ 
        # buld distance from start point array [km]
        transect = _do_compute_distance_from_startpoint(transect)
        
        #___________________________________________________________________________    
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
    transect['path_dx'      ] = [] # dx of path sections
    transect['path_dy'      ] = [] # dy of path sections
    transect['path_dist'    ] = [] # dy of path sections
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
    transect['path_dx'      ] = sub_transect['path_dx'      ][0]
    transect['path_dy'      ] = sub_transect['path_dy'      ][0]
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
            transect['path_xy'      ] = np.vstack((transect['path_xy'      ], sub_transect['path_xy'      ][ii]))
            transect['path_ei'      ] = np.hstack((transect['path_ei'      ], sub_transect['path_ei'      ][ii]))
            transect['path_ni'      ] = np.vstack((transect['path_ni'      ], sub_transect['path_ni'      ][ii]))
            transect['path_dx'      ] = np.hstack((transect['path_dx'      ], sub_transect['path_dx'      ][ii]))
            transect['path_dy'      ] = np.hstack((transect['path_dy'      ], sub_transect['path_dy'      ][ii]))
            transect['path_nvec_cs' ] = np.vstack((transect['path_nvec_cs' ], sub_transect['path_nvec_cs' ][ii]))
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
        if ((X0>=0) & (X0<=normA0               +np.finfo(np.float32).eps) &
            (X1>=0) & (X1<=transect['e_norm'][-1]+np.finfo(np.float32).eps) ):
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
    alpha = -np.arctan2(transect['e_vec'][-1][1], transect['e_vec'][-1][0])
    transect['alpha'].append(alpha*180/np.pi)
    
    #___________________________________________________________________________
    # loop over intersected edges 
    nced = transect['edge_cut_i'][-1].size
    for edi in range(0,nced):
        #print(' --> ', edi)
        
        #_______________________________________________________________________
        # --> rotate edge with bearing angle -alpha
        # determine if edge shows to the left or to the right with 
        # respect to cross-section line, if theta<0, edge shows to the right
        # in this case use [L]eft triangle, if theta>0 edge shows to the left
        # in this case use right triangle (its that the "downstream" triangle
        # with respect to cross-section direction)
        auxx = transect['edge_cut_evec'][-1][edi,0]*np.cos(alpha)-transect['edge_cut_evec'][-1][edi,1]*np.sin(alpha)
        auxy = transect['edge_cut_evec'][-1][edi,0]*np.sin(alpha)+transect['edge_cut_evec'][-1][edi,1]*np.cos(alpha)
        theta= np.arctan2(auxy,auxx)
        del(auxx, auxy)
                
        # indices of [L]eft and [R]ight triangle with respect to the edge
        edge_elem  = edge_tri[:, transect['edge_cut_i'][-1][edi]]
        
        #_______________________________________________________________________
        # add upsection element to path if it exist --> path_xy coodinate points 
        # for the element always come from downsection triangle 
        path_xy, path_ei, path_ni, path_dx, path_dy = __add_upsection_elem2path(mesh, transect, edi, nced, theta, 
                                                  edge_elem, edge_dxdy_l, edge_dxdy_r,
                                                  path_xy, path_ei, path_ni, path_dx, path_dy)
        
        # add edge mid point of cutted edge
        path_xy.append(transect['edge_cut_midP'][-1][edi])
        
        # add downsection element to path if it exist
        path_xy, path_ei, path_ni, path_dx, path_dy = __add_downsection_elem2path(mesh, transect, edi, nced, theta, 
                                                  edge_elem, edge_dxdy_l, edge_dxdy_r,
                                                  path_xy, path_ei, path_ni, path_dx, path_dy)
        
    #___________________________________________________________________________    
    # reformulate from list --> np.array
    transect['path_xy'].append(np.asarray(path_xy))
    transect['path_ei'].append(np.asarray(path_ei))
    transect['path_ni'].append(np.asarray(path_ni))
    transect['path_dx'].append(np.asarray(path_dx))
    transect['path_dy'].append(np.asarray(path_dy))
    
    aux = np.ones((transect['path_dx'][-1].size,2))
    aux[:,0], aux[:,1] = transect['n_vec'][-1][0], transect['n_vec'][-1][1]
    transect['path_nvec_cs'].append(aux)
    del(aux)
    
    # !!! Make sure positive Transport is defined S-->N and W-->E
    # --> Preliminary --> not 100% sure its universal
    rad = np.pi/180
    #print(alpha/rad)
    #if (alpha/rad>90 and alpha/rad<180 ) or (alpha/rad>=-180 and alpha/rad<=-90 ):
    if (alpha/rad>=-180 and alpha/rad<=-90 ) or (alpha/rad>90 and alpha/rad<=180 ):
        #print(' >-))))°>.°oO :1')
        transect['path_dx'][-1] = -transect['path_dx'][-1]
        transect['path_dy'][-1] = -transect['path_dy'][-1]
        
        transect['path_nvec_cs'][-1] = transect['path_nvec_cs'][-1]
    del(path_xy, path_ei, path_ni, path_dx, path_dy, edge_elem)

    #___________________________________________________________________________
    return(transect)

#
#
#_______________________________________________________________________________
def __add_downsection_elem2path(mesh, transect, edi, nced, theta, edge_elem, edge_dxdy_l, edge_dxdy_r,
                                path_xy, path_ei, path_ni, path_dx, path_dy):
    
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
        
        # edge is boundary edge right triangle does not exist--> put dummy values
        # instead of centroid position (transect['edge_cut_midP'][-1][edi])  
        else:
            #print(' >-)))°> .°oO: downsection', edi)
            path_xy.append(transect['edge_cut_midP'][-1][edi])  #(***)
            path_ei.append(-1)                     #--> -1 is here dummy index
            path_ni.append(np.array([-1, -1, -1])) #--> -1 is here dummy index
            path_dx.append(np.nan)
            path_dy.append(np.nan)
    
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
        
        
        
    #___________________________________________________________________________
    return(path_xy, path_ei, path_ni, path_dx, path_dy)
#
#
#_______________________________________________________________________________
def __add_upsection_elem2path(mesh, transect, edi, nced, theta, edge_elem, edge_dxdy_l, edge_dxdy_r,
                            path_xy, path_ei, path_ni, path_dx, path_dy):
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
            
            if edi!=0 and edi!=nced-1:
                path_ei.append(-1)                     
                path_ni.append(np.array([-1, -1, -1])) 
                path_dx.append(np.nan)
                path_dy.append(np.nan)
            elif transect['ncsi'][-1]!=0 and edi==0:        
                path_ei.append(-1)                     
                path_ni.append(np.array([-1, -1, -1])) 
                path_dx.append(np.nan)
                path_dy.append(np.nan)
            elif transect['ncsi'][-1]!=transect['ncs'] and edi==nced-1:        
                path_ei.append(-1)                     
                path_ni.append(np.array([-1, -1, -1])) 
                path_dx.append(np.nan)
                path_dy.append(np.nan)
                
    #___________________________________________________________________________
    return(path_xy, path_ei, path_ni, path_dx, path_dy)



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
def calc_transect_transp(mesh, data, transects, do_transectattr=False, do_rot=True, do_info=True):
    """
    --> Compute fesom2 modell accurate transport through defined transect
    
    Parameters:
    
        :mesh:          fesom2 tripyview mesh object,  with all mesh information 
        
        :data:          object, with xarray dataset object with 3d element zonal and meridional
                        velocities
        
        :transects:     list with analysed transect dictionary information computed by 
                        do_analyse _trasnsects
                        
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
    vname, vname2 = list(data.keys())
    for transect in transects:
        #_______________________________________________________________________
        # select only necessary elements 
        data_uv = data.isel(elem=transect['path_ei']).load()
        vel_u   = data_uv[vname ].values
        vel_v   = data_uv[vname2].values
        
        #_______________________________________________________________________
        # rotate vectors insert topography as nan
        if 'time' in list(data.dims):
            # Here i introduce the vertical topography, we replace the zeros that
            # describe the topography with nan
            for ii, ei in enumerate(transect['path_ei']):
                vel_u[:,ii,mesh.e_iz[ei]:], vel_v[:,ii,mesh.e_iz[ei]:] = np.nan, np.nan
            
            # here rotate the velocities from roted frame to geo frame 
            if do_rot:
                for nti in range(data.dims['time']):
                    vel_u[nti,:,:], vel_v[nti,:,:] = vec_r2g(mesh.abg, 
                                        mesh.n_x[transect['path_ni']].sum(axis=1)/3.0, 
                                        mesh.n_y[transect['path_ni']].sum(axis=1)/3.0,
                                        vel_u[nti,:,:], vel_v[nti,:,:])            
        else: 
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
        
        #_______________________________________________________________________
        # multiply with dz
        if   'nz1'  in data.dims: dz = -np.diff(mesh.zlev)
        elif 'nz_1' in data.dims: dz = -np.diff(mesh.zlev)    
        elif 'nz'   in data.dims: dz = np.hstack(((mesh.zlev[0]-mesh.zlev[1])/2.0, mesh.zmid[:-1]-mesh_zmid[1:], (mesh.zlev[-2]-mesh.zlev[-1])/2.0))
        vel_u, vel_v = vel_u*dz, vel_v*dz
        
        #_______________________________________________________________________
        # multiple u*dy and v*(-dx) --> vec_v * vec_n
        dx, dy     = transect['path_dx'], transect['path_dy']
        
        # proper transport sign with respect to normal vector of section 
        # normal vector definition --> norm vector definition (-dy,dx) --> shows 
        # to the left of the e_vec
        if 'time' in list(data.dims):
            aux_transp = (vel_u.transpose((0,2,1))*(-dy) + vel_v.transpose((0,2,1))*(dx))*1e-6 
        else:
            aux_transp = (vel_u.T*(-dy) + vel_v.T*(dx))*1e-6
            
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
            list_dimname['time'] , list_dimsize['time'] , list_dimval['time']  = 'time', data.dims['time'], data.time.data #pd.to_datetime(data.time)            
        if   'nz1'  in data.dims: 
            list_dimname['depth'], list_dimsize['depth'], list_dimval['depth'] = 'nz1' , mesh.zmid.size   , mesh.zmid
        elif 'nz'   in data.dims: 
            list_dimname['depth'], list_dimsize['depth'], list_dimval['depth'] = 'nz'  , mesh.zlev.size   , mesh.zlev
        list_dimname['horiz'], list_dimsize['horiz'] = 'npts', transect['path_dx'].size    
        
        #_______________________________________________________________________
        # define variable 
        gattrs = data.attrs
        gattrs['proj']          = 'index+depth+xy'
        
        data_vars = dict()
        aux_attr  = data[vname].attrs
        #aux_attr['long_name'], aux_attr['units'] = 'Transport through cross-section', 'Sv'
        aux_attr['long_name'], aux_attr['units'] = 'Volume Transport', 'Sv'
        aux_attr['transect_name'] = transect['Name']
        aux_attr['transect_lon']  = transect['lon']
        aux_attr['transect_lat']  = transect['lat']
        if do_transectattr: aux_attr['transect'] = transect
        data_vars['transp'] = (list(list_dimname.values()), aux_transp, aux_attr) 
        
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
                print('neto transport:', transp[-1]['transp'].sum(dim=('npts','nz1'), skipna=True).data,' [Sv]')
                print(' (+) transport:', transp[-1]['transp'].where(transp[-1]['transp']>0).sum(dim=('npts','nz1'), skipna=True).data,' [Sv]')
                print(' (-) transport:', transp[-1]['transp'].where(transp[-1]['transp']<0).sum(dim=('npts','nz1'), skipna=True).data,' [Sv]')
        else:
            transp = xr.Dataset(data_vars=data_vars, coords=coords, attrs=gattrs )
            # we have to set the time here with assign_coords otherwise if its 
            # setted in xr.Dataset(..., coords=dict(...),...)xarray does not 
            # recognize the cfttime format and things like data['time.year']
            # are not possible
            if 'time' in data.dims: transp = transp.assign_coords(time=data.time)  
            if do_info:
                print('neto transport:', transp['transp'].sum(dim=('npts','nz1'), skipna=True).data,' [Sv]')
                print(' (+) transport:', transp['transp'].where(transp['transp']>0).sum(dim=('npts','nz1'), skipna=True).data,' [Sv]')
                print(' (-) transport:', transp['transp'].where(transp['transp']<0).sum(dim=('npts','nz1'), skipna=True).data,' [Sv]')
    #___________________________________________________________________________
    #if do_info: print('        elapsed time: {:3.2f}min.'.format((clock.time()-t1)/60.0))
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
                elem_i_P = nodeinelem[:,transects[0]['edge_cut_ni'][:,0]]
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
                elem_i_P = nodeinelem[:,transects[0]['edge_cut_ni'][:,1]]
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
            print(dist)
            #zoom = (np.pi*6367.5)/transect['edge_cut_dist'].max()
            zoom = (np.pi*Rearth)/dist
            del(dist, x, y, z)
        else:
            if proj == 'nears': proj = 'rob'
    
    #___________________________________________________________________________
    orig = ccrs.PlateCarree()
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
        points = orig.transform_points(proj, 
                                    tri.x[tri.triangles].sum(axis=1)/3, 
                                    tri.y[tri.triangles].sum(axis=1)/3)
        xpts, ypts = points[:,0].flatten().tolist(), points[:,1].flatten().tolist()
        crs_pts = list(zip(xpts,ypts))
        fig_pts = ax.transData.transform(crs_pts)
        ax_pts  = ax.transAxes.inverted().transform(fig_pts)
        x, y    =  ax_pts[:,0], ax_pts[:,1]
        e_idxbox= (x>=-0.05) & (x<=1.05) & (y>=-0.05) & (y<=1.05)
        ax.triplot(tri.x, tri.y, tri.triangles[e_idxbox,:], color='w', linewidth=0.25, alpha=0.35, transform=orig)
    
    #___________________________________________________________________________
    #intersected edges 
    if edge is not None:
        ax.plot(mesh.n_x[edge[:,transect['edge_cut_i']]], mesh.n_y[edge[:,transect['edge_cut_i']]],'-', color=[0.75,0.75,0.75], linewidth=1.5, transform=orig)

    # intersection points
    ax.plot(transect['edge_cut_P'][ :, 0], transect['edge_cut_P'][ :, 1],'limegreen', marker='None', linestyle='-', markersize=7, markerfacecolor='w', linewidth=3.0, transform=orig)

    #transport path
    if do_path:
        ax.plot(transect['path_xy'   ][ :, 0], transect['path_xy'   ][ :, 1],'m', marker='None', linestyle='-', markersize=7, markerfacecolor='w', linewidth=1.0, transform=orig)

    # end start point [green] and end point [red]
    ax.plot(transect['edge_cut_P'][ 0, 0], transect['edge_cut_P'][ 0, 1],'k', marker='o', linestyle='None', markersize=5, markerfacecolor='g', transform=orig)
    ax.plot(transect['edge_cut_P'][-1, 0], transect['edge_cut_P'][-1, 1],'k', marker='o', linestyle='None', markersize=5, markerfacecolor='r', transform=orig)
    
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
