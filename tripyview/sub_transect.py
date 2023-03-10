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

#+___PRE-ANALYSE ARBITRARY CROSS-SECTION LINE__________________________________+
#|                                                                             |
#+_____________________________________________________________________________+
def do_analyse_transects(input_transect, mesh, edge, edge_tri, edge_dxdy_l, edge_dxdy_r, 
                          which_res='res', res=1.0, npts=500):
    
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
    # normal vector 
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
        # print(theta*180/np.pi)
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
            e_xR, e_yR = np.sum(mesh.n_x[mesh.e_i[edge_elem[1],:]])/3.0, np.sum(mesh.n_y[mesh.e_i[edge_elem[1],:]])/3.0
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
        e_xL, e_yL = np.sum(mesh.n_x[mesh.e_i[edge_elem[0],:]])/3.0, np.sum(mesh.n_y[mesh.e_i[edge_elem[0],:]])/3.0
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
                    transect_bnd['edge_cut_i'   ] = np.hstack((transect['edge_cut_i'   ][ni_s:ni_e  ], np.ones((2,), dtype=np.int)*-1 ))
                    transect_bnd['edge_cut_evec'] = np.vstack((transect['edge_cut_evec'][ni_s:ni_e,:], np.ones((2,2))*np.nan))
                    transect_bnd['edge_cut_P'   ] = np.vstack((transect['edge_cut_P'   ][ni_s:ni_e,:], transect['edge_cut_P'   ][[ni,ni+1],:]))
                    transect_bnd['edge_cut_midP'] = np.vstack((transect['edge_cut_midP'][ni_s:ni_e,:], transect['edge_cut_midP'][[ni,ni+1],:]))
                    transect_bnd['edge_cut_lint'] = np.hstack((transect['edge_cut_lint'][ni_s:ni_e  ], np.ones((2,))*np.nan ))
                    transect_bnd['edge_cut_ni'  ] = np.vstack((transect['edge_cut_ni'  ][ni_s:ni_e,:], np.ones((2,2), dtype=np.int)*-1))
                else:
                    transect_bnd['edge_cut_i'   ] = np.hstack((transect_bnd['edge_cut_i'   ], transect['edge_cut_i'   ][ni_s:ni_e  ],np.ones((2,), dtype=np.int)*-1 ))
                    transect_bnd['edge_cut_evec'] = np.vstack((transect_bnd['edge_cut_evec'], transect['edge_cut_evec'][ni_s:ni_e,:],np.ones((2,2))*np.nan))
                    transect_bnd['edge_cut_P'   ] = np.vstack((transect_bnd['edge_cut_P'   ], transect['edge_cut_P'   ][ni_s:ni_e,:],transect['edge_cut_P'   ][[ni,ni+1],:]))
                    transect_bnd['edge_cut_midP'] = np.vstack((transect_bnd['edge_cut_midP'], transect['edge_cut_midP'][ni_s:ni_e,:],transect['edge_cut_midP'][[ni,ni+1],:]))
                    transect_bnd['edge_cut_lint'] = np.hstack((transect_bnd['edge_cut_lint'], transect['edge_cut_lint'][ni_s:ni_e  ],np.ones((2,))*np.nan ))
                    transect_bnd['edge_cut_ni'  ] = np.vstack((transect_bnd['edge_cut_ni'  ], transect['edge_cut_ni'  ][ni_s:ni_e,:],np.ones((2,2), dtype=np.int)*-1))
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
    


#+___COMPUTE VOLUME TRANSPORT THROUGH TRANSECT_________________________________+
#|                                                                             |
#+_____________________________________________________________________________+ 
def calc_transect_transp(mesh, data, transects, do_transectattr=False, do_info=True):
    t1=clock.time()
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
        vel_u = data[vname ].isel(elem=transect['path_ei']).load().values
        vel_v = data[vname2].isel(elem=transect['path_ei']).load().values
        
        #_______________________________________________________________________
        # rotate vectors 
        if 'time' in list(data.dims):
            for ii, ei in enumerate(transect['path_ei']):
                vel_u[:,ii,mesh.e_iz[ei]:], vel_v[:,ii,mesh.e_iz[ei]:] = np.nan, np.nan
                
            for nti in range(data.dims['time']):
                vel_u[nti,:,:], vel_v[nti,:,:] = vec_r2g(mesh.abg, 
                                       mesh.n_x[transect['path_ni']].sum(axis=1)/3.0, 
                                       mesh.n_y[transect['path_ni']].sum(axis=1)/3.0,
                                       vel_u[nti,:,:], vel_v[nti,:,:])            
        else: 
            print(vel_u.shape)
            print(vel_v.shape)
            print(transect['path_ei'].shape)
            for ii, ei in enumerate(transect['path_ei']):
                vel_u[ii, mesh.e_iz[ei]:], vel_v[ii,mesh.e_iz[ei]:] = np.nan, np.nan
                 
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
        # aux_transp = (vel_u.T*dy + vel_v.T*(-dx))*1e-6
        
        # proper transport sign with respect to normal vector of section 
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
        list_dimname, list_dimsize, list_dimval = list(), list(), list()
        
        if   'time' in data.dims: list_dimname.append('time'), list_dimsize.append(data.dims['time']), list_dimval.append(data['time'].values)             
        if   'nz1'  in data.dims: list_dimname.append('nz1' ), list_dimsize.append(mesh.zmid.size   ), list_dimval.append(mesh.zmid) 
        elif 'nz_1' in data.dims: list_dimname.append('nz1' ), list_dimsize.append(mesh.zmid.size   ), list_dimval.append(mesh.zmid)  
        elif 'nz'   in data.dims: list_dimname.append('nz'  ), list_dimsize.append(mesh.zlev.size   ), list_dimval.append(mesh.zlev)  
        list_dimname.append('npts'), list_dimsize.append(transect['path_dx'].size)
        
        # define variable 
        data_vars = dict()
        aux_attr  = data[vname].attrs
        #aux_attr['long_name'], aux_attr['units'] = 'Transport through cross-section', 'Sv'
        aux_attr['long_name'], aux_attr['units'] = 'Volume Transport', 'Sv'
        aux_attr['transect_name'] = transect['Name']
        aux_attr['transect_lon']  = transect['lon']
        aux_attr['transect_lat']  = transect['lat']
        if do_transectattr: aux_attr['transect'] = transect
        
        data_vars['transp'] = (list_dimname, aux_transp, aux_attr) 
        # define coordinates
        if 'time' in data.dims:
            coords = {'time ' : (list_dimname[0], list_dimval[0]),
                      'depth' : (list_dimname[1], list_dimval[1]),
                      'lon'   : (list_dimname[2], lon),
                      'lat'   : (list_dimname[2], lat),
                      'dst'   : (list_dimname[2], dst)}
        else:    
            coords = {'depth' : (list_dimname[0], list_dimval[0]),
                      'lon'   : (list_dimname[1], lon),
                      'lat'   : (list_dimname[1], lat),
                      'dst'   : (list_dimname[1], dst)}
         
        # create dataset
        if is_list:
            transp.append(xr.Dataset(data_vars=data_vars, coords=coords, attrs=data.attrs))
            if do_info:
                print('neto transport:', transp[-1]['transp'].sum(dim=('npts','nz1'), skipna=True).data,' [Sv]')
                print(' (+) transport:', transp[-1]['transp'].where(transp[-1]['transp']>0).sum(dim=('npts','nz1'), skipna=True).data,' [Sv]')
                print(' (-) transport:', transp[-1]['transp'].where(transp[-1]['transp']<0).sum(dim=('npts','nz1'), skipna=True).data,' [Sv]')
        else:
            transp = xr.Dataset(data_vars=data_vars, coords=coords, attrs=data.attrs)
            if do_info:
                print('neto transport:', transp['transp'].sum(dim=('npts','nz1'), skipna=True).data,' [Sv]')
                print(' (+) transport:', transp['transp'].where(transp['transp']>0).sum(dim=('npts','nz1'), skipna=True).data,' [Sv]')
                print(' (-) transport:', transp['transp'].where(transp['transp']<0).sum(dim=('npts','nz1'), skipna=True).data,' [Sv]')
    #___________________________________________________________________________
    if do_info: print('        elapsed time: {:3.2f}min.'.format((clock.time()-t1)/60.0))
    return(transp)




#+___COMPUTE TRANSECT OF SCALAR VERTICE/ELEMENTAL VARIABLE_____________________+
#|                                                                             |
#+_____________________________________________________________________________+ 
def calc_transect_scalar(mesh, data, transects, nodeinelem=None, 
                         do_transectattr=False, do_info=True):
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
                
                # define lon, lat , distance arrays
                lon = transect['edge_cut_P'][:-1,0]
                lat = transect['edge_cut_P'][:-1,1]
                dst = transect['edge_cut_dist'][:-1]
                
            else:
                # average elemental values to edge node 1
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
        if   'nz1'  in data.dims: list_dimname.append('nz1' ), list_dimsize.append(mesh.zmid.size), list_dimval.append(mesh.zmid) 
        elif 'nz_1' in data.dims: list_dimname.append('nz_1'), list_dimsize.append(mesh.zmid.size), list_dimval.append(mesh.zmid)  
        elif 'nz'   in data.dims: list_dimname.append('nz'  ), list_dimsize.append(mesh.zlev.size), list_dimval.append(mesh.zlev)  
        list_dimname.append('npts'), list_dimsize.append(dst.size)
        
        # define variable 
        data_vars = dict()
        aux_attr  = data[vname].attrs
        aux_attr['transect_name'] = transect['Name']
        aux_attr['transect_lon']  = transect['lon']
        aux_attr['transect_lat']  = transect['lat']
        if do_transectattr: aux_attr['transect'] = transect
        
        data_vars['transp'] = (list_dimname, scalarPcut, aux_attr) 
        
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
            csects.append(xr.Dataset(data_vars=data_vars, coords=coords, attrs=data.attrs))
        else:
            csects = xr.Dataset(data_vars=data_vars, coords=coords, attrs=data.attrs)
            
    #___________________________________________________________________________
    if do_info: print('        elapsed time: {:3.2f}min.'.format((clock.time()-t1)/60.0))
    return(csects)



#+___PLOT TRANSECT POSITION____________________________________________________+
#|                                                                             |
#+_____________________________________________________________________________+ 
def plot_transect_position(mesh, transect, edge=None, zoom=None, fig=None,  figsize=[10,10],
                           resolution='low', do_path=True, do_labels=True, do_title=True,
                           do_grid=False, ax_pos=[0.90, 0.05, 0.45, 0.45]):
    #___________________________________________________________________________
    # compute zoom factor based on the length of the transect
    if zoom is None: 
        Rearth = 6367.5  # [km]
        x,y,z  = grid_cart3d(transect['path_xy'][[0,-1],0], transect['path_xy'][[0,-1],1], is_deg=True)
        dist   = np.arccos(x[0]*x[1] + y[0]*y[1] + z[0]*z[1])*Rearth
        #zoom = (np.pi*6367.5)/transect['edge_cut_dist'].max()
        zoom = (np.pi*Rearth)/dist
        del(dist, x, y, z)
    
    #___________________________________________________________________________
    orig = ccrs.PlateCarree()
    proj = ccrs.NearsidePerspective(central_longitude=transect['edge_cut_P'][:,0].mean(), 
                                    central_latitude =transect['edge_cut_P'][:,1].mean(), 
                                    satellite_height =35785831/zoom)

    #___________________________________________________________________________
    if fig is None: 
        fig = plt.figure(figsize=figsize)
        ax  = plt.axes(projection=proj)
    else:
        ax = fig.add_axes(ax_pos, projection=proj)
        
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
    ax.coastlines()
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



#+___PLOT TRANSECT LINE________________________________________________________+
#|                                                                             |
#+_____________________________________________________________________________+
def plot_transect(data, transects, figsize=[12, 6], 
              n_rc=[1, 1], do_grid=True, cinfo=None, do_rescale=False,
              do_reffig=False, ref_cinfo=None, ref_rescale=False,
              cbar_nl=8, cbar_orient='vertical', cbar_label=None, cbar_unit=None,
              do_bottom=True, max_dep=[], color_bot=[0.6, 0.6, 0.6], 
              pos_fac=1.0, pos_gap=[0.02, 0.02], do_save=None, save_dpi=600, 
              do_contour=True, do_clabel=True, title='descript', which_xaxis='lat', 
              pos_extend=[0.05, 0.08, 0.95,0.95], do_ylog=True, do_smooth=False, 
              do_position=True, mesh=None, 
            ):
    #____________________________________________________________________________
    fontsize = 12
    rescale_str = None
    
    #___________________________________________________________________________
    # make matrix with row colum index to know where to put labels
    rowlist = np.zeros((n_rc[0], n_rc[1]))
    collist = np.zeros((n_rc[0], n_rc[1]))       
    for ii in range(0,n_rc[0]): rowlist[ii,:]=ii
    for ii in range(0,n_rc[1]): collist[:,ii]=ii
    rowlist = rowlist.flatten()
    collist = collist.flatten()
    
    #___________________________________________________________________________    
    # create figure and axes
    fig, ax = plt.subplots( n_rc[0],n_rc[1], figsize=figsize, 
                            gridspec_kw=dict(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.05,),
                            constrained_layout=False, sharex=True, sharey=True)
    
    #___________________________________________________________________________    
    # flatt axes if there are more than 1
    if isinstance(ax, np.ndarray): ax = ax.flatten()
    else:                          ax = [ax] 
    nax = len(ax)
     
    #___________________________________________________________________________
    # data must be list filled with xarray data
    if not isinstance(data  , list): data  = [data]
    ndata = len(data) 
    
    #___________________________________________________________________________
    # set up color info 
    if do_reffig:
        ref_cinfo = do_setupcinfo(ref_cinfo, [data[0]], ref_rescale, do_index=True)
        cinfo     = do_setupcinfo(cinfo    , data[1:] , do_rescale , do_index=True)
    else:    
        cinfo     = do_setupcinfo(cinfo, data, do_rescale, do_index=True)

    #_______________________________________________________________________
    # setup normalization log10, symetric log10, None
    which_norm = do_compute_scalingnorm(cinfo, do_rescale)
    if do_reffig:
        which_norm_ref = do_compute_scalingnorm(ref_cinfo, ref_rescale)
        
    #___________________________________________________________________________
    # loop over axes
    hpall=list()
    for ii in range(0,ndata):
        
        #_______________________________________________________________________
        # limit data to color range
        vname= list(data[ii][0].keys())[0]
        data_plot = data[ii][0][vname].values.copy()
        
        #_______________________________________________________________________
        # setup x-coord and y-coord
        # determine if lon and lat are valid axis for plotting if not use distance
        # as x-axis
        auxdst = np.diff(data[0][0]['dst'].values)
        auxlat, auxlon = data[0][0]['lat'].values[1:], data[0][0]['lon'].values[1:]
        auxlat, auxlon = auxlat[auxdst!=0.0], auxlon[auxdst!=0.0]
        is_ok = (np.isnan(auxlon)==False)
        auxlat, auxlon = auxlat[is_ok], auxlon[is_ok]
        if   np.all(np.diff(auxlat)==0) : which_xaxis='lon'
        elif np.all(np.diff(auxlon)==0) : which_xaxis='lat'
        elif np.any(np.diff(auxlat)==0) and not np.any(np.diff(auxlon)==0): which_xaxis='lon'
        elif np.any(np.diff(auxlon)==0) and not np.any(np.diff(auxlat)==0): which_xaxis='lat'
        else                            : which_xaxis='dist' 
        
        if   which_xaxis=='lat' : data_x, str_xlabel = data[ii][0]['lat'].values, 'Latitude [deg]'
        elif which_xaxis=='lon' : data_x, str_xlabel = data[ii][0]['lon'].values, 'Longitude [deg]'
        elif which_xaxis=='dist': data_x, str_xlabel = data[ii][0]['dst'].values, 'Distance from start point [km]'
        else: raise ValueError('these definition for which_xaxis is not supported')
        
        data_y, str_ylabel = data[ii][0]['depth'].values, 'Depth [m]'    
        data_y = np.abs(data_y)
        
        #_______________________________________________________________________
        if do_reffig: 
            if ii==0: cinfo_plot, which_norm_plot = ref_cinfo, which_norm_ref
            else    : cinfo_plot, which_norm_plot = cinfo    , which_norm
        else        : cinfo_plot, which_norm_plot = cinfo    , which_norm
        
        #_______________________________________________________________________
        # be sure there are no holes
        data_plot[data_plot<cinfo_plot['clevel'][ 0]] = cinfo_plot['clevel'][ 0]+np.finfo(np.float32).eps
        data_plot[data_plot>cinfo_plot['clevel'][-1]] = cinfo_plot['clevel'][-1]-np.finfo(np.float32).eps
        
        #___________________________________________________________________________
        # apply horizontal smoothing filter
        if do_smooth: 
            filt=np.array([1,2,1])
            filt=filt[np.newaxis,:]
            #filt=np.ones((3,3))
            #filt=np.array([[0.5,1,0.5],[1,2,1],[0.5,1,0.5]])
            filt=filt/np.sum(filt.flatten())
            is_nan = np.isnan(data_plot)
            data_plot[is_nan] = 0.0
            data_plot = convolve2d(data_plot, filt, mode='same', boundary='symm') 
            data_plot[is_nan] = np.nan
            del(is_nan, filt)
            
        #_______________________________________________________________________
        # plot MOC
        hp=ax[ii].contourf(data_x, data_y, data_plot, 
                           levels=cinfo_plot['clevel'], extend='both', cmap=cinfo_plot['cmap'],
                           norm = which_norm_plot)
        hpall.append(hp)
        
        if do_contour: 
            tickl    = cinfo_plot['clevel']
            ncbar_l  = len(tickl)
            idx_cref = np.where(cinfo_plot['clevel']==cinfo_plot['cref'])[0]
            idx_cref = np.asscalar(idx_cref)
            nstep    = ncbar_l/cbar_nl
            nstep    = np.max([np.int(np.floor(nstep)),1])
            
            idx = np.arange(0, ncbar_l, 1)
            idxb = np.ones((ncbar_l,), dtype=bool)                
            idxb[idx_cref::nstep]  = False
            idxb[idx_cref::-nstep] = False
            idx_yes = idx[idxb==False]
            
            cont=ax[ii].contour(data_x, data_y, data_plot,
                            levels=cinfo_plot['clevel'][idx_yes], colors='k', linewidths=[0.5],
                            norm = which_norm_plot) #linewidths=[0.5,0.25])
            #if do_clabel: 
                #ax[ii].clabel(cont, cont.levels, inline=1, inline_spacing=1, fontsize=6, fmt='%1.1f Sv')
                #ax[ii].clabel(cont, cont.levels[np.where(cont.levels!=cinfo_plot['cref'])], 
                            #inline=1, inline_spacing=1, fontsize=6, fmt='%1.1f Sv')
            
        #___________________________________________________________________
        #ylim = np.sum(~np.isnan(data_plot),axis=0).max()-1
        ylim = np.sum(~np.isnan(data_plot),axis=0).max()
        if ylim<data_y.shape[0]-1: ylim=ylim+1
        if np.isscalar(max_dep)==False: max_dep=data_y[ylim]
        
        # plot bottom patch
        aux_bot = np.ones(data_plot.shape,dtype='int16')
        aux_bot[np.isnan(data_plot)]=0
        aux_bot = aux_bot.sum(axis=0)
        #aux[aux!=0]=aux[aux!=0]-1
        aux_bot=aux_bot-1
        bottom = np.abs(data_y[aux_bot])
        bottom[aux_bot<0]=0.1
        
        # smooth bottom patch
        filt   = np.array([1,2,1]) #np.array([1,2,3,2,1])
        filt   = filt/np.sum(filt)
        aux    = np.concatenate( (np.ones((filt.size,))*bottom[0],bottom,np.ones((filt.size,))*bottom[-1] ) )
        aux    = np.convolve(aux,filt,mode='same')
        bottom = aux[filt.size:-filt.size]
        bottom[aux_bot<0]=0.1
        
        # plot bottom patch 
        ax[ii].fill_between(data_x, bottom, max_dep,color=color_bot)#,alpha=0.95)
        ax[ii].plot(data_x, bottom, color='k')
        ax[ii].set_facecolor(color_bot) 
        #_______________________________________________________________________
        # fix color range
        for im in ax[ii].get_images(): im.set_clim(cinfo_plot['clevel'][ 0], cinfo_plot['clevel'][-1])
        
        #_______________________________________________________________________
        # plot grid lines 
        if do_grid: ax[ii].grid(color='k', linestyle='-', linewidth=0.25,alpha=1.0)
        
        #_______________________________________________________________________
        # set description string plus x/y labels
        isnotnan = np.isnan(data_plot[:,0])==False
        isnotnan = isnotnan.sum()-1
        
        if title is not None: 
            #txtx, txty = data_x[0]+(data_x[-1]-data_x[0])*0.025, data_y[isnotnan]-(data_y[isnotnan]-data_y[0])*0.025                    
            txtx, txty = data_x[0]+(data_x[-1]-data_x[0])*0.015, max_dep-(max_dep-data_y[0])*0.015
            if   isinstance(title,str) : 
                # if title string is 'descript' than use descript attribute from 
                # data to set plot title 
                if title=='descript' and ('descript' in data[ii][0][vname].attrs.keys() ):
                    txts = data[ii][0][vname].attrs['descript']
                else:
                    txts = title
            # is title list of string        
            elif isinstance(title,list):   
                txts = title[ii]
            ax[ii].text(txtx, txty, txts, fontsize=12, fontweight='bold',
                        horizontalalignment='left', verticalalignment='bottom')
        
        #_______________________________________________________________________
        # set x/y label
        if collist[ii]==0        : ax[ii].set_ylabel(str_ylabel, fontsize=12)
        if rowlist[ii]==n_rc[0]-1: ax[ii].set_xlabel(str_xlabel, fontsize=12)
        
        #_______________________________________________________________________
        if do_ylog: 
            ax[ii].grid(True,which='major')
            #ax[ii].set_yscale('log')
            ax[ii].set_yscale('function', functions=(forward, inverse))
            #yticklog = np.array([5,10,25,50,100,250,500,1000,2000,4000,6000])
            yticklog = np.array([10,25,50,100,250,500,1000,2000,4000,6000])
            ax[ii].set_yticks(yticklog)
            if data_y[0]==0: ax[ii].set_ylim(data_y[1],max_dep)
            else           : ax[ii].set_ylim(data_y[0],max_dep)
            ax[ii].invert_yaxis()
            
        else:
            ax[ii].set_ylim(data_y[0],max_dep)
            ax[ii].invert_yaxis()
            ax[ii].grid(True,which='major')
            
        #ax[ii].set_yticks([5,10,25,50,100,250,500,1000,2000,4000,6000])
        ax[ii].get_yaxis().set_major_formatter(ScalarFormatter())
        
        xmin, xmax = np.nanmin(data_x), np.nanmax(data_x)
        
        ax[ii].set_xlim(xmin-(xmax-xmin)*0.05, xmax+(xmax-xmin)*0.05)
    nax_fin = ii+1
    
    #___________________________________________________________________________
    # set superior title
    if 'transect_name' in data[ii][0][vname].attrs.keys():
        fig.suptitle( data[ii][0][vname].attrs['transect_name'], x=0.5, y=0.925, #y=1.04, 
                     fontsize=14, fontweight='bold',
                     horizontalalignment='center', verticalalignment='bottom')
    
    #___________________________________________________________________________
    # delete axes that are not needed
    #for jj in range(nax_fin, nax): fig.delaxes(ax[jj])
    for jj in range(ndata, nax): fig.delaxes(ax[jj])
    if nax != nax_fin-1: ax = ax[0:nax_fin]
    
    #___________________________________________________________________________
    # delete axes that are not needed
    if do_reffig==False:
        cbar = fig.colorbar(hp, orientation=cbar_orient, ax=ax, ticks=cinfo['clevel'], 
                        extendrect=False, extendfrac=None,
                        drawedges=True, pad=0.025, shrink=1.0)
        
        # do formatting of colorbar 
        cbar = do_cbar_formatting(cbar, do_rescale, cbar_nl, fontsize, cinfo['clevel'])
        
        # do labeling of colorbar
        if cbar_label is None : 
            if   'short_name' in data[0][0][vname].attrs:
                cbar_label = data[0][0][ vname ].attrs['short_name']
            elif 'long_name' in data[0][0][vname].attrs:
                cbar_label = data[0][0][ vname ].attrs['long_name']
        if cbar_unit  is None : cbar_label = cbar_label+' ['+data[0][0][ vname ].attrs['units']+']'
        else                  : cbar_label = cbar_label+' ['+cbar_unit+']'
        if 'str_ltim' in data[0][0][vname].attrs.keys():
            cbar_label = cbar_label+'\n'+data[0][0][vname].attrs['str_ltim']
        cbar.set_label(cbar_label, size=fontsize+2)
    else:
        cbar=list()
        for ii, aux_ax in enumerate(ax): 
            cbar_label =''
            if ii==0: 
                aux_cbar = plt.colorbar(hpall[ii], orientation=cbar_orient, ax=aux_ax, ticks=ref_cinfo['clevel'], 
                            extendrect=False, extendfrac=None, drawedges=True, pad=0.025, shrink=1.0,)  
                aux_cbar = do_cbar_formatting(aux_cbar, ref_rescale, cbar_nl, fontsize, ref_cinfo['clevel'])
            else:     
                aux_cbar = plt.colorbar(hpall[ii], orientation=cbar_orient, ax=aux_ax, ticks=cinfo['clevel'], 
                            extendrect=False, extendfrac=None, drawedges=True, pad=0.025, shrink=1.0,)  
                aux_cbar = do_cbar_formatting(aux_cbar, do_rescale, cbar_nl, fontsize, cinfo['clevel'])
                #cbar_label ='anom. '
            # do labeling of colorbar
            #if cbar_label is None : 
            if   'short_name' in data[ii][0][vname].attrs:
                cbar_label = cbar_label+data[ii][0][ vname ].attrs['short_name']
            elif 'long_name' in data[ii][0][vname].attrs:
                cbar_label = cbar_label+data[ii][0][ vname ].attrs['long_name']
            if cbar_unit  is None : cbar_label = cbar_label+' ['+data[ii][0][ vname ].attrs['units']+']'
            else                  : cbar_label = cbar_label+' ['+cbar_unit+']'
            if 'str_ltim' in data[ii][0][vname].attrs.keys():
                cbar_label = cbar_label+'\n'+data[ii][0][vname].attrs['str_ltim']
            aux_cbar.set_label(cbar_label, size=fontsize+2)
            cbar.append(aux_cbar)
    #___________________________________________________________________________
    # repositioning of axes and colorbar
    if do_reffig==False:
        ax, cbar = do_reposition_ax_cbar(ax, cbar, rowlist, collist, pos_fac, pos_gap, 
                                        title=None, extend=pos_extend)
    
        if do_position==True and mesh is not None:
            cbar_pos = cbar.ax.get_tightbbox(fig.canvas.get_renderer())
            fsize_dpi= fig.get_size_inches()*fig.dpi
            cbar_x, cbar_y = cbar_pos.x1/fsize_dpi[0], cbar_pos.y0/fsize_dpi[1]
            fig, axp = plot_transect_position(mesh, transects[0], fig=fig, 
                                            do_labels=False, do_path=False, do_title=False, 
                                            ax_pos=[cbar_x-0.05, cbar_y, 0.25, 0.25])
        
    plt.show()
    fig.canvas.draw()
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, fig, dpi=save_dpi)
    
    #___________________________________________________________________________
    return(fig, ax, cbar)



#+___PLOT TIME SERIES OF TRANSPORT THROUGH SECTION_____________________________+
#|                                                                             |
#+_____________________________________________________________________________+
def plot_transect_transp_t(time, tseries_list, input_names, transect, which_cycl=None, 
                       do_allcycl=False, do_concat=False, str_descript='', str_time='', figsize=[], 
                       do_save=None, save_dpi=600, do_pltmean=True, do_pltstd=False,
                       ymaxstep=None, xmaxstep=5, ylabel=None):    
    
    if len(figsize)==0: figsize=[13,6.5]
    if do_concat: figsize[0] = figsize[0]*2
    fig,ax= plt.figure(figsize=figsize),plt.gca()
    
    #___________________________________________________________________________
    # setup colormap
    if do_allcycl: 
        if which_cycl is not None:
            cmap = categorical_cmap(np.int32(len(tseries_list)/which_cycl), which_cycl, cmap="tab10")
        else:
            cmap = categorical_cmap(len(tseries_list), 1, cmap="tab10")
    else:
        if do_concat: do_concat=False
        cmap = categorical_cmap(len(tseries_list), 1, cmap="tab10")
    
    #___________________________________________________________________________
    ii=0
    ii_cycle=1
    for ii_ts, (tseries, tname) in enumerate(zip(tseries_list, input_names)):
        
        if tseries.ndim>1: tseries = tseries.squeeze()
        auxtime = time.copy()
        if np.mod(ii_ts+1,which_cycl)==0 or do_allcycl==False:
            
            if do_concat: auxtime = auxtime + (time[-1]-time[0]+1)*(ii_cycle-1)
            hp=ax.plot(auxtime,tseries, 
                   linewidth=1.5, label=tname, color=cmap.colors[ii_ts,:], 
                   marker='o', markerfacecolor='w', markersize=5, #path_effects=[path_effects.SimpleLineShadow(offset=(1.5,-1.5),alpha=0.3),path_effects.Normal()],
                   zorder=2)
            
            if do_pltmean: 
                # plot mean value with triangle 
                plt.plot(time[0]-(time[-1]-time[0])*0.0120, tseries.mean(),
                        marker='<', markersize=8, markeredgecolor='k', markeredgewidth=0.5,
                        color=hp[0].get_color(),clip_box=False,clip_on=False, zorder=3)
            if do_pltstd:
                # plot std. range
                plt.plot(time[0]-(time[-1]-time[0])*0.015, tseries.mean()+tseries.std(),
                        marker='^', markersize=6, markeredgecolor='k', markeredgewidth=0.5,
                        color=hp[0].get_color(),clip_box=False,clip_on=False, zorder=3)
                
                plt.plot(time[0]-(time[-1]-time[0])*0.015, tseries.mean()-tseries.std(),
                        marker='v', markersize=6, markeredgecolor='k', markeredgewidth=0.5,
                        color=hp[0].get_color(),clip_box=False,clip_on=False, zorder=3)
        
        else:
            if do_concat: auxtime = auxtime + (time[-1]-time[0]+1)*(ii_cycle-1)
            hp=ax.plot(auxtime, tseries, 
                   linewidth=1.5, label=tname, color=cmap.colors[ii_ts,:],
                   zorder=1) #marker='o', markerfacecolor='w', 
                   # path_effects=[path_effects.SimpleLineShadow(offset=(1.5,-1.5),alpha=0.3),path_effects.Normal()])
        
        ii_cycle=ii_cycle+1
        if ii_cycle>which_cycl: ii_cycle=1
        
    #___________________________________________________________________________
    ax.legend(shadow=True, fancybox=True, frameon=True, #mode='None', 
              bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
              #bbox_to_anchor=(1.04, 1.0), ncol=1) #loc='lower right', 
    ax.set_xlabel('Time [years]',fontsize=12)
    
    if ylabel is None: ax.set_ylabel('{:s} in [Sv]'.format('Transport'),fontsize=12)
    else             : ax.set_ylabel('{:s} in [Sv]'.format(ylabel),fontsize=12)
    ax.set_title(transect['Name'], fontsize=12, fontweight='bold')
    
    #___________________________________________________________________________
    if do_concat: xmaxstep=20
    xmajor_locator = MultipleLocator(base=xmaxstep) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(xmajor_locator)
    
    if ymaxstep is not None: 
        ymajor_locator = MultipleLocator(base=ymaxstep) # this locator puts ticks at regular intervals
        ax.yaxis.set_major_locator(ymajor_locator)
    
    if not do_concat:
        xminor_locator = AutoMinorLocator(5)
        yminor_locator = AutoMinorLocator(4)
        ax.yaxis.set_minor_locator(yminor_locator)
        ax.xaxis.set_minor_locator(xminor_locator)
    
    plt.grid(which='major')
    if not do_concat:
        plt.xlim(time[0]-(time[-1]-time[0])*0.015,time[-1]+(time[-1]-time[0])*0.015)    
    else:    
        plt.xlim(time[0]-(time[-1]-time[0])*0.015,time[-1]+(time[-1]-time[0]+1)*(which_cycl-1)+(time[-1]-time[0])*0.015)    
    
    #___________________________________________________________________________
    plt.show()
    fig.canvas.draw()
    
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, fig, dpi=save_dpi)
    
    #___________________________________________________________________________
    return(fig,ax)



#+___COMPUTE ZONAL MEAN SECTION BASED ON BINNING_______________________________+
#|                                                                             |
#+_____________________________________________________________________________+
def load_zmeantransect_fesom2(mesh, data, box_list, dlat=0.5, boxname=None, do_harithm='mean', 
                                do_compute=True, do_outputidx=False, diagpath=None, do_onelem=False, 
                                do_info=False, do_smooth=True, 
                                **kwargs,):
    
    #___________________________________________________________________________
    # str_anod    = ''
    index_list  = []
    idxin_list  = []
    cnt         = 0
    
    #___________________________________________________________________________
    # loop over box_list
    vname = list(data.keys())[0]
    if   'nz'  in list(data[vname].dims): which_ddim, ndi, depth = 'nz' , mesh.nlev  , mesh.zlev     
    elif 'nz1' in list(data[vname].dims) or 'nz_1' in list(data[vname].dims): 
        which_ddim, ndi, depth = 'nz1', mesh.nlev-1, mesh.zmid
    
    
    for box in box_list:
        if not isinstance(box, shp.Reader) and not box =='global' and not box==None :
            if len(box)==2: boxname, box = box[1], box[0]
        
        #_______________________________________________________________________
        # compute box mask index for nodes 
        n_idxin=do_boxmask(mesh,box)
        if do_onelem: 
            e_idxin = n_idxin[mesh.e_i].sum(axis=1)>=1  
        
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
        if do_onelem:
            #_______________________________________________________________________
            # load elem area from diag file
            if ( os.path.isfile(diagpath)):
                mat_area = xr.open_mfdataset(diagpath, parallel=True, **kwargs)['elem_area']
                mat_area = mat_area.isel(elem=e_idxin).compute()   
                mat_area = mat_area.expand_dims({which_ddim:depth}).transpose()
                mat_iz   = xr.open_mfdataset(diagpath, parallel=True, **kwargs)['nlevels']-1
                mat_iz   = mat_iz.isel(elem=e_idxin).compute()   
                if which_ddim=='nz1': mat_iz=mat_iz-1    
            else: 
                raise ValueError('could not find ...mesh.diag.nc file')
            
            #_______________________________________________________________________
            # create meridional bins
            e_y   = mesh.n_y[mesh.e_i[e_idxin,:]].sum(axis=1)/3.0
            lat   = np.arange(np.floor(e_y.min())+dlat/2, 
                            np.ceil( e_y.max())-dlat/2, 
                            dlat)
            lat_i = (( e_y-lat[0])/dlat ).astype('int')
            
            #_______________________________________________________________________    
            # mean over elements + select MOC basin 
            if 'time' in list(data.dims): 
                wdim = ['time','elem',which_ddim]
                wdum = data[vname].data[:, mesh.e_i[e_idxin,:], :].sum(axis=2)/3.0 * 1e-6
            else                        : 
                wdim = ['elem',which_ddim]
                wdum = data[vname].data[mesh.e_i[e_idxin,:], :].sum(axis=1)/3.0 * 1e-6
            mat_mean = xr.DataArray(data=wdum, dims=wdim)
            mat_mean = mat_mean.fillna(0.0)
            del wdim, wdum
            
            #_______________________________________________________________________
            # calculate area weighted mean
            if 'time' in list(data.dims):
                nt = data['time'].values.size
                for nti in range(nt):
                    mat_mean.data[nti,:,:] = np.multiply(mat_mean.data[nti,:,:], mat_area.data)    
                    
                # be sure ocean floor is setted to zero 
                for di in range(0,ndi): 
                    mat_mean.data[:, np.where(di>mat_iz)[0], di]=0.0
                
            else:
                mat_mean.data = np.multiply(mat_mean.data, mat_area.data)
                
                # be sure ocean floor is setted to zero 
                for di in range(0,ndi): 
                    mat_mean.data[np.where(di>mat_iz)[0], di]=0.0
            del mat_area
        
        # compute area weighted vertical velocities on vertices
        else:     
            #_______________________________________________________________________
            # load vertice cluster area from diag file
            if ( os.path.isfile(diagpath)):
                mat_area = xr.open_mfdataset(diagpath, parallel=True, **kwargs)['nod_area'].transpose() 
                if   'nod_n' in list(mat_area.dims): mat_area = mat_area.isel(nod_n=n_idxin).compute()  
                elif 'nod2'  in list(mat_area.dims): mat_area = mat_area.isel(nod2=n_idxin).compute()     
                
                mat_iz   = xr.open_mfdataset(diagpath, parallel=True, **kwargs)['nlevels_nod2D']-1
                if   'nod_n' in list(mat_area.dims): mat_iz   = mat_iz.isel(nod_n=n_idxin).compute()
                elif 'nod2'  in list(mat_area.dims): mat_iz   = mat_iz.isel(nod2=n_idxin).compute()   
            else: 
                raise ValueError('could not find ...mesh.diag.nc file')
            
            # data are on mid depth levels
            if which_ddim=='nz1': 
                mat_iz   = mat_iz - 1
                mat_area = mat_area[:,:-1]
            
            #_______________________________________________________________________
            # create meridional bins
            lat   = np.arange(np.floor(mesh.n_y[n_idxin].min())+dlat/2, 
                            np.ceil( mesh.n_y[n_idxin].max())-dlat/2, 
                            dlat)
            lat_i = ( (mesh.n_y[n_idxin]-lat[0])/dlat ).astype('int')
                
            #_______________________________________________________________________    
            # select MOC basin 
            mat_mean = data[vname].isel(nod2=n_idxin)
            isnan = np.isnan(mat_mean.values)
            mat_mean.values[isnan] = 0.0
            mat_area.values[isnan] = 0.0
            del(isnan)
            #mat_mean = mat_mean.fillna(0.0)
            
            #_______________________________________________________________________
            # calculate area weighted mean
            if 'time' in list(data.dims):
                nt = data['time'].values.size
                for nti in range(nt):
                    mat_mean.data[nti,:,:] = np.multiply(mat_mean.data[nti,:,:], mat_area.data)    
                    
                # be sure ocean floor is setted to zero 
                for di in range(0,ndi): 
                    #mat_mean.data[:, np.where(di>=mat_iz)[0], di]=0.0
                    mat_mean.data[:, np.where(di>mat_iz)[0], di]=0.0
            else:
                mat_mean.data = np.multiply(mat_mean.data, mat_area.data)
                
                # be sure ocean floor is setted to zero 
                for di in range(0,ndi): 
                    #mat_mean.data[np.where(di>=mat_iz)[0], di]=0.0
                    mat_mean.data[np.where(di>mat_iz)[0], di]=0.0
                
        #___________________________________________________________________________
        # This approach is five time faster than the original from dima at least for
        # COREv2 mesh but needs probaply a bit more RAM
        if 'time' in list(data.dims): 
            aux_zonmean  = np.zeros([nt, ndi, lat.size])
            aux_zonarea  = np.zeros([nt, ndi, lat.size])
        else                        : 
            aux_zonmean  = np.zeros([ndi, lat.size])
            aux_zonarea  = np.zeros([ndi, lat.size])
        bottom  = np.zeros([lat.size,])
        numbtri = np.zeros([lat.size,])
    
        #  switch topo beteen computation on nodes and elements
        if do_onelem: topo    = np.float16(mesh.zlev[mesh.e_iz[e_idxin]])
        else        : topo    = np.float16(mesh.n_z[n_idxin])
        
        # this is more or less required so bottom patch looks aceptable
        topo[np.where(topo>-30.0)[0]]=np.nan  
        
        # loop over meridional bins
        if 'time' in list(data.dims):
            for bini in range(lat_i.min(), lat_i.max()):
                numbtri[bini]= np.sum(lat_i==bini)
                aux_zonmean[:,:, bini]=mat_mean[:,lat_i==bini,:].sum(axis=1)
                aux_zonarea[:,:, bini]=mat_area[:,lat_i==bini,:].sum(axis=1)
                bottom[bini] = np.nanpercentile(topo[lat_i==bini],15)
                
            
            # kickout outer bins where eventually no triangles are found
            idx    = numbtri>0
            aux_zonmean = aux_zonmean[:,:,idx]
            aux_zonarea = aux_zonarea[:,:,idx]
            del(mat_mean, mat_area, topo)
            
        else:        
            for bini in range(lat_i.min(), lat_i.max()):
                numbtri[bini]= np.sum(lat_i==bini)
                aux_zonmean[:, bini]=mat_mean[lat_i==bini,:].sum(axis=0)
                aux_zonarea[:, bini]=mat_area[lat_i==bini,:].sum(axis=0)
                #bottom[bini] = np.nanpercentile(topo[lat_i==bini],15)
                bottom[bini] = np.nanpercentile(topo[lat_i==bini],10)
                
            # kickout outer bins where eventually no triangles are found
            idx    = numbtri>0
            aux_zonmean = aux_zonmean[:,idx]
            aux_zonarea = aux_zonarea[:,idx]
            del(mat_mean, mat_area, topo)
            
        bottom = bottom[idx]
        lat    = lat[idx]
        aux_zonmean[aux_zonarea!=0]= aux_zonmean[aux_zonarea!=0]/aux_zonarea[aux_zonarea!=0]
        
        #___________________________________________________________________________
        if do_smooth: 
            filt=np.array([1,2,1])
            filt=filt/np.sum(filt)
            filt=filt[np.newaxis,:]
            aux_zonmean[aux_zonarea==0] = 0.0
            aux_zonmean = convolve2d(aux_zonmean, filt, mode='same', boundary='symm') 
        
        #___________________________________________________________________________
        aux_zonmean[aux_zonarea==0]= np.nan
        del(aux_zonarea)
        
        #___________________________________________________________________________
        # smooth bottom line a bit 
        filt=np.array([1,2,3,2,1])
        filt=filt/np.sum(filt)
        aux = np.concatenate( (np.ones((filt.size,))*bottom[0],bottom,np.ones((filt.size,))*bottom[-1] ) )
        aux = np.convolve(aux,filt,mode='same')
        bottom = aux[filt.size:-filt.size]
        del aux, filt
        
        
        
        #___________________________________________________________________________
        # Create Xarray Datasert for moc_basins
        # copy global attributes from dataset
        global_attr = data.attrs
        
        # copy local attributes from dataset
        local_attr  = data[vname].attrs
        if 'long_name' in local_attr:
            local_attr['long_name'] = " zonal mean {}".format(local_attr['long_name']) 
        else:
            local_attr['long_name'] = " zonal mean {}".format(vname) 
        
        # create coordinates
        if 'time' in list(data.dims):
            coords    = {'depth' : ([which_ddim], depth), 
                        'lat'   : (['ny'], lat      ), 
                        'bottom': (['ny'], bottom   ),
                        'time'  : (['time'], data['time'].values)}
            dims = ['time', which_ddim, 'ny']
        else:    
            coords    = {'depth' : ([which_ddim], depth), 
                        'lat'   : (['ny'], lat      ), 
                        'bottom': (['ny'], bottom   )}
            dims = [which_ddim,'ny']
            # create coordinates
        data_vars = {vname   : (dims, aux_zonmean, local_attr)} 
        index_list.append( xr.Dataset(data_vars=data_vars, coords=coords, attrs=global_attr) )
        
        #_______________________________________________________________________
        if box is None or box is 'global': 
            index_list[cnt][vname].attrs['transect_name'] = 'global zonal mean'
        elif isinstance(box, shp.Reader):
            str_name = box.shapeName.split('/')[-1].replace('_',' ')
            index_list[cnt][vname].attrs['transect_name'] = '{} zonal mean'.format(str_name.lower())
        
        #_______________________________________________________________________
        cnt = cnt + 1
    #___________________________________________________________________________
    if do_outputidx:
        return(index_list, idxin_list)
    else:
        return(index_list)



#+___COMPUTE ZONAL MEAN SECTION________________________________________________+
#|                                                                             |
#+_____________________________________________________________________________+
def plot_zmeantransects(data, figsize=[12, 6], 
              n_rc=[1, 1], do_grid=True, cinfo=None, do_rescale=False,
              do_reffig=False, ref_cinfo=None, ref_rescale=False,
              cbar_nl=8, cbar_orient='vertical', cbar_label=None, cbar_unit=None,
              do_bottom=True, max_dep=[], color_bot=[0.6, 0.6, 0.6], 
              pos_fac=1.0, pos_gap=[0.02, 0.02], do_save=None, save_dpi=600, 
              do_contour=True, do_clabel=False, title='descript', do_ylog=True,
              pos_extend=[0.05, 0.08, 0.95,0.95],
            ):
    #____________________________________________________________________________
    fontsize = 12
    
    #___________________________________________________________________________
    # make matrix with row colum index to know where to put labels
    rowlist = np.zeros((n_rc[0], n_rc[1]))
    collist = np.zeros((n_rc[0], n_rc[1]))       
    for ii in range(0,n_rc[0]): rowlist[ii,:]=ii
    for ii in range(0,n_rc[1]): collist[:,ii]=ii
    rowlist = rowlist.flatten()
    collist = collist.flatten()
    
    #___________________________________________________________________________    
    # create figure and axes
    fig, ax = plt.subplots( n_rc[0],n_rc[1], figsize=figsize, 
                            gridspec_kw=dict(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.05,),
                            constrained_layout=False, sharex=True, sharey=True)
    
    #___________________________________________________________________________    
    # flatt axes if there are more than 1
    if isinstance(ax, np.ndarray): ax = ax.flatten()
    else:                          ax = [ax] 
    nax = len(ax)
     
    #___________________________________________________________________________
    # data must be list filled with xarray data
    if not isinstance(data  , list): data  = [data]
    ndata = len(data) 
    
    #___________________________________________________________________________
    # set up color info 
    if do_reffig:
        ref_cinfo = do_setupcinfo(ref_cinfo, [data[0]], ref_rescale, do_index=True)
        cinfo     = do_setupcinfo(cinfo    , data[1:] , do_rescale , do_index=True)
    else:    
        cinfo     = do_setupcinfo(cinfo, data, do_rescale, do_index=True)

    #_______________________________________________________________________
    # setup normalization log10, symetric log10, None
    which_norm = do_compute_scalingnorm(cinfo, do_rescale)
    if do_reffig:
        which_norm_ref = do_compute_scalingnorm(ref_cinfo, ref_rescale)
    
    #___________________________________________________________________________
    # loop over axes
    hpall=list()
    for ii in range(0,ndata):
        
        #_______________________________________________________________________
        # limit data to color range
        vname= list(data[ii][0].keys())[0]
        data_plot = data[ii][0][vname].values.copy()
        
        #_______________________________________________________________________
        # setup x-coorod and y-coord
        lat  , str_xlabel = data[ii][0]['lat'].values   , 'Latitude [deg]'
        depth, str_ylabel = data[ii][0]['depth'].values , 'Depth [m]'
        depth = np.abs(depth)
        
        #_______________________________________________________________________
        if do_reffig: 
            if ii==0: cinfo_plot, which_norm_plot = ref_cinfo, which_norm_ref
            else    : cinfo_plot, which_norm_plot = cinfo    , which_norm
        else        : cinfo_plot, which_norm_plot = cinfo    , which_norm
        
        #_______________________________________________________________________
        # be sure there are no holes
        data_plot[data_plot<cinfo_plot['clevel'][ 0]] = cinfo_plot['clevel'][ 0]+np.finfo(np.float32).eps
        data_plot[data_plot>cinfo_plot['clevel'][-1]] = cinfo_plot['clevel'][-1]-np.finfo(np.float32).eps
        
        #_______________________________________________________________________
        # plot zonal mean data
        hp=ax[ii].contourf(lat, depth, data_plot, levels=cinfo_plot['clevel'], extend='both', cmap=cinfo_plot['cmap'],
                           norm = which_norm)
        hpall.append(hp) 
        
        if do_contour: 
            tickl    = cinfo_plot['clevel']
            ncbar_l  = len(tickl)
            idx_cref = np.where(cinfo_plot['clevel']==cinfo_plot['cref'])[0]
            idx_cref = np.asscalar(idx_cref)
            nstep    = ncbar_l/cbar_nl
            nstep    = np.max([np.int(np.floor(nstep)),1])
            
            idx = np.arange(0, ncbar_l, 1)
            idxb = np.ones((ncbar_l,), dtype=bool)                
            idxb[idx_cref::nstep]  = False
            idxb[idx_cref::-nstep] = False
            idx_yes = idx[idxb==False]
            cont=ax[ii].contour(lat, depth, data_plot, levels=cinfo_plot['clevel'][idx_yes], colors='k', linewidths=[0.5],
                                norm = which_norm) #linewidths=[0.5,0.25])
            
            if do_clabel: 
                ax[ii].clabel(cont, cont.levels, inline=1, inline_spacing=1, fontsize=6, fmt='%1.1f',zorder=1)
                #ax[ii].clabel(cont, cont.levels[np.where(cont.levels!=cinfo_plot['cref'])], 
                            #inline=1, inline_spacing=1, fontsize=6, fmt='%1.1f Sv')
            
        #___________________________________________________________________
        # plot bottom patch   
        if do_bottom:
            aux = np.ones(data_plot.shape,dtype='int16')
            aux[np.isnan(data_plot)]=0
            aux = aux.sum(axis=0)
            aux[aux!=0]=aux[aux!=0]-1
            bottom = depth[aux]
            # smooth bottom patch
            filt=np.array([1,2,3,2,1]) #np.array([1,2,1])
            filt=filt/np.sum(filt)
            aux = np.concatenate( (np.ones((filt.size,))*bottom[0],bottom,np.ones((filt.size,))*bottom[-1] ) )
            aux = np.convolve(aux,filt,mode='same')
            bottom = aux[filt.size:-filt.size]
            bottom = np.maximum(bottom, data[ii][0]['bottom'].values)
            
            ax[ii].fill_between(lat, bottom, depth[-1],color=[0.5,0.5,0.5], zorder=2)#,alpha=0.95)
            ax[ii].plot(lat, bottom, color='k')
        
        #_______________________________________________________________________
        # fix color range
        for im in ax[ii].get_images(): im.set_clim(cinfo_plot['clevel'][ 0], cinfo_plot['clevel'][-1])
        
        #_______________________________________________________________________
        # plot grid lines 
        if do_grid: ax[ii].grid(color='k', linestyle='-', linewidth=0.25,alpha=1.0)
        
        #_______________________________________________________________________
        # set description string plus x/y labels
        if title is not None: 
            txtx, txty = lat[0]+(lat[-1]-lat[0])*0.015, depth[-1]-(depth[-1]-depth[0])*0.015
            if   isinstance(title,str) : 
                # if title string is 'descript' than use descript attribute from 
                # data to set plot title 
                if title=='descript' and ('descript' in data[ii][0][vname].attrs.keys() ):
                    txts = data[ii][0][vname].attrs['descript']
                else:
                    txts = title
            # is title list of string        
            elif isinstance(title,list):   
                txts = title[ii]
            ax[ii].text(txtx, txty, txts, fontsize=12, fontweight='bold', horizontalalignment='left')
        
        #_______________________________________________________________________
        # set x/y label
        if collist[ii]==0        : ax[ii].set_ylabel(str_ylabel, fontsize=12)
        if rowlist[ii]==n_rc[0]-1: ax[ii].set_xlabel(str_xlabel, fontsize=12)
        
        #_______________________________________________________________________
        if do_ylog: 
            if depth[0]==0: ax[ii].set_ylim(depth[1],depth[-1])
            else          : ax[ii].set_ylim(depth[0],depth[-1])
            ax[ii].set_yscale('function', functions=(forward, inverse))
            yticklog = np.array([10,25,50,100,250,500,1000,2000,4000,6000])
            ax[ii].set_yticks(yticklog)
            ax[ii].invert_yaxis()
            ax[ii].grid(True,which='major')
            
        else:
            ax[ii].set_ylim(depth[0],depth[-1])
            ax[ii].invert_yaxis()
            ax[ii].grid(True,which='major')
        
        #ax[ii].set_yscale('log')
        #ax[ii].set_yticks([5,10,25,50,100,250,500,1000,2000,4000,6000])
        ax[ii].get_yaxis().set_major_formatter(ScalarFormatter())
        
    nax_fin = ii+1
    
    #___________________________________________________________________________
    # set superior title
    if 'transect_name' in data[ii][0][vname].attrs.keys():
        fig.suptitle( data[ii][0][vname].attrs['transect_name'], x=0.5, y=1.04, fontsize=16, 
                     horizontalalignment='center', verticalalignment='bottom')
    
    #___________________________________________________________________________
    # delete axes that are not needed
    #for jj in range(nax_fin, nax): fig.delaxes(ax[jj])
    for jj in range(ndata, nax): fig.delaxes(ax[jj])
    if nax != nax_fin-1: ax = ax[0:nax_fin]
    
    #___________________________________________________________________________
    # delete axes that are not needed
    if do_reffig==False:
        cbar = fig.colorbar(hp, orientation=cbar_orient, ax=ax, ticks=cinfo['clevel'], 
                        extendrect=False, extendfrac=None,
                        drawedges=True, pad=0.025, shrink=1.0)
        
        # do formatting of colorbar 
        cbar = do_cbar_formatting(cbar, do_rescale, cbar_nl, fontsize, cinfo['clevel'])
        
        # do labeling of colorbar
        if cbar_label is None : 
            if   'short_name' in data[0][0][vname].attrs:
                cbar_label = data[0][0][ vname ].attrs['short_name']
            elif 'long_name' in data[0][0][vname].attrs:
                cbar_label = data[0][0][ vname ].attrs['long_name']
        if cbar_unit  is None : cbar_label = cbar_label+' ['+data[0][0][ vname ].attrs['units']+']'
        else                  : cbar_label = cbar_label+' ['+cbar_unit+']'
        if 'str_ltim' in data[0][0][vname].attrs.keys():
            cbar_label = cbar_label+'\n'+data[0][0][vname].attrs['str_ltim']
        cbar.set_label(cbar_label, size=fontsize+2)
    else:
        cbar=list()
        for ii, aux_ax in enumerate(ax): 
            cbar_label =''
            if ii==0: 
                aux_cbar = plt.colorbar(hpall[ii], orientation=cbar_orient, ax=aux_ax, ticks=ref_cinfo['clevel'], 
                            extendrect=False, extendfrac=None, drawedges=True, pad=0.025, shrink=1.0,)  
                aux_cbar = do_cbar_formatting(aux_cbar, ref_rescale, cbar_nl, fontsize, ref_cinfo['clevel'])
            else:     
                aux_cbar = plt.colorbar(hpall[ii], orientation=cbar_orient, ax=aux_ax, ticks=cinfo['clevel'], 
                            extendrect=False, extendfrac=None, drawedges=True, pad=0.025, shrink=1.0,)  
                aux_cbar = do_cbar_formatting(aux_cbar, do_rescale, cbar_nl, fontsize, cinfo['clevel'])
                #cbar_label ='anom. '
            if   'short_name' in data[ii][0][vname].attrs:
                cbar_label = cbar_label+data[ii][0][ vname ].attrs['short_name']
            elif 'long_name' in data[ii][0][vname].attrs:
                cbar_label = cbar_label+data[ii][0][ vname ].attrs['long_name']
            if cbar_unit  is None : cbar_label = cbar_label+' ['+data[ii][0][ vname ].attrs['units']+']'
            else                  : cbar_label = cbar_label+' ['+cbar_unit+']'
            if 'str_ltim' in data[ii][0][vname].attrs.keys():
                cbar_label = cbar_label+'\n'+data[ii][0][vname].attrs['str_ltim']
            aux_cbar.set_label(cbar_label, size=fontsize+2)
            cbar.append(aux_cbar)
            
    #___________________________________________________________________________
    # repositioning of axes and colorbar
    if do_reffig==False:
        ax, cbar = do_reposition_ax_cbar(ax, cbar, rowlist, collist, pos_fac, pos_gap, 
                                        title=None, extend=pos_extend)
    
    plt.show()
    fig.canvas.draw()
    
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, fig, dpi=save_dpi)
    
    #___________________________________________________________________________
    return(fig, ax, cbar)



# ___DO ANOMALY________________________________________________________________
#| compute anomaly between two xarray Datasets                                 |
#| ___INPUT_________________________________________________________________   |
#| data1        :   xarray dataset object                                      |
#| data2        :   xarray dataset object                                      |
#| ___RETURNS_______________________________________________________________   |
#| anom         :   xarray dataset object, data1-data2                         |
#|_____________________________________________________________________________|
def do_transect_anomaly(index1,index2):
    
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
                        anom_idx[vname].attrs[key] = 'anomalous '+anom_idx[vname].attrs[key] 
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
