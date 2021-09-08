import os
import xarray as xr
import pandas as pa
import numpy  as np

#
#
#_____________________________________________________________________________________________
def do_node_neighbour(mesh, do_nghbr_e=False):
    print(' --> compute node neighbourhood')
    #_________________________________________________________________________________________
    # compute  infrastructure
    # nmb_n_in_e ...number of elements with node
    # n_nghbr_e  ...element index with node   
    nmb_n_in_e = np.zeros((mesh.n2dn,), dtype=np.int32)
    n_nghbr_e     =  [ [] for _ in range(mesh.n2dn)] 
    for elemi in range(0,mesh.n2de):
        nmb_n_in_e[mesh.e_i[elemi,:]] = nmb_n_in_e[mesh.e_i[elemi,:]] + 1
        nodes = mesh.e_i[elemi,:]
        for ni in range(0,3): n_nghbr_e[nodes[ni]].append(elemi)  

    # n_nghbr_n   ...index of neighbouring nodes         
    n_nghbr_n     =  [ [] for _ in range(mesh.n2dn)] 
    for nodei in range(0,mesh.n2dn):

        # loop over neigbouring elements
        for nie in range(nmb_n_in_e[nodei]):
            elemi  = n_nghbr_e[nodei][nie]

            # loop over vertice indecies of element elemi 
            for ni in mesh.e_i[elemi,:]:
                if (ni==nodei) or (ni in n_nghbr_n[nodei]): continue
                n_nghbr_n[nodei].append(ni)    

        # sort indices (not realy necessary ?!)
        n_nghbr_n[nodei].sort()
    
    #_________________________________________________________________________________________
    if do_nghbr_e: 
        return(n_nghbr_n, n_nghbr_e)
    else:
        return(n_nghbr_n)

#
#
#_____________________________________________________________________________________________
def do_elem_neighbour(mesh):
    #_________________________________________________________________________________________
    # compute node neighbourhood
    # n_nghbr_n ... neighboring nodes
    # n_nghbr_e ... neighboring elem
    n_nghbr_n, n_nghbr_e = do_node_neighbour(mesh, do_nghbr_e=True)
    
    #_________________________________________________________________________________________
    print(' --> compute edge neighbourhood')
    # compute edge and neighbouring elements with respect to edge
    # edges     ... list with node indices that form edge
    # ed_nghbr_e... neighbouring elements with respect to edge 
    edges      = list()
    ed_nghbr_e = list()
    # loop over nodes
    for n in range(0,mesh.n2dn):
        # loop over the neighbouring nodes 
        for nghbr_n in n_nghbr_n[n]:
            
            # do not dublicate edges
            if nghbr_n<n: continue
            
            # loop over the neighbouring elements
            elems=[]
            for nghbr_e in n_nghbr_e[n]:
                elnodes = mesh.e_i[nghbr_e,:]
                
                # if neighbouring nodes to node n is found in neighbouring elements
                if nghbr_n in elnodes:
                    elems.append(nghbr_e)
                
                if len(elems)==2: break
                    
            # write node indices and elem indices that contribute to edge
            if len(elems)==1: elems.append(-999)
            edges.append([n,nghbr_n])
            ed_nghbr_e.append(elems)
            
    #_________________________________________________________________________________________
    print(' --> compute elem neighbourhood')        
    # compute edge indices with respect to element
    # e_nghbr_ed ... neighbouring edges with respect to element 
    n2ded = len(edges)   
    e_nghbr_ed =  [ [] for _ in range(mesh.n2de)]
    # loop over edges
    for edi in range(0,n2ded):
        # loop over neighbouring elements with respect to edge
        for nghbr_e in ed_nghbr_e[edi]:
            
            # its a boundary edge, has only one valid elem neighbour
            if nghbr_e<0: 
                continue 
            else:    
                e_nghbr_ed[nghbr_e].append(edi)
                
                
    # compute neighbouring elem indecies with respect to elem
    # e_nghbr_e ... neighbouring elements with respect to elem
    e_nghbr_e =  [ [] for _ in range(mesh.n2de)]
    for ei in range(0,mesh.n2de):
        for nghbr_ed in e_nghbr_ed[ei]:
            elems = ed_nghbr_e[nghbr_ed]     
            if elems[0]==ei: e_nghbr_e[ei].append(elems[1])
            else:            e_nghbr_e[ei].append(elems[0])
    
    #_________________________________________________________________________________________
    return(e_nghbr_e)

#
#
#_____________________________________________________________________________________________
def do_node_smoothing(mesh, data_orig, n_nghbr_n, special_region, special_boxlist, rel_cent_weight, num_iter):
    print(' --> compute node smoothing')
    #_________________________________________________________________________________________
    data_smooth = data_orig.copy()
    # compute smoothing
    for it in range(num_iter):
        print('     iter: {}'.format(str(it)))
        # loop over nodes
        data_done = data_smooth.copy()
        for nodei in range(0,mesh.n2dn):
            # smooth from down to top 
            idx = np.argmax(data_done)
            data_done[idx] = -99999.0

            # set coeff
            coeff1=1.0

            # include special region with less smoothing by increaing the 
            # weight of the center
            if(special_region): 
                for box in special_boxlist:
                    if (mesh.n_x[idx]>=box[0] and mesh.n_x[idx]<=box[1] and
                        mesh.n_y[idx]>=box[2] and mesh.n_y[idx]<=box[3] ): 
                        coeff1=3.0
            
            # do convolution over neighbours (weight of neighbours = 1)
            # sum depth over neighbouring nodes 
            dsum=0.0
            nmb_n_nghbr=0
            for ni in n_nghbr_n[idx]: 
                dsum=dsum+data_smooth[ni]
                nmb_n_nghbr=nmb_n_nghbr+1

            # add contribution from center weight 
            dsum=dsum + np.real(coeff1*rel_cent_weight*(nmb_n_nghbr-1.0)-1.0)*data_smooth[idx]
            data_smooth[idx]=dsum/np.real( nmb_n_nghbr + coeff1*rel_cent_weight*(nmb_n_nghbr-1)-1.0 )
            #                                  |                       | 
            #                   sum over weight (=1)          center weight (depends
            #                   of neighbours                 on number of neighbours)
    #_________________________________________________________________________________________
    return(data_smooth)

#
#
#_____________________________________________________________________________________________
def do_elem_smoothing(mesh, data_orig, e_nghbr_e, special_region, special_boxlist, rel_cent_weight, num_iter):
    print(' --> compute elem smoothing')
    #_________________________________________________________________________________________
    data_smooth = data_orig.copy()
    # compute smoothing
    for it in range(num_iter):
        print('     iter: {}'.format(str(it)))
        # loop over nodes
        data_done = data_smooth.copy()
        for elemi in range(0,mesh.n2de):
            # smooth from down to top 
            idx = np.argmax(data_done)
            data_done[idx] = -99999.0

            # set coeff
            coeff1=1.0

            # include special region with less smoothing
            if(special_region): 
                for box in special_boxlist:
                    if (mesh.n_x[mesh.e_i[idx,:]].sum()/3 >=box[0] and mesh.n_x[mesh.e_i[idx,:]].sum()/3 <=box[1] and
                        mesh.n_y[mesh.e_i[idx,:]].sum()/3 >=box[2] and mesh.n_y[mesh.e_i[idx,:]].sum()/3 <=box[3] ): 
                        coeff1=3.0

            # sum depth over neighbouring elements 
            dsum=0.0
            nmb_e_nghbr = 0
            for ei in e_nghbr_e[idx]: 
                if ei<0: continue
                dsum=dsum+data_smooth[ei]
                nmb_e_nghbr = nmb_e_nghbr+1

            # add contribution from center weight 
            dsum=dsum + (coeff1*rel_cent_weight*(nmb_e_nghbr-1.0)-1.0)*data_smooth[idx]
            data_smooth[idx]=dsum/np.real( nmb_e_nghbr + coeff1*rel_cent_weight*(nmb_e_nghbr-1)-1.0 )
            #                                  |                        | 
            #                   sum over weight (=1)          center weight (depends
            #                   of neighbours                 on number of neighbours)
            
    #_________________________________________________________________________________________
    return(data_smooth)