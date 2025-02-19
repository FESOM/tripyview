from dask.distributed import Client
import gc
import dask
import os
import tripyview
import shapefile as shp

#
#
#_______________________________________________________________________________
# start parallel dask client
def shortcut_setup_daskclient(client, use_existing_client, do_parallel, parallel_nprc, parallel_tmem):
    """
    --> shortcut to setup dask client in a note book 
    
    Parameters: 
    
        :client:    None, dask client object (default: None) If None no dask client was 
                    started do_parallel=False
                    
        :use_existing_client:   str, (default:"tcp://0.0.0.0:0000") You can give here the 
                                adress of an already running dask client e.g.
                                "tcp://127.0.0.1:42465" that can be re used by the notebook.
                                default is "tcp://0.0.0.0:0000" as an non existent dummy client
                                which means that first a new client will be started if none
                                is already attributed to the notebook 
                                
        :do_parallel:   bool, (default: False) True/False if a parallel dask client 
                        should be started                       
                        
                        
        :parallel_nprc:  int, (default:48) How many dask worker should be started
        
        :parallel_tmem:  int, (default:200) Available memory that will be distributed 
                         between the started dask workers   
    
     __________________________________________________
    
    Returns:
    
        :client:    returns None or dask client object
    
    ____________________________________________________________________________
    """
    
    #___________________________________________________________________________
    if do_parallel:
        dask.config.config.get('distributed').get('dashboard').update({'link':'{JUPYTERHUB_SERVICE_PREFIX}/proxy/{port}/status'})
        
        #_______________________________________________________________________
        # check for existing client via adress
        try:
            client = Client(use_existing_client)
            client.run(gc.collect)  # Run garbage collection on all workers
            print("Connected to existing Dask cluster:", client)
            
        #_______________________________________________________________________    
        except OSError:
            print("No existing Dask cluster found at:", use_existing_client)
            try:
                # Check if an existing client is connected
                client = Client.current()
                client.run(gc.collect)  # Run garbage collection on all workers
                print("Dask client already running:", client)
            except ValueError:
                # No active client, start a new one
                client = Client(n_workers=parallel_nprc, threads_per_worker=1, memory_limit='{:3.3f} GB'.format(parallel_tmem/parallel_nprc))
                print("Started a new Dask client:", client)
        display(client)
    #___________________________________________________________________________
    return(client)



#
#
#_______________________________________________________________________________
# start parallel dask client
def shortcut_setup_pathwithspinupcycles(input_paths, input_names, ref_path, ref_name, n_cycl, do_allcycl):
    """
    --> shortcut to setup up input_paths with specific spinup cycle structure
        if n_cycl=5, do_allcycl=True
        
            input_paths = [ input_paths[0]/1/, 
                            input_paths[0]/2/, 
                            input_paths[0]/3/, 
                            input_paths[0]/4/, 
                            input_paths[0]/5/,
                            input_paths[1]/1/,
                            input_paths[1]/2/, 
                            ...]
                        
        if n_cycl=5, do_allcycl=False
        
            input_paths = [ input_paths[0]/5/, 
                            input_paths[1]/5/, 
                            input_paths[2]/5/, 
                            input_paths[3]/5/, 
                            input_paths[4]/5/,
                            ]                
    
    Parameters: 
    
        :input_paths:   list, list with data path strings
                    
        :input_names:   list, list with data name tag strings
                                
        :ref_path:      list, list with reference data path strings                  
                        
        :ref_path:      list, list with reference data name tag strings
        
        :n_cycl:        int, (default:None) which/how many spinupcycles should be considered
        
        :do_allcycl:    bool, (default:False ) True/False id entire spinup cycle structure 
                        from  1:n_cycle should  be considered
    
    __________________________________________________
    
    Returns:
    
        :input_paths: 
        
        :input_names: 
        
        :ref_path:  
        
        :ref_path:
    ____________________________________________________________________________
    """
    #___________________________________________________________________________
    if (n_cycl is not None): 
        cycl_s=1 if do_allcycl else n_cycl
        #_______________________________________________________________________
        aux_path, aux_name = list(), list()
        input_paths_old, input_names_old = input_paths, input_names
        for ii, (ipath,iname) in enumerate(zip(input_paths,input_names)):
            for ii_cycl in range(cycl_s, n_cycl+1):
                aux_path.append(os.path.join(ipath,'{:d}/'.format(ii_cycl)))
                if not do_allcycl: aux_name.append('{}'.format(iname))
                else             : aux_name.append('{:d}) {}'.format(ii_cycl, iname))
                print(ii, aux_path[-1],aux_name[-1])
        input_paths, input_names = aux_path, aux_name
        
        #_______________________________________________________________________
        if (ref_path is not None): 
            aux_path, aux_name = list(), list()
            ref_path_old, ref_name_old = ref_path, ref_name
            for ii_cycl in range(cycl_s, n_cycl+1):
                aux_path.append(os.path.join(ref_path,'{:d}/'.format(ii_cycl)))
                if not do_allcycl: aux_name.append('{}'.format(ref_name))
                else             : aux_name.append('{:d}) {}'.format(ii_cycl, ref_name))
                print('R', ref_path[-1])        
            ref_path, ref_name = aux_path, aux_name
        del(aux_path, aux_name)
    #___________________________________________________________________________
    return(input_paths, input_names, ref_path, ref_name)  




#
#
#_______________________________________________________________________________
# concatenate ref_path and input_path together if is not None
def shortcut_setup_concatinputrefpath(input_paths, input_names, ref_path, ref_name):
    """
    --> shortcut to setup concat ref_path and input_path together, ref_path
        becomes first list entry, input_path comes behind. like that only need to cycle 
        over inputpath list 
    
    Parameters: 
    
        :input_paths:   list, list with data path strings
                    
        :input_names:   list, list with data name tag strings
                                
        :ref_path:      list, list with reference data path strings                  
                        
        :ref_path:      list, list with reference data name tag strings
        
    __________________________________________________
    
    Returns:
    
        :input_paths: 
        
        :input_names: 
        
    ____________________________________________________________________________
    """
    #___________________________________________________________________________
    # concatenate list = list1+list2
    if (ref_path != None): 
        if isinstance(ref_path, list): 
            input_paths, input_names = ref_path + input_paths        , ref_name + input_names
        else:    
            input_paths, input_names = list([ref_path]) + input_paths, list([ref_name]) + input_names
    #___________________________________________________________________________
    return(input_paths, input_names)  



#
#_______________________________________________________________________________
# concatenate ref_path and input_path together if is not None
def shortcut_setup_boxregion(box_region):
    """
    --> shortcut to setup concat ref_path and input_path together, ref_path
        becomes first list entry, input_path comes behind. like that only need to cycle 
        over inputpath list 
    
    Parameters: 
    
        :box_region:   list, with strings from inputregions or list with box definittion 
        
    __________________________________________________
    
    Returns:
    
        :box: returns list where regions string a converted into shp.Reader object
        
    ____________________________________________________________________________
    """
    #___________________________________________________________________________
    # define index regions --> reading shape files
    box = list()
    shp_path = os.path.join(tripyview.__path__[0],'shapefiles/')
    for region in box_region:
        if region == 'global' or isinstance(region,list): 
            print('global')
            box.append(region)
        else: 
            print(tripyview.__path__[0],region)
            box.append(shp.Reader(os.path.join(shp_path,region))) 
    #___________________________________________________________________________
    return(box)
