from dask.distributed import Client, LocalCluster
import numpy as np
import gc
import dask
import os
import tripyview
import shapefile as shp
import logging
#
#
#_______________________________________________________________________________
# start parallel dask client
def shortcut_setup_daskclient(client, use_existing_client, do_parallel, parallel_nprc, parallel_tmem,
                              threads_per_worker=4,
                              memory_thresh=0.90, # hoch much memory from total mem should be distributed
                              memory_target=0.85, # Start spilling at 85% usage (default 60%)
                              memory_spill =0.90,  # Spill to disk at 90% usage (default 70%)
                              memory_pause =0.95, # Pause execution at 95% usage (default 80%)
                              memory_termin=0.98, # Pause execution at 95% usage (default 80%)
                              do_dashbrdlnk=True,
                              scheduler_file=None,
                              ):
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

        :parallel_tmem:  int, (default:256) Available memory that will be distributed
                         between the started dask workers

        :scheduler_file: str or None, (default: None) path to a dask scheduler-info
                        file, e.g. '~/.tripyview_dask_scheduler.json'. Unlike
                        use_existing_client this needs no address typed in by hand:
                        if the file exists and a scheduler is actually listening at
                        the address it records, that cluster is reused (this is the
                        only one of the three reuse checks here that survives a
                        kernel restart, since use_existing_client's default and
                        Client.current() both need the original kernel process to
                        still be alive). If the file is missing, or present but
                        stale (no scheduler there anymore, e.g. after a crashed
                        kernel), it is removed and a new cluster is started and
                        written to that path for the next notebook to find.

     __________________________________________________

    Returns:

        :client:    returns None or dask client object

    ____________________________________________________________________________
    """
    
    #___________________________________________________________________________
    if do_parallel:
        
        if "JUPYTERHUB_SERVICE_PREFIX" not in os.environ:
            # the dashboard link template below already starts with '/proxy/...';
            # defaulting this to '/' would produce a leading '//', which browsers
            # parse as a protocol-relative URL (host 'proxy') instead of a same-
            # origin path, and the dashboard fails with "Not Found"
            os.environ["JUPYTERHUB_SERVICE_PREFIX"] = ""

        if do_dashbrdlnk:
            dask.config.config.get('distributed').get('dashboard').update({'link':'{JUPYTERHUB_SERVICE_PREFIX}/proxy/{port}/status'})
                        
        # - work-stealing: True or False ... Enables/disables the feature (default: True).
        # - work-stealing-interval: str (default: "100ms") ... How often the scheduler 
        #   checks for imbalanced task loads across workers. Lower values (e.g., "50ms") 
        #   make work stealing more aggressive but increase scheduling overhead.
        # - work-stealing-distance: int (default: 2) ... Determines how far tasks can 
        #   be stolen across the cluster. A higher value allows more aggressive 
        #   stealing, which is useful for heterogeneous clusters. A lower value 
        #   keeps tasks local, reducing data transfer overhead.
        # - work-stealing-threshold: float (default: 0.3) ... Defines how much imbalance 
        #   (CPU/memory usage) is needed before stealing happens. A lower value (e.g., 0.2) 
        #   (makes stealing more frequent. A higher value (e.g., 0.5) makes workers 
        #   (hold onto tasks longer, reducing movement overhead.
        dask.config.set({
                "distributed.scheduler.work-stealing"                : True   ,  # Enable work stealing
                "distributed.scheduler.work-stealing-interval"       : "50ms" ,  # Frequency of stealing checks
                "distributed.scheduler.work-stealing-distance"       : 10     ,  # Limits how far tasks can be stolen
                "distributed.scheduler.work-stealing-threshold"      : 0.01   ,  # Load imbalance threshold before stealing starts
                })
        
        # - "managed" ensures Dask only tracks Python-related memory (Dask arrays, xarray, NumPy).
        # - memory.spill: 0.9 prevents excessive disk I/O but avoids memory overload.
        # - memory.pause: 0.95 ensures workers don’t overcommit memory.
        # - memory.terminate: 0.98 prevents system crashes due to OOM errors.
        # - memory.chunk-size: 256MB improves performance for xarray/Dask array computations.
        # - connections.outgoing: 20 Limits the number of concurrent outgoing connections 
        #   a worker can have to other workers or the scheduler. When a worker needs to 
        #   send data (e.g., during client.scatter(), client.rebalance(), or task 
        #   dependencies), it can only open up to 20 simultaneous connections.
        #   Increased worker connections help with large data transfers in multi-worker setups.
        # - connections.incoming: 50. Limits the number of concurrent incoming connections a
        #   worker can handle from other workers or the scheduler. If multiple workers are 
        #   trying to send data to a single worker (e.g., during a reduce operation like sum() 
        #   or mean() over a large dataset), it will only accept data from up to 50 workers
        #   at a time.
        dask.config.set({
                "distributed.scheduler.active-memory-manager.measure": "managed",  # Track Dask-managed memory only
                "distributed.worker.memory.rebalance.measure"        : "managed",  # Rebalance based on managed memory
                "distributed.worker.memory.target"                   : memory_target,  # Start spilling at 80%
                "distributed.worker.memory.spill"                    : memory_spill,  # Spill only when 90% of memory is used
                "distributed.worker.memory.pause"                    : memory_pause,  # Pause new tasks if memory exceeds 95%
                "distributed.worker.memory.terminate"                : memory_termin,  # Kill worker if memory exceeds 98%
                "distributed.array.chunk-size"                       : "256MB",  # Optimize chunk size for large arrays
                "distributed.worker.connections.outgoing"            : 15,  # Improve parallelism for large data transfers
                "distributed.worker.connections.incoming"            : 40
                })
        
        #logger = logging.getLogger('distributed')
        #logger.setLevel(logging.DEBUG)
                
        #_______________________________________________________________________
        # resolve scheduler_file to an absolute path up front so both the
        # reconnect attempt and the new-cluster branch below agree on it
        if scheduler_file is not None:
            scheduler_file = os.path.expanduser(scheduler_file)

        client = None

        #_______________________________________________________________________
        # check for an existing cluster via scheduler_file -- this is the only one
        # of the three reuse checks that survives a kernel restart, since it reads
        # the address from disk instead of from this process' or a typed-in state
        if scheduler_file is not None and os.path.isfile(scheduler_file):
            try:
                client = Client(scheduler_file=scheduler_file, timeout="10s")
                client.run(gc.collect)  # Run garbage collection on all workers
                print("Reconnected to existing Dask cluster via scheduler_file:", client)
            except (OSError, TimeoutError):
                print("scheduler_file", scheduler_file, "is stale (no cluster there anymore), removing it")
                try: os.remove(scheduler_file)
                except OSError: pass
                client = None

        #_______________________________________________________________________
        # check for existing client via adress
        if client is None:
            try:
                client = Client(use_existing_client)
                client.run(gc.collect)  # Run garbage collection on all workers
                print("Connected to existing Dask cluster:", client)

            #___________________________________________________________________
            except OSError:
                print("No existing Dask cluster found at:", use_existing_client)
                try:
                    # Check if an existing client is connected
                    client = Client.current()
                    client.run(gc.collect)  # Run garbage collection on all workers
                    print("Dask client already running:", client)
                except ValueError:
                    # No active client, start a new one.
                    #
                    # NOTE: passing scheduler_file= directly to Client(..., n_workers=...)
                    # does NOT start a new cluster and write the file -- dask's own
                    # Client._start() treats scheduler_file as "connect to a cluster
                    # described by this file" unconditionally and busy-waits for it to
                    # appear (`while not os.path.exists(self.scheduler_file): sleep`),
                    # regardless of n_workers also being given; that path never reaches
                    # cluster creation at all, so the call hangs forever. The file is
                    # only ever written by the Scheduler itself, which is reached by
                    # constructing the LocalCluster explicitly (scheduler_kwargs is
                    # forwarded to distributed.Scheduler, which does accept
                    # scheduler_file) and handing that cluster to Client().
                    worker_kwargs = dict(
                        n_workers         = np.int16(parallel_nprc/threads_per_worker),
                        threads_per_worker= threads_per_worker,
                        memory_limit      = '{:3.3f} GB'.format(parallel_tmem/parallel_nprc*threads_per_worker*memory_thresh),
                        )
                    if scheduler_file is not None:
                        cluster = LocalCluster(scheduler_kwargs={'scheduler_file': scheduler_file},
                                               **worker_kwargs)
                        client  = Client(cluster, timeout="300s")
                        print("Started a new Dask client:", client)
                        print("Wrote scheduler_file for reuse by later notebooks:", scheduler_file)
                    else:
                        client = Client(timeout="300s", **worker_kwargs)
                        print("Started a new Dask client:", client)
        
        #_______________________________________________________________________            
        display(client)
        
        #_______________________________________________________________________            
        # Rebalance data within network. Move data between workers to roughly balance 
        # memory burden. This either affects a subset of the keys/workers or the 
        # entire network, depending on keyword arguments.
        client.rebalance()
        
        
    #___________________________________________________________________________
    return(client)



#
#
#_______________________________________________________________________________
# start parallel dask client
def shortcut_setup_pathwithspinupcycles(input_paths, input_names, ref_path, ref_name, n_cycl, do_allcycl, fmtstr='scycle {:d}'):
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
            
            if ipath is None: 
                aux_path.append(None)
                aux_name.append('{}'.format(iname))
                print('{:02d}'.format(ii), aux_path[-1],aux_name[-1])  
                continue
                
            for ii_cycl in range(cycl_s, n_cycl+1):
                aux_path.append(os.path.join(ipath,'{:d}/'.format(ii_cycl)))
                if not do_allcycl: aux_name.append('{}'.format(iname))
                #else             : aux_name.append('{:d}) {}'.format(ii_cycl, iname))
                else             : aux_name.append('{}, {}'.format(fmtstr.format(ii_cycl), iname))
                print('{:02d}'.format(ii), aux_path[-1],aux_name[-1])
        input_paths, input_names = aux_path, aux_name
        
        #_______________________________________________________________________
        if (ref_path is not None): 
            aux_path, aux_name = list(), list()
            ref_path_old, ref_name_old = ref_path, ref_name
            for ii_cycl in range(cycl_s, n_cycl+1):
                aux_path.append(os.path.join(ref_path,'{:d}/'.format(ii_cycl)))
                if not do_allcycl: aux_name.append('{}'.format(ref_name))
                #else             : aux_name.append('{:d}) {}'.format(ii_cycl, ref_name))
                else             : aux_name.append('{}, {}'.format(fmtstr.format(ii_cycl), ref_name))
                print('R', aux_path[-1], aux_name[-1])        
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
