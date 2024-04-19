import yaml
import papermill as pm
import argparse
import json
import shutil
import sys
import os

#os.environ['PATH_TRIPYVIEW'] = '/home/ollie/pscholz/tripyview_github/'

#try: pkg_path = os.environ['PATH_TRIPYVIEW']
#except: pkg_path='.'    
#sys.path.append(os.path.join(pkg_path,"src/"))
from .sub_tripydriver import *

#_________________________________________________________________________________________________            
# open htnl template file
#try: pkg_path = os.environ['PATH_TRIPYVIEW']
#except: pkg_path='' 
pkg_path          = os.path.dirname(os.path.dirname(__file__))
templates_path    = os.path.join(pkg_path,'templates_html')
templates_nb_path = os.path.join(pkg_path,'templates_notebooks')
print(pkg_path)

#
#
#_______________________________________________________________________________
class cd:
    """Context manager for changing the current working directory.
    https://stackoverflow.com/questions/431684/how-do-i-change-the-working-directory-in-python
    """
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

#
#
#_______________________________________________________________________________
# command line input --> diagrun() 
def tripyrun():
    
    # command line input arguments
    parser = argparse.ArgumentParser(prog='tripyrun', description='do FESOM tripyview diagnostics in command line')
    parser.add_argument('workflow_file', help='name of work flow yaml file')
    parser.add_argument('--diagnostics',
                        '-d',
                        nargs='+',
                        help='run only particular diagnostics from the yml file.',)
    
    #parser.add_argument('--nworkers',
                        #'-nw',
                        #nargs=1,
                        #type=int, 
                        #help='start dask client with #workers',)
    
    #parser.add_argument('--memory',
                            #'-mem',
                            #nargs=1,
                            #default=100,
                            #type=int,
                            #help='total memory avaible, become distributed over workers',)
    
    # input arguments from command line
    inargs = parser.parse_args()
    if inargs.diagnostics:
        print(f"selective diagnostics will be run: {inargs.diagnostics}")
    #if inargs.nworkers:
        #print(f" --> Number of dask workers to ask for: {inargs.nworkers}")    
        #if inargs.memory:
            #print(f" --> Total memory avaiable: {inargs.memory}")        

    #___________________________________________________________________________
    # open selected yaml files
    yaml_filename = inargs.workflow_file
    with open(yaml_filename) as file:
        # load yaml file dictionaries
        yaml_settings = yaml.load(file, Loader=yaml.FullLoader)

    #____DICTIONARY CONTENT_____________________________________________________
    # name of workflow runs --> also folder name 
    tripyrun_name = yaml_settings['tripyrun_name']

    # setup data input paths & input names
    input_paths = yaml_settings['input_paths']
    if 'input_names' in yaml_settings:
        input_names = yaml_settings['input_names']
    else:
        input_names = list()
        for path in input_paths: 
            input_names.append(path.split('/')[-3])
    if len(input_names) != len(input_paths): raise ValueError("The size of input_names & input_paths is not equal") 
    
    # setup save path    
    if 'save_path' in yaml_settings: 
        save_path = f"{yaml_settings['save_path']}/{tripyrun_name}"
    else:
        save_path = os.path.join(pkg_path, f"Results/{tripyrun_name}") 
    save_path = os.path.expanduser(save_path)
    save_path = os.path.abspath(save_path)

        
    #_________________________________________________________________________________________________    
    # actualize settings dictionary    
    yaml_settings['tripyrun_name']     = tripyrun_name
    yaml_settings['input_paths']       = input_paths
    yaml_settings['input_names']       = input_names
    yaml_settings['do_papermill']      = True
    #yaml_settings['workflow_settings'] = None
    
    yaml_settings['save_path']    = save_path
    yaml_settings['tripyrun_spath_nb' ]= os.path.join(save_path, "notebooks")
    yaml_settings['tripyrun_spath_fig']= os.path.join(save_path, "figures")  
    
    # papermiller imports python None variable declaration as "None" string this causes 
    # trouble in the if VAR is None: comparison:
    for key in yaml_settings: 
        if yaml_settings[key] == "None": yaml_settings[key]=None

    #_________________________________________________________________________________________________
    # create save directory if they do not exist
    if not os.path.exists(yaml_settings['save_path']):
        print(f' --> mkdir: {yaml_settings["save_path"]}')
        os.makedirs(yaml_settings['save_path'])
        print(f' --> mkdir: {yaml_settings["tripyrun_spath_nb"]}')
        os.makedirs(yaml_settings['tripyrun_spath_nb'])
        print(f' --> mkdir: {yaml_settings["tripyrun_spath_fig"]}')
        os.makedirs(yaml_settings['tripyrun_spath_fig'])
        
    #___________________________________________________________________________
    # initialise/create webpage interface .json file
    fname_json = f"{yaml_settings['tripyrun_name']}.json"
    save_path_json = os.path.join(yaml_settings['save_path'], fname_json)
    if os.path.exists(save_path_json):
        with open(save_path_json) as json_file:
            webpages = json.load(json_file)
            print(f"Jupyter Notebooks from previous runs exist in {save_path_json},")
            print(f"they will be used to generate output for diagnostics you do not run this time.")
    else:
        webpages = {}
        webpages["analyses"] = {}
    
    #___________________________________________________________________________
    webpages["general"] = {}
    webpages["general"]["name"] = yaml_settings["tripyrun_name"]
    webpages["logo"] = {}
    webpages["logo"]["path"] =os.path.join(templates_path, 'fesom2_logo.png')

    #___________________________________________________________________________
    # define analyses drivers
    analyses_driver_list = {}
    analyses_driver_list["hslice"             ] = drive_hslice
    analyses_driver_list["hslice_np"          ] = drive_hslice
    analyses_driver_list["hslice_sp"          ] = drive_hslice
    analyses_driver_list["hslice_clim"        ] = drive_hslice_clim
    analyses_driver_list["hslice_clim_np"     ] = drive_hslice_clim
    analyses_driver_list["hslice_clim_sp"     ] = drive_hslice_clim
    analyses_driver_list["hslice_isotdep"     ] = drive_hslice_isotdep
    
    analyses_driver_list["hovm"               ] = drive_hovm
    analyses_driver_list["hovm_clim"          ] = drive_hovm_clim
    
    analyses_driver_list["transect"           ] = drive_transect
    analyses_driver_list["transect_clim"      ] = drive_transect_clim
    analyses_driver_list["transect_transp"    ] = drive_transect_transp
    analyses_driver_list["transect_transp_t"  ] = drive_transect_transp_t
    analyses_driver_list["transect_transp_t_OSNAP"] = drive_transect_transp_t_OSNAP
    analyses_driver_list["transect_zmean"     ] = drive_transect_zmean
    analyses_driver_list["transect_zmean_clim"] = drive_transect_zmean_clim
    
    analyses_driver_list["vprofile"           ] = drive_vprofile
    analyses_driver_list["vprofile_clim"      ] = drive_vprofile_clim
    
    analyses_driver_list["var_t"              ] = drive_var_t
    
    analyses_driver_list["zmoc"               ] = drive_zmoc
    analyses_driver_list["zmoc_t"             ] = drive_zmoc_t
    analyses_driver_list["dmoc"               ] = drive_dmoc
    analyses_driver_list["dmoc_z"             ] = drive_dmoc
    analyses_driver_list["dmoc_srf"           ] = drive_dmoc
    analyses_driver_list["dmoc_srf_z"         ] = drive_dmoc
    analyses_driver_list["dmoc_inner"         ] = drive_dmoc
    analyses_driver_list["dmoc_inner_z"       ] = drive_dmoc
    analyses_driver_list["dmoc_wdiap"         ] = drive_dmoc_wdiap
    analyses_driver_list["dmoc_srfcbflx"      ] = drive_dmoc_srfcbflx
    analyses_driver_list["dmoc_t"             ] = drive_dmoc_t
    
    analyses_driver_list["hbarstreamf"        ] = drive_hbarstreamf
    analyses_driver_list["ghflx"              ] = drive_ghflx
    analyses_driver_list["mhflx"              ] = drive_mhflx
    
    #___________________________________________________________________________
    # loop over available diagnostics and run the one selected in the yaml file
    # loop over all analyses
    for analysis_name in analyses_driver_list:
        
        # check if analysis is in input yaml settings
        if analysis_name in yaml_settings:
            
            #___________________________________________________________________
            # if no -d ... flag is setted perform full yml file diagnostic
            if inargs.diagnostics is None:
                print(f" --> compute {analysis_name}:")
                # drive specific analysis from analyses_driver_list
                webpage = analyses_driver_list[analysis_name](yaml_settings, analysis_name)
                webpages["analyses"][analysis_name] = webpage
            
            #___________________________________________________________________
            # if -d ... flag is setted perform specific yml file diagnostic    
            else:
                if analysis_name in inargs.diagnostics:
                    print(f" --> compute {analysis_name}:")
                    # drive specific analysis from analyses_driver_list
                    webpage = analyses_driver_list[analysis_name](yaml_settings, analysis_name)
                    webpages["analyses"][analysis_name] = webpage
                
            # write linked analysis to .json file
            with open(save_path_json, "w") as fp: json.dump(webpages, fp)
            
    #___________________________________________________________________________
    # save everything to .html
    fname_html     = f"{yaml_settings['tripyrun_name']}.html"
    save_path_html = os.path.join(yaml_settings['save_path'], fname_html)
    ofile          = open(save_path_html, "w")
    template       = env.get_template("experiment.html")
    output         = template.render(webpages)
    ofile.write(output)
    ofile.close()   
    
    #___________________________________________________________________________
    # save everything to .html
    #render_main_page()
    
#
#
#_______________________________________________________________________________
if __name__ == "__main__":
    # args = parser.parse_args()
    # args.func(args)
    tripyrun()    
