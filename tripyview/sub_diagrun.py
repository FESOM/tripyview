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
from .sub_diagdriver import *

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
def diagrun():
    
    # command line input arguments
    parser = argparse.ArgumentParser(prog='diagrun', description='un FESOM tripyview diagnostics in command line')
    parser.add_argument('workflow_file', help='name of work flow yaml file')
    parser.add_argument('--diagnostics',
                        '-d',
                        nargs='+',
                        help='run only particilar diagnostics from the yml file.',)
    
    # input arguments from command line
    inargs = parser.parse_args()
    if inargs.diagnostics:
        print(f"selective diagnostics will be run: {inargs.diagnostics}")

    #___________________________________________________________________________
    # open selected yaml files
    yaml_filename = inargs.workflow_file
    with open(yaml_filename) as file:
        # load yaml file dictionaries
        yaml_settings = yaml.load(file, Loader=yaml.FullLoader)

    #____DICTIONARY CONTENT_____________________________________________________
    # name of workflow runs --> also folder name 
    workflow_name = yaml_settings['workflow_name']

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
        save_path = f"{yaml_settings['save_path']}/{workflow_name}"
    else:
        save_path = f"./Results_diagrun/{workflow_name}" 
    save_path = os.path.expanduser(save_path)
    save_path = os.path.abspath(save_path)

        
    #_________________________________________________________________________________________________    
    # actualize settings dictionary    
    yaml_settings['input_paths']       = input_paths
    yaml_settings['input_names']       = input_names
    yaml_settings['workflow_name']     = workflow_name
    yaml_settings['workflow_settings'] = None
    yaml_settings['save_path']         = save_path
    yaml_settings['save_path_nb' ]     = os.path.join(save_path, "notebooks")
    yaml_settings['save_path_fig']     = os.path.join(save_path, "figures")    

    #_________________________________________________________________________________________________
    # create save directory if they do not exist
    if not os.path.exists(yaml_settings['save_path']):
        print(f' --> mkdir: {yaml_settings["save_path"]}')
        os.makedirs(yaml_settings["save_path"])
        print(f' --> mkdir: {yaml_settings["save_path_nb"]}')
        os.makedirs(yaml_settings["save_path_nb"])
        print(f' --> mkdir: {yaml_settings["save_path_fig"]}')
        os.makedirs(yaml_settings["save_path_fig"])
        
    #___________________________________________________________________________
    # initialise/create webpage interface .json file
    fname_json = f"{yaml_settings['workflow_name']}.json"
    save_path_json = os.path.join(yaml_settings['save_path'], fname_json)
    if os.path.exists(save_path_json):
        with open(save_path_json) as json_file:
            webpages = json.load(json_file)
            print(f"Data on previous runs exist in {save_path_json}, \n")
            print("they will be used to generate output for diagnostics you do not run this time.")
    else:
        webpages = {}
        webpages["analyses"] = {}

    webpages["general"] = {}
    webpages["general"]["name"] = yaml_settings["workflow_name"]

    #___________________________________________________________________________
    # define analyses drivers
    analyses_opt = {}
    analyses_opt["hslice"         ] = drive_hslice
    analyses_opt["hslice_np"      ] = drive_hslice
    analyses_opt["hslice_sp"      ] = drive_hslice
    analyses_opt["hslice_clim"    ] = drive_hslice_clim
    analyses_opt["hovm"           ] = drive_hovm
    analyses_opt["hovm_clim"      ] = drive_hovm_clim
    analyses_opt["xmoc"           ] = drive_xmoc
    analyses_opt["vprofile"       ] = drive_vprofile
    analyses_opt["vprofile_clim"  ] = drive_vprofile_clim
    analyses_opt["transect"       ] = drive_transect
    analyses_opt["transect_clim"  ] = drive_transect_clim
    analyses_opt["zmeantrans"     ] = drive_zmeantrans
    analyses_opt["zmeantrans_clim"] = drive_zmeantrans_clim
    
    #___________________________________________________________________________
    # loop over available diagnostics and run the one selected in the yaml file
    # loop over all analyses
    for analysis in analyses_opt:
        # check if analysis is in input yaml settings
        if analysis in yaml_settings:
            print(f" --> compute {analysis}:")
            # drive specific analysis
            webpage = analyses_opt[analysis](yaml_settings, analysis)
            webpages["analyses"][analysis] = webpage
            
            # write linked analysis to .json file
            with open(save_path_json, "w") as fp: json.dump(webpages, fp)
            
    #___________________________________________________________________________
    # save everything to .html
    fname_html     = f"{yaml_settings['workflow_name']}.html"
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
    diagrun()    
