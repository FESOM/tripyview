import yaml
import papermill as pm
import argparse
import json
import shutil
import sys
import os
import numpy     as np
import time      as clock 
#os.environ['PATH_TRIPYVIEW'] = '/home/ollie/pscholz/tripyview_github/'

#try: pkg_path = os.environ['PATH_TRIPYVIEW']
#except: pkg_path='.'    
#sys.path.append(os.path.join(pkg_path,"src/"))
from .sub_tripyrundriver import *

#_______________________________________________________________________________       
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
# render webpage collection
def render_experiment_html(webpages, yaml_settings):
    fname_html     = f"{yaml_settings['tripyrun_name']}.html"
    save_path_html = os.path.join(yaml_settings['save_path'], fname_html)
    ofile          = open(save_path_html, "w")
    template       = env.get_template("experiment.html")
    output         = template.render(webpages)
    ofile.write(output)
    ofile.close()   
    return
    
    
    
#
#
#_______________________________________________________________________________
# command line input --> diagrun() 
def tripyrun():
    ts = clock.time()
    print(" --> start time:", clock.strftime("%Y-%m-%d %H:%M:%S", clock.localtime()))

    # command line input arguments
    parser = argparse.ArgumentParser(prog='tripyrun', description='do FESOM tripyview diagnostics in command line')
    
    # input tripyrun_main_all.yml yaml diagnostic file 
    parser.add_argument('workflow_file', help='name of work flow yaml file')
    
    # only run specific diagnostic driver
    parser.add_argument('--diagnostics',
                        '-d',
                        nargs='+',
                        help='run only particular driver diagnostics from the yml file.'+ \
                             'Possible diagnostics are: \n'+ \
                             ' - hmesh \n'+ \
                             ' - hslice, hslice_np, hslice_sp \n'+ \
                             ' - hslice_clim, hslice_clim_np, hslice_clim_sp \n'+ \
                             ' - hslice_isotdep \n'+ \
                             ' - hquiver \n'+ \
                             ' - hovm, hovm_clim \n'+ \
                             ' - transect, transect_clim \n'+ \
                             ' - transect_transp, transect_transp_t \n'+ \
                             ' - transect_hflx, transect_hflx_t \n'+ \
                             ' - transect_zmean, transect_zmean_clim \n'+ \
                             ' - vprofile, vprofile_clim \n'+ \
                             ' - var_t \n'+ \
                             ' - zmoc, zmoc_t \n'+ \
                             ' - dmoc, dmoc_srf, dmoc_inner, dmoc_t \n'+ \
                             ' - dmoc_z, dmoc_srf_z, dmoc_inner_z \n'+ \
                             ' - dmoc_wdiap, dmoc_srfcbflx \n'+ \
                             ' - hbarstreamf \n'+ \
                             ' - ghflx, mhflx, zhflx \n')
    
    # only run specific variable within diagnostic driver
    parser.add_argument('--variable',
                        '-v',
                        nargs='+',
                        help=('run only particular variable in a particular driver'  
                        'diagnostic from the yml file. \n In that case only one driver'
                        'diagnostic is possible but it can be several variables'))
    
    # re-render ,html based on saved json file
    parser.add_argument('--render',
                        '-r',
                        action='store_true',  # This makes it a flag, no need for nargs=0
                        help='run only rendering of .html based on saved json file.',)
    
    # input arguments from command line
    inargs = parser.parse_args()
    if inargs.diagnostics:
        print(f" --> selected diagnostics will be run: {inargs.diagnostics}")
        if inargs.variable:
            print(f"     --> selected variable will be run: {inargs.diagnostics}")
    
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

        
    #___________________________________________________________________________
    # actualize settings dictionary    
    yaml_settings['tripyrun_name']     = tripyrun_name
    yaml_settings['input_paths']       = input_paths
    yaml_settings['input_names']       = input_names
    yaml_settings['do_papermill']      = True
    
    yaml_settings['save_path']         = save_path
    yaml_settings['tripyrun_spath_nb' ]= os.path.join(save_path, "notebooks")
    yaml_settings['tripyrun_spath_fig']= os.path.join(save_path, "figures")  
    
    # papermiller imports python None variable declaration as "None" string this causes 
    # trouble in the if VAR is None: comparison:
    for key in yaml_settings: 
        if yaml_settings[key] == "None": yaml_settings[key]=None

    #___________________________________________________________________________
    # create save directory if they do not exist
    if not os.path.exists(yaml_settings['save_path']):
        print(f' --> mkdir: {yaml_settings["save_path"]}')
        os.makedirs(yaml_settings['save_path'])
        print(f' --> mkdir: {yaml_settings["tripyrun_spath_nb"]}')
        os.makedirs(yaml_settings['tripyrun_spath_nb'])
        print(f' --> mkdir: {yaml_settings["tripyrun_spath_fig"]}')
        os.makedirs(yaml_settings['tripyrun_spath_fig'])
        
    #___________________________________________________________________________
    # define all the analyses drivers
    analyses_driver_list = {}
    analyses_driver_list["hmesh"              ] = drive_hmesh
    
    analyses_driver_list["hslice"             ] = drive_hslice
    analyses_driver_list["hslice_np"          ] = drive_hslice
    analyses_driver_list["hslice_sp"          ] = drive_hslice
    analyses_driver_list["hslice_clim"        ] = drive_hslice_clim
    analyses_driver_list["hslice_clim_np"     ] = drive_hslice_clim
    analyses_driver_list["hslice_clim_sp"     ] = drive_hslice_clim
    analyses_driver_list["hslice_isotdep"     ] = drive_hslice_isotdep
    
    analyses_driver_list["hquiver"            ] = drive_hquiver
    
    analyses_driver_list["hovm"               ] = drive_hovm
    analyses_driver_list["hovm_clim"          ] = drive_hovm_clim
    
    analyses_driver_list["transect"           ] = drive_transect
    analyses_driver_list["transect_clim"      ] = drive_transect_clim
    analyses_driver_list["transect_transp"    ] = drive_transect_Xtransp
    analyses_driver_list["transect_transp_t"  ] = drive_transect_Xtransp_t
    analyses_driver_list["transect_hflx"      ] = drive_transect_Xtransp
    analyses_driver_list["transect_hflx_t"    ] = drive_transect_Xtransp_t
    analyses_driver_list["transect_zmean"     ] = drive_transect_zm_mean
    analyses_driver_list["transect_zmean_clim"] = drive_transect_zm_mean_clim
    analyses_driver_list["transect_mmean"     ] = drive_transect_zm_mean
    analyses_driver_list["transect_mmean_clim"] = drive_transect_zm_mean_clim
    
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
    # initialise/create webpage interface based on .json file, if it exist
    fname_json = f"{yaml_settings['tripyrun_name']}.json"
    save_path_json = os.path.join(yaml_settings['save_path'], fname_json)
    if os.path.exists(save_path_json):
        with open(save_path_json) as json_file:
            webpages = json.load(json_file)
            print(f"Jupyter Notebooks from previous runs exist in {save_path_json},")
            print(f"they will be used to generate output for diagnostics you do not run this time.")
            #___________________________________________________________________
            if inargs.render:
                print(f"\n --> render experment .html:  {yaml_settings['tripyrun_name']}:")
                render_experiment_html(webpages, yaml_settings)
                print(" --> end time:", clock.strftime("%Y-%m-%d %H:%M:%S", clock.localtime()))
                print(" --> elapsed time: {:2.2f} min.".format((clock.time()-ts)/60))
                return
    else:
        # json file doesnt exist, freshly initialise webpage
        webpages = {}
        webpages["analyses"] = {}
    
    #___________________________________________________________________________
    webpages["general"] = {}
    webpages["general"]["name"] = yaml_settings["tripyrun_name"]
    webpages["logo"] = {}
    webpages["logo"]["path"] =os.path.join(templates_path, 'fesom2_logo.png')

    #___________________________________________________________________________
    # loop over available diagnostics and run the one selected in the yaml file
    # loop over all analyses
    for analysis_name in analyses_driver_list:
        # check if analysis is in input yaml settings
        if analysis_name in yaml_settings:
            
            #___________________________________________________________________
            # if -d ... flag is setted perform specific yml file diagnostic    
            if inargs.diagnostics:
                if analysis_name in inargs.diagnostics:
                    print(f"\n\n --> compute {analysis_name}:")
                    #___________________________________________________________
                    # if -v vname1 vname2 ... flag is setted perform specific 
                    # variables in specific yml file diagnostic    
                    if inargs.variable:
                        for vname in inargs.variable:
                            cnt, cnt_max = -1, -1
                            
                            # drive specific analysis from analyses_driver_list only
                            # for one specifc anlysis
                            for values in webpages["analyses"][analysis_name].values():
                                cnt_max = np.maximum(cnt_max,values['cnt'])
                                if values['variable']==vname: 
                                    cnt = values['cnt']
                                    break
                            if cnt==-1: 
                                print(" --> could not find variable: {vname} in loaded webpage. This variable will be attached if it exist")
                                cnt=cnt_max+1
                            webpage = analyses_driver_list[analysis_name](yaml_settings, analysis_name, webpage=webpages["analyses"][analysis_name], image_count=cnt, vname=vname)
                            webpages["analyses"][analysis_name] = webpage
                    
                    #___________________________________________________________
                    # if no -v vname1 vname2 ... flag is setted
                    # execute entire specfic driver section as it is defined in the yaml file
                    else:  
                        webpage = analyses_driver_list[analysis_name](yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None)
                        webpages["analyses"][analysis_name] = webpage
            
            #___________________________________________________________________
            # if no -d ... flag is setted perform full yml file diagnostic
            else:
                print(f"\n\n --> compute {analysis_name}:")
                # drive specific analysis from analyses_driver_list
                webpage = analyses_driver_list[analysis_name](yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None)
                webpages["analyses"][analysis_name] = webpage
            
            #___________________________________________________________________
            # write linked analysis to .json file
            with open(save_path_json, "w") as fp: json.dump(webpages, fp)
            
    #___________________________________________________________________________
    # save everything to .html and render it 
    render_experiment_html(webpages, yaml_settings)
    
    #___________________________________________________________________________
    # save everything to .html
    #render_main_page()
    print(" --> end time:", clock.strftime("%Y-%m-%d %H:%M:%S", clock.localtime()))
    print(" --> elapsed time: {:2.2f} min.".format((clock.time()-ts)/60))
    
#
#
#_______________________________________________________________________________
if __name__ == "__main__":
    # args = parser.parse_args()
    # args.func(args)
    tripyrun()
