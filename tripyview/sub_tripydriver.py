import yaml
import papermill as pm
import math
import pkg_resources
from jinja2 import Environment, FileSystemLoader
import yaml
import argparse
import json
import glob
import shutil
import sys
import os

#_______________________________________________________________________________       
# open htnl template file
#try: pkg_path = os.environ['PATH_TRIPYVIEW']
#except: pkg_path='' 
pkg_path          = os.path.dirname(os.path.dirname(__file__))
templates_path    = os.path.join(pkg_path,'templates_html')
templates_nb_path = os.path.join(pkg_path,'templates_notebooks')
file_loader       = FileSystemLoader(templates_path)
env               = Environment(loader=file_loader)



#
#
#_______________________________________________________________________________
def hslice_exec_papermill(webpage, cnt, vname_params, exec_template='hslice'):
    #___________________________________________________________________________
    # create strings for fname and labels in the webpage
    str_vname1, str_vname2 = f"_{vname_params['vname'].replace('/',':')}", f"{vname_params['vname'].replace('/',':')}"
    str_proj               = f"_{vname_params['proj']}" if 'proj' in vname_params else ''
    str_dep1, str_dep2     = '', ''
    if 'depth' in vname_params: str_dep1, str_dep2 = f"_z{vname_params['depth']}", f", @dep:{vname_params['depth']}"
    str_mon1, str_mon2     = '', ''
    if 'mon'   in vname_params: str_mon1, str_mon2 = f"_m{vname_params['mon']}"  , f", @mon:{vname_params['mon']}"
    str_isotd1, str_isotd2 = '', ''
    if 'which_isotherm' in vname_params: str_isotd1, str_isotd2 = f"_{vname_params['which_isotherm']}", f"depth of {vname_params['which_isotherm']} °C isotherm"
    #___________________________________________________________________________
    # create filepaths for notebook and figures 
    save_fname    = f"{vname_params['tripyrun_name']}_{vname_params['tripyrun_analysis']}{str_vname1}{str_proj}{str_isotd1}{str_dep1}{str_mon1}.png"
    save_fname_nb = f"{vname_params['tripyrun_name']}_{vname_params['tripyrun_analysis']}{str_vname1}{str_proj}{str_isotd1}{str_dep1}{str_mon1}.ipynb"
    short_name    = f"{vname_params['tripyrun_name']}_{vname_params['tripyrun_analysis']}{str_vname1}{str_proj}{str_isotd1}{str_dep1}{str_mon1}"
    vname_params["save_fname"] = os.path.join(vname_params['tripyrun_spath_fig'], save_fname)
                
    #___________________________________________________________________________
    # execute notebook with papermill
    pm.execute_notebook(f"{templates_nb_path}/template_{exec_template}.ipynb",
                        os.path.join(vname_params['tripyrun_spath_nb'], save_fname_nb),
                        parameters=vname_params,
                        nest_asyncio=True,)
                
    #___________________________________________________________________________
    # attach created figures to webpage collection
    webpage[f"image_{cnt}"] = {}
    webpage[f"image_{cnt}"]["name"]       = f"{str_isotd2}{str_vname2}{str_dep2}{str_mon2}"
    webpage[f"image_{cnt}"]["path"]       = os.path.join('./figures/', save_fname)
    webpage[f"image_{cnt}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
    webpage[f"image_{cnt}"]["short_name"] = short_name
    cnt += 1
    
    #___________________________________________________________________________
    return(webpage, cnt)



#
#
#_______________________________________________________________________________
def drive_hslice(yaml_settings, analysis_name):
    print(' --> drive_hslice:',analysis_name)
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            if value=="None": value=None 
            current_params[key] = value
    
    #___________________________________________________________________________
    # loop over variable name  
    webpage, image_count = {}, 0
    for vname in driver_settings:
        print(f'         --> compute: {vname}')
        vname_params = current_params.copy()
        vname_params.update(driver_settings[vname])
        vname_params["vname"] = vname
        vname_params["tripyrun_analysis"] = analysis_name
        
        #_______________________________________________________________________
        # make no loop over depths
        if 'depths' not in driver_settings[vname]:
            #___________________________________________________________________
            # make no loops over the months
            if 'months' not in driver_settings[vname]:
                webpage, image_count = hslice_exec_papermill(webpage, image_count, vname_params)
                
            #___________________________________________________________________
            # make loop over list of months
            else:
                del vname_params["months"]
                for mon in driver_settings[vname]["months"]:
                    print(f'                 --> mon: {mon}')
                    vname_params["mon"] = mon
                    webpage, image_count = hslice_exec_papermill(webpage, image_count, vname_params)
        
        #_______________________________________________________________________
        # make loop over list of depths
        else:    
            del vname_params["depths"] # --> delete depth list [0, 100, 1000,...] from current_param dict()
            for depth in driver_settings[vname]["depths"]:
                print(f'             --> depth: {depth}')
                vname_params["depth"] = depth
                
                #_______________________________________________________________
                # make no loops over the months
                if 'months' not in driver_settings[vname]:
                    webpage, image_count = hslice_exec_papermill(webpage, image_count, vname_params)
                
                #_______________________________________________________________
                # make loop over list of months
                else:
                    del vname_params["months"]
                    for mon in driver_settings[vname]["months"]:
                        print(f'                 --> mon: {mon}')
                        vname_params["mon"] = mon
                        webpage, image_count = hslice_exec_papermill(webpage, image_count, vname_params)
    #___________________________________________________________________________        
    return webpage



#
#
#_______________________________________________________________________________
def drive_hslice_clim(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            if value=="None": value=None
            current_params[key] = value
    
    #___________________________________________________________________________
    # loop over variable name  
    webpage, image_count = {}, 0
    for vname in driver_settings:
        print(f'         --> compute:{vname}')
        vname_params = current_params.copy()
        vname_params.update(driver_settings[vname])
        vname_params["vname"] = vname
        vname_params["tripyrun_analysis"] = analysis_name
        
        #_______________________________________________________________________
        # make no loop over depths
        if 'depths' not in driver_settings[vname]:
            webpage, image_count = hslice_exec_papermill(webpage, image_count, vname_params, exec_template='hslice_clim')
        #_______________________________________________________________________
        # make loop over list of depths
        else: 
            del vname_params["depths"] # --> delete depth list [0, 100, 1000,...] from current_param dict()
            for depth in driver_settings[vname]["depths"]:
                print(f'             --> depth: {depth}')
                vname_params["depth"] = depth
                webpage, image_count = hslice_exec_papermill(webpage, image_count, vname_params, exec_template='hslice_clim')
    return webpage



#
#
#_______________________________________________________________________________
def drive_hslice_isotdep(yaml_settings, analysis_name):
    print(' --> drive_hslice:',analysis_name)
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            if value=="None": value=None 
            current_params[key] = value
    
    #___________________________________________________________________________
    # loop over isotherms
    #_______________________________________________________________________
    # make no loop over isotherms
    webpage, image_count = {}, 0
    vname_params = current_params.copy()
    vname_params.update(driver_settings)
    vname_params["tripyrun_analysis"] = analysis_name
    vname_params["vname"] = 'isotdep'
    if 'which_isotherms' not in driver_settings:
        print(f'         --> compute isotherm depth @: {which_isotherm}')
        vname_params["which_isotherm"]    = driver_settings['which_isotherm']
    #_______________________________________________________________________
    # make loop over list of isotherms    
    else:   
        del vname_params["which_isotherms"]
        for which_isotherm in driver_settings['which_isotherms']:
            print(f'         --> compute isotherm depth @: {which_isotherm}')
            vname_params["which_isotherm"] = which_isotherm
            
            #___________________________________________________________________
            # make no loops over the months
            if 'months' not in driver_settings:
                webpage, image_count = hslice_exec_papermill(webpage, image_count, vname_params, exec_template='hslice_isotdep')
                
            #___________________________________________________________________
            # make loop over list of months
            else:
                del vname_params["months"]
                for mon in driver_settings["months"]:
                    print(f'                 --> mon: {mon}')
                    vname_params["mon"] = mon
                    webpage, image_count = hslice_exec_papermill(webpage, image_count, vname_params, exec_template='hslice_isotdep')
    return webpage



#
#
#_______________________________________________________________________________
def drive_hovm(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    # loop over variable name  
    for vname in driver_settings:
        print(f'         --> compute: {vname}')
        auxvname = vname.replace('/',':')
        
        # loop over depths
        for box_region in driver_settings[vname]["box_regions"]:
            print(f'             --> compute: {box_region}')
            current_params2 = {}
            current_params2 = current_params.copy()
            current_params2["vname"] = vname
            current_params2["box_region"] = list([box_region])
            current_params2.update(driver_settings[vname])
            del current_params2["box_regions"] # --> delete depth list [0, 100, 1000,...] from current_param dict()
            str_boxregion = box_region.split('/')[-1].split('.')[0]
            
            #___________________________________________________________________
            save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{str_boxregion}.png"
            save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{str_boxregion}.ipynb"
            current_params2["save_fname"] = os.path.join(yaml_settings['tripyrun_spath_fig'], save_fname)
            
            #___________________________________________________________________
            pm.execute_notebook(
                f"{templates_nb_path}/template_hovm.ipynb",
                os.path.join(yaml_settings['tripyrun_spath_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True,
            )
            
            #___________________________________________________________________
            webpage[f"image_{image_count}"] = {}
            webpage[f"image_{image_count}"][
                "name"
            ] = f"{auxvname.capitalize()} at {str_boxregion} m"
            webpage[f"image_{image_count}"]["path"] = os.path.join('./figures/', save_fname)
            webpage[f"image_{image_count}"]["path_nb"] = os.path.join('./notebooks/', save_fname_nb)
            webpage[f"image_{image_count}"][
                "short_name"
            ] = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{str_boxregion}"
            image_count += 1
    return webpage



#
#
#_______________________________________________________________________________
def drive_hovm_clim(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    # loop over variable name  
    for vname in driver_settings:
        print(f'         --> compute: {vname}')
        
        # loop over depths
        for box_region in driver_settings[vname]["box_regions"]:
            print(f'             --> compute: {box_region}')
            current_params2 = {}
            current_params2 = current_params.copy()
            current_params2["vname"] = vname
            current_params2["box_region"] = list([box_region])
            current_params2.update(driver_settings[vname])
            del current_params2["box_regions"] # --> delete depth list [0, 100, 1000,...] from current_param dict()
            str_boxregion = box_region.split('/')[-1].split('.')[0]
            
            #___________________________________________________________________
            save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{vname}_{str_boxregion}.png"
            save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{vname}_{str_boxregion}.ipynb"
            current_params2["save_fname"] = os.path.join(yaml_settings['tripyrun_spath_fig'], save_fname)
            
            #___________________________________________________________________
            pm.execute_notebook(
                f"{templates_nb_path}/template_hovm_clim.ipynb",
                os.path.join(yaml_settings['tripyrun_spath_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True,
            )
            
            #___________________________________________________________________
            webpage[f"image_{image_count}"] = {}
            webpage[f"image_{image_count}"]["name"      ] = f"{vname.capitalize()} at {str_boxregion} m"
            webpage[f"image_{image_count}"]["path"      ] = os.path.join('./figures/', save_fname)
            webpage[f"image_{image_count}"]["path_nb"   ] = os.path.join('./notebooks/', save_fname_nb)
            webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{vname}_{str_boxregion}"
            image_count += 1
    return webpage



#
#
#_______________________________________________________________________________
def drive_zmoc(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    # loop over variable name  
    for vname in driver_settings:
        print(f'         --> compute: {vname}')
        
        current_params2 = {}
        current_params2 = current_params.copy()
        current_params2["vname"] = vname
        if driver_settings[vname] is not None:
            current_params2.update(driver_settings[vname])
            
        #_______________________________________________________________________
        save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{vname}.png"
        save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{vname}.ipynb"
        current_params2["save_fname"] = os.path.join(yaml_settings['tripyrun_spath_fig'], save_fname)
            
        #_______________________________________________________________________
        pm.execute_notebook(
                f"{templates_nb_path}/template_transp_zmoc.ipynb",
                os.path.join(yaml_settings['tripyrun_spath_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True)
            
        #_______________________________________________________________________
        webpage[f"image_{image_count}"] = {}
        webpage[f"image_{image_count}"]["name"]       = f"{vname.upper()}"
        webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
        webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
        webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{vname}"
        image_count += 1
    return webpage



#
#
#_______________________________________________________________________________
def drive_zmoc_t(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    # loop over variable name  
    for which_lat in driver_settings['which_lats']:
        print(f'         --> compute tseries @: {str(which_lat)}')
        
        current_params2 = {}
        current_params2 = current_params.copy()
        current_params2["which_lat"] = [which_lat]
        current_params2.update(driver_settings)
        del current_params2["which_lats"]
            
        #_______________________________________________________________________
        save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{which_lat}.png"
        save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{which_lat}.ipynb"
        current_params2["save_fname"] = os.path.join(yaml_settings['tripyrun_spath_fig'], save_fname)
            
        #_______________________________________________________________________
        pm.execute_notebook(
                f"{templates_nb_path}/template_transp_zmoc_t.ipynb",
                os.path.join(yaml_settings['tripyrun_spath_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True)
            
        #_______________________________________________________________________
        webpage[f"image_{image_count}"] = {}
        if which_lat == 'max':
            webpage[f"image_{image_count}"]["name"]       = f" max AMOC @ 30°N<lat<45°N"
        else:
            webpage[f"image_{image_count}"]["name"]       = f" AMOC @ {which_lat}°N"
        webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
        webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
        webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{which_lat}"
        image_count += 1
    return webpage



#
#
#_______________________________________________________________________________
def drive_dmoc(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    # loop over variable name  
    for vname in driver_settings:
        print(f'         --> compute: {vname}')
        current_params2 = {}
        current_params2 = current_params.copy()
        current_params2["vname"] = vname
        if   analysis_name == 'dmoc'      : 
            current_params2["which_transf"], str_mode = 'dmoc' , ''
        elif analysis_name == 'dmoc_srf'  : 
            current_params2["which_transf"], str_mode = 'srf'  , '_srf'
        elif analysis_name == 'dmoc_inner': 
            current_params2["which_transf"], str_mode = 'inner', '_inner'
        elif analysis_name == 'dmoc_z': 
            current_params2["which_transf"], str_mode = 'dmoc', '_z'    
            current_params2["do_zcoord"] = True
        elif analysis_name == 'dmoc_srf_z'  : 
            current_params2["do_zcoord"] = True
            current_params2["which_transf"], str_mode = 'srf'  , '_srf_z'
        elif analysis_name == 'dmoc_inner_z': 
            current_params2["which_transf"], str_mode = 'inner', '_inner_z'
            current_params2["do_zcoord"] = True
        if driver_settings[vname] is not None:
            current_params2.update(driver_settings[vname])
        #_______________________________________________________________________
        save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{vname}.png"
        save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{vname}.ipynb"
        current_params2["save_fname"] = os.path.join(yaml_settings['tripyrun_spath_fig'], save_fname)
            
        #_______________________________________________________________________
        pm.execute_notebook(
                f"{templates_nb_path}/template_transp_dmoc.ipynb",
                os.path.join(yaml_settings['tripyrun_spath_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True)
            
        #_______________________________________________________________________
        webpage[f"image_{image_count}"] = {}
        webpage[f"image_{image_count}"]["name"]       = f"Density-{vname.upper()}{str_mode}"
        webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
        webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
        webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{vname}"
        image_count += 1
    return webpage


#
#
#_______________________________________________________________________________
def drive_dmoc_t(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    # loop over variable name  
    for which_lat in driver_settings['which_lats']:
        print(f'         --> compute tseries @: {which_lat}')
        
        current_params2 = {}
        current_params2 = current_params.copy()
        current_params2["which_lat"] = [which_lat]
        current_params2.update(driver_settings)
        del current_params2["which_lats"]
            
        #_______________________________________________________________________
        save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{which_lat}.png"
        save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{which_lat}.ipynb"
        current_params2["save_fname"] = os.path.join(yaml_settings['tripyrun_spath_fig'], save_fname)
            
        #_______________________________________________________________________
        pm.execute_notebook(
                f"{templates_nb_path}/template_transp_dmoc_t.ipynb",
                os.path.join(yaml_settings['tripyrun_spath_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True)
            
        #_______________________________________________________________________
        webpage[f"image_{image_count}"] = {}
        if which_lat == 'max':
            webpage[f"image_{image_count}"]["name"]       = f" max density-AMOC @ 45°N<lat<60°N"
        else:
            webpage[f"image_{image_count}"]["name"]       = f" density AMOC @ {which_lat}°N"
        webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
        webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
        webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{which_lat}"
        image_count += 1
    return webpage


#
#
#_______________________________________________________________________________
def drive_dmoc_wdiap(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    # loop over diffrent ispycnals   
    for which_isopyc in driver_settings['which_isopycs']:
        
        current_params2 = {}
        current_params2 = current_params.copy()
        current_params2["which_isopyc"] = [which_isopyc]
        current_params2.update(driver_settings)
        del current_params2["which_isopycs"] # --> delete depth list [0, 100, 1000,...] from current_param dict()
            
        #_______________________________________________________________________
        if 'proj' in current_params2.keys(): 
            save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{current_params2['proj']}_{which_isopyc}.png"
            save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{current_params2['proj']}_{which_isopyc}.ipynb"
            short_name    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{current_params2['proj']}_{which_isopyc}"
        else:
            save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{which_isopyc}.png"
            save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{which_isopyc}.ipynb"
            short_name    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{which_isopyc}"
        current_params2["save_fname"] = os.path.join(yaml_settings['tripyrun_spath_fig'], save_fname)
            
        #_______________________________________________________________________
        pm.execute_notebook(
            f"{templates_nb_path}/template_transp_dmoc_wdiap.ipynb",
            os.path.join(yaml_settings['tripyrun_spath_nb'], save_fname_nb),
            parameters=current_params2,
            nest_asyncio=True,
        )
            
        #_______________________________________________________________________
        webpage[f"image_{image_count}"] = {}
        webpage[f"image_{image_count}"]["name"]       = f"{'W_diap'} at sigma2={which_isopyc} kg/m^3"
        webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
        webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
        webpage[f"image_{image_count}"]["short_name"] = short_name
        image_count += 1
    return webpage


#
#
#_______________________________________________________________________________
def drive_dmoc_srfcbflx(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    # loop over diffrent ispycnals   
    for which_isopyc in driver_settings['which_isopycs']:
        
        current_params2 = {}
        current_params2 = current_params.copy()
        current_params2["which_isopyc"] = [which_isopyc]
        current_params2.update(driver_settings)
        del current_params2["which_isopycs"] # --> delete depth list [0, 100, 1000,...] from current_param dict()
            
        #_______________________________________________________________________
        if 'proj' in current_params2.keys(): 
            save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{current_params2['proj']}_{which_isopyc}.png"
            save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{current_params2['proj']}_{which_isopyc}.ipynb"
            short_name    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{current_params2['proj']}_{which_isopyc}"
        else:
            save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{which_isopyc}.png"
            save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{which_isopyc}.ipynb"
            short_name    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{which_isopyc}"
        current_params2["save_fname"] = os.path.join(yaml_settings['tripyrun_spath_fig'], save_fname)
        
        #_______________________________________________________________________
        pm.execute_notebook(
            f"{templates_nb_path}/template_transp_dmoc_srfcbflx.ipynb",
            os.path.join(yaml_settings['tripyrun_spath_nb'], save_fname_nb),
            parameters=current_params2,
            nest_asyncio=True,
        )
        
        #_______________________________________________________________________
        webpage[f"image_{image_count}"] = {}
        webpage[f"image_{image_count}"]["name"]       = f"{'srf. buoyancy transf.'} at sigma2={which_isopyc} kg/m^3"
        webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
        webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
        webpage[f"image_{image_count}"]["short_name"] = short_name
        image_count += 1
    return webpage


#
#
#_______________________________________________________________________________
def drive_hbarstreamf(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #            
    if not yaml_settings[analysis_name]: driver_settings = dict()
    else                               : driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = dict()
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            current_params[key] = value
    # initialse webpage for analyis
    webpage = {}
    image_count = 0
    print(f'         --> compute: hbarstreamf')
    if not current_params: current_params2 = dict()
    else                 : current_params2 = current_params.copy()
    current_params2.update(driver_settings)
            
    #_______________________________________________________________________
    save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}.png"
    save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}.ipynb"
    current_params2["save_fname"] = os.path.join(yaml_settings['tripyrun_spath_fig'], save_fname)
            
    #_______________________________________________________________________
    pm.execute_notebook(
            f"{templates_nb_path}/template_transp_hbstreamf.ipynb",
            os.path.join(yaml_settings['tripyrun_spath_nb'], save_fname_nb),
            parameters=current_params2,
            nest_asyncio=True)
            
    #_______________________________________________________________________
    webpage[f"image_{image_count}"] = {}
    webpage[f"image_{image_count}"]["name"]       = f"{'horiz. barotrop. streamfunction'}"
    webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
    webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
    webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['tripyrun_name']}_{analysis_name}"
    image_count += 1
    return webpage



#
#
#_______________________________________________________________________________
def drive_vprofile(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    # loop over variable name  
    for vname in driver_settings:
        print(f'         --> compute: {vname}')
        
        current_params2 = {}
        current_params2 = current_params.copy()
        current_params2["vname"] = vname
        current_params2.update(driver_settings[vname])
            
        #_______________________________________________________________________
        save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{vname}.png"
        save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{vname}.ipynb"
        current_params2["save_fname"] = os.path.join(yaml_settings['tripyrun_spath_fig'], save_fname)
            
        #_______________________________________________________________________
        pm.execute_notebook(
                f"{templates_nb_path}/template_vprofile.ipynb",
                os.path.join(yaml_settings['tripyrun_spath_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True)
            
        #_______________________________________________________________________
        webpage[f"image_{image_count}"] = {}
        webpage[f"image_{image_count}"]["name"]       = f"{vname.upper()}"
        webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
        webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
        webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{vname}"
        image_count += 1
    return webpage



#
#
#_______________________________________________________________________________
def drive_vprofile_clim(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    # loop over variable name  
    for vname in driver_settings:
        print(f'         --> compute: {vname}')
        
        current_params2 = {}
        current_params2 = current_params.copy()
        current_params2["vname"] = vname
        current_params2.update(driver_settings[vname])
            
        #_______________________________________________________________________
        save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{vname}.png"
        save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{vname}.ipynb"
        current_params2["save_fname"] = os.path.join(yaml_settings['tripyrun_spath_fig'], save_fname)
            
        #_______________________________________________________________________
        pm.execute_notebook(
                f"{templates_nb_path}/template_vprofile_clim.ipynb",
                os.path.join(yaml_settings['tripyrun_spath_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True)
            
        #_______________________________________________________________________
        webpage[f"image_{image_count}"] = {}
        webpage[f"image_{image_count}"]["name"]       = f"{vname.capitalize()}"
        webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
        webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
        webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{vname}"
        image_count += 1
    return webpage



#
#
#
#_______________________________________________________________________________
def drive_transect(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    #___________________________________________________________________________
    which_transects = driver_settings['which_transects']
    driver_settings2 = yaml_settings[analysis_name].copy()
    del driver_settings2['which_transects']
    
    #___________________________________________________________________________
    # loop over variable name
    for vname in driver_settings2: 
        auxvname = vname.replace('/',':')
        print(f'     -->{vname}')
        
        #_______________________________________________________________________
        # loop over transect name  
        for transect in which_transects:
            tname = transect[2]
            tname = tname.replace(' ','_')
            tname = tname.replace(',','')
            tname = tname.replace('°','')
            print(f'         -->{tname}')
            
            #___________________________________________________________________
            current_params2 = {}
            current_params2 = current_params.copy()
            current_params2["vname"] = vname
            current_params2["input_transect"] = list([transect])
            current_params2.update(driver_settings2[vname])
            
            #___________________________________________________________________
            save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{tname}.png"
            save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{tname}.ipynb"
            current_params2["save_fname"] = os.path.join(yaml_settings['tripyrun_spath_fig'], save_fname)
            
            #___________________________________________________________________
            pm.execute_notebook(
                    f"{templates_nb_path}/template_transect.ipynb",
                    os.path.join(yaml_settings['tripyrun_spath_nb'], save_fname_nb),
                    parameters=current_params2,
                    nest_asyncio=True)
            
            #___________________________________________________________________
            webpage[f"image_{image_count}"] = {}
            webpage[f"image_{image_count}"]["name"]       = f"{auxvname.upper()} @ {tname}"
            webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
            webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
            webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{tname}"
            image_count += 1
    return webpage



#
#
#_______________________________________________________________________________
def drive_transect_clim(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    #___________________________________________________________________________
    which_transects = driver_settings['which_transects']
    driver_settings2 = yaml_settings[analysis_name].copy()
    del driver_settings2['which_transects']
    
    #___________________________________________________________________________
    # loop over variable name
    for vname in driver_settings2: 
        auxvname = vname.replace('/',':')
        print(f'            -->{vname}')
        
        #_______________________________________________________________________
        # loop over transect name  
        for transect in which_transects:
            tname = transect[2]
            tname = tname.replace(' ','_')
            tname = tname.replace(',','')
            tname = tname.replace('°','')
            print(f'         -->{tname}')
            
            #___________________________________________________________________
            current_params2 = {}
            current_params2 = current_params.copy()
            current_params2["vname"] = vname
            current_params2["input_transect"] = list([transect])
            current_params2.update(driver_settings2[vname])
            
            #___________________________________________________________________
            save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{tname}.png"
            save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{tname}.ipynb"
            current_params2["save_fname"] = os.path.join(yaml_settings['tripyrun_spath_fig'], save_fname)
            
            #___________________________________________________________________
            pm.execute_notebook(
                    f"{templates_nb_path}/template_transect_clim.ipynb",
                    os.path.join(yaml_settings['tripyrun_spath_nb'], save_fname_nb),
                    parameters=current_params2,
                    nest_asyncio=True)
            
            #___________________________________________________________________
            webpage[f"image_{image_count}"] = {}
            webpage[f"image_{image_count}"]["name"]       = f"{auxvname.upper()} @ {tname}"
            webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
            webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
            webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{tname}"
            image_count += 1
    return webpage



#
#
#_______________________________________________________________________________
def drive_transect_transp(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    #___________________________________________________________________________
    which_transects = driver_settings['which_transects']
    driver_settings2 = yaml_settings[analysis_name].copy()
    del driver_settings2['which_transects']
    
    #___________________________________________________________________________
    # loop over variable name
    for vname in driver_settings2: 
        auxvname = vname.replace('/',':')
        print(f'            -->{vname}')
        
        #_______________________________________________________________________
        # loop over transect name  
        for transect in which_transects:
            tname = transect[2]
            tname = tname.replace(' ','_')
            tname = tname.replace(',','')
            tname = tname.replace('°','')
            print(f'         -->{tname}')
            
            #___________________________________________________________________
            current_params2 = {}
            current_params2 = current_params.copy()
            current_params2["vname"] = vname
            current_params2["input_transect"] = list([transect])
            current_params2.update(driver_settings2[vname])
            
            #___________________________________________________________________
            save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{tname}.png"
            save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{tname}.ipynb"
            current_params2["save_fname"] = os.path.join(yaml_settings['tripyrun_spath_fig'], save_fname)
            
            #___________________________________________________________________
            pm.execute_notebook(
                    f"{templates_nb_path}/template_transect_transp.ipynb",
                    os.path.join(yaml_settings['tripyrun_spath_nb'], save_fname_nb),
                    parameters=current_params2,
                    nest_asyncio=True)
            
            #___________________________________________________________________
            webpage[f"image_{image_count}"] = {}
            webpage[f"image_{image_count}"]["name"]       = f"{auxvname.upper()} @ {tname}"
            webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
            webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
            webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{tname}"
            image_count += 1
    return webpage



#
#
#_______________________________________________________________________________
def drive_transect_transp_t(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    #___________________________________________________________________________
    which_transects = driver_settings['which_transects']
    driver_settings2 = yaml_settings[analysis_name].copy()
    del driver_settings2['which_transects']
    
    #___________________________________________________________________________
    # loop over variable name
    for vname in driver_settings2: 
        auxvname = vname.replace('/',':')
        print(f'            -->{vname}')
        
        #_______________________________________________________________________
        # loop over transect name  
        for transect in which_transects:
            tname = transect[2]
            tname = tname.replace(' ','_')
            print(f'         -->{tname}')
            
            #___________________________________________________________________
            current_params2 = {}
            current_params2 = current_params.copy()
            current_params2["vname"] = vname
            current_params2["input_transect"] = list([transect])
            current_params2.update(driver_settings2[vname])    
            
            #_______________________________________________________________________
            save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{tname}.png"
            save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{tname}.ipynb"
            current_params2["save_fname"] = os.path.join(yaml_settings['tripyrun_spath_fig'], save_fname)
            
            #_______________________________________________________________________
            pm.execute_notebook(
                    f"{templates_nb_path}/template_transect_transp_t.ipynb",
                    os.path.join(yaml_settings['tripyrun_spath_nb'], save_fname_nb),
                    parameters=current_params2,
                    nest_asyncio=True)
            
            #_______________________________________________________________________
            webpage[f"image_{image_count}"] = {}
            webpage[f"image_{image_count}"]["name"]       = f"{auxvname.upper()} @ {tname}"
            webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
            webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
            webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{tname}"
            image_count += 1
    return webpage

#
#
#_______________________________________________________________________________
def drive_transect_transp_t_OSNAP(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    #___________________________________________________________________________
    which_transects = driver_settings['which_transects']
    driver_settings2 = yaml_settings[analysis_name].copy()
    del driver_settings2['which_transects']
    
    #___________________________________________________________________________
    # loop over variable name
    for vname in driver_settings2: 
        auxvname = vname.replace('/',':')
        print(f'            -->{vname}')
        
        #_______________________________________________________________________
        # loop over transect name  
        for transect in which_transects:
            tname = transect[2]
            tname = tname.replace(' ','_')
            print(f'         -->{tname}')
            
            #___________________________________________________________________
            current_params2 = {}
            current_params2 = current_params.copy()
            current_params2["vname"] = vname
            current_params2["input_transect"] = list([transect])
            current_params2.update(driver_settings2[vname])    
            
            #_______________________________________________________________________
            save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{tname}.png"
            save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{tname}.ipynb"
            current_params2["save_fname"] = os.path.join(yaml_settings['tripyrun_spath_fig'], save_fname)
            
            #_______________________________________________________________________
            pm.execute_notebook(
                    f"{templates_nb_path}/template_transect_transp_t_OSNAP.ipynb",
                    os.path.join(yaml_settings['tripyrun_spath_nb'], save_fname_nb),
                    parameters=current_params2,
                    nest_asyncio=True)
            
            #_______________________________________________________________________
            webpage[f"image_{image_count}"] = {}
            webpage[f"image_{image_count}"]["name"]       = f"{auxvname.upper()} @ {tname}"
            webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
            webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
            webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{tname}"
            image_count += 1
    return webpage


#
#
#_______________________________________________________________________________
def drive_transect_zmean(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    #___________________________________________________________________________
    which_box_regions = driver_settings['which_box_regions']
    driver_settings2 = yaml_settings[analysis_name].copy()
    del driver_settings2['which_box_regions']
    
    #___________________________________________________________________________
    # loop over variable name
    for vname in driver_settings2: 
        auxvname = vname.replace('/',':')
        print(f'            -->{vname}')
        
        #_______________________________________________________________________
        # loop over transect name  
        for box_region in which_box_regions:
            aux_boxname = box_region.split('/')[-1].split('.')[0]
            print(f'         -->{box_region}')
            
            current_params2 = {}
            current_params2 = current_params.copy()
            current_params2["vname"] = vname
            current_params2["box_region"] = list([box_region])
            current_params2.update(driver_settings2[vname])
            
            #___________________________________________________________________
            save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{aux_boxname}.png"
            save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{aux_boxname}.ipynb"
            current_params2["save_fname"] = os.path.join(yaml_settings['tripyrun_spath_fig'], save_fname)
            
            #___________________________________________________________________
            pm.execute_notebook(
                f"{templates_nb_path}/template_transect_zmean.ipynb",
                os.path.join(yaml_settings['tripyrun_spath_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True,
            )
            
            #___________________________________________________________________
            webpage[f"image_{image_count}"] = {}
            webpage[f"image_{image_count}"][
                "name"
            ] = f"{auxvname.capitalize()} at {aux_boxname} m"
            webpage[f"image_{image_count}"]["path"] = os.path.join('./figures/', save_fname)
            webpage[f"image_{image_count}"]["path_nb"] = os.path.join('./notebooks/', save_fname_nb)
            webpage[f"image_{image_count}"][
                "short_name"
            ] = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{aux_boxname}"
            image_count += 1
    return webpage



#
#
#_______________________________________________________________________________
def drive_transect_zmean_clim(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    #___________________________________________________________________________
    which_box_regions = driver_settings['which_box_regions']
    driver_settings2 = yaml_settings[analysis_name].copy()
    del driver_settings2['which_box_regions']
    
    #___________________________________________________________________________
    # loop over variable name
    for vname in driver_settings2: 
        auxvname = vname.replace('/',':')
        print(f'            -->{vname}')
        
        #_______________________________________________________________________
        # loop over transect name  
        for box_region in which_box_regions:
            aux_boxname = box_region.split('/')[-1].split('.')[0]
            print(f'         -->{box_region}')
            
            current_params2 = {}
            current_params2 = current_params.copy()
            current_params2["vname"] = vname
            current_params2["box_region"] = list([box_region])
            current_params2.update(driver_settings2[vname])
            
            #___________________________________________________________________
            save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{vname}_{aux_boxname}.png"
            save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{vname}_{aux_boxname}.ipynb"
            current_params2["save_fname"] = os.path.join(yaml_settings['tripyrun_spath_fig'], save_fname)
            
            #___________________________________________________________________
            pm.execute_notebook(
                f"{templates_nb_path}/template_transect_zmean_clim.ipynb",
                os.path.join(yaml_settings['tripyrun_spath_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True,
            )
            
            #___________________________________________________________________
            webpage[f"image_{image_count}"] = {}
            webpage[f"image_{image_count}"][
                "name"
            ] = f"{vname.capitalize()} at {aux_boxname} m"
            webpage[f"image_{image_count}"]["path"] = os.path.join('./figures/', save_fname)
            webpage[f"image_{image_count}"]["path_nb"] = os.path.join('./notebooks/', save_fname_nb)
            webpage[f"image_{image_count}"][
                "short_name"
            ] = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{vname}_{aux_boxname}"
            image_count += 1
    return webpage



#
#
#_______________________________________________________________________________
def drive_ghflx(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    print(f'         --> compute ghflx:')
    current_params2 = {}
    current_params2 = current_params.copy()
            
    #___________________________________________________________________________
    save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}.png"
    save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}.ipynb"
    current_params2["save_fname"] = os.path.join(yaml_settings['tripyrun_spath_fig'], save_fname)
            
    #___________________________________________________________________________
    pm.execute_notebook(
            f"{templates_nb_path}/template_transp_ghflx.ipynb",
            os.path.join(yaml_settings['tripyrun_spath_nb'], save_fname_nb),
            parameters=current_params2,
            nest_asyncio=True)
            
    #___________________________________________________________________________
    webpage[f"image_{image_count}"] = {}
    webpage[f"image_{image_count}"]["name"]       = f" GHFLX"
    webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
    webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
    webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['tripyrun_name']}_{analysis_name}"
    image_count += 1
    return webpage



#
#
#_______________________________________________________________________________
def drive_mhflx(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    print(f'         --> compute mhflx:')
    current_params2 = {}
    current_params2 = current_params.copy()
            
    #___________________________________________________________________________
    save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}.png"
    save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}.ipynb"
    current_params2["save_fname"] = os.path.join(yaml_settings['tripyrun_spath_fig'], save_fname)
            
    #___________________________________________________________________________
    pm.execute_notebook(
            f"{templates_nb_path}/template_transp_mhflx.ipynb",
            os.path.join(yaml_settings['tripyrun_spath_nb'], save_fname_nb),
            parameters=current_params2,
            nest_asyncio=True)
            
    #___________________________________________________________________________
    webpage[f"image_{image_count}"] = {}
    webpage[f"image_{image_count}"]["name"]       = f" MHFLX"
    webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
    webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
    webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['tripyrun_name']}_{analysis_name}"
    image_count += 1
    return webpage


#
#
#_______________________________________________________________________________
def drive_var_t(yaml_settings, analysis_name):
    # copy yaml settings for  analysis driver --> hslice: 
    #                                         
    driver_settings = yaml_settings[analysis_name].copy()
    
    # create current primary parameter from yaml settings
    current_params = {}
    for key, value in yaml_settings.items():
        # if value is a dictionary its not a primary paramter anymore e.g.
        # hslice: --> dict(...)
        #    temp:
        #        levels: [-2, 30, 41]
        #        depths: [0, 100, 400, 1000]
        # ....
        if isinstance(value, dict):
            pass
        else:
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    # loop over variable name  
    for vname in driver_settings:
        print(f'         --> compute: {vname}')
        auxvname = vname.replace('/',':')
        
        # loop over depths
        for box_region in driver_settings[vname]["box_regions"]:
            print(f'             --> compute: {box_region}')
            current_params2 = {}
            current_params2 = current_params.copy()
            current_params2["vname"] = vname
            current_params2["box_region"] = list([box_region])
            current_params2.update(driver_settings[vname])
            del current_params2["box_regions"] # --> delete depth list [0, 100, 1000,...] from current_param dict()
            str_boxregion = box_region.split('/')[-1].split('.')[0]
            
            #___________________________________________________________________
            save_fname    = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{str_boxregion}.png"
            save_fname_nb = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{str_boxregion}.ipynb"
            current_params2["save_fname"] = os.path.join(yaml_settings['tripyrun_spath_fig'], save_fname)
            
            #___________________________________________________________________
            pm.execute_notebook(
                f"{templates_nb_path}/template_var_t.ipynb",
                os.path.join(yaml_settings['tripyrun_spath_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True,
            )
            
            #___________________________________________________________________
            webpage[f"image_{image_count}"] = {}
            webpage[f"image_{image_count}"][
                "name"
            ] = f"{auxvname.capitalize()} at {str_boxregion} m"
            webpage[f"image_{image_count}"]["path"] = os.path.join('./figures/', save_fname)
            webpage[f"image_{image_count}"]["path_nb"] = os.path.join('./notebooks/', save_fname_nb)
            webpage[f"image_{image_count}"][
                "short_name"
            ] = f"{yaml_settings['tripyrun_name']}_{analysis_name}_{auxvname}_{str_boxregion}"
            image_count += 1
    return webpage
