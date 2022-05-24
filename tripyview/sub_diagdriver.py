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

#_________________________________________________________________________________________________            
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
#_______________________________________________________________________________________________________
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
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    # loop over variable name  
    for vname in driver_settings:
        
        # loop over depths
        for depth in driver_settings[vname]["depths"]:
            current_params2 = {}
            current_params2 = current_params.copy()
            current_params2["vname"] = vname
            current_params2["depth"] = depth
            current_params2.update(driver_settings[vname])
            del current_params2["depths"] # --> delete depth list [0, 100, 1000,...] from current_param dict()
            
            #__________________________________________________________________________________________
            if 'proj' in current_params2.keys(): 
                save_fname    = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{current_params2['proj']}_{depth}.png"
                save_fname_nb = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{current_params2['proj']}_{depth}.ipynb"
                short_name    = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{current_params2['proj']}_{depth}"
            else:
                save_fname    = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{depth}.png"
                save_fname_nb = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{depth}.ipynb"
                short_name    = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{depth}"
                
            current_params2["save_fname"] = os.path.join(yaml_settings['save_path_fig'], save_fname)
            
            #__________________________________________________________________________________________
            pm.execute_notebook(
                f"{templates_nb_path}/template_hslice.ipynb",
                os.path.join(yaml_settings['save_path_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True,
            )
            
            #__________________________________________________________________________________________
            webpage[f"image_{image_count}"] = {}
            webpage[f"image_{image_count}"]["name"]       = f"{vname.capitalize()} at {depth} m"
            webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
            webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
            webpage[f"image_{image_count}"]["short_name"] = short_name
            image_count += 1
    return webpage



#
#
#_______________________________________________________________________________________________________
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
            current_params[key] = value
    # initialse webpage for analyis 
    webpage = {}
    image_count = 0
    
    # loop over variable name  
    for vname in driver_settings:
        print(f'         --> compute:{vname}')
        # loop over depths
        for depth in driver_settings[vname]["depths"]:
            current_params2 = {}
            current_params2 = current_params.copy()
            current_params2["vname"] = vname
            current_params2["depth"] = depth
            current_params2.update(driver_settings[vname])
            del current_params2["depths"] # --> delete depth list [0, 100, 1000,...] from current_param dict()
            # print(current_params2)
            #__________________________________________________________________________________________
            if 'proj' in current_params2.keys(): 
                save_fname    = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{current_params2['proj']}_{depth}.png"
                save_fname_nb = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{current_params2['proj']}_{depth}.ipynb"
                short_name    = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{current_params2['proj']}_{depth}"
            else:
                save_fname    = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{depth}.png"
                save_fname_nb = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{depth}.ipynb"
                short_name    = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{depth}"
                
            current_params2["save_fname"] = os.path.join(yaml_settings['save_path_fig'], save_fname)
            #__________________________________________________________________________________________
            pm.execute_notebook(
                f"{templates_nb_path}/template_hslice_clim.ipynb",
                os.path.join(yaml_settings['save_path_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True,
            )
            
            #__________________________________________________________________________________________
            webpage[f"image_{image_count}"] = {}
            webpage[f"image_{image_count}"]["name"]       = f"{vname.capitalize()} at {depth} m"
            webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
            webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
            webpage[f"image_{image_count}"]["short_name"] = short_name
            image_count += 1
    return webpage



#
#
#_______________________________________________________________________________________________________
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
            
            #__________________________________________________________________________________________
            save_fname    = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{str_boxregion}.png"
            save_fname_nb = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{str_boxregion}.ipynb"
            current_params2["save_fname"] = os.path.join(yaml_settings['save_path_fig'], save_fname)
            
            #__________________________________________________________________________________________
            pm.execute_notebook(
                f"{templates_nb_path}/template_hovm.ipynb",
                os.path.join(yaml_settings['save_path_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True,
            )
            
            #__________________________________________________________________________________________
            webpage[f"image_{image_count}"] = {}
            webpage[f"image_{image_count}"][
                "name"
            ] = f"{vname.capitalize()} at {str_boxregion} m"
            webpage[f"image_{image_count}"]["path"] = os.path.join('./figures/', save_fname)
            webpage[f"image_{image_count}"]["path_nb"] = os.path.join('./notebooks/', save_fname_nb)
            webpage[f"image_{image_count}"][
                "short_name"
            ] = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{str_boxregion}"
            image_count += 1
    return webpage



#
#
#_______________________________________________________________________________________________________
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
            
            #__________________________________________________________________________________________
            save_fname    = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{str_boxregion}.png"
            save_fname_nb = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{str_boxregion}.ipynb"
            current_params2["save_fname"] = os.path.join(yaml_settings['save_path_fig'], save_fname)
            
            #__________________________________________________________________________________________
            pm.execute_notebook(
                f"{templates_nb_path}/template_hovm_clim.ipynb",
                os.path.join(yaml_settings['save_path_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True,
            )
            
            #__________________________________________________________________________________________
            webpage[f"image_{image_count}"] = {}
            webpage[f"image_{image_count}"][
                "name"
            ] = f"{vname.capitalize()} at {str_boxregion} m"
            webpage[f"image_{image_count}"]["path"] = os.path.join('./figures/', save_fname)
            webpage[f"image_{image_count}"]["path_nb"] = os.path.join('./notebooks/', save_fname_nb)
            webpage[f"image_{image_count}"][
                "short_name"
            ] = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{str_boxregion}"
            image_count += 1
    return webpage



#
#
#_______________________________________________________________________________________________________
def drive_xmoc(yaml_settings, analysis_name):
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
#         current_params2.update(driver_settings[vname])
            
        #__________________________________________________________________________________________
        save_fname    = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}.png"
        save_fname_nb = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}.ipynb"
        current_params2["save_fname"] = os.path.join(yaml_settings['save_path_fig'], save_fname)
            
        #__________________________________________________________________________________________
        pm.execute_notebook(
                f"{templates_nb_path}/template_xmoc.ipynb",
                os.path.join(yaml_settings['save_path_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True)
            
        #__________________________________________________________________________________________
        webpage[f"image_{image_count}"] = {}
        webpage[f"image_{image_count}"]["name"]       = f"{vname.upper()}"
        webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
        webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
        webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}"
        image_count += 1
    return webpage



#
#
#_______________________________________________________________________________________________________
def drive_xmoc_tseries(yaml_settings, analysis_name):
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
            
        #__________________________________________________________________________________________
        save_fname    = f"{yaml_settings['workflow_name']}_{analysis_name}_{str(which_lat)}.png"
        save_fname_nb = f"{yaml_settings['workflow_name']}_{analysis_name}_{str(which_lat)}.ipynb"
        current_params2["save_fname"] = os.path.join(yaml_settings['save_path_fig'], save_fname)
            
        #__________________________________________________________________________________________
        pm.execute_notebook(
                f"{templates_nb_path}/template_xmoc_tseries.ipynb",
                os.path.join(yaml_settings['save_path_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True)
            
        #__________________________________________________________________________________________
        webpage[f"image_{image_count}"] = {}
        if which_lat == 'max':
            webpage[f"image_{image_count}"]["name"]       = f" max AMOC @ 30°N<lat<45°N"
        else:
            webpage[f"image_{image_count}"]["name"]       = f" AMOC @ {str(which_lat)}°N"
        webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
        webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
        webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['workflow_name']}_{analysis_name}_{str(which_lat)}"
        image_count += 1
    return webpage



#
#
#_______________________________________________________________________________________________________
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
        
        #__________________________________________________________________________________________
        save_fname    = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}.png"
        save_fname_nb = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}.ipynb"
        current_params2["save_fname"] = os.path.join(yaml_settings['save_path_fig'], save_fname)
            
        #__________________________________________________________________________________________
        pm.execute_notebook(
                f"{templates_nb_path}/template_dmoc.ipynb",
                os.path.join(yaml_settings['save_path_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True)
            
        #__________________________________________________________________________________________
        webpage[f"image_{image_count}"] = {}
        webpage[f"image_{image_count}"]["name"]       = f"Density-{vname.upper()}{str_mode}"
        webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
        webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
        webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}"
        image_count += 1
    return webpage


#
#
#_______________________________________________________________________________________________________
def drive_dmoc_tseries(yaml_settings, analysis_name):
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
            
        #__________________________________________________________________________________________
        save_fname    = f"{yaml_settings['workflow_name']}_{analysis_name}_{str(which_lat)}.png"
        save_fname_nb = f"{yaml_settings['workflow_name']}_{analysis_name}_{str(which_lat)}.ipynb"
        current_params2["save_fname"] = os.path.join(yaml_settings['save_path_fig'], save_fname)
            
        #__________________________________________________________________________________________
        pm.execute_notebook(
                f"{templates_nb_path}/template_dmoc_tseries.ipynb",
                os.path.join(yaml_settings['save_path_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True)
            
        #__________________________________________________________________________________________
        webpage[f"image_{image_count}"] = {}
        if which_lat == 'max':
            webpage[f"image_{image_count}"]["name"]       = f" max density-AMOC @ 45°N<lat<60°N"
        else:
            webpage[f"image_{image_count}"]["name"]       = f" density AMOC @ {str(which_lat)}°N"
        webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
        webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
        webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['workflow_name']}_{analysis_name}_{str(which_lat)}"
        image_count += 1
    return webpage


#
#
#_______________________________________________________________________________________________________
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
            
        #__________________________________________________________________________________________
        if 'proj' in current_params2.keys(): 
            save_fname    = f"{yaml_settings['workflow_name']}_{analysis_name}_{current_params2['proj']}_{which_isopyc}.png"
            save_fname_nb = f"{yaml_settings['workflow_name']}_{analysis_name}_{current_params2['proj']}_{which_isopyc}.ipynb"
            short_name    = f"{yaml_settings['workflow_name']}_{analysis_name}_{current_params2['proj']}_{which_isopyc}"
        else:
            save_fname    = f"{yaml_settings['workflow_name']}_{analysis_name}_{which_isopyc}.png"
            save_fname_nb = f"{yaml_settings['workflow_name']}_{analysis_name}_{which_isopyc}.ipynb"
            short_name    = f"{yaml_settings['workflow_name']}_{analysis_name}_{which_isopyc}"
        current_params2["save_fname"] = os.path.join(yaml_settings['save_path_fig'], save_fname)
            
        #__________________________________________________________________________________________
        pm.execute_notebook(
            f"{templates_nb_path}/template_dmoc_wdiap.ipynb",
            os.path.join(yaml_settings['save_path_nb'], save_fname_nb),
            parameters=current_params2,
            nest_asyncio=True,
        )
            
        #__________________________________________________________________________________________
        webpage[f"image_{image_count}"] = {}
        webpage[f"image_{image_count}"]["name"]       = f"{'W_diap'} at sigma2={which_isopyc} kg/m^3"
        webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
        webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
        webpage[f"image_{image_count}"]["short_name"] = short_name
        image_count += 1
    return webpage


#
#
#_______________________________________________________________________________________________________
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
            
        #__________________________________________________________________________________________
        if 'proj' in current_params2.keys(): 
            save_fname    = f"{yaml_settings['workflow_name']}_{analysis_name}_{current_params2['proj']}_{which_isopyc}.png"
            save_fname_nb = f"{yaml_settings['workflow_name']}_{analysis_name}_{current_params2['proj']}_{which_isopyc}.ipynb"
            short_name    = f"{yaml_settings['workflow_name']}_{analysis_name}_{current_params2['proj']}_{which_isopyc}"
        else:
            save_fname    = f"{yaml_settings['workflow_name']}_{analysis_name}_{which_isopyc}.png"
            save_fname_nb = f"{yaml_settings['workflow_name']}_{analysis_name}_{which_isopyc}.ipynb"
            short_name    = f"{yaml_settings['workflow_name']}_{analysis_name}_{which_isopyc}"
        current_params2["save_fname"] = os.path.join(yaml_settings['save_path_fig'], save_fname)
        
        #__________________________________________________________________________________________
        pm.execute_notebook(
            f"{templates_nb_path}/template_dmoc_srfcbflx.ipynb",
            os.path.join(yaml_settings['save_path_nb'], save_fname_nb),
            parameters=current_params2,
            nest_asyncio=True,
        )
        
        #__________________________________________________________________________________________
        webpage[f"image_{image_count}"] = {}
        webpage[f"image_{image_count}"]["name"]       = f"{'srf. buoyancy transf.'} at sigma2={which_isopyc} kg/m^3"
        webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
        webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
        webpage[f"image_{image_count}"]["short_name"] = short_name
        image_count += 1
    return webpage



#
#
#_______________________________________________________________________________________________________
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
            
        #__________________________________________________________________________________________
        save_fname    = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}.png"
        save_fname_nb = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}.ipynb"
        current_params2["save_fname"] = os.path.join(yaml_settings['save_path_fig'], save_fname)
            
        #__________________________________________________________________________________________
        pm.execute_notebook(
                f"{templates_nb_path}/template_vprofile.ipynb",
                os.path.join(yaml_settings['save_path_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True)
            
        #__________________________________________________________________________________________
        webpage[f"image_{image_count}"] = {}
        webpage[f"image_{image_count}"]["name"]       = f"{vname.upper()}"
        webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
        webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
        webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}"
        image_count += 1
    return webpage



#
#
#_______________________________________________________________________________________________________
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
            
        #__________________________________________________________________________________________
        save_fname    = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}.png"
        save_fname_nb = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}.ipynb"
        current_params2["save_fname"] = os.path.join(yaml_settings['save_path_fig'], save_fname)
            
        #__________________________________________________________________________________________
        pm.execute_notebook(
                f"{templates_nb_path}/template_vprofile_clim.ipynb",
                os.path.join(yaml_settings['save_path_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True)
            
        #__________________________________________________________________________________________
        webpage[f"image_{image_count}"] = {}
        webpage[f"image_{image_count}"]["name"]       = f"{vname.capitalize()}"
        webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
        webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
        webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}"
        image_count += 1
    return webpage



#
#
#_______________________________________________________________________________________________________
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
    
    # loop over variable name  
    for vname in driver_settings:
        print(f'         --> compute: {vname}')
        for tname in driver_settings[vname]: 
            
            current_params2 = {}
            current_params2 = current_params.copy()
            current_params2["vname"] = vname
            current_params2.update(driver_settings[vname][tname])

            #__________________________________________________________________________________________
            save_fname    = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{tname}.png"
            save_fname_nb = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{tname}.ipynb"
            current_params2["save_fname"] = os.path.join(yaml_settings['save_path_fig'], save_fname)

            #__________________________________________________________________________________________
            pm.execute_notebook(
                    f"{templates_nb_path}/template_transect.ipynb",
                    os.path.join(yaml_settings['save_path_nb'], save_fname_nb),
                    parameters=current_params2,
                    nest_asyncio=True)

            #__________________________________________________________________________________________
            webpage[f"image_{image_count}"] = {}
            webpage[f"image_{image_count}"]["name"]       = f"{vname.upper()} @ {tname}"
            webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
            webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
            webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{tname}"
            image_count += 1
    return webpage



#
#
#_______________________________________________________________________________________________________
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
    
    # loop over variable name  
    for vname in driver_settings:
        print(f'         --> compute: {vname}')
        for tname in driver_settings[vname]: 
            
            current_params2 = {}
            current_params2 = current_params.copy()
            current_params2["vname"] = vname
            current_params2.update(driver_settings[vname][tname])

            #__________________________________________________________________________________________
            save_fname    = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{tname}.png"
            save_fname_nb = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{tname}.ipynb"
            current_params2["save_fname"] = os.path.join(yaml_settings['save_path_fig'], save_fname)

            #__________________________________________________________________________________________
            pm.execute_notebook(
                    f"{templates_nb_path}/template_transect_clim.ipynb",
                    os.path.join(yaml_settings['save_path_nb'], save_fname_nb),
                    parameters=current_params2,
                    nest_asyncio=True)

            #__________________________________________________________________________________________
            webpage[f"image_{image_count}"] = {}
            webpage[f"image_{image_count}"]["name"]       = f"{vname.capitalize()} @ {tname} m"
            webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
            webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
            webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{tname}"
            image_count += 1
    return webpage



#
#
#_______________________________________________________________________________________________________
def drive_transect_vflx_t(yaml_settings, analysis_name):
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
    for tname in driver_settings:
        print(f'         --> compute: {tname}')
        current_params2 = {}
        current_params2 = current_params.copy()
        current_params2.update(driver_settings[tname])

        #__________________________________________________________________________________________
        save_fname    = f"{yaml_settings['workflow_name']}_{analysis_name}_{tname}.png"
        save_fname_nb = f"{yaml_settings['workflow_name']}_{analysis_name}_{tname}.ipynb"
        current_params2["save_fname"] = os.path.join(yaml_settings['save_path_fig'], save_fname)

        #__________________________________________________________________________________________
        pm.execute_notebook(
                f"{templates_nb_path}/template_transect_vflx_t.ipynb",
                os.path.join(yaml_settings['save_path_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True)

        #__________________________________________________________________________________________
        webpage[f"image_{image_count}"] = {}
        webpage[f"image_{image_count}"]["name"]       = f" @ {tname}"
        webpage[f"image_{image_count}"]["path"]       = os.path.join('./figures/', save_fname)
        webpage[f"image_{image_count}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
        webpage[f"image_{image_count}"]["short_name"] = f"{yaml_settings['workflow_name']}_{analysis_name}_{tname}"
        image_count += 1
    return webpage



#
#
#_______________________________________________________________________________________________________
def drive_zmeantrans(yaml_settings, analysis_name):
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
            
            #__________________________________________________________________________________________
            save_fname    = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{str_boxregion}.png"
            save_fname_nb = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{str_boxregion}.ipynb"
            current_params2["save_fname"] = os.path.join(yaml_settings['save_path_fig'], save_fname)
            
            #__________________________________________________________________________________________
            pm.execute_notebook(
                f"{templates_nb_path}/template_zmeantransect.ipynb",
                os.path.join(yaml_settings['save_path_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True,
            )
            
            #__________________________________________________________________________________________
            webpage[f"image_{image_count}"] = {}
            webpage[f"image_{image_count}"][
                "name"
            ] = f"{vname.capitalize()} at {str_boxregion} m"
            webpage[f"image_{image_count}"]["path"] = os.path.join('./figures/', save_fname)
            webpage[f"image_{image_count}"]["path_nb"] = os.path.join('./notebooks/', save_fname_nb)
            webpage[f"image_{image_count}"][
                "short_name"
            ] = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{str_boxregion}"
            image_count += 1
    return webpage



#
#
#_______________________________________________________________________________________________________
def drive_zmeantrans_clim(yaml_settings, analysis_name):
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
            
            #__________________________________________________________________________________________
            save_fname    = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{str_boxregion}.png"
            save_fname_nb = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{str_boxregion}.ipynb"
            current_params2["save_fname"] = os.path.join(yaml_settings['save_path_fig'], save_fname)
            
            #__________________________________________________________________________________________
            pm.execute_notebook(
                f"{templates_nb_path}/template_zmeantransect_clim.ipynb",
                os.path.join(yaml_settings['save_path_nb'], save_fname_nb),
                parameters=current_params2,
                nest_asyncio=True,
            )
            
            #__________________________________________________________________________________________
            webpage[f"image_{image_count}"] = {}
            webpage[f"image_{image_count}"][
                "name"
            ] = f"{vname.capitalize()} at {str_boxregion} m"
            webpage[f"image_{image_count}"]["path"] = os.path.join('./figures/', save_fname)
            webpage[f"image_{image_count}"]["path_nb"] = os.path.join('./notebooks/', save_fname_nb)
            webpage[f"image_{image_count}"][
                "short_name"
            ] = f"{yaml_settings['workflow_name']}_{analysis_name}_{vname}_{str_boxregion}"
            image_count += 1
    return webpage

