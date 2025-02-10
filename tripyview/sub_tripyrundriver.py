import papermill as pm
from jinja2 import Environment, FileSystemLoader
import sys
import os
import warnings

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
def exec_papermill(webpage, cnt, params_vname, exec_template='hslice'):
    #___________________________________________________________________________
    # create strings for fname and labels in the webpage
    str_vname1, str_vname2, str_dep1, str_dep2, str_mon1, str_mon2, str_proj1, str_proj2 = '', '', '', '', '', '', '',''
    if 'vname' in params_vname:
        str_vname1, str_vname2 = f"_{params_vname['vname'].replace('/',':')}", f"{params_vname['vname'].replace('/',':')}"
    if 'proj' in params_vname:
        str_proj1, str_proj2 = f"_{params_vname['proj']}",  f", proj:{params_vname['proj']}" 
    if 'depth' in params_vname: 
        str_dep1, str_dep2 = f"_z{params_vname['depth']}", f", @dep:{params_vname['depth']}"
    if 'mon'   in params_vname: 
        str_mon1, str_mon2 = f"_m{params_vname['mon']}"  , f", @mon:{params_vname['mon']}"
    
    #___________________________________________________________________________
    # assemble total strings
    str_all1 = f"{params_vname['tripyrun_name']}_{params_vname['tripyrun_analysis']}{str_vname1}"
    if exec_template in ['hslice', 'hslice_clim', 'hquiver']:
        str_all1 = f"{str_all1}{str_proj1}{str_dep1}{str_mon1}"
        str_all2 = f"{str_vname2}{str_proj2}{str_dep2}{str_mon2}"
        
    if exec_template in ['hmesh']:
        str_all1 = f"{str_all1}{str_proj1}"
        str_all2 = f"{str_vname2}{str_proj2}"    
        
    elif exec_template in ['hslice_isotdep']:
        str_isotd1, str_isotd2 = '', ''
        if 'which_isotherm' in params_vname: str_isotd1, str_isotd2 = f"_{params_vname['which_isotherm']}", f"depth of {params_vname['which_isotherm']}C isotherm"
        str_all1 = f"{str_all1}{str_proj1}{str_isotd1}{str_dep1}{str_mon1}"
        str_all2 = f"{str_isotd2}{str_proj2}{str_mon2}"
        
    elif exec_template in ['hovm', 'hovm_clim', 'transect_zmean', 'transect_zmean_clim', 'var_t']:
        str_box1, str_box2 = '', ''
        if 'box_region' in params_vname: 
            auxboxr = params_vname['box_region'][0].split('/')[-1].split('.')[0]            
            str_box1, str_box2 = f"_{auxboxr}", f"@ {auxboxr}"
        str_all1 = f"{str_all1}{str_box1}"
        str_all2 = f"{str_vname2}{str_box2}"
        
    elif exec_template in ['vprofile', 'vprofile_clim', 'transp_zmoc', 'transp_zmoc_t']:
        if exec_template in ['transp_zmoc_t']: 
            str_all1 = f"{str_all1}{params_vname['which_lat']}"
            if params_vname['which_lat'] == 'max': str_all2 = f" max AMOC @ 30°N<lat<50°N"
            else                                 : str_all2 = f" AMOC @ {params_vname['which_lat']}°N"
        else:
            str_all1 = f"{str_all1}"
            str_all2 = f"{str_vname2}"
        
    elif exec_template in ['transect', 'transect_clim', 'transect_transp', 'transect_transp_t']:
        str_tra1, str_tra2 = '', ''
        if 'input_transect' in params_vname: 
            auxtra = params_vname['input_transect'][0][2].replace(' ','_').replace(',','').replace('°','')
            str_tra1, str_tra2 = f"_{auxtra}", f"@ {params_vname['input_transect'][0][2]}"
        str_all1 = f"{str_all1}{str_tra1}"
        str_all2 = f"{str_vname2}{str_tra2}"
    
    elif exec_template in ['transp_dmoc', 'transp_dmoc_t']:
        str_moc1=''
        if params_vname["which_transf"] != 'dmoc': str_moc1 = f"_{params_vname['which_transf']}"
        if params_vname["do_zcoord"]             : str_moc1 = f"{str_moc1}z"
        if exec_template in ['transp_dmoc_t']: 
            str_all1 = f"{str_all1}{str_moc1}{params_vname['which_lat']}" 
            if params_vname['which_lat'] == 'max': str_all2 = f" max Density-AMOC @ 40°N<lat<60°N"
            else                                 : str_all2 = f" Density-AMOC @ {params_vname['which_lat']}°N"
        else:
            str_all1 = f"{str_all1}{str_moc1}"
            str_all2 = f"Density-{str_vname2}{str_moc1}"
    
    elif exec_template in ['transp_dmoc_wdiap', 'transp_dmoc_srfcbflx']:
        str_isop1, str_isop2 = '', ''        
        if exec_template in ['transp_dmoc_wdiap']: 
            if 'which_isopyc' in params_vname: str_isop1, str_isop2 = f"_{params_vname['which_isopyc']}", f"W_diap @ sigma2= {params_vname['which_isopyc']} kg/m^3"
        elif exec_template in ['transp_dmoc_srfcbflx']:
            if 'which_isopyc' in params_vname: str_isop1, str_isop2 = f"_{params_vname['which_isopyc']}", f"Srf. buoyancy transf. @ sigma2= {params_vname['which_isopyc']} kg/m^3"
        str_all1 = f"{str_all1}{str_proj1}{str_isop1}{str_mon1}"
        str_all2 = f"{str_isop2}{str_proj1}{str_mon2}"
    
    elif exec_template in ['transp_hbstreamf']:
        str_all1 = f"{str_all1}{str_mon1}"
        str_all2 = f"Horiz. barotrop. streamfunction {str_mon2}"
    
    elif exec_template in ['transp_ghflx', 'transp_mhflx']:
        str_all1 = f"{str_all1}{str_mon1}"
        if   exec_template in ['transp_ghflx']:  str_all2 = f"Global. Merid. Heatflx. {str_mon2}"
        elif exec_template in ['transp_mhflx']:  str_all2 = f"Dyn. Merid. Heatflx. {str_mon2}"
    
    #___________________________________________________________________________
    # create filepaths for notebook and figures 
    save_fname    = f"{str_all1}.png"
    save_fname_nb = f"{str_all1}.ipynb"
    short_name    = f"{str_all1}"
    params_vname["save_fname"] = os.path.join(params_vname['tripyrun_spath_fig'], save_fname)
    
    #___________________________________________________________________________
    # execute notebook with papermill
    pm.execute_notebook(f"{templates_nb_path}/template_{exec_template}.ipynb",
                        os.path.join(params_vname['tripyrun_spath_nb'], save_fname_nb),
                        parameters=params_vname,
                        nest_asyncio=True,)
                
    #___________________________________________________________________________
    # attach created figures to webpage collection
    webpage[f"image_{cnt}"] = {}
    if 'vname' in params_vname: webpage[f"image_{cnt}"]["variable"] = params_vname['vname']
    else                      : webpage[f"image_{cnt}"]["variable"] = ''
    webpage[f"image_{cnt}"]["cnt"]        = cnt
    webpage[f"image_{cnt}"]["name"]       = str_all2
    webpage[f"image_{cnt}"]["path"]       = os.path.join('./figures/', save_fname)
    webpage[f"image_{cnt}"]["path_nb"]    = os.path.join('./notebooks/', save_fname_nb)
    webpage[f"image_{cnt}"]["short_name"] = short_name
    cnt += 1
    
    #___________________________________________________________________________
    return(webpage, cnt)



#
#
#_______________________________________________________________________________
def extract_params(yaml_settings):
    # create current parameter level from yaml settings, these are either the parameter 
    # that are setted at the very beginning of the yaml file (1st level) or the 
    # parameterssetted at the beginning of the driver section (2nd level) or the 
    # parameters that are setted in each variable section. These parameters 
    # can than be overwritten param_1lvl --> param_2lvl --> param_3lvl
    params_current = dict()
    if isinstance(yaml_settings,dict):
        for key, value in yaml_settings.items():
            # if value is a dictionary its not a primary paramter anymore e.g.
            # hslice: --> dict(...)
            #    temp:
            #        levels: [-2, 30, 41]
            #        depths: [0, 100, 400, 1000]
            # ....
            if isinstance(value, dict) or value is None:
                pass
            else:
                if value=="None": value=None 
                params_current[key] = value
    else:
        pass
    return(params_current)

  

#
#
#_______________________________________________________________________________
# define subroutine for loop over box_regions list 
def loop_over_param(webpage, image_count, params_vname, target='box_region', source_loop='box_regions', source_single=None, exec_template='hovm'):
    #___________________________________________________________________________
    # make loop over list of variables
    if source_loop != None and source_loop in params_vname:
        var_loop = params_vname[source_loop]
        del params_vname[source_loop]
        if var_loop is not None:
            for var in var_loop:
                if   isinstance(var,list) : params_vname[target] = [var]
                elif isinstance(var,str)  : params_vname[target] = [var]
                else                      : params_vname[target] =  var  
                
                #___________________________________________________________________
                if isinstance(params_vname[target],list) and len(params_vname[target][0])==3:
                    print(f"          --> compute {target}: {params_vname[target][0][2]}")
                else:
                    print(f"          --> compute {target}: {params_vname[target]}")
                    
                #___________________________________________________________________
                webpage, image_count = exec_papermill(webpage, image_count, params_vname, exec_template=exec_template)
                
        else:
            var_single = params_vname[source_single]
            del params_vname[source_single]
            params_vname[target] = var_single
            webpage, image_count = exec_papermill(webpage, image_count, params_vname, exec_template=exec_template)
            
    #___________________________________________________________________________
    # only single var is defined or use list as single input 
    elif source_single != None and source_single in params_vname:    
        var_single = params_vname[source_single]
        del params_vname[source_single]
        params_vname[target] = var_single
        webpage, image_count = exec_papermill(webpage, image_count, params_vname, exec_template=exec_template)
        
    #___________________________________________________________________________
    # no variables defined or found use the one defined in the notebook 
    else:
        #warnings.warn(' -WARNING-> box_regions is not defined, use the default on defined in the notebook')
        webpage, image_count = exec_papermill(webpage, image_count, params_vname, exec_template=exec_template)
        
    return(webpage, image_count)   



#
#
#_______________________________________________________________________________
def drive_hslice(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # kickout secondary parameters from yaml_settings[analysis_name] only leave 
    # the dictionaries of the variables
    driver_vars = yaml_settings[analysis_name].copy()
    for key in params_2lvl.keys():
        del driver_vars[key]
    
    # execute only specfic driver with a specific variable 
    if vname != None: 
        if vname in driver_vars: driver_vars = dict({vname: driver_vars[vname]})
    
    #___________________________________________________________________________
    # loop over variable name  
    for vname in driver_vars:
        print(f'     --> compute: {vname}')
        params_3lvl  = extract_params(driver_vars[vname])
        params_vname = dict({'tripyrun_analysis':analysis_name})
        params_vname.update(params_1lvl)
        params_vname.update(params_2lvl)
        params_vname.update(params_3lvl)
        params_vname["vname"] = vname
        
        #_______________________________________________________________________
        # make loop over depths
        if 'depths' in params_vname:
            depths = params_vname["depths"]
            del params_vname["depths"] # --> delete depth list [0, 100, 1000,...] from current_param dict()
            if depths is not None:
                for depth in depths:
                    print(f'          --> depth: {depth}')
                    params_vname["depth"] = depth
                    # make loops over the months or not 
                    webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='mon', 
                                                        source_loop='months', exec_template='hslice')
            else:
                webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='mon', 
                                                   source_loop='months', exec_template='hslice')
        #_______________________________________________________________________
        # only single boxregion defined 
        elif 'depth' in params_vname:    
            # make loops over the months or not 
            webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='mon', 
                                                   source_loop='months', exec_template='hslice')
            
        #_______________________________________________________________________    
        # no depth defined use the one defined in the notebook 
        else:
            #warnings.warn(' -WARNING-> depths is not defined, use the default on defined in the notebook')
            webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='mon', 
                                                   source_loop='months', exec_template='hslice')
    return webpage



#
#
#_______________________________________________________________________________
def drive_hslice_clim(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # kickout secondary parameters from yaml_settings[analysis_name] only leave 
    # the dictionaries of the variables
    driver_vars = yaml_settings[analysis_name].copy()
    for key in params_2lvl.keys():
        del driver_vars[key]
    
    # execute only specfic driver with a specific variable 
    if vname != None: 
        if vname in driver_vars: driver_vars = dict({vname: driver_vars[vname]})
    
    #___________________________________________________________________________
    # loop over variable name  
    for vname in driver_vars:
        print(f'     --> compute: {vname}')
        params_3lvl  = extract_params(driver_vars[vname])
        params_vname = dict({'tripyrun_analysis':analysis_name})
        params_vname.update(params_1lvl)
        params_vname.update(params_2lvl)
        params_vname.update(params_3lvl)
        params_vname["vname"] = vname
        
        #_______________________________________________________________________
        # make loop over depths
        webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='depth', 
                                               source_loop='depths', exec_template='hslice_clim')
    return webpage



#
#
#_______________________________________________________________________________
def drive_hslice_isotdep(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # kickout secondary parameters from yaml_settings[analysis_name] only leave 
    # the dictionaries of the variables
    driver_vars = yaml_settings[analysis_name].copy()
    for key in params_2lvl.keys():
        del driver_vars[key]
    
    # execute only specfic driver with a specific variable 
    if vname != None: 
        if vname in driver_vars: driver_vars = dict({vname: driver_vars[vname]})
    
    #___________________________________________________________________________
    # loop over variable name  
    for vname in driver_vars:
        print(f'     --> compute: {vname}')
        params_3lvl  = extract_params(driver_vars[vname])
        params_vname = dict({'tripyrun_analysis':analysis_name})
        params_vname.update(params_1lvl)
        params_vname.update(params_2lvl)
        params_vname.update(params_3lvl)
        params_vname["vname"] = vname
        
        #_______________________________________________________________________
        # make loop over which_isotherms
        webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='which_isotherm', 
                                               source_loop='which_isotherms', exec_template='hslice_isotdep')
    return webpage



#
#
#_______________________________________________________________________________
def drive_hmesh(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # kickout secondary parameters from yaml_settings[analysis_name] only leave 
    # the dictionaries of the variables
    driver_vars = yaml_settings[analysis_name].copy()
    for key in params_2lvl.keys():
        del driver_vars[key]
    
    # execute only specfic driver with a specific variable 
    if vname != None: 
        if vname in driver_vars: driver_vars = dict({vname: driver_vars[vname]})
    
    #___________________________________________________________________________
    # loop over variable name  
    for vname in driver_vars:
        print(f'     --> compute: {vname}')
        params_3lvl  = extract_params(driver_vars[vname])
        params_vname = dict({'tripyrun_analysis':analysis_name})
        params_vname.update(params_1lvl)
        params_vname.update(params_2lvl)
        params_vname.update(params_3lvl)
        params_vname["vname"] = vname
        
        #_______________________________________________________________________
        # make loop over which_isotherms
        webpage, image_count = exec_papermill(webpage, image_count, params_vname, exec_template='hmesh')
    return webpage



#
#
#_______________________________________________________________________________
def drive_hquiver(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # kickout secondary parameters from yaml_settings[analysis_name] only leave 
    # the dictionaries of the variables
    driver_vars = yaml_settings[analysis_name].copy()
    for key in params_2lvl.keys():
        del driver_vars[key]
    
    # execute only specfic driver with a specific variable 
    if vname != None: 
        if vname in driver_vars: driver_vars = dict({vname: driver_vars[vname]})
    
    #___________________________________________________________________________
    # loop over variable name  
    for vname in driver_vars:
        print(f'     --> compute: {vname}')
        params_3lvl  = extract_params(driver_vars[vname])
        params_vname = dict({'tripyrun_analysis':analysis_name})
        params_vname.update(params_1lvl)
        params_vname.update(params_2lvl)
        params_vname.update(params_3lvl)
        params_vname["vname"] = vname
        
        #_______________________________________________________________________
        # make loop over depths
        if 'depths' in params_vname:
            depths = params_vname["depths"]
            del params_vname["depths"] # --> delete depth list [0, 100, 1000,...] from current_param dict()
            if depths is not None:
                for depth in depths:
                    print(f'          --> depth: {depth}')
                    params_vname["depth"] = depth
                    # make loops over the months or not 
                    webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='mon', 
                                                        source_loop='months', exec_template='hquiver')
            else:
                webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='mon', 
                                                   source_loop='months', exec_template='hquiver')
        #_______________________________________________________________________
        # only single boxregion defined 
        elif 'depth' in params_vname:    
            # make loops over the months or not 
            webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='mon', 
                                                   source_loop='months', exec_template='hquiver')
            
        #_______________________________________________________________________    
        # no depth defined use the one defined in the notebook 
        else:
            #warnings.warn(' -WARNING-> depths is not defined, use the default on defined in the notebook')
            webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='mon', 
                                                   source_loop='months', exec_template='hslice')
    return webpage



#
#
#_______________________________________________________________________________
def drive_hovm(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # kickout secondary parameters from yaml_settings[analysis_name] only leave 
    # the dictionaries of the variables
    driver_vars = yaml_settings[analysis_name].copy()
    for key in params_2lvl.keys():
        del driver_vars[key]
    
    # execute only specfic driver with a specific variable 
    if vname != None: 
        if vname in driver_vars: driver_vars = dict({vname: driver_vars[vname]})
    
    #___________________________________________________________________________
    # loop over variable name  
    for vname in driver_vars:
        print(f'     --> compute: {vname}')
        params_3lvl  = extract_params(driver_vars[vname])
        params_vname = dict({'tripyrun_analysis':analysis_name})
        params_vname.update(params_1lvl)
        params_vname.update(params_2lvl)
        params_vname.update(params_3lvl)
        params_vname["vname"] = vname
        
        #_______________________________________________________________________
        # make loop over box_regions
        webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='box_region', 
                                               source_loop='box_regions', exec_template='hovm')
    return webpage



#
#
#_______________________________________________________________________________
def drive_hovm_clim(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # kickout secondary parameters from yaml_settings[analysis_name] only leave 
    # the dictionaries of the variables
    driver_vars = yaml_settings[analysis_name].copy()
    for key in params_2lvl.keys():
        del driver_vars[key]
    
    # execute only specfic driver with a specific variable 
    if vname != None: 
        if vname in driver_vars: driver_vars = dict({vname: driver_vars[vname]})
    
    #___________________________________________________________________________
    # loop over variable name  
    for vname in driver_vars:
        print(f'     --> compute: {vname}')
        params_3lvl  = extract_params(driver_vars[vname])
        params_vname = dict({'tripyrun_analysis':analysis_name})
        params_vname.update(params_1lvl)
        params_vname.update(params_2lvl)
        params_vname.update(params_3lvl)
        params_vname["vname"] = vname
        
        #_______________________________________________________________________
        # make loop over box_regions
        webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='box_region', 
                                               source_loop='box_regions', exec_template='hovm_clim')
    return webpage



#
#
#_______________________________________________________________________________
def drive_zmoc(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # kickout secondary parameters from yaml_settings[analysis_name] only leave 
    # the dictionaries of the variables
    driver_vars = yaml_settings[analysis_name].copy()
    for key in params_2lvl.keys():
        del driver_vars[key]
    
    # execute only specfic driver with a specific variable 
    if vname != None: 
        if vname in driver_vars: driver_vars = dict({vname: driver_vars[vname]})
    
    #___________________________________________________________________________
    # loop over variable name  
    for vname in driver_vars:
        print(f'         --> compute: {vname}')
        params_3lvl  = extract_params(driver_vars[vname])
        params_vname = dict({'tripyrun_analysis':analysis_name})
        params_vname.update(params_1lvl)
        params_vname.update(params_2lvl)
        params_vname.update(params_3lvl)
        params_vname["vname"] = vname
        webpage, image_count = exec_papermill(webpage, image_count, params_vname, exec_template='transp_zmoc')
    return webpage



#
#
#_______________________________________________________________________________
def drive_zmoc_t(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # kickout secondary parameters from yaml_settings[analysis_name] only leave 
    # the dictionaries of the variables
    driver_vars = yaml_settings[analysis_name].copy()
    for key in params_2lvl.keys():
        del driver_vars[key]
    
    # execute only specfic driver with a specific variable 
    if vname != None: 
        if vname in driver_vars: driver_vars = dict({vname: driver_vars[vname]})
    
    #___________________________________________________________________________
    # loop over variable name  
    for vname in driver_vars:
        print(f'         --> compute: {vname}')
        params_3lvl  = extract_params(driver_vars[vname])
        params_vname = dict({'tripyrun_analysis':analysis_name})
        params_vname.update(params_1lvl)
        params_vname.update(params_2lvl)
        params_vname.update(params_3lvl)
        params_vname["vname"] = vname
        
        #_______________________________________________________________________
        # make loop over which_lats
        webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='which_lat', 
                                           source_loop='which_lats', source_single=None, exec_template='transp_zmoc_t')
    return webpage



#
#
#_______________________________________________________________________________
def drive_dmoc(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # kickout secondary parameters from yaml_settings[analysis_name] only leave 
    # the dictionaries of the variables
    driver_vars = yaml_settings[analysis_name].copy()
    for key in params_2lvl.keys():
        del driver_vars[key]
    
    # execute only specfic driver with a specific variable 
    if vname != None: 
        if vname in driver_vars: driver_vars = dict({vname: driver_vars[vname]})
    
    #___________________________________________________________________________
    # loop over variable name  
    for vname in driver_vars:
        params_3lvl  = extract_params(driver_vars[vname])
        params_vname = dict({'tripyrun_analysis':analysis_name})
        params_vname.update(params_1lvl)
        params_vname.update(params_2lvl)
        params_vname.update(params_3lvl)
        params_vname["vname"] = vname
        params_vname["do_zcoord"] = False
        
        #_______________________________________________________________________
        if   analysis_name == 'dmoc'        : params_vname["which_transf"] = 'dmoc' 
        elif analysis_name == 'dmoc_srf'    : params_vname["which_transf"] = 'srf'  
        elif analysis_name == 'dmoc_inner'  : params_vname["which_transf"] = 'inner'
        elif analysis_name == 'dmoc_z'      : params_vname["which_transf"], params_vname["do_zcoord"] = 'dmoc' , True
        elif analysis_name == 'dmoc_srf_z'  : params_vname["which_transf"], params_vname["do_zcoord"] = 'srf'  , True
        elif analysis_name == 'dmoc_inner_z': params_vname["which_transf"], params_vname["do_zcoord"] = 'inner', True
        
        #_______________________________________________________________________
        webpage, image_count = exec_papermill(webpage, image_count, params_vname, exec_template='transp_dmoc')
    return webpage



#
#
#_______________________________________________________________________________
def drive_dmoc_t(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # kickout secondary parameters from yaml_settings[analysis_name] only leave 
    # the dictionaries of the variables
    driver_vars = yaml_settings[analysis_name].copy()
    for key in params_2lvl.keys():
        del driver_vars[key]
    
    # execute only specfic driver with a specific variable 
    if vname != None: 
        if vname in driver_vars: driver_vars = dict({vname: driver_vars[vname]})
    
    #___________________________________________________________________________
    # loop over variable name  
    for vname in driver_vars:
        print(f'         --> compute: {vname}')
        params_3lvl  = extract_params(driver_vars[vname])
        params_vname = dict({'tripyrun_analysis':analysis_name})
        params_vname.update(params_1lvl)
        params_vname.update(params_2lvl)
        params_vname.update(params_3lvl)
        params_vname["vname"]        = vname
        params_vname["which_transf"] = 'dmoc'
        params_vname["do_zcoord"]    = False
        
        #_______________________________________________________________________
        # make loop over which_lats
        webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='which_lat', 
                                           source_loop='which_lats', source_single=None, exec_template='transp_dmoc_t')
    return webpage



#
#
#_______________________________________________________________________________
def drive_dmoc_wdiap(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # make no loop over isopycnals
    params_vname = dict({'tripyrun_analysis':analysis_name})
    params_vname.update(params_1lvl)
    params_vname.update(params_2lvl)
    params_vname["vname"] = 'wdiap_isopyc'
    
    #___________________________________________________________________________
    # make loop over which_isopycs
    webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='which_isopyc', 
                                           source_loop='which_isopycs', source_single=None, exec_template='transp_dmoc_wdiap')
    return webpage



#
#
#_______________________________________________________________________________
def drive_dmoc_srfcbflx(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # make no loop over isopycnals
    params_vname = dict({'tripyrun_analysis':analysis_name})
    params_vname.update(params_1lvl)
    params_vname.update(params_2lvl)
    params_vname["vname"] = 'srfbflx_isopyc'
    
    #___________________________________________________________________________
    # make loop over which_isopycs
    webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='which_isopyc', 
                                           source_loop='which_isopycs', source_single=None, exec_template='transp_dmoc_srfcbflx')
    return webpage



#
#
#_______________________________________________________________________________
def drive_hbarstreamf(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
     #__________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # initialse webpage for analyis
    params_vname = dict({'tripyrun_analysis':analysis_name})
    params_vname.update(params_1lvl)
    params_vname.update(params_2lvl)
    params_vname["vname"] = 'hbstreamf'
    webpage, image_count = exec_papermill(webpage, image_count, params_vname, exec_template='transp_hbstreamf')
    return webpage



#
#
#_______________________________________________________________________________
def drive_vprofile(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # kickout secondary parameters from yaml_settings[analysis_name] only leave 
    # the dictionaries of the variables
    driver_vars = yaml_settings[analysis_name].copy()
    for key in params_2lvl.keys():
        del driver_vars[key]
    
    # execute only specfic driver with a specific variable 
    if vname != None: 
        if vname in driver_vars: driver_vars = dict({vname: driver_vars[vname]})
    
    #___________________________________________________________________________
    # loop over variable name  
    for vname in driver_vars:
        print(f'     --> compute: {vname}')
        params_3lvl  = extract_params(driver_vars[vname])
        params_vname = dict({'tripyrun_analysis':analysis_name})
        params_vname.update(params_1lvl)
        params_vname.update(params_2lvl)
        params_vname.update(params_3lvl)
        params_vname["vname"] = vname   
        
        #_______________________________________________________________________
        # make no loop over box_regions
        webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='box_region', 
                                               source_loop=None, source_single='box_regions', exec_template='vprofile')
    return webpage



#
#
#_______________________________________________________________________________
def drive_vprofile_clim(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # kickout secondary parameters from yaml_settings[analysis_name] only leave 
    # the dictionaries of the variables
    driver_vars = yaml_settings[analysis_name].copy()
    for key in params_2lvl.keys():
        del driver_vars[key]
    
    # execute only specfic driver with a specific variable 
    if vname != None: 
        if vname in driver_vars: driver_vars = dict({vname: driver_vars[vname]})
    
    #___________________________________________________________________________
    # loop over variable name  
    for vname in driver_vars:
        print(f'     --> compute: {vname}')
        params_3lvl  = extract_params(driver_vars[vname])
        params_vname = dict({'tripyrun_analysis':analysis_name})
        params_vname.update(params_1lvl)
        params_vname.update(params_2lvl)
        params_vname.update(params_3lvl)
        params_vname["vname"] = vname   
        
        #_______________________________________________________________________
        # make no loop over box_regions
        webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='box_region', 
                                               source_loop=None, source_single='box_regions', exec_template='vprofile_clim')
    return webpage



#
#
#_______________________________________________________________________________
def drive_transect(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # kickout secondary parameters from yaml_settings[analysis_name] only leave 
    # the dictionaries of the variables
    driver_vars = yaml_settings[analysis_name].copy()
    for key in params_2lvl.keys():
        del driver_vars[key]
    
    # execute only specfic driver with a specific variable 
    if vname != None: 
        if vname in driver_vars: driver_vars = dict({vname: driver_vars[vname]})
    
    #___________________________________________________________________________
    # loop over variable name  
    for vname in driver_vars:
        print(f'     --> compute: {vname}')
        params_3lvl  = extract_params(driver_vars[vname])
        params_vname = dict({'tripyrun_analysis':analysis_name})
        params_vname.update(params_1lvl)
        params_vname.update(params_2lvl)
        params_vname.update(params_3lvl)
        params_vname["vname"] = vname
        
        #_______________________________________________________________________
        # make loop over transects
        webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='input_transect', 
                                               source_loop="transects", source_single='transect', exec_template='transect')
    return webpage



#
#
#_______________________________________________________________________________
def drive_transect_clim(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # kickout secondary parameters from yaml_settings[analysis_name] only leave 
    # the dictionaries of the variables
    driver_vars = yaml_settings[analysis_name].copy()
    for key in params_2lvl.keys():
        del driver_vars[key]
    
    # execute only specfic driver with a specific variable 
    if vname != None: 
        if vname in driver_vars: driver_vars = dict({vname: driver_vars[vname]})
    
    #___________________________________________________________________________
    # loop over variable name  
    for vname in driver_vars:
        print(f'     --> compute: {vname}')
        params_3lvl  = extract_params(driver_vars[vname])
        params_vname = dict({'tripyrun_analysis':analysis_name})
        params_vname.update(params_1lvl)
        params_vname.update(params_2lvl)
        params_vname.update(params_3lvl)
        params_vname["vname"] = vname
        
        #_______________________________________________________________________
        # make loop over transects
        webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='input_transect', 
                                               source_loop="transects", source_single='transect', exec_template='transect_clim')
    return webpage



#
#
#_______________________________________________________________________________
def drive_transect_Xtransp(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    # analysis_name: 
    #  -'transect_transp_t'
    #  -'transect_hflx_t'
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # loop over variable name  
    params_vname = dict({'tripyrun_analysis':analysis_name})
    params_vname.update(params_1lvl)
    params_vname.update(params_2lvl)
    #params_vname["vname"] = vname
        
    #___________________________________________________________________________
    # make loop over transects
    webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='input_transect', 
                                               source_loop="transects", source_single='transect', exec_template=analysis_name)
    return webpage



#
#
#_______________________________________________________________________________
def drive_transect_Xtransp_t(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    # analysis_name: 
    #  -'transect_transp_t'
    #  -'transect_hflx_t'
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # loop over variable name  
    params_vname = dict({'tripyrun_analysis':analysis_name})
    params_vname.update(params_1lvl)
    params_vname.update(params_2lvl)
    #params_vname["vname"] = vname
        
    #___________________________________________________________________________
    # make loop over transects
    webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='input_transect', 
                                               source_loop="transects", source_single='transect', exec_template=analysis_name)
    return webpage



#
#
#_______________________________________________________________________________
def drive_transect_zmean(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # kickout secondary parameters from yaml_settings[analysis_name] only leave 
    # the dictionaries of the variables
    driver_vars = yaml_settings[analysis_name].copy()
    for key in params_2lvl.keys():
        del driver_vars[key]
    
    # execute only specfic driver with a specific variable 
    if vname != None: 
        if vname in driver_vars: driver_vars = dict({vname: driver_vars[vname]})
    
    #___________________________________________________________________________
    # loop over variable name  
    for vname in driver_vars:
        print(f'     --> compute: {vname}')
        params_3lvl  = extract_params(driver_vars[vname])
        params_vname = dict({'tripyrun_analysis':analysis_name})
        params_vname.update(params_1lvl)
        params_vname.update(params_2lvl)
        params_vname.update(params_3lvl)
        params_vname["vname"] = vname
        
        #_______________________________________________________________________
        # make loop over box_regions
        webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='box_region', 
                                               source_loop='box_regions', exec_template='transect_zmean')
    return webpage



#
#
#_______________________________________________________________________________
def drive_transect_zmean_clim(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # kickout secondary parameters from yaml_settings[analysis_name] only leave 
    # the dictionaries of the variables
    driver_vars = yaml_settings[analysis_name].copy()
    for key in params_2lvl.keys():
        del driver_vars[key]
    
    # execute only specfic driver with a specific variable 
    if vname != None: 
        if vname in driver_vars: driver_vars = dict({vname: driver_vars[vname]})
    
    #___________________________________________________________________________
    # loop over variable name  
    for vname in driver_vars:
        print(f'     --> compute: {vname}')
        params_3lvl  = extract_params(driver_vars[vname])
        params_vname = dict({'tripyrun_analysis':analysis_name})
        params_vname.update(params_1lvl)
        params_vname.update(params_2lvl)
        params_vname.update(params_3lvl)
        params_vname["vname"] = vname
        
        #_______________________________________________________________________
        # make loop over box_regions
        webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='box_region', 
                                               source_loop='box_regions', exec_template='transect_zmean_clim')
    return webpage



#
#
#_______________________________________________________________________________
def drive_ghflx(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # initialse webpage for analyis
    params_vname = dict({'tripyrun_analysis':analysis_name})
    params_vname.update(params_1lvl)
    params_vname.update(params_2lvl)
    params_vname["vname"] = 'ghflx'
    webpage, image_count = exec_papermill(webpage, image_count, params_vname, exec_template='transp_ghflx')
    return webpage



#
#
#_______________________________________________________________________________
def drive_mhflx(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # initialse webpage for analyis
    params_vname = dict({'tripyrun_analysis':analysis_name})
    params_vname.update(params_1lvl)
    params_vname.update(params_2lvl)
    params_vname["vname"] = 'mhflx'
    webpage, image_count = exec_papermill(webpage, image_count, params_vname, exec_template='transp_mhflx')
    return webpage



#
#
#_______________________________________________________________________________
def drive_var_t(yaml_settings, analysis_name, webpage=dict(), image_count=0, vname=None):
    #___________________________________________________________________________
    # create 1st-level parameter from yaml_settings
    params_1lvl = extract_params(yaml_settings)
    
    # create 2nd-level parameter from yaml_settings[analysis_name]
    params_2lvl = extract_params(yaml_settings[analysis_name])
    
    #___________________________________________________________________________
    # kickout secondary parameters from yaml_settings[analysis_name] only leave 
    # the dictionaries of the variables
    driver_vars = yaml_settings[analysis_name].copy()
    for key in params_2lvl.keys():
        del driver_vars[key]
    
    # execute only specfic driver with a specific variable 
    if vname != None: 
        if vname in driver_vars: driver_vars = dict({vname: driver_vars[vname]})
    
    #___________________________________________________________________________
    # loop over variable name  
    for vname in driver_vars:
        print(f'     --> compute: {vname}')
        params_3lvl  = extract_params(driver_vars[vname])
        params_vname = dict({'tripyrun_analysis':analysis_name})
        params_vname.update(params_1lvl)
        params_vname.update(params_2lvl)
        params_vname.update(params_3lvl)
        params_vname["vname"] = vname
        
        #_______________________________________________________________________
        # make loop over box_regions
        webpage, image_count = loop_over_param(webpage, image_count, params_vname, target='box_region', 
                                               source_loop='box_regions', exec_template='var_t')
    return webpage
