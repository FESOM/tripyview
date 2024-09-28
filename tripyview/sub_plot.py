import os
import sys
import numpy                    as np
import time                     as clock
import xarray                   as xr
import cartopy.crs              as ccrs
import cartopy.feature          as cfeature
from   cartopy.mpl.ticker       import (LongitudeFormatter, LatitudeFormatter)
from   cartopy.mpl.gridliner    import Gridliner
from   mpl_toolkits.axes_grid1  import make_axes_locatable
import matplotlib
#import matplotlib.pylab         as plt
import matplotlib.pyplot        as plt
from   matplotlib.tri           import Triangulation,TriAnalyzer
import matplotlib.ticker        as mticker
import matplotlib.path          as mpath
import matplotlib.colors        as mcolors
from   matplotlib.colors        import ListedColormap
from   matplotlib.ticker        import MultipleLocator, AutoMinorLocator, ScalarFormatter, LogLocator, LogFormatter
from   scipy.signal             import convolve2d
from   scipy.interpolate        import interp1d
import textwrap
import warnings
import copy                     as cp
from .sub_mesh     import *
from .sub_data     import *
from .sub_colormap import *
from .sub_utility  import *



#
#
#_______________________________________________________________________________
# --> do plotting of horizontal slices
def plot_hslice(mesh                   , 
                data                   , 
                box        = None      , 
                cinfo      = None      , # colormap info and defintion
                nrow       = 1         , # number of row in figures panel
                ncol       = 1         , # number of column in figure panel
                proj       = 'pc'      ,
                do_ie2n    = False     , # interpolate element data to vertices
                do_rescale = False     ,
                #--- data -----------
                do_plt     = 'tpc'     , # tpc:tripcolor, tcf:tricontourf
                plt_opt    = dict()    ,
                plt_contb  = False     , # plot background contour lines: True/False 
                pltcb_opt  = dict()    , # background contour line option
                plt_contf  = False     , # plot foreground contour lines: True/False 
                pltcf_opt  = dict()    , # foreground contour line option
                plt_contr  = False     , # plot reference contour lines: True/False 
                pltcr_opt  = dict()    , # reference contour line option
                plt_contl  = False     , # do contourline labels 
                pltcl_opt  = dict()    , # contour line label options
                #--- mesh -----------
                do_mesh    = False     , 
                mesh_opt   = dict()    , 
                #--- bottom mask ----
                do_bot     = True      , 
                bot_opt    = dict()    ,
                #--- landsea mask ---
                do_lsm     = 'fesom'   , 
                lsm_opt    = dict()    , 
                lsm_res    = 'low'     ,
                #--- gridlines ------
                do_grid    = True      , 
                do_boundbox= True      , 
                grid_opt   = dict()    ,
                #--- colorbar -------
                cb_label   = None      ,
                cb_lunit   = None      ,
                cb_ltime   = None      ,
                cb_ldep    = None      , 
                cb_opt     = dict()    , # colorbar option
                cbl_opt    = dict()    , # colorbar label option, fontsize ,...
                cbtl_opt   = dict()    , # colorbar ticklabel option, fontsize ,...
                #--- axes -----------
                ax_title   = 'descript',
                ax_opt     = dict()    ,
                axl_opt    = dict()    ,# dictionary that defines axes and colorbar arangement
                #--- enumerate axes -
                do_enum    = False     ,
                enum_opt   = dict()    , 
                enum_str   = []        , 
                enum_x     = [0.005]   , 
                enum_y     = [1.00]    ,
                enum_dir   = 'lr'    ,# prescribed list of enumeration strings
                #--- save figure ----
                do_save    = None      , 
                save_dpi   = 300       ,
                save_opt   = dict()    ,
                #--- set output -----
                nargout    =['hfig', 'hax', 'hcb'],
                ):
    """
    --> plot FESOM2 horizontal data slice:

    __________________________________________________

    Parameters: 
    
        :mesh:      fesom2 mesh object, with all mesh information

        :data:      xarray dataset object, or list of xarray dataset object

        :box:       None, list (default: None) regional limitation of plot. For ortho...
                    box=[lonc, latc], nears...box=[lonc, latc, zoom], for all others box = 
                    [lonmin, lonmax, latmin, latmax]

        :cinfo:     None, dict() (default: None), dictionary with colorbar information. 
                    Information that are given are used, others are computed. cinfo dictionary 
                    entries can be,
                
                    - cinfo['cmin'], cinfo['cmax'], cinfo['cref'] ... scalar min, max, reference value
                    - cinfo['crange'] ... list with [cmin, cmax, cref] overrides scalar values 
                    - cinfo['cnum']   ... minimum number of colors
                    - cinfo['cstr']   ... name of colormap see in sub_colormap_c2c.py
                    - cinfo['cmap']   ... colormap object ('wbgyr', 'blue2red, 'jet' ...)
                    - cinfo['clevel'] ... color level array

        :nrow:      int, (default: 1) number of columns when plotting multiple data panels

        :ncol:      int, (default: 1) number of rows when plotting multiple data panels

        :proj:      str, (default: 'pc') which projection should be used, 
                    - pc     ... PlateCarree         (box=[lonmin, lonmax, latmin, latmax])
                    - merc   ... Mercator            (box=[lonmin, lonmax, latmin, latmax])
                    - rob    ... Robinson            (box=[lonmin, lonmax, latmin, latmax])
                    - eqearth... EqualEarth          (box=[lonmin, lonmax, latmin, latmax])
                    - mol    ... Mollweide           (box=[lonmin, lonmax, latmin, latmax])
                    - nps    ... NorthPolarStereo    (box=[-180, 180, >0, latmax])
                    - sps    ... SouthPolarStereo    (box=[-180, 180, latmin, <0])
                    - ortho  ... Orthographic        (box=[loncenter, latcenter]) 
                    - nears  ... NearsidePerspective (box=[loncenter, latcenter, zoom]) 
                    - channel... PlateCaree

        :do_ie2n:   bool, (default: False) do interpolation of data on elements towards nodes

        :do_rescale: bool, str, np.array (defaul: False) do scaling of colorbar
                    - False      ... scale data automatically scientifically by 10^x, for data data larger 10^3 and smaller 10^-3
                    - log10      ... do logaritmic scaling
                    - slog10     ... do symetric logarithmic scaling
                    - np.array() ... scale colorbar stepwise according to values in np.array allows also for non-linear colortick steps

        ___plot data parameters_____________________________

        :do_plt:    str, (default: tpc)
                    - tpc ... make pseudocolor plot (tripcolor)
                    - tcf ... make contourf color plot (tricontourf)

        :plt_opt:   dict, (default: dict()) additional options that are given to tripcolor 
                    or tricontourf via the kwarg argument

        :plt_contb: bool, (default: False) overlay thin contour lines of all colorbar steps (background)

        :pltcb_opt: dict, (default: dict()) background contour line option

        :plt_contf: bool, (default: False) overlay thicker contour lines of the main colorbar steps (foreground)

        :pltcf_opt: dict, (default: dict()) foreground contour line option

        :plt_contr: bool, (default: False) overlay thick contour lines of reference color steps (reference)

        :pltcr_opt: dict, (default: dict()) reference contour line option

        :plt_contl: bool, (default: False) label overlayed  contour linec plot

        :pltcl_opt: dict, (default: dict()) additional options that are given to clabel via the kwarg argument

        ___plot mesh________________________________________

        :do_mesh:   bool, (default: True), overlay FESOM grid over dataplot

        :mesh_opt:  dict, (default: dict()) additional options that are given to the mesh plotting via kwarg

        ___plot bottom mask_________________________________

        :do_bot:    bool, (default: True), overlay topographic bottom mask

        :bot_opt:   dict, (default: dict()) additional options that are given to the bottom mask plotting via kwarg

        ___plot lsmask______________________________________

        :do_lsm:    str, (default: 'fesom'), overlay FESOM grid inverted land sea mask
                    option are here:
                
                    - fesom      ... grey fesom landsea mask
                    - stock      ... uses cartopy stock image
                    - bluemarble ... uses bluemarble image in folder tripyview/background/
                    - etopo      ... uses etopo image in folder tripyview/background/

        :lsm_opt:   dict, (default: dict()) additional options that are given to 
                    the landsea mask plotting via kwarg

        :lsm_res:   str, (default='low') resolution of bluemarble texture file either 'low' or 'high'

        ___plot cartopy gridlines____________________________

        :do_grid:   bool, (default: True) plot cartopy grid lines
        
        :grid_opt:  dict, (default: dict()) additional options that are given to 
                    the cartopy gridline plotting via kwarg

        :do_boundbox: bool, (default: True) plot cartopy black bounding box. If you 
                    make plots as a texture to use in blender, unity or pyvista the bounding box 
                    has to be removed with do_boundbox=False

        ___colorbar_________________________________________

        :cb_label:  str, (default: None) if string its used as colorbar label, otherwise 
                    information from data ('long_name, short_name) are used

        :cb_lunit:  str, (default: None) if string its used as colorbar unit label, 
                    otherwise info from data are used

        :cb_ltime:  str, (default: None) if string its used as colorbar time label, 
                    otherwise info from data are used

        :cb_ldep:   str, (default: None) if string its used as colorbar depth label, 
                    otherwise info from data are used                

        :cb_opt:    dict, (default: dict()) direct option for colorbar via kwarg

        :cbl_opt:   dict, (default: dict()) direct option for colorbar labels (fontsize, fontweight, ...) via kwarg

        :cbtl_opt:  dict, (default: dict()) direct option for colorbar tick labels (fontsize, fontweight, ...) via kwarg

        ___axes______________________________________

        :ax_title:  str, (default: 'descript') If 'descript' use descript attribute 
                    in data to title label axes, If 'str' use this string to label axes

        :ax_opt:    dict, (default: dict()) set option for axes arangement see subroutine 
                    do_axes_arrange

        :axl_opt:   dict, (default: dict()) set option for axes labels (fontsize, ...)

        ___enumerate_________________________________

        :do_enum:   bool, (default: False) do enumeration of axes with a), b), c) ...

        :enum_opt:  dict, (default: dict()) direct option for enumeration strings via kwarg 

        :enum_str:  list, (default: []) overwrite default enumeration strings        , 

        :enum_x:    float, (default: 0.005)  x position of enumeration string in axes coordinates

        :enum_y:    float, (default: 1.000)  y position of enumeration string in axes coordinates

        :enum_dir:  str, (default: 'lr')  direction of numbering, 'lr' from left to right, 'ud' from up to down

        ___save figure________________________________

        :do_save:   str, (default: None) if None figure will by not saved, if string figure 
                    will be saved, strings must give directory and filename  where to save.

        :save_dpi:  int, (default: 300) dpi resolution at which the figure is saved

        :save_opt:  dict, (default: dict()) direct option for saving via kwarg

        ___set output_________________________________

        :nargout:   list, (default: ['hfig', 'hax', 'hcb']) list of variables that are given 
                    out from the routine. 
                    Default: 
                    
                    - hfig ... figure handle
                    - hax  ... list of axes handle 
                    - hcb  ... list of colorbar handles (every variable that is defined in this subroutine can become output parameter)
    
    __________________________________________________
    
    Returns:
    
        :hfig: returns figure handle 

        :hax: returns list with axes handle 

        :hcb: returns colorbar handle
    
    ____________________________________________________________________________
    """
    #___________________________________________________________________________
    # --> create box
    if (box is None or box=="None") and proj!='channel': box = [ -180+mesh.focus, 180+mesh.focus, -90, 90 ]
    
    #___________________________________________________________________________
    # --> check if input data is a list
    if not isinstance(data, list): data = [data]
    ndat = len(data)
    
    #___________________________________________________________________________
    # --> create projection
    proj_to = None
    # proj is string 
    if   isinstance(proj, str): 
        proj_to, box = do_projection(mesh, proj, box)
    # proj is cartopy projection object
    elif isinstance(proj, ccrs.CRS): 
        proj_to = proj
    # proj is list of cartopy projection objects or string
    elif isinstance(proj, list): 
        proj_to = list()
        for proj_ii in proj:
            if   isinstance(proj_ii, str): 
                proj_dum, box = do_projection(mesh, proj_ii, box)
                proj_to.append(proj_dum)                
            elif isinstance(proj_ii, ccrs.CRS):    
                proj_to.append(proj_ii)    
    
    #___________________________________________________________________________
    # --> pre-arange axes
    ax_optdefault=dict({'projection': proj_to, 'box': box, 'proj':proj})
    ax_optdefault.update(ax_opt)
    hfig, hax, hcb, cb_plt_idx = do_axes_arrange(ncol, nrow, **ax_optdefault)
    cb_plt_idx=cb_plt_idx[:ndat]
    
    #___________________________________________________________________________
    # --> axes enumeration 
    do_axes_enum(hax[:ndat], do_enum, nrow, ncol, enum_opt=enum_opt, enum_str=enum_str, 
                       enum_x=enum_x, enum_y=enum_y, enum_dir=enum_dir)
    
    #___________________________________________________________________________
    # --> create mesh triangulation, when input data are unstructured, if not fall
    #     back to regular plotting
    tri = None
    if 'nod2' in data[0].dims or 'elem' in data[0].dims:
        #_______________________________________________________________________
        tri = do_triangulation(hax[0], mesh, proj_to, box)
        
    #___________________________________________________________________________
    # --> set up color info
    # --> setup normalization log10, symetric log10, None
    cinfo_plot, norm_plot, idxs = list(), list(), 0
    for ii in np.unique(cb_plt_idx):
        idsel = np.where(cb_plt_idx==ii)[0]
        #_______________________________________________________________________
        if isinstance(cinfo, list) and isinstance(do_rescale, list):
            cinfo_plot.append( do_setupcinfo(cinfo[ii-1], [data[jj] for jj in idsel], do_rescale[ii-1], mesh=mesh, tri=tri) )
        elif isinstance(cinfo, list):
            cinfo_plot.append( do_setupcinfo(cinfo[ii-1], [data[jj] for jj in idsel], do_rescale, mesh=mesh, tri=tri) )
        else:    
            cinfo_plot.append( do_setupcinfo(cinfo, [data[jj] for jj in idsel], do_rescale, mesh=mesh, tri=tri) )
        
        #_______________________________________________________________________
        if isinstance(do_rescale, list):
            norm_plot.append(  do_data_norm(cinfo_plot[-1], do_rescale[ii-1]) )
        else:
            norm_plot.append(  do_data_norm(cinfo_plot[-1], do_rescale) )
        
    #___________________________________________________________________________
    # --> loop over axes
    hp, hbot, hmsh, hlsm, hgrd, hall = list(), list(), list(), list(), list(), list()
    for ii, (hax_ii, hcb_ii) in enumerate(zip(hax, hcb)):
        # if there are no ddatra to fill axes, make it invisible 
        if ii>=ndat: 
            hax_ii.axis('off')
        elif data[ii] is None: 
            hax_ii.axis('off')    
        # axis is normally fillt with data    
        else:    
            ii_valid=ii
            #___________________________________________________________________
            # switch between unstrucutred and regular plotting
            # plot unstructured data
            if tri is not None:
                #_______________________________________________________________
                # prepare unstructured data for plotting, augment periodic 
                # boundaries, interpolate from elements to nodes, kick out nan 
                # values from plotting that are bottom topo  
                vname     = list(data[ii].keys())[0]
                data_plot = data[ii][ vname ].data.copy()
                data_plot, tri = do_data_prepare_unstruct(mesh, tri, data_plot, do_ie2n)
                
                #_______________________________________________________________
                # add color for ocean bottom
                h0 = do_plt_bot(hax_ii, do_bot, tri=tri, bot_opt=bot_opt)
                hbot.append(h0)
                
                #_______________________________________________________________
                # add tripcolor or tricontourf plot 
                h0 = do_plt_data(hax_ii, do_plt, tri, data_plot, 
                                cinfo_plot[ cb_plt_idx[ii]-1 ], norm_plot[ cb_plt_idx[ii]-1 ], 
                                plt_opt  = plt_opt,
                                plt_contb=plt_contb, pltcb_opt=pltcb_opt,
                                plt_contf=plt_contf, pltcf_opt=pltcf_opt,
                                plt_contr=plt_contr, pltcr_opt=pltcr_opt,
                                plt_contl=plt_contl, pltcl_opt=pltcl_opt)
                hp.append(h0)
                
                #___________________________________________________________________
                # add grid mesh on top
                h0 = do_plt_mesh(hax_ii, do_mesh, tri, mesh_opt=mesh_opt)
                hmsh.append(h0)
            
            # plot regular gridded data
            else:
                #_______________________________________________________________
                # prepare regular gridded data for plotting
                vname = list(data[ii].keys())[0]
                data_plot = data[ii][vname].data.copy()
                if   do_plt in ['tpc','pc'] and ('lon_bnd' in data[ii] and 'lat_bnd' in data[ii]) : 
                    data_x, data_y = data[ii]['lon_bnd'], data[ii]['lat_bnd']
                else               : data_x, data_y = data[ii]['lon'    ], data[ii]['lat'    ]
                
                #_______________________________________________________________
                # add tripcolor or tricontourf plot 
                h0 = do_plt_datareg(hax_ii, do_plt, data_x, data_y, data_plot, 
                                cinfo_plot[ cb_plt_idx[ii]-1 ], norm_plot[ cb_plt_idx[ii]-1 ],
                                which_transf=ccrs.PlateCarree(), 
                                plt_opt  = plt_opt,
                                plt_contb=plt_contb, pltcb_opt=pltcb_opt,
                                plt_contf=plt_contf, pltcf_opt=pltcf_opt,
                                plt_contr=plt_contr, pltcr_opt=pltcr_opt,
                                plt_contl=plt_contl, pltcl_opt=pltcl_opt)
                hp.append(h0)
                
            #___________________________________________________________________
            # add mesh land-sea mask
            h0 = do_plt_lsmask(hax_ii, do_lsm, mesh, lsm_opt=lsm_opt, resolution=lsm_res)
            hlsm.append(h0)  
            
            #___________________________________________________________________
            # add grids lines 
            h0 = do_plt_gridlines(hax_ii, do_grid, box, ndat, grid_opt=grid_opt, proj=proj)
            hgrd.append(h0)
            
            #___________________________________________________________________
            # add title and axes labels
            axl_optdefault=dict({'fontsize':hax_ii.fs_label})
            if ax_title is not None: 
                # is title  string:
                if   isinstance(ax_title,str) : 
                    # if title string is 'descript' than use descript attribute from 
                    # data to set plot title 
                    if ax_title=='descript' and ('descript' in data[ii][vname].attrs.keys() ):
                        axl_optdefault.update({'verticalalignment':'top'})
                        axl_optdefault.update(axl_opt)
                        hax_ii.set_title(data[ii][ vname ].attrs['descript'], **axl_optdefault )
                        
                    else:
                        axl_optdefault.update(axl_opt)
                        hax_ii.set_title(ax_title, **axl_optdefault )
                # is title list of string        
                elif isinstance(ax_title,list): 
                    axl_optdefault.update(axl_opt)
                    hax_ii.set_title(ax_title[ii], **axl_optdefault )
                
        #_______________________________________________________________________
        # add colorbar 
        if hcb_ii != 0 and hp[-1] is not None: 
            if isinstance(do_rescale, list):
                hcb_ii = do_cbar(hcb_ii, hax_ii, hp, data[ii_valid], cinfo_plot[cb_plt_idx[ii_valid]-1], do_rescale[cb_plt_idx[ii_valid]-1], 
                                cb_label, cb_lunit, cb_ltime, cb_ldep, norm=norm_plot[ cb_plt_idx[ii_valid]-1 ], 
                                cb_opt=cb_opt, cbl_opt=cbl_opt, cbtl_opt=cbtl_opt)
            else:    
                hcb_ii = do_cbar(hcb_ii, hax_ii, hp, data[ii_valid], cinfo_plot[cb_plt_idx[ii_valid]-1], do_rescale, 
                                cb_label, cb_lunit, cb_ltime, cb_ldep, norm=norm_plot[ cb_plt_idx[ii_valid]-1 ], 
                                cb_opt=cb_opt, cbl_opt=cbl_opt, cbtl_opt=cbtl_opt)
        
        #___________________________________________________________________
        # add all handles together 
        hall = hp + hbot + hmsh + hlsm + hgrd
            
        #_______________________________________________________________________
        # hfig.canvas.draw() 
        
        
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, hfig, dpi=save_dpi, save_opt=save_opt)
    plt.show()
    
    #___________________________________________________________________________
    list_argout=[]
    if len(nargout)>0:
        for stri in nargout:
            try   : list_argout.append(eval(stri))
            except: print(f" -warning-> variable {stri} was not found, could not be added as output argument") 
            
        return(list_argout)
    else:
        return



#
#
#_______________________________________________________________________________
# --> do plotting of horizontal meshes
def plot_hmesh( mesh                   , 
                data       = None      , # None, resolution, depth
                box        = None      , 
                cinfo      = None      , # colormap info and defintion
                nrow       = 1         , # number of row in figures panel
                ncol       = 1         , # number of column in figure panel
                proj       = 'pc'      ,
                do_ie2n    = False     ,
                do_rescale = False     ,
                #--- data -----------
                do_plt     = 'tpc'     , # tpc:tripcolor, tcf:tricontourf
                plt_opt    = dict()    ,
                plt_contb  = False     , # plot background contour lines: True/False 
                pltcb_opt  = dict()    , # background contour line option
                plt_contf  = False     , # plot foreground contour lines: True/False 
                pltcf_opt  = dict()    , # foreground contour line option
                plt_contr  = False     , # plot reference contour lines: True/False 
                pltcr_opt  = dict()    , # reference contour line option
                plt_contl  = False     , # do contourline labels 
                pltcl_opt  = dict()    , # contour line label options
                #--- mesh -----------
                do_mesh    = True      , 
                mesh_opt   = dict()    , 
                #--- landsea mask ---
                do_lsm     = 'fesom'   , 
                lsm_opt    = dict()    , 
                lsm_res    = 'low'     ,
                #--- gridlines ------
                do_grid    = True      ,
                do_boundbox= True      , 
                grid_opt   = dict()    ,
                #--- colorbar -------
                cb_label   = None      ,
                cb_lunit   = None      ,
                cb_ltime   = None      ,
                cb_ldep    = None      , 
                cb_opt     = dict()    , # colorbar option
                cbl_opt    = dict()    , # colorbar label option, fontsize ,...
                cbtl_opt   = dict()    , # colorbar ticklabel option, fontsize ,...
                #--- axes -----------
                ax_title   = None,
                ax_opt     = dict()    , # dictionary that defines axes and colorbar arangement
                #--- enumerate axes -
                do_enum    = False     ,
                enum_opt   = dict()    , 
                enum_str   = []        , 
                enum_x     = [0.005]   , 
                enum_y     = [1.00]    ,
                enum_dir   = 'lr'    ,# prescribed list of enumeration strings
                #--- save figure ----
                do_save    = None      , 
                save_dpi   = 300       ,
                save_opt   = dict()    ,
                #--- set output -----
                nargout=['hfig', 'hax', 'hcb'],
                ):
    """
    --> plot horizontal mesh and mesh paramters on vertices and elements
    
    __________________________________________________
    
    Parameters:
    
        :mesh:      fesom2 mesh object, with all mesh information

        :data:      str, (default: None) string can be:

                    - 'resolution', 'resol', 'n_resol', 'nresol'
                    - 'narea', 'n_area', 'clusterarea', 'scalararea'
                    - 'eresol', 'e_resol', 'triresolution', 'triresol'
                    - 'earea', 'e_area', 'triarea'
                    - 'ndepth', 'ntopo', 'n_depth', 'n_topo', 'topography', 'zcoord'
                    - 'edepth', 'etopo', 'e_depth', 'e_topo' 

        :box:       None, list (default: None) regional limitation of plot. For ortho...
                    box=[lonc, latc], nears...box=[lonc, latc, zoom], for all others box = 
                    [lonmin, lonmax, latmin, latmax]

        :cinfo:     None, dict() (default: None), dictionary with colorbar information. 
                    Information that are given are used, others are computed. cinfo dictionary 
                    entries can be,
                    
                    - cinfo['cmin'], cinfo['cmax'], cinfo['cref'] ... scalar min, max, reference value
                    - cinfo['crange'] ... list with [cmin, cmax, cref] overrides scalar values 
                    - cinfo['cnum']   ... minimum number of colors
                    - cinfo['cstr']   ... name of colormap see in sub_colormap_c2c.py
                    - cinfo['cmap']   ... colormap object ('wbgyr', 'blue2red, 'jet' ...)
                    - cinfo['clevel'] ... color level array

        :nrow:      int, (default: 1) number of columns when plotting multiple data panels

        :ncol:      int, (default: 1) number of rows when plotting multiple data panels

        :proj:      str, (default: 'pc') which projection should be used, 
                    - pc     ... PlateCarree         (box=[lonmin, lonmax, latmin, latmax])
                    - merc   ... Mercator            (box=[lonmin, lonmax, latmin, latmax])
                    - rob    ... Robinson            (box=[lonmin, lonmax, latmin, latmax])
                    - eqearth... EqualEarth          (box=[lonmin, lonmax, latmin, latmax])
                    - mol    ... Mollweide           (box=[lonmin, lonmax, latmin, latmax])
                    - nps    ... NorthPolarStereo    (box=[-180, 180, >0, latmax])
                    - sps    ... SouthPolarStereo    (box=[-180, 180, latmin, <0])
                    - ortho  ... Orthographic        (box=[loncenter, latcenter]) 
                    - nears  ... NearsidePerspective (box=[loncenter, latcenter, zoom]) 
                    - channel... PlateCaree

        :do_ie2n:   bool, (default: False) do interpolation of data on elements towards nodes

        :do_rescale: bool, str, np.array (defaul: False) do scaling of colorbar
                    - False      ... scale data automatically scientifically by 10^x, for data data larger 10^3 and smaller 10^-3
                    - log10      ... do logaritmic scaling
                    - slog10     ... do symetric logarithmic scaling
                    - np.array() ... scale colorbar stepwise according to values in np.array allows also for non-linear colortick steps

        ___plot data parameters_____________________________

        :do_plt:    str, (default: tpc)
                    - tpc ... make pseudocolor plot (tripcolor)
                    - tcf ... make contourf color plot (tricontourf)

        :plt_opt:   dict, (default: dict()) additional options that are given to tripcolor 
                    or tricontourf via the kwarg argument

        :plt_contb: bool, (default: False) overlay thin contour lines of all colorbar steps (background)

        :pltcb_opt: dict, (default: dict()) background contour line option

        :plt_contf: bool, (default: False) overlay thicker contour lines of the main colorbar steps (foreground)

        :pltcf_opt: dict, (default: dict()) foreground contour line option

        :plt_contr: bool, (default: False) overlay thick contour lines of reference color steps (reference)

        :pltcr_opt: dict, (default: dict()) reference contour line option

        :plt_contl: bool, (default: False) label overlayed  contour linec plot

        :pltcl_opt: dict, (default: dict()) additional options that are given to clabel via the kwarg argument

        ___plot mesh________________________________________

        :do_mesh:   bool, (default: True), overlay FESOM grid over dataplot

        :mesh_opt:  dict, (default: dict()) additional options that are given to the mesh plotting via kwarg

        ___plot bottom mask_________________________________

        :do_bot:    bool, (default: True), overlay topographic bottom mask

        :bot_opt:   dict, (default: dict()) additional options that are given to the bottom mask plotting via kwarg

        ___plot lsmask______________________________________

        :do_lsm:    str, (default: 'fesom'), overlay FESOM grid inverted land sea mask
                    option are here:
                    
                    - fesom      ... grey fesom landsea mask
                    - stock      ... uses cartopy stock image
                    - bluemarble ... uses bluemarble image in folder tripyview/background/
                    - etopo      ... uses etopo image in folder tripyview/background/

        :lsm_opt:   dict, (default: dict()) additional options that are given to 
                    the landsea mask plotting via kwarg

        :lsm_res:   str, (default='low') resolution of bluemarble texture file either 'low' or 'high'

        ___plot cartopy gridlines____________________________

        :do_grid:   bool, (default: True) plot cartopy grid lines

        :grid_opt:  dict, (default: dict()) additional options that are given to 
                    the cartopy gridline plotting via kwarg

        :do_boundbox: bool, (default: True) plot cartopy black bounding box. If you 
                    make plots as a texture to use in blender, unity or pyvista the bounding box 
                    has to be removed with do_boundbox=False

        ___colorbar_________________________________________

        :cb_label:  str, (default: None) if string its used as colorbar label, otherwise 
                    information from data ('long_name, short_name) are used

        :cb_lunit:  str, (default: None) if string its used as colorbar unit label, 
                    otherwise info from data are used

        :cb_ltime:  str, (default: None) if string its used as colorbar time label, 
                    otherwise info from data are used

        :cb_ldep:   str, (default: None) if string its used as colorbar depth label, 
                    otherwise info from data are used                

        :cb_opt:    dict, (default: dict()) direct option for colorbar via kwarg

        :cbl_opt:   dict, (default: dict()) direct option for colorbar labels (fontsize, fontweight, ...) via kwarg

        :cbtl_opt:  dict, (default: dict()) direct option for colorbar tick labels (fontsize, fontweight, ...) via kwarg

        ___axes______________________________________

        :ax_title:  str, (default: 'descript') If 'descript' use descript attribute 
                    in data to title label axes, If 'str' use this string to label axes

        :ax_opt:    dict, (default: dict()) set option for axes arangement see subroutine 
                    do_axes_arrange

        :axl_opt:   dict, (default: dict()) set option for axes labels (fontsize, ...)

        ___enumerate_________________________________

        :do_enum:   bool, (default: False) do enumeration of axes with a), b), c) ...

        :enum_opt:  dict, (default: dict()) direct option for enumeration strings via kwarg 

        :enum_str:  list, (default: []) overwrite default enumeration strings        , 

        :enum_x:    float, (default: 0.005)  x position of enumeration string in axes coordinates

        :enum_y:    float, (default: 1.000)  y position of enumeration string in axes coordinates

        :enum_dir:  str, (default: 'lr')  direction of numbering, 'lr' from left to right, 'ud' from up to down

        ___save figure________________________________

        :do_save:   str, (default: None) if None figure will by not saved, if string figure 
                    will be saved, strings must give directory and filename  where to save.

        :save_dpi:  int, (default: 300) dpi resolution at which the figure is saved

        :save_opt:  dict, (default: dict()) direct option for saving via kwarg

        ___set output_________________________________

        :nargout:   list, (default: ['hfig', 'hax', 'hcb']) list of variables that are given 
                    out from the routine. 
                    Default: 
                    
                    - hfig ... figure handle
                    - hax  ... list of axes handle 
                    - hcb  ... list of colorbar handles (every variable that is defined in this subroutine can become output parameter)
    
    __________________________________________________
    
    Returns:
    
        :hfig: returns figure handle 

        :hax: returns list with axes handle 

        :hcb: returns colorbar handle
    
    ____________________________________________________________________________
    """    
    #___________________________________________________________________________
    # --> create box
    if box is None or box=="None": box = [ -180+mesh.focus, 180+mesh.focus, -90, 90 ]
    
    #___________________________________________________________________________
    # --> check if input data is a list
    if not isinstance(mesh, list): mesh = [mesh]
    ndat = len(mesh)
    
    #___________________________________________________________________________
    # --> create projection
    proj_to = None
    # proj is string 
    if   isinstance(proj, str): 
        proj_to, box = do_projection(mesh, proj, box)
    # proj is cartopy projection object
    elif isinstance(proj, ccrs.CRS): 
        proj_to = proj
    # proj is list of cartopy projection objects or string
    elif isinstance(proj, list): 
        proj_to = list()
        for proj_ii in proj:
            if   isinstance(proj_ii, str): 
                proj_dum, box = do_projection(mesh, proj_ii, box)
                proj_to.append(proj_dum)                
            elif isinstance(proj_ii, ccrs.CRS):    
                proj_to.append(proj_ii)    
    
    #___________________________________________________________________________
    # --> look if you need data
    aux_cb_plt = False
    if data is not None: aux_cb_plt = True
    
    #___________________________________________________________________________
    # --> pre-arange axes
    ax_optdefault=dict({'projection': proj_to, 'box': box, 'cb_plt':aux_cb_plt})
    ax_optdefault.update(ax_opt)
    hfig, hax, hcb, cb_plt_idx = do_axes_arrange(ncol, nrow, **ax_optdefault)
    cb_plt_idx=cb_plt_idx[:ndat]
    
    #___________________________________________________________________________
    # --> axes enumeration 
    do_axes_enum(hax[:ndat], do_enum, nrow, ncol, enum_opt=enum_opt, enum_str=enum_str, 
                       enum_x=enum_x, enum_y=enum_y, enum_dir=enum_dir)
    
    #___________________________________________________________________________
    # --> loop over axes
    hp, hmsh, hlsm, hgrd = list(), list(), list(), list()
    for ii, (hax_ii, hcb_ii) in enumerate(zip(hax, hcb)):
        # if there are no ddatra to fill axes, make it invisible 
        if ii>=ndat: 
            hax_ii.axis('off')
        # axis is normally fillt with data    
        else:    
            #___________________________________________________________________
            # build triangulation
            tri = do_triangulation(hax, mesh[ii], proj_to, box)
            
            #___________________________________________________________________
            print(data)
            if data != None and data != 'None':
                if  data in ['resolution', 'resol', 'n_resol', 'nresol']:
                    if len(mesh[ii].n_resol)==0: mesh[ii]=mesh[ii].compute_n_resol()
                    data_plot = mesh[ii].n_resol/1000
                    #data_plot = mesh[ii].n_resol[0,:]/1000
                    cb_label, cb_lunit = 'vertice resolution', 'km'
                elif data in ['narea', 'n_area', 'clusterarea', 'scalararea']:    
                    if len(mesh[ii].n_area)==0: mesh[ii]=mesh[ii].compute_n_area()
                    data_plot = mesh[ii].n_area[0,:]
                    cb_label, cb_lunit = 'vertice area', 'm^2'
                elif data in ['eresol', 'e_resol', 'triresolution', 'triresol']:
                    if len(mesh[ii].e_resol)==0: mesh[ii]=mesh[ii].compute_e_resol()
                    data_plot = mesh[ii].e_resol/1000
                    cb_label, cb_lunit = 'element resolution', 'km'
                elif data in ['earea', 'e_area', 'triarea']:
                    if len(mesh[ii].e_area)==0: mesh[ii]=mesh[ii].compute_e_area()
                    data_plot =  mesh[ii].e_area
                    cb_label, cb_lunit = 'element area', 'm^2'
                elif data in ['ndepth', 'ntopo', 'n_depth', 'n_topo', 'topography', 'zcoord']:
                    data_plot =  np.abs(mesh[ii].n_z)
                    cb_label, cb_lunit = 'vertice depth', 'm'
                elif data in ['edepth', 'etopo', 'e_depth', 'e_topo' ]:
                    data_plot = np.abs(mesh[ii].zlev[mesh[ii].e_iz])
                    cb_label, cb_lunit = 'element depth', 'm'
                
                print(data_plot.shape)
                #_______________________________________________________________
                cinfo_plot = do_setupcinfo(cinfo, [data_plot], do_rescale, mesh=mesh[ii], tri=tri)
                norm_plot  = do_data_norm(cinfo_plot, do_rescale)
                
                #_______________________________________________________________
                # prepare unstructured data for plotting, 
                data_plot, e_ok_mask = do_data_prepare_unstruct(mesh[ii], tri, data_plot, do_ie2n)

                #_______________________________________________________________
                # add tripcolor or tricontourf plot 
                h0 = do_plt_data(hax_ii, do_plt, tri, data_plot, cinfo_plot, norm_plot, 
                                 plt_opt  =plt_opt, 
                                 plt_contb=plt_contb, pltcb_opt=pltcb_opt, 
                                 plt_contf=plt_contf, pltcf_opt=pltcf_opt,
                                 plt_contr=plt_contr, pltcr_opt=pltcr_opt,
                                 plt_contl=plt_contl, pltcl_opt=pltcl_opt)
                hp.append(h0)
                
                
            #___________________________________________________________________
            # add grid mesh on top
            h0 = do_plt_mesh(hax_ii, do_mesh, tri, mesh_opt=mesh_opt)
            hmsh.append(h0)
                
            #___________________________________________________________________
            # add mesh land-sea mask
            h0 = do_plt_lsmask(hax_ii, do_lsm, mesh[ii], lsm_opt=lsm_opt, resolution=lsm_res)
            hlsm.append(h0)  
            
            #___________________________________________________________________
            # add grids lines 
            h0 = do_plt_gridlines(hax_ii, do_grid, box, len(mesh), grid_opt=grid_opt)
            hgrd.append(h0)
            
            #___________________________________________________________________
            # add title and axes labels
            if ax_title != None: 
                # is title  string:
                if   isinstance(ax_title,str) : hax_ii.set_title(ax_title, fontsize=hax_ii.fs_label)
                # is title list of string        
                elif isinstance(ax_title,list): hax_ii.set_title(ax_title[ii], fontsize=hax_ii.fs_label)
            
            #___________________________________________________________________
            # remove platecaree bounding box when used as image alpha map or texture
            if not do_boundbox: hax_ii.spines['geo'].set_visible(False)
            
        #_______________________________________________________________________
        # add colorbar 
        if (data != None and data != 'None'):
            if hcb_ii != 0 and hp[-1] is not None: 
                hcb_ii = do_cbar(hcb_ii, hax_ii, hp, data, cinfo_plot, norm_plot, 
                                cb_label, cb_lunit, cb_ltime, cb_ldep, cb_opt=cb_opt, cbl_opt=cbl_opt, cbtl_opt=cbtl_opt)
        else:
            hcb_ii.remove()
        #_______________________________________________________________________
        # hfig.canvas.draw()   
        
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, hfig, dpi=save_dpi, save_opt=save_opt)
    
    #___________________________________________________________________________
    list_argout=[]
    if len(nargout)>0:
        for stri in nargout:
            try   : list_argout.append(eval(stri))
            except: print(f" -warning-> variable {stri} was not found, could not be added as output argument") 
            
        return(list_argout)
    else:
        return



#
#
#_______________________________________________________________________________
# --> do plotting of horizontal slices
def plot_hquiver(mesh                  , 
                data                   , 
                box        = None      , 
                cinfo      = None      , # colormap info and defintion
                nrow       = 1         , # number of row in figures panel
                ncol       = 1         , # number of column in figure panel
                proj       = 'pc'      ,
                do_ie2n    = False     , # interpolate element data to vertices
                do_rescale = False     ,
                #--- quiver ---------
                do_quiv    = True      , # tpc:tripcolor, tcf:tricontourf
                quiv_opt   = dict()    ,
                quiv_scalfac  = 1      , # bigger means larger arrows  
                quiv_arrwidth = 0.25   ,
                quiv_dens  = 0.5       , # larger mean more excluded arrows
                quiv_smax  = 10        , # small arrow are scaled strong with factor smax, its off when smax=1
                quiv_shiftL= 2         , # shift smothing function to the left
                quiv_smooth= 2         , # slope of transitions zone, smaller value steeper transition
                #--- mesh -----------
                do_mesh    = False     , 
                mesh_opt   = dict()    , 
                #--- bottom mask ----
                do_bot     = True      , 
                bot_opt    = dict()    ,
                #--- topography -----
                do_topo    = 'tpc'     , 
                topo_opt   = dict()    ,
                topo_cont  = True      , # plot contour lines: True/False 
                topoc_opt  = dict()    , # contour line option
                topo_contl = False     , # do contourline labels 
                topocl_opt = dict()    , # contour line label options
                #--- landsea mask ---
                do_lsm     = 'fesom'   , 
                lsm_opt    = dict()    , 
                lsm_res    = 'low'     ,
                #--- gridlines ------
                do_grid    = True      , 
                do_boundbox= True      , 
                grid_opt   = dict()    ,
                #--- colorbar -------
                cb_label   = None      ,
                cb_lunit   = None      ,
                cb_ltime   = None      ,
                cb_ldep    = None      , 
                cb_opt     = dict()    , # colorbar option
                cbl_opt    = dict()    , # colorbar label option, fontsize ,...
                cbtl_opt   = dict()    , # colorbar ticklabel option, fontsize ,...
                #--- axes -----------
                ax_title   = 'descript',
                ax_opt     = dict()    , # dictionary that defines axes and colorbar arangement
                #--- enumerate axes -
                do_enum    = False     ,
                enum_opt   = dict()    , 
                enum_str   = []        , 
                enum_x     = [0.005]   , 
                enum_y     = [1.00]    ,
                enum_dir   = 'lr'      , # prescribed list of enumeration strings
                #--- save figure ----
                do_save    = None      , 
                save_dpi   = 300       ,
                save_opt   = dict()    ,
                #--- set output -----
                nargout=['hfig', 'hax', 'hcb'],
                ):
    """
    --> plot FESOM2 horizontal data slice as quiver plot:
    
    __________________________________________________
    
    Parameters:
        
        :mesh:      fesom2 mesh object, with all mesh information

        :data:      xarray dataset object, or list of xarray dataset object

        :box:       None, list (default: None) regional limitation of plot. For ortho...
                    box=[lonc, latc], nears...box=[lonc, latc, zoom], for all others box = 
                    [lonmin, lonmax, latmin, latmax]

        :cinfo:     None, dict() (default: None), dictionary with colorbar information. 
                    Information that are given are used, others are computed. cinfo dictionary 
                    entries can be,
                    
                    - cinfo['cmin'], cinfo['cmax'], cinfo['cref'] ... scalar min, max, reference value
                    - cinfo['crange'] ... list with [cmin, cmax, cref] overrides scalar values 
                    - cinfo['cnum']   ... minimum number of colors
                    - cinfo['cstr']   ... name of colormap see in sub_colormap_c2c.py
                    - cinfo['cmap']   ... colormap object ('wbgyr', 'blue2red, 'jet' ...)
                    - cinfo['clevel'] ... color level array
 
        :nrow:      int, (default: 1) number of columns when plotting multiple data panels

        :ncol:      int, (default: 1) number of rows when plotting multiple data panels

        :proj:      str, (default: 'pc') which projection should be used, 
                    - pc     ... PlateCarree         (box=[lonmin, lonmax, latmin, latmax])
                    - merc   ... Mercator            (box=[lonmin, lonmax, latmin, latmax])
                    - rob    ... Robinson            (box=[lonmin, lonmax, latmin, latmax])
                    - eqearth... EqualEarth          (box=[lonmin, lonmax, latmin, latmax])
                    - mol    ... Mollweide           (box=[lonmin, lonmax, latmin, latmax])
                    - nps    ... NorthPolarStereo    (box=[-180, 180, >0, latmax])
                    - sps    ... SouthPolarStereo    (box=[-180, 180, latmin, <0])
                    - ortho  ... Orthographic        (box=[loncenter, latcenter]) 
                    - nears  ... NearsidePerspective (box=[loncenter, latcenter, zoom]) 
                    - channel... PlateCaree

        :do_ie2n:    bool, (default: False) do interpolation of data on elements towards nodes

        :do_rescale: bool, str, np.array (defaul: False) do scaling of colorbar
                    - False      ... scale data automatically scientifically by 10^x, for data data larger 10^3 and smaller 10^-3
                    - log10      ... do logaritmic scaling
                    - slog10     ... do symetric logarithmic scaling
                    - np.array() ... scale colorbar stepwise according to values in np.array allows also for non-linear colortick steps

        ___plot quiver parameters___________________________

        :do_quiv:   bool, (default: True), do cartopy quiver plot

        :quiv_opt:  dict, (default: dict()) additional options that are given to quiver plot routine

        :quiv_scalfac:  float, (default: 1.0)  bigger means larger arrows

        :quiv_arrwidth: float, (default: 0.25) scale arrow width

        :quiv_dens: float, (default: 0.5)  larger mean more excluded arrows

        :quiv_smax: float, (default: 10) small arrow are scaled strong with factor smax, its off when smax=1

        :quiv_shiftL: float, (default: 2) shift smothing function to the left

        :quiv_smooth: float, (default: 2) slope of transitions zone, smaller value steeper transition

        ___plot mesh________________________________________

        :do_mesh:   bool, (default: True), overlay FESOM grid over dataplot

        :mesh_opt:  dict, (default: dict()) additional options that are given to the mesh plotting via kwarg

        ___plot bottom mask_________________________________

        :do_bot:    bool, (default: True), overlay topographic bottom mask

        :bot_opt:   dict, (default: dict()) additional options that are given to the bottom mask plotting via kwarg

        ___plot topo________________________________________

        :do_topo:   str, (default: tpc) = 
                    - tpc ... make pseudocolor plot (tripcolor)
                    - tcf ... make contourf coor plot (tricontourf)  , # tpc:tripcolor, tcf:tricontourf    

        :topo_opt:  dict, (default: dict()) additional options that are given to 
                    tripcolor or tricontourf via the kwarg argument

        :topo_cont: bool, (default: False) overlay contour line plot of data

        :topoc_opt: dict, (default: dict()) additional options that are given to 
                    tricontour via the kwarg argument

        :topo_contl: bool, (default: False) label overlayed  contour linec plot

        :topocl_opt: dict, (default: dict()) additional options that are given to 
                    clabel via the kwarg argument

         ___plot lsmask______________________________________

        :do_lsm:    str, (default: 'fesom'), overlay FESOM grid inverted land sea mask
                    option are here:
                    
                    - fesom      ... grey fesom landsea mask
                    - stock      ... uses cartopy stock image
                    - bluemarble ... uses bluemarble image in folder tripyview/background/
                    - etopo      ... uses etopo image in folder tripyview/background/

        :lsm_opt:   dict, (default: dict()) additional options that are given to 
                    the landsea mask plotting via kwarg

        :lsm_res:   str, (default='low') resolution of bluemarble texture file either 'low' or 'high'

        ___plot cartopy gridlines____________________________

        :do_grid:   bool, (default: True) plot cartopy grid lines

        :grid_opt:  dict, (default: dict()) additional options that are given to 
                    the cartopy gridline plotting via kwarg

        :do_boundbox: bool, (default: True) plot cartopy black bounding box. If you 
                    make plots as a texture to use in blender, unity or pyvista the bounding box 
                    has to be removed with do_boundbox=False

        ___colorbar_________________________________________

        :cb_label:  str, (default: None) if string its used as colorbar label, otherwise 
                    information from data ('long_name, short_name) are used

        :cb_lunit:  str, (default: None) if string its used as colorbar unit label, 
                    otherwise info from data are used

        :cb_ltime:  str, (default: None) if string its used as colorbar time label, 
                    otherwise info from data are used

        :cb_ldep:   str, (default: None) if string its used as colorbar depth label, 
                    otherwise info from data are used                

        :cb_opt:    dict, (default: dict()) direct option for colorbar via kwarg

        :cbl_opt:   dict, (default: dict()) direct option for colorbar labels (fontsize, fontweight, ...) via kwarg

        :cbtl_opt:  dict, (default: dict()) direct option for colorbar tick labels (fontsize, fontweight, ...) via kwarg

        ___axes______________________________________

        :ax_title:  str, (default: 'descript') If 'descript' use descript attribute 
                    in data to title label axes, If 'str' use this string to label axes

        :ax_opt:    dict, (default: dict()) set option for axes arangement see subroutine 
                    do_axes_arrange

        :axl_opt:   dict, (default: dict()) set option for axes labels (fontsize, ...)

        ___enumerate_________________________________

        :do_enum:   bool, (default: False) do enumeration of axes with a), b), c) ...

        :enum_opt:  dict, (default: dict()) direct option for enumeration strings via kwarg 

        :enum_str:  list, (default: []) overwrite default enumeration strings        , 

        :enum_x:    float, (default: 0.005)  x position of enumeration string in axes coordinates

        :enum_y:    float, (default: 1.000)  y position of enumeration string in axes coordinates

        :enum_dir:  str, (default: 'lr')  direction of numbering, 'lr' from left to right, 'ud' from up to down

        ___save figure________________________________

        :do_save:   str, (default: None) if None figure will by not saved, if string figure 
                    will be saved, strings must give directory and filename  where to save.

        :save_dpi:  int, (default: 300) dpi resolution at which the figure is saved

        :save_opt:  dict, (default: dict()) direct option for saving via kwarg

        ___set output_________________________________

        :nargout:   list, (default: ['hfig', 'hax', 'hcb']) list of variables that are given 
                    out from the routine. 
                    Default: 
                    
                    - hfig ... figure handle
                    - hax  ... list of axes handle 
                    - hcb  ... list of colorbar handles (every variable that is defined in this subroutine can become output parameter)
    
    __________________________________________________
    
    Returns:
    
        hfig: returns figure handle 

        hax: returns list with axes handle 

        hcb: returns colorbar handle
    
    ____________________________________________________________________________
    """
    #___________________________________________________________________________
    # --> create box
    if box is None or box=="None": box = [ -180+mesh.focus, 180+mesh.focus, -90, 90 ]
    ts=clock.time()
    #___________________________________________________________________________
    # --> check if input data is a list
    if not isinstance(data, list): data = [data]
    ndat = len(data)
    
    #___________________________________________________________________________
    # --> create projection
    proj_to = None
    # proj is string 
    if   isinstance(proj, str): 
        proj_to, box = do_projection(mesh, proj, box)
    # proj is cartopy projection object
    elif isinstance(proj, ccrs.CRS): 
        proj_to = proj
    # proj is list of cartopy projection objects or string
    elif isinstance(proj, list): 
        proj_to = list()
        for proj_ii in proj:
            if   isinstance(proj_ii, str): 
                proj_dum, box = do_projection(mesh, proj_ii, box)
                proj_to.append(proj_dum)                
            elif isinstance(proj_ii, ccrs.CRS):    
                proj_to.append(proj_ii)    
    
    #___________________________________________________________________________
    # --> pre-arange axes
    ax_optdefault=dict({'projection': proj_to, 'box': box})
    ax_optdefault.update(ax_opt)
    hfig, hax, hcb, cb_plt_idx = do_axes_arrange(ncol, nrow, **ax_optdefault)
    cb_plt_idx=cb_plt_idx[:ndat]
    
    #___________________________________________________________________________
    # --> axes enumeration 
    do_axes_enum(hax[:ndat], do_enum, nrow, ncol, enum_opt=enum_opt, enum_str=enum_str, 
                       enum_x=enum_x, enum_y=enum_y, enum_dir=enum_dir)
    
    #___________________________________________________________________________
    # --> create mesh triangulation, when input data are unstructured
    tri =  do_triangulation(hax, mesh, proj_to, box, do_triorig=True, do_earea=True)
    
    #___________________________________________________________________________
    # --> set up color info
    # --> setup normalization log10, symetric log10, None
    cinfo_plot, norm_plot, idxs = list(), list(), 0
    for ii in np.unique(cb_plt_idx):
        idsel = np.where(cb_plt_idx==ii)[0]
        #_______________________________________________________________________
        if isinstance(cinfo, list):
            cinfo_plot.append( do_setupcinfo(cinfo[ii-1], [data[jj] for jj in idsel], do_rescale, mesh=mesh, tri=tri, do_vec=True) )
        else:    
            cinfo_plot.append( do_setupcinfo(cinfo, [data[jj] for jj in idsel], do_rescale, mesh=mesh, tri=tri, do_vec=True) )
        
        #_______________________________________________________________________
        if isinstance(do_rescale, list):
            norm_plot.append(  do_data_norm(cinfo_plot[-1], do_rescale[ii-1]) )
        else:
            norm_plot.append(  do_data_norm(cinfo_plot[-1], do_rescale) )
    
    #___________________________________________________________________________
    # --> loop over axes
    hp, hbot, hmsh, hlsm, hgrd, htop = list(), list(), list(), list(), list(), list()
    for ii, (hax_ii, hcb_ii) in enumerate(zip(hax, hcb)):
        # if there are no ddatra to fill axes, make it invisible 
        if ii>=ndat: 
            hax_ii.axis('off')
        # axis is normally fillt with data    
        else:
            ii_valid=ii
            ts = clock.time()
            
            #___________________________________________________________________
            # prepare unstructured data for plotting, augment periodic 
            # boundaries, interpolate from elements to nodes, kick out nan 
            # values from plotting that are bottom topo  
            vname = list(data[ii].keys())
            data_plot_u, data_plot_v = data[ii][ vname[0] ].data.copy(), data[ii][ vname[1] ].data.copy()
            data_plot_u, tri   = do_data_prepare_unstruct(mesh, tri, data_plot_u, do_ie2n)
            data_plot_v, tri   = do_data_prepare_unstruct(mesh, tri, data_plot_v, do_ie2n)
            
            #___________________________________________________________________
            # add color for ocean bottom
            h0 = do_plt_bot(hax_ii, do_bot, tri=tri, bot_opt=bot_opt)
            hbot.append(h0)
            
            #___________________________________________________________________
            # add grey topo
            h0 = do_plt_topo(hax_ii, do_topo, abs(mesh.n_z), mesh, cp.copy(tri), 
                             plt_opt=topo_opt,
                             plt_contb=topo_cont , pltcb_opt=topoc_opt,
                             plt_contl=topo_contl, pltcl_opt=topocl_opt)
            htop.append(h0)
            
            #___________________________________________________________________
            # add grid mesh on top
            h0 = do_plt_mesh(hax_ii, do_mesh, tri, mesh_opt=mesh_opt)
            hmsh.append(h0)
            
            #___________________________________________________________________
            # do quiver computations
            h0 = do_plt_quiver(hax_ii, do_quiv, tri, data_plot_u, data_plot_v, 
                               cinfo_plot[ cb_plt_idx[ii]-1 ], norm_plot[ cb_plt_idx[ii]-1 ], 
                               quiv_scalfac=quiv_scalfac, quiv_arrwidth=quiv_arrwidth, quiv_dens=quiv_dens,
                               quiv_smax=quiv_smax, quiv_shiftL=quiv_shiftL, 
                               quiv_smooth=quiv_smooth, quiv_opt=quiv_opt)
            hp.append(h0)
            
            #___________________________________________________________________
            # add mesh land-sea mask
            h0 = do_plt_lsmask(hax_ii, do_lsm, mesh, lsm_opt=lsm_opt, resolution=lsm_res)
            hlsm.append(h0)  
            
            #___________________________________________________________________
            # add grids lines 
            h0 = do_plt_gridlines(hax_ii, do_grid, box, ndat, grid_opt=grid_opt)
            hgrd.append(h0)
            
            #___________________________________________________________________
            # add title and axes labels
            if ax_title is not None: 
                # is title  string:
                if   isinstance(ax_title,str) : 
                    # if title string is 'descript' than use descript attribute from 
                    # data to set plot title 
                    if ax_title=='descript' and ('descript' in data[ii][ vname[0] ].attrs.keys() ):
                        hax_ii.set_title(data[ii][ vname[0] ].attrs['descript'], fontsize=hax_ii.fs_label, verticalalignment='top')
                        
                    else:
                        hax_ii.set_title(ax_title, fontsize=hax_ii.fs_label)
                # is title list of string        
                elif isinstance(ax_title,list): hax_ii.set_title(ax_title[ii], fontsize=hax_ii.fs_label)
            
            #___________________________________________________________________
            # remove platecaree bounding box when used as image alpha map or texture
            if not do_boundbox: hax_ii.spines['geo'].set_visible(False)
            
        #_______________________________________________________________________
        # add colorbar 
        if hcb_ii != 0 and hp[-1] is not None: 
            hcb_ii = do_cbar(hcb_ii, hax_ii, hp, data[ii_valid], cinfo_plot[cb_plt_idx[ii_valid]-1], do_rescale, 
                             cb_label, cb_lunit, cb_ltime, cb_ldep, cb_opt=cb_opt, cbl_opt=cbl_opt, cbtl_opt=cbtl_opt)
            
        #_______________________________________________________________________
        # hfig.canvas.draw()   
        
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, hfig, dpi=save_dpi, save_opt=save_opt)
    
    #___________________________________________________________________________
    list_argout=[]
    if len(nargout)>0:
        for stri in nargout:
            try   : list_argout.append(eval(stri))
            except: print(f" -warning-> variable {stri} was not found, could not be added as output argument") 
            
        return(list_argout)
    else:
        return
    

    
#
#
#_______________________________________________________________________________
# --> do plotting of horizontal slices
def plot_vslice(mesh                   , 
                data                   , 
                box        = None      , 
                box_idx    = None      ,
                box_label  = None      , 
                boxl_opt   = dict()    , # option for box label string 
                cinfo      = None      , # colormap info and defintion
                nrow       = 1         , # number of row in figures panel
                ncol       = 1         , # number of column in figure panel
                proj       = None      ,
                do_ie2n    = False     , # interpolate element data to vertices
                do_rescale = False     ,
                #--- data -----------
                do_plt     = 'tpc'     , # tpc:tripcolor, tcf:tricontourf
                plt_opt    = dict()    ,
                plt_contb  = False     , # plot background contour lines: True/False 
                pltcb_opt  = dict()    , # background contour line option
                plt_contf  = False     , # plot foreground contour lines: True/False 
                pltcf_opt  = dict()    , # foreground contour line option
                plt_contr  = False     , # plot reference contour lines: True/False 
                pltcr_opt  = dict()    , # reference contour line option
                plt_contl  = False     , # do contourline labels 
                pltcl_opt  = dict()    , # contour line label options
                #--- mesh -----------
                do_mesh    = False     , 
                mesh_opt   = dict()    , 
                #--- bottom mask ----
                do_bot     = True      , 
                bot_opt    = dict()    ,
                #--- landsea mask ---
                do_lsm     = 'fesom'   , 
                lsm_opt    = dict()    , 
                lsm_res    = 'low'     ,
                #--- gridlines ------
                do_grid    = True      , 
                grid_opt   = dict({'yexp':True})    ,
                #--- colorbar -------
                cb_label   = None      ,
                cb_lunit   = None      ,
                cb_ltime   = None      ,
                cb_ldep    = None      , 
                cb_opt     = dict()    , # colorbar option
                cbl_opt    = dict()    , # colorbar label option, fontsize ,...
                cbtl_opt   = dict()    , # colorbar ticklabel option, fontsize ,...
                #--- axes -----------
                ax_title   = 'descript',
                ax_opt     = dict()    , # dictionary that defines axes and colorbar arangement
                ax_xlim    = None      ,
                ax_ylim    = None      ,
                #--- enumerate axes -
                do_enum    = False     ,
                enum_opt   = dict({'horizontalalignment':'center'})    , 
                enum_str   = []        , 
                enum_x     = [0.000]   , 
                enum_y     = [1.005]    ,
                enum_dir   = 'lr'    ,# prescribed list of enumeration strings
                #--- save figure ----
                do_save    = None      , 
                save_dpi   = 300       ,
                save_opt   = dict()    ,
                #--- set output -----
                nargout=['hfig', 'hax', 'hcb'],
                ):
    """
    --> plot FESOM2 horizontal data slice:
    
    __________________________________________________
    
    Parameters: 
    
        :mesh:      fesom2 mesh object,  with all mesh information 

        :data:      xarray dataset object, or list of xarray dataset object

        :box:       None, list (default: None)  list with regional box limitation for index computation box can be: 

                    - ['global']   ... compute global index 
                    - [shp.Reader] ... index region defined by shapefile 
                    - [ [lonmin,lonmax,latmin, latmax], boxname] index region defined by rect box 
                    - [ [ [px1,px2...], [py1,py2,...]], boxname] index region defined by polygon
                    - [ np.array(2 x npts), boxname] index region defined by polygon

        :box_idx:   int, (default: None) index in boxlist 

        :box_label: str  (default: None) overwrites boxname string 

        :boxl_opt:  dict (default: dict() additional options for boxlabel string (fontsize...)

        :cinfo:     None, dict() (default: None), dictionary with colorbar information. 
                    Information that are given are used, others are computed. cinfo dictionary 
                    entries can be,
                    
                    - cinfo['cmin'], cinfo['cmax'], cinfo['cref'] ... scalar min, max, reference value
                    - cinfo['crange'] ... list with [cmin, cmax, cref] overrides scalar values 
                    - cinfo['cnum']   ... minimum number of colors
                    - cinfo['cstr']   ... name of colormap see in sub_colormap_c2c.py
                    - cinfo['cmap']   ... colormap object ('wbgyr', 'blue2red, 'jet' ...)
                    - cinfo['clevel'] ... color level array

        :nrow:      int, (default: 1) number of columns when plotting multiple data panels

        :ncol:      int, (default: 1) number of rows when plotting multiple data panels

        :proj:      str, (default: None) is choosen here autometically by data attribute
                    proj, can be setted also from hand to ...
                    - index+depth+xy    
                    - index+depth+time  
                    - zmoc   
                    - dmoc
                    - dmoc+depth
                    - dmoc+dens

        :do_ie2n:   bool, (default: False) do interpolation of data on elements towards nodes

        :do_rescale: bool, str, np.array (defaul: False) do scaling of colorbar
                    - False      ... scale data automatically scientifically by 10^x, for data data larger 10^3 and smaller 10^-3
                    - log10      ... do logaritmic scaling
                    - slog10     ... do symetric logarithmic scaling
                    - np.array() ... scale colorbar stepwise according to values in np.array allows also for non-linear colortick steps

        ___plot data parameters_____________________________

        :do_plt:    str, (default: tpc)
                    - tpc ... make pseudocolor plot (tripcolor)
                    - tcf ... make contourf color plot (tricontourf)

        :plt_opt:   dict, (default: dict()) additional options that are given to tripcolor 
                    or tricontourf via the kwarg argument

        :plt_contb: bool, (default: False) overlay thin contour lines of all colorbar steps (background)

        :pltcb_opt: dict, (default: dict()) background contour line option

        :plt_contf: bool, (default: False) overlay thicker contour lines of the main colorbar steps (foreground)

        :pltcf_opt: dict, (default: dict()) foreground contour line option

        :plt_contr: bool, (default: False) overlay thick contour lines of reference color steps (reference)

        :pltcr_opt: dict, (default: dict()) reference contour line option

        :plt_contl: bool, (default: False) label overlayed  contour linec plot

        :pltcl_opt: dict, (default: dict()) additional options that are given to clabel via the kwarg argument

        ___plot mesh________________________________________

        :do_mesh:   bool, (default: True), overlay FESOM grid over dataplot

        :mesh_opt:  dict, (default: dict()) additional options that are given to the mesh plotting via kwarg

        ___plot bottom mask_________________________________
        
        :do_bot:    bool, (default: True), overlay topographic bottom mask

        :bot_opt:   dict, (default: dict()) additional options that are given to the bottom mask plotting via kwarg
 
        ___plot cartopy gridlines____________________________

        :do_grid:   bool, (default: True) plot cartopy grid lines

        :grid_opt:  dict, (default: dict()) additional options that are given to 
                    the cartopy gridline plotting via kwarg

        ___colorbar_________________________________________

        :cb_label:  str, (default: None) if string its used as colorbar label, otherwise 
                    information from data ('long_name, short_name) are used

        :cb_lunit:  str, (default: None) if string its used as colorbar unit label, 
                    otherwise info from data are used

        :cb_ltime:  str, (default: None) if string its used as colorbar time label, 
                    otherwise info from data are used

        :cb_ldep:   str, (default: None) if string its used as colorbar depth label, 
                    otherwise info from data are used                

        :cb_opt:    dict, (default: dict()) direct option for colorbar via kwarg

        :cbl_opt:   dict, (default: dict()) direct option for colorbar labels (fontsize, fontweight, ...) via kwarg

        :cbtl_opt:  dict, (default: dict()) direct option for colorbar tick labels (fontsize, fontweight, ...) via kwarg

        ___axes______________________________________

        :ax_title:  str, (default: 'descript') If 'descript' use descript attribute 
                    in data to title label axes, If 'str' use this string to label axes

        :ax_opt:    dict, (default: dict()) set option for axes arangement see subroutine 
                    do_axes_arrange

        :axl_opt:   dict, (default: dict()) set option for axes labels (fontsize, ...)

        ___enumerate_________________________________

        :do_enum:   bool, (default: False) do enumeration of axes with a), b), c) ...

        :enum_opt:  dict, (default: dict()) direct option for enumeration strings via kwarg 

        :enum_str:  list, (default: []) overwrite default enumeration strings        , 

        :enum_x:    float, (default: 0.005)  x position of enumeration string in axes coordinates

        :enum_y:    float, (default: 1.000)  y position of enumeration string in axes coordinates

        :enum_dir:  str, (default: 'lr')  direction of numbering, 'lr' from left to right, 'ud' from up to down

        ___save figure________________________________

        :do_save:   str, (default: None) if None figure will by not saved, if string figure 
                    will be saved, strings must give directory and filename  where to save.

        :save_dpi:  int, (default: 300) dpi resolution at which the figure is saved

        :save_opt:  dict, (default: dict()) direct option for saving via kwarg

    
        ___set output_________________________________

        :nargout:   list, (default: ['hfig', 'hax', 'hcb']) list of variables that are given 
                    out from the routine. 
                    Default: 
                    
                    - hfig ... figure handle
                    - hax  ... list of axes handle 
                    - hcb  ... list of colorbar handles (every variable that is defined in this subroutine can become output parameter)
    
    __________________________________________________
    
    Returns:
    
        :hfig: returns figure handle 
        
        :hax: returns list with axes handle 
        
        :hcb: returns colorbar handle
    
    ____________________________________________________________________________
    """
    #___________________________________________________________________________
    # --> check if input data is a list
    if not isinstance(data, list): data = [data]
    ndat = len(data)
    
    #___________________________________________________________________________
    # check vertical plotting mode if index+depth+xy, zmoc, dmoc
    if proj is None:
        if isinstance(data[0], xr.Dataset):
            if 'proj' in data[0].attrs: proj=data[0].attrs['proj']
        elif isinstance(data[0], list):
            if isinstance(data[0][box_idx], xr.Dataset):
                if 'proj' in data[0][box_idx].attrs: proj=data[0][box_idx].attrs['proj']
    
    #___________________________________________________________________________
    # --> create projection
    proj_to = None
    # proj is string 
    if   isinstance(proj, str): 
        proj_to, box = do_projection(mesh, proj, box)
    # proj is cartopy projection object
    elif isinstance(proj, ccrs.CRS): 
        proj_to = proj
    # proj is list of cartopy projection objects or string
    elif isinstance(proj, list): 
        proj_to = list()
        for proj_ii in proj:
            if   isinstance(proj_ii, str): 
                proj_dum, box = do_projection(mesh, proj_ii, box)
                proj_to.append(proj_dum)                
            elif isinstance(proj_ii, ccrs.CRS):    
                proj_to.append(proj_ii)    
    
    #___________________________________________________________________________
    # --> pre-arange axes
    ax_optdefault=dict({'projection': proj_to})
    ax_optdefault.update(ax_opt)
    #print(ncol, nrow)
    
    hfig, hax, hcb, cb_plt_idx = do_axes_arrange(ncol, nrow, **ax_optdefault)
    cb_plt_idx=cb_plt_idx[:ndat]
    
    #___________________________________________________________________________
    # --> axes enumeration 
    do_axes_enum(hax[:ndat], do_enum, nrow, ncol, enum_opt=enum_opt, enum_str=enum_str, 
                       enum_x=enum_x, enum_y=enum_y, enum_dir=enum_dir)
        
    #___________________________________________________________________________
    # --> set up color info
    # --> setup normalization log10, symetric log10, None
    cinfo_plot, norm_plot, idxs = list(), list(), 0
    for ii in np.unique(cb_plt_idx):
        idsel = np.where(cb_plt_idx==ii)[0]
        
        #_______________________________________________________________________
        # setup my own colormap definition dictionary
        cinfo_optdefault=dict()
        if   hax[0].projection in ['index+depth+xy', 'index+depth+time']:
            cinfo_optdefault.update({'do_index':True, 'box_idx':box_idx})
        elif hax[0].projection == 'zmoc' :
            cinfo_optdefault.update({'do_moc':True})
        elif hax[0].projection == 'dmoc+depth' or hax[0].projection == 'dmoc+dens':
            cinfo_optdefault.update({'do_dmoc':True})
            
        if isinstance(cinfo, list):
            cinfo_plot.append( do_setupcinfo(cinfo[ii-1], [data[jj] for jj in idsel], do_rescale, **cinfo_optdefault) )
        else:    
            cinfo_plot.append( do_setupcinfo(cinfo, [data[jj] for jj in idsel], do_rescale, **cinfo_optdefault) )
        
        #_______________________________________________________________________
        # setup matplotlib rescaling options for, log10, slog10, or predefined array
        if isinstance(do_rescale, list):
            norm_plot.append(  do_data_norm(cinfo_plot[-1], do_rescale[ii-1]) )
        else:
            norm_plot.append(  do_data_norm(cinfo_plot[-1], do_rescale) )
        
    #___________________________________________________________________________
    # --> loop over axes
    hp, hbot, hmsh, hlsm, hgrd = list(), list(), list(), list(), list()
    count_cb = 0
    for ii, (hax_ii, hcb_ii) in enumerate(zip(hax, hcb)):
        # if there are no ddatra to fill axes, make it invisible 
        if ii>=ndat: 
            hax_ii.axis('off')
        elif data[ii] is None: 
            hax_ii.axis('off')        
        # axis is normally fillt with data    
        else: 
            ii_valid=ii
            #___________________________________________________________________
            # prepare regular gridded data for plotting
            data_x, data_y, data_plot = do_data_prepare_vslice(hax_ii, data[ii], box_idx)
           
            #___________________________________________________________________
            # add tripcolor or tricontourf plot 
            h0 = do_plt_datareg(hax_ii, do_plt, data_x, data_y, data_plot, 
                                cinfo_plot[ cb_plt_idx[ii]-1 ], norm_plot[ cb_plt_idx[ii]-1 ], 
                                plt_opt  =plt_opt,
                                plt_contb=plt_contb, pltcb_opt=pltcb_opt,
                                plt_contf=plt_contf, pltcf_opt=pltcf_opt,
                                plt_contr=plt_contr, pltcr_opt=pltcr_opt,
                                plt_contl=plt_contl, pltcl_opt=pltcl_opt)
            hp.append(h0)
            
            ##__________________________________________________________________
            ## add bottom  mask
            ax_xlim0, ax_ylim0 = ax_xlim, ax_ylim
            if   hax_ii.projection=='index+depth+xy':
                h0 = do_plt_bot(hax_ii, do_bot, data_x=data_x, data_y=data_y, 
                                data_plot=data_plot, bot_opt=bot_opt)
        
            # zmoc bottom patch
            elif hax_ii.projection=='zmoc':
                ax_ylim0 = [0, abs(mesh.zlev[-1])]
                h0 = do_plt_bot(hax_ii, do_bot, data_x=data_x, data_y=data_y, 
                                data_plot=data[ii]['botmax'].values, ylim=ax_ylim0, 
                                bot_opt=bot_opt) 
                
            # dmoc when doeing z-coordinate projection bottom patch                   
            elif 'dmoc' in hax_ii.projection and \
                ('ndens_zfh' in data[ii].coords or 'nz_rho' in data[ii].coords or'ndens_z' in data[ii].coords) :
                ax_ylim0 = [0, abs(mesh.zlev[-1])]
                h0 = do_plt_bot(hax_ii, do_bot, data_x=data[ii]['lat'].values, data_y=data_y, 
                                data_plot=data[ii]['botmax'].values, ylim=ax_ylim0, 
                                bot_opt=bot_opt)    
            hbot.append(h0)
            
            #___________________________________________________________________
            # add grids lines 
            h0 = do_plt_gridlines(hax_ii, do_grid, box, ndat, data_x=data_x, 
                                  data_y=data_y, xlim=ax_xlim0, ylim=ax_ylim0, 
                                  grid_opt=grid_opt)
            hgrd.append(h0)
            
            #___________________________________________________________________
            # add title and axes labels
            if ax_title is not None: 
                # is title  string:
                if   isinstance(ax_title,str) : 
                    # if title string is 'descript' than use descript attribute from 
                    # data to set plot title 
                    if box_idx is not None: 
                        vname     = list(data[ii][box_idx].keys())[0]
                        loc_attrs = data[ii][box_idx][vname].attrs
                    else:
                        vname     = list(data[ii].keys())[0] 
                        loc_attrs = data[ii][vname].attrs
                    
                    if ax_title=='descript' and ('descript' in loc_attrs.keys() ):
                        hax_ii.set_title(loc_attrs['descript'], fontsize=hax_ii.fs_label, verticalalignment='top')
                    else:
                        hax_ii.set_title(ax_title, fontsize=hax_ii.fs_label)
                        
                # is title list of string        
                elif isinstance(ax_title,list): hax_ii.set_title(ax_title[ii], fontsize=hax_ii.fs_label)
            
            #___________________________________________________________________
            # set superior title
            boxl_optdefault = dict({'x':0.99, 'y':0.99, 's':'', \
                                    'fontsize':12, 'fontweight':'bold', 'transform':hax_ii.transAxes,\
                                    'horizontalalignment':'right', 'verticalalignment':'top', 'zorder':5})                
            # print transect labels
            if  box_idx is not None:
                vname     = list(data[ii][box_idx].keys())[0]
                loc_attrs = data[ii][box_idx][vname].attrs
                if   'transect_name' in loc_attrs: boxl_optdefault.update({'s':loc_attrs['transect_name']})
                elif 'boxname'       in loc_attrs: boxl_optdefault.update({'s':loc_attrs['boxname']})
            
            # print precribed box_label
            elif isinstance(box_label, str):
                boxl_optdefault.update({'s':box_label})
            
            # print list of prescribed box_label
            elif isinstance(box_label, list):
                boxl_optdefault.update({'s':box_label[ii]}) 
            
            # print moc labels
            elif hax_ii.projection in ['zmoc', 'dmoc+depth', 'dmoc+dens']:
                vname     = list(data[ii].keys())[0]
                loc_attrs = data[ii][vname].attrs
                if 'short_name' in loc_attrs: boxl_optdefault.update({'s':loc_attrs['short_name']})
                boxl_optdefault.update({'x':0.01, 'y':0.01, 'horizontalalignment':'left', 'verticalalignment':'bottom'})
                
            boxl_optdefault.update(boxl_opt)
            ht = hax_ii.text(**boxl_optdefault)
            
        #_______________________________________________________________________
        # add colorbar 
        
        if hcb_ii != 0 and hp[-1] is not None: 
            if isinstance(cb_label,list): cb_label2 = cb_label[count_cb]
            else: cb_label2 = cb_label
            hcb_ii = do_cbar(hcb_ii, hax_ii, hp, data[ii_valid], cinfo_plot[cb_plt_idx[ii_valid]-1], do_rescale, 
                             cb_label2, cb_lunit, cb_ltime, cb_ldep, norm=norm_plot[ cb_plt_idx[ii_valid]-1 ], 
                             box_idx=box_idx, cb_opt=cb_opt, cbl_opt=cbl_opt, cbtl_opt=cbtl_opt)
            
            count_cb=count_cb+1
        #_______________________________________________________________________
        # hfig.canvas.draw()   
        
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, hfig, dpi=save_dpi, save_opt=save_opt)
    plt.show()
    
    #___________________________________________________________________________
    list_argout=[]
    if len(nargout)>0:
        for stri in nargout:
            try   : list_argout.append(eval(stri))
            except: print(f" -warning-> variable {stri} was not found, could not be added as output argument") 
            
        return(list_argout)
    else:
        return
 
 

#
#
#_______________________________________________________________________________
# --> do plotting of horizontal lines over index region (e.g. heatflux vs lon, lat)
def plot_hline(data                   , 
                box        = None      , 
                box_idx    = None      ,
                box_label  = None      , 
                boxl_opt   = dict()    , # option for box label string 
                nrow       = 1         , # number of row in figures panel
                ncol       = 1         , # number of column in figure panel
                proj       = 'index+xy',
                n_cycl     = None      ,
                do_allcycl = False     , 
                do_shdw    = True      ,
                do_mean    = False     ,
                do_rescale = False     ,
                #--- data -----------
                plt_opt    = dict()    ,
                mark_opt   = dict()    ,
                #--- gridlines ------
                do_grid    = True      , 
                grid_opt   = dict()    ,
                #--- axes -----------
                ax_title   = 'descript',
                ax_xlabel  = None      ,
                ax_ylabel  = None      ,
                ax_xunit   = None      ,
                ax_yunit   = None      ,
                ax_opt     = dict()    , # dictionary that defines axes and colorbar arangement
                ax_xlim    = None      ,
                ax_ylim    = None      ,
                #--- enumerate axes -
                do_enum    = False     ,
                enum_opt   = dict({'horizontalalignment':'center'})    , 
                enum_str   = []        , 
                enum_x     = [0.000]   , 
                enum_y     = [1.005]    ,
                enum_dir   = 'lr'    ,# prescribed list of enumeration strings
                #--- save figure ----
                do_save    = None      , 
                save_dpi   = 300       ,
                save_opt   = dict()    ,
                #--- set output -----
                nargout=['hfig', 'hax'],
                ):
    """
    --> do plotting of horizontal lines over index region (e.g. heatflux vs lon, lat)
    
    __________________________________________________
    
    Parameters:

        :data:      xarray dataset object, or list of xarray dataset object

        :box:       None, list (default: None)  list with regional box limitation for index computation box can be: 

                    - ['global']   ... compute global index 
                    - [shp.Reader] ... index region defined by shapefile 
                    - [ [lonmin,lonmax,latmin, latmax], boxname] index region defined by rect box 
                    - [ [ [px1,px2...], [py1,py2,...]], boxname] index region defined by polygon
                    - [ np.array(2 x npts) boxname] index region defined by polygon

        :box_idx:   int, (default: None) index in boxlist 

        :box_label: str  (default: None) overwrites boxname string 

        :boxl_opt:  dict (default: dict() additional options for boxlabel string (fontsize...)
        
        :nrow:      int, (default: 1) number of columns when plotting multiple data panels

        :ncol:      int, (default: 1) number of rows when plotting multiple data panels

        :proj:      str, (default: 'index+xy')
                        
        :n_cycl:    int, (default: None) How many spinup cycles are contained
                    in the data_list. Info is needed when do_allcycl=True,

        :do_allcycl: bool, (default: False) plot all spinupcycles based on colormap value 

        :do_shdw:   bool, (default: True) give lines a black outline

        :do_mean:   bool, (default: False) plot triangle on the left side that indicates 
                    the mean value

        :do_rescale: bool, str, np.array (defaul: False) do scaling of colorbar
                    - False      ... scale data automatically scientifically by 10^x, for data data larger 10^3 and smaller 10^-3
                    - log10      ... do logaritmic scaling
                    - slog10     ... do symetric logarithmic scaling

        ___plot data parameters_____________________________

        :plt_opt:   dict, (default: dict()) additional options that are given to 
                    line plot routine via the kwarg argument

        :mark_opt:  dict, (default: dict()) additional options that are given to 
                    control the line markers via the kwarg argument

        ___plot gridlines___________________________________

        :do_grid:   bool, (default: True) plot cartopy grid lines

        :grid_opt:  dict, (default: dict()) additional options that are given to 
                    the cartopy gridline plotting via kwarg
                    

        ___axes______________________________________

        :ax_title:  str, (default: 'descript') If 'descript' use descript attribute 
                    in data to title label axes, If 'str' use this string to label axes

        :ax_xlabel: str, (default: None) overwrites default xlabel

        :ax_ylabel: str, (default: None) overwrites default ylabel

        :ax_xunit:  str, (default: None) overwrites default xlabel unit

        :ax_yunit:  str, (default: None) overwrites default ylabel unit

        :ax_opt:    dict, (default: dict()) set option for axes arangement see subroutine do_axes_arrange

        :ax_xlim:   list (default: None) overright data related xlimits

        :ax_ylim:   list (default: None) overright data related ylimits  

        ___enumerate_________________________________

        :do_enum:   bool, (default: False) do enumeration of axes with a), b), c) ...

        :enum_opt:  dict, (default: dict()) direct option for enumeration strings via kwarg 

        :enum_str:  list, (default: []) overwrite default enumeration strings

        :enum_x:    float, (default: 0.005)  x position of enumeration string in axes coordinates

        :enum_y:    float, (default: 1.000)  y position of enumeration string in axes coordinates

        :enum_dir:  str, (default: 'lr')  direction of numbering, 'lr' from left to right, 'ud' from up to down

        ___save figure________________________________

        :do_save:   str, (default: None) if None figure will by not saved, if string figure 
                    will be saved, strings must give directory and filename  where to save.

        :save_dpi:  int, (default: 300) dpi resolution at which the figure is saved

        :save_opt:  dict, (default: dict()) direct option for saving via kwarg

        ___set output_________________________________

        :nargout:   list, (default: ['hfig', 'hax', 'hcb']) list of variables that are given 
                    out from the routine. 
                    Default: 
                    
                    - hfig ... figure handle
                    - hax  ... list of axes handle 
                    - (every variable that is defined in this subroutine can become output parameter)
    
    __________________________________________________
    
    Returns:
    
        :hfig: returns figure handle 

        :hax: returns list with axes handle 
    
    ____________________________________________________________________________
    """
    #___________________________________________________________________________
    # --> check if input data is a list
    if not isinstance(data, list): data = [data]
    # keep in mind ndat is here the number index boxes in thre list, not the number 
    # of datas that are loaded
    nbox = len(data[0])
    ndat = len(data)
    
    #___________________________________________________________________________
    # check vertical plotting mode if index+depth+xy, zmoc, dmoc
    if proj is None:
        if isinstance(data[0], xr.Dataset):
            if 'proj' in data[0].attrs: proj=data[0].attrs['proj']
        elif isinstance(data[0], list):
            if isinstance(data[0][0], xr.Dataset):
                if 'proj' in data[0][0].attrs: proj=data[0][0].attrs['proj']
    
    #___________________________________________________________________________
    # --> create projection
    proj_to = None
    # proj is string 
    if   isinstance(proj, str): 
        proj_to, box = do_projection(None, proj, box)
    # proj is cartopy projection object
    elif isinstance(proj, ccrs.CRS): 
        proj_to = proj
    # proj is list of cartopy projection objects or string
    elif isinstance(proj, list): 
        proj_to = list()
        for proj_ii in proj:
            if   isinstance(proj_ii, str): 
                proj_dum, box = do_projection(None, proj_ii, box)
                proj_to.append(proj_dum)                
            elif isinstance(proj_ii, ccrs.CRS):    
                proj_to.append(proj_ii)    
    
    #___________________________________________________________________________
    # --> pre-arange axes
    ax_optdefault=dict({'projection': proj_to, 'ax_sharex':False, 'ax_sharey':False, 'ax_dt':1.75})
    ax_optdefault.update(ax_opt)
    hfig, hax, hcb, cb_plt_idx = do_axes_arrange(ncol, nrow, **ax_optdefault)
    cb_plt_idx=cb_plt_idx[:nbox]
    
    #___________________________________________________________________________
    # --> axes enumeration 
    do_axes_enum(hax[:nbox], do_enum, nrow, ncol, enum_opt=enum_opt, enum_str=enum_str, 
                       enum_x=enum_x, enum_y=enum_y, enum_dir=enum_dir)
    
    #___________________________________________________________________________
    # setup colormap
    if do_allcycl: 
        if n_cycl is not None:
            cmap = categorical_cmap(np.int32(ndat/n_cycl), n_cycl, cmap="tab10")
        else:
            cmap = categorical_cmap(ndat, 1, cmap="tab10")
    else:
        cmap = categorical_cmap(ndat, 1, cmap="tab10")
        
    #___________________________________________________________________________
    # --> loop over axes
    hp, hbot, hmsh, hlsm, hgrd = list(), list(), list(), list(), list()
    list_strdatalabel,list_strboxlabel = list(), list()
    for ii, (hax_ii, hcb_ii) in enumerate(zip(hax, hcb)):
        # if there are no ddatra to fill axes, make it invisible 
        if ii>=nbox: 
            hax_ii.axis('off')
        # axis is normally fillt with data    
        else: 
            #___________________________________________________________________
            # prepare regular gridded data for plotting
            # prepare regular gridded data for plotting
            optline = dict({'linewidth':1.5, 'marker':'None', 'markerfacecolor':'w', 'markersize':5, 'zorder':2})
            optmark = dict({'markersize':8, 'markeredgecolor':'k', 'markeredgewidth':0.5,
                            'clip_on':False, 'zorder':3}) #'clip_box':False,
            list_lstyle=['solid', 'dashed', 'dotted', 'dashdot', 'dashdotdotted']
            
            #___________________________________________________________________
            allinone = False
            if   nrow*ncol == 1:
                if   nbox == 1: box_idx = [0]
                elif nbox >  1: 
                    box_idx, allinone = list(range(0,nbox)), True
                    
            elif nrow*ncol > 1 and nrow*ncol <= nbox:   
                box_idx, allinone = [ii], False
            xmin, xmax, ymin, ymax = np.inf, -np.inf, np.inf, -np.inf
            
            #___________________________________________________________________
            # If nrow*ncol=1    && nbox > 1: plot all box lines in one figure panel
            #    nrow*ncol=nbox && nbox > 1: plot each box lines in separate figure panel
            for bi in box_idx:
            
                #___________________________________________________________________
                cnt, cnt_cycl = 0, 0
                if not allinone: xmin, xmax, ymin, ymax = np.inf, -np.inf, np.inf, -np.inf
                for jj in range(0,ndat):
                    #_______________________________________________________________
                    vname = list(data[jj][bi].data_vars)[0]
                    data_y = data[jj][bi][vname].data.copy()
                    ymin, ymax = np.min([ymin, data_y.min()]), np.max([ymax, data_y.max()])
                    
                    #_______________________________________________________________
                    if   'lat' in data[jj][bi]: 
                        data_x     =  data[jj][bi]['lat'].values 
                        str_xlabel = 'Latitude' if ax_xlabel is None else ax_xlabel
                    elif 'lon' in data[jj][bi]: 
                        data_x     = data[jj][bi]['lon'].values 
                        str_xlabel = 'Longitude' if ax_xlabel is None else ax_xlabel 
                    if ax_xunit is None: str_xlabel = str_xlabel + ' / deg'
                    else               : str_xlabel = str_xlabel + ' / ' + ax_xunit
                    xmin, xmax = np.min([xmin, data_x.min()]), np.max([xmax, data_x.max()])
                    
                    #_______________________________________________________________
                    loc_attrs= data[jj][bi][vname].attrs
                    str_ylabel, str_llabel, str_blabel = '', '', ''
                    if ax_ylabel is None:
                        if   'long_name'     in loc_attrs: str_ylabel = str_ylabel + loc_attrs['long_name']
                        elif 'short_name'    in loc_attrs: str_ylabel = str_ylabel + loc_attrs['short_name']
                    else: str_ylabel = ax_ylabel
                    if ax_yunit is None:
                            if 'units' in loc_attrs: str_ylabel = str_ylabel+' / '+loc_attrs['units']
                    else: str_ylabel = str_ylabel+' / '+ ax_yunit
                        
                    if   'descript'      in loc_attrs: str_llabel = str_llabel + loc_attrs['descript']
                    if   'boxname'       in loc_attrs: str_blabel = str_blabel + loc_attrs['boxname']
                    if   'transect_name' in loc_attrs: str_blabel = str_blabel + loc_attrs['transect_name']    
                    str_blabel = str_blabel.replace('MOC','').replace('_','')
                    if bi==0: list_strdatalabel.append(str_llabel)
                    if jj==0: list_strboxlabel.append(str_blabel)
                    
                    #_______________________________________________________________
                    # plot lines 
                    optline.update({'color':cmap.colors[cnt,:]})
                    optline.update(plt_opt)
                    if allinone and nrow*ncol==1: optline.update({'linestyle':list_lstyle[bi]})
                    optmark.update({'color':cmap.colors[cnt,:]})
                    optmark.update(mark_opt)
                    
                    # plot black underlying shadow slightly wider than line on top 
                    if do_shdw:
                        optline2 = optline.copy()
                        optline2.update({'color':'k', 'linewidth':optline['linewidth']*1.2, 'zorder':1})
                        hax_ii.plot(data_x, data_y, **optline2)
                    
                    # plot actual line with color 
                    h0 = hax_ii.plot(data_x, data_y, label=str_llabel, **optline)
                    hp.append(h0)
                    
                    # plot mean value with left triangle 
                    if do_mean: 
                        hax_ii.plot(xmin-(data_x[-1]-data_x[0])*0.00, data_y.mean(), marker='<', **optmark)
                    
                    #_______________________________________________________________
                    cnt      = cnt+1
                    cnt_cycl = cnt_cycl+1
                    if n_cycl is not None:
                        if cnt_cycl>= n_cycl: cnt_cycl=0
                
            #___________________________________________________________________
            if hax_ii.do_xlabel: hax_ii.set_xlabel(str_xlabel)
            if hax_ii.do_ylabel: hax_ii.set_ylabel(str_ylabel)
            
            #___________________________________________________________________
            # add grids lines 
            ax_xlim0, ax_ylim0 = ax_xlim, ax_ylim 
            if ax_xlim is None: ax_xlim0=[xmin,xmax]
            if ax_ylim is None: ax_ylim0=[ymin,ymax]
            h0 = do_plt_gridlines(hax_ii, do_grid, None, None, data_x=None, 
                                  data_y=data_y, xlim=ax_xlim0, ylim=ax_ylim0, 
                                  grid_opt=grid_opt, do_rescale=do_rescale)
            hgrd.append(h0)
            
            #_______________________________________________________________________
            # do legend and legend positioning 
            # -->bbox_to_anchor=(1.0, 1.0), loc='upper left' this should make sure 
            #    that the legend is always outside in the upper right corner 
            if hax_ii.coli==ncol-1 and hax_ii.rowi==0 : 
                if allinone and nbox>1:
                    # make data legend:
                    legend1 = plt.legend([hp[i][0] for i in range(0,ndat,1)], list_strdatalabel, 
                                         frameon=True, fancybox=True, shadow=True, fontsize=10, ncol=1,
                                         labelspacing=0.5, bbox_to_anchor=(1.0, 0.0), loc='lower right')
                    hax_ii.add_artist(legend1)
                    
                    # make box list legend:
                    import copy as cp
                    lines = [cp.copy(hp[i][0]) for i in range(0,ndat*nbox, ndat)]
                    # make legend box lines always black, just show linestyle 
                    for li in range(0,len(lines)): lines[li].set(color='k')
                    legend2 = plt.legend(lines, list_strboxlabel, 
                                         frameon=True, fancybox=True, shadow=True, fontsize=10, ncol=1,
                                         labelspacing=0.5, bbox_to_anchor=(0.0, 1.0), loc='upper left')
                    hax_ii.add_artist(legend2)
                else:
                    # make data legend:
                    hax_ii.legend(frameon=True, fancybox=True, shadow=True, fontsize=10, ncol=1,
                                labelspacing=0.5, bbox_to_anchor=(1.0, 1.0), loc='upper left') #bbox_to_anchor=(1.5, 1.5))
                    # box label becomes here axes title
            #___________________________________________________________________
            # add title and axes labels
            if ax_title is not None: 
                if isinstance(ax_title,list): hax_ii.set_title(ax_title[ii], fontsize=hax_ii.fs_label)
                else                        : 
                    if not allinone and nbox>1: hax_ii.set_title(str_blabel, fontsize=hax_ii.fs_label)
                    
         
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, hfig, dpi=save_dpi, save_opt=save_opt)
    
    #___________________________________________________________________________
    list_argout=[]
    if len(nargout)>0:
        for stri in nargout:
            try   : list_argout.append(eval(stri))
            except: print(f" -warning-> variable {stri} was not found, could not be added as output argument") 
            
        return(list_argout)
    else:
        return
    

    
#
#
#_______________________________________________________________________________
# --> do plotting of mean indices over depth (e.g. vertical profiles)
def plot_vline(data                   , 
                box        = None      , 
                box_idx    = None      ,
                box_label  = None      , 
                boxl_opt   = dict()    , # option for box label string 
                cinfo      = None      , # colormap info and defintion
                nrow       = 1         , # number of row in figures panel
                ncol       = 1         , # number of column in figure panel
                proj       = 'index+depth',
                n_cycl     = None      ,
                do_allcycl = False     , 
                do_shdw    = True      ,
                do_mean    = False     ,
                do_rescale = False     ,
                #--- data -----------
                plt_opt    = dict()    ,
                mark_opt   = dict()    ,
                #--- gridlines ------
                do_grid    = True      , 
                grid_opt   = dict({'yexp':True})    ,
                #--- axes -----------
                ax_title   = 'descript',
                ax_xlabel  = None      ,
                ax_ylabel  = None      ,
                ax_xunit   = None      ,
                ax_yunit   = None      ,
                ax_opt     = dict()    , # dictionary that defines axes and colorbar arangement
                ax_xlim    = None      ,
                ax_ylim    = None      ,
                #--- enumerate axes -
                do_enum    = False     ,
                enum_opt   = dict({'horizontalalignment':'center'})    , 
                enum_str   = []        , 
                enum_x     = [0.000]   , 
                enum_y     = [1.005]    ,
                enum_dir   = 'lr'    ,# prescribed list of enumeration strings
                #--- save figure ----
                do_save    = None      , 
                save_dpi   = 300       ,
                save_opt   = dict()    ,
                #--- set output -----
                nargout=['hfig', 'hax'],
                ):
    """
    --> do plotting of mean indices over depth (e.g. vertical profiles)
    
    __________________________________________________
    
    Parameters:

        :data:      xarray dataset object, or list of xarray dataset object

        :box:       None, list (default: None)  list with regional box limitation for index computation box can be: 

                    - ['global']   ... compute global index 
                    - [shp.Reader] ... index region defined by shapefile 
                    - [ [lonmin,lonmax,latmin, latmax], boxname] index region defined by rect box 
                    - [ [ [px1,px2...], [py1,py2,...]], boxname] index region defined by polygon
                    - [ np.array(2 x npts) boxname] index region defined by polygon

        :box_idx:   int, (default: None) index in boxlist 

        :box_label: str, (default: None) overwrites boxname string 

        :boxl_opt:  dict, (default: dict() additional options for boxlabel string (fontsize...)
        
        :nrow:      int, (default: 1) number of columns when plotting multiple data panels

        :ncol:      int, (default: 1) number of rows when plotting multiple data panels

        :proj:      str, (default: 'index+xy')

        :n_cycl:    int, (default: None) How many spinup cycles are contained
                    in the data_list. Info is needed when do_allcycl=True,

        :do_allcycl: bool, (default: False) plot all spinupcycles based on colormap value 

        :do_shdw:   bool, (default: True) give lines a black outline

        :do_mean:   bool, (default: False) plot triangle on the left side that indicates 
                    the mean value

        :do_rescale: bool, str, np.array (defaul: False) do scaling of colorbar
                    - False      ... scale data automatically scientifically by 10^x, for data data larger 10^3 and smaller 10^-3
                    - log10      ... do logaritmic scaling
                    - slog10     ... do symetric logarithmic scaling

         ___plot data parameters_____________________________

        :plt_opt:   dict, (default: dict()) additional options that are given to 
                    line plot routine via the kwarg argument

        :mark_opt:  dict, (default: dict()) additional options that are given to 
                    control the line markers via the kwarg argument

        ___plot gridlines___________________________________
    
        :do_grid:   bool, (default: True) plot cartopy grid lines

        :grid_opt:  dict, (default: dict()) additional options that are given to 
                    the cartopy gridline plotting via kwarg
                    

        ___axes______________________________________

        :ax_title:  str, (default: 'descript') If 'descript' use descript attribute 
                    in data to title label axes, If 'str' use this string to label axes

        :ax_xlabel: str, (default: None) overwrites default xlabel

        :ax_ylabel: str, (default: None) overwrites default ylabel

        :ax_xunit:  str, (default: None) overwrites default xlabel unit

        :ax_yunit:  str, (default: None) overwrites default ylabel unit

        :ax_opt:    dict, (default: dict()) set option for axes arangement see subroutine do_axes_arrange

        :ax_xlim:   list (default: None) overright data related xlimits

        :ax_ylim:   list (default: None) overright data related ylimits  

        ___enumerate_________________________________

        :do_enum:   bool, (default: False) do enumeration of axes with a), b), c) ...

        :enum_opt:  dict, (default: dict()) direct option for enumeration strings via kwarg 

        :enum_str:  list, (default: []) overwrite default enumeration strings

        :enum_x:    float, (default: 0.005)  x position of enumeration string in axes coordinates

        :enum_y:    float, (default: 1.000)  y position of enumeration string in axes coordinates

        :enum_dir:  str, (default: 'lr')  direction of numbering, 'lr' from left to right, 'ud' from up to down

        ___save figure________________________________

        :do_save:   str, (default: None) if None figure will by not saved, if string figure 
                    will be saved, strings must give directory and filename  where to save.

        :save_dpi:  int, (default: 300) dpi resolution at which the figure is saved

        :save_opt:  dict, (default: dict()) direct option for saving via kwarg

    
        ___set output_________________________________

        nargout:    list, (default: ['hfig', 'hax', 'hcb']) list of variables that are given 
                    out from the routine. 
                    Default: 
                    
                    - hfig ... figure handle
                    - hax  ... list of axes handle 
                    - (every variable that is defined in this subroutine can become output parameter)
    
    __________________________________________________
    
    Returns:
    
        :hfig: returns figure handle 

        :hax: returns list with axes handle 
    
    ____________________________________________________________________________
    """
    #___________________________________________________________________________
    # --> check if input data is a list
    if not isinstance(data, list): data = [data]
    # keep in mind ndat is here the number index boxes in thre list, not the number 
    # of datas that are loaded
    nbox = len(data[0])
    ndat = len(data)
    
    #___________________________________________________________________________
    # check vertical plotting mode if index+depth+xy, zmoc, dmoc
    if proj is None:
        if isinstance(data[0], xr.Dataset):
            if 'proj' in data[0].attrs: proj=data[0].attrs['proj']
        elif isinstance(data[0], list):
            if isinstance(data[0][0], xr.Dataset):
                if 'proj' in data[0][0].attrs: proj=data[0][0].attrs['proj']
    
    #___________________________________________________________________________
    # --> create projection
    proj_to = None
    # proj is string 
    if   isinstance(proj, str): 
        proj_to, box = do_projection(None, proj, box)
    # proj is cartopy projection object
    elif isinstance(proj, ccrs.CRS): 
        proj_to = proj
    # proj is list of cartopy projection objects or string
    elif isinstance(proj, list): 
        proj_to = list()
        for proj_ii in proj:
            if   isinstance(proj_ii, str): 
                proj_dum, box = do_projection(None, proj_ii, box)
                proj_to.append(proj_dum)                
            elif isinstance(proj_ii, ccrs.CRS):    
                proj_to.append(proj_ii)    
    
    #___________________________________________________________________________
    # --> pre-arange axes
    ax_optdefault=dict({'projection': proj_to, 'ax_sharex':False, 'ax_sharey':True, 'ax_dt':2.0})
    ax_optdefault.update(ax_opt)
    hfig, hax, hcb, cb_plt_idx = do_axes_arrange(ncol, nrow, **ax_optdefault)
    cb_plt_idx=cb_plt_idx[:nbox]
    
    #___________________________________________________________________________
    # --> axes enumeration 
    do_axes_enum(hax[:nbox], do_enum, nrow, ncol, enum_opt=enum_opt, enum_str=enum_str, 
                       enum_x=enum_x, enum_y=enum_y, enum_dir=enum_dir)
    
    #___________________________________________________________________________
    # setup colormap
    if do_allcycl: 
        if n_cycl is not None:
            cmap = categorical_cmap(np.int32(ndat/n_cycl), n_cycl, cmap="tab10")
        else:
            cmap = categorical_cmap(ndat, 1, cmap="tab10")
    else:
        cmap = categorical_cmap(ndat, 1, cmap="tab10")
        
    #___________________________________________________________________________
    # --> loop over axes
    hp, hbot, hmsh, hlsm, hgrd = list(), list(), list(), list(), list()
    list_strdatalabel,list_strboxlabel = list(), list()
    for ii, (hax_ii, hcb_ii) in enumerate(zip(hax, hcb)):
        # if there are no ddatra to fill axes, make it invisible 
        if ii>=nbox: 
            hax_ii.axis('off')
        # axis is normally fillt with data    
        else: 
            #___________________________________________________________________
            # prepare regular gridded data for plotting
            # prepare regular gridded data for plotting
            optline = dict({'linewidth':1.5, 'marker':'None', 'markerfacecolor':'w', 'markersize':5, 'zorder':2})
            optmark = dict({'markersize':8, 'markeredgecolor':'k', 'markeredgewidth':0.5,
                            'clip_on':False, 'zorder':3}) #'clip_box':False, 
            list_lstyle=['solid', 'dashed', 'dotted', 'dashdot', 'dashdotdotted']
            
            #___________________________________________________________________
            allinone = False
            if   nrow*ncol == 1:
                if   nbox == 1: box_idx = [0]
                elif nbox >  1: 
                    box_idx, allinone = list(range(0,nbox)), True
                    
            elif nrow*ncol > 1 and nrow*ncol >= nbox:   
                box_idx, allinone = [ii], False
            xmin, xmax, ymin, ymax = np.inf, -np.inf, np.inf, -np.inf
            
            #___________________________________________________________________
            # If nrow*ncol=1    && nbox > 1: plot all box lines in one figure panel
            #    nrow*ncol=nbox && nbox > 1: plot each box lines in separate figure panel
            for bi in box_idx:
                
                #___________________________________________________________________
                # prepare regular gridded data for plotting
                cnt, cnt_cycl = 0, 0
                if not allinone: xmin, xmax, ymin, ymax = np.inf, -np.inf, np.inf, -np.inf
                for jj in range(0,ndat):
                    #_______________________________________________________________
                    vname = list(data[jj][bi].data_vars)[0]
                    data_x = data[jj][bi][vname].data.copy()
                    xmin, xmax = np.min([xmin, data_x.min()]), np.max([xmax, data_x.max()])
                    
                    #_______________________________________________________________
                    data_y, str_ylabel = np.abs(data[jj][bi]['depth'].values) , 'Depth / m'
                    ymin, ymax = np.min([ymin, data_y.min()]), np.max([ymax, data_y.max()])
                    
                    #_______________________________________________________________
                    loc_attrs= data[jj][bi][vname].attrs
                    str_xlabel, str_llabel, str_blabel = '', '', ''
                    if ax_xlabel is None:
                        if   'long_name'     in loc_attrs: str_xlabel = str_xlabel + loc_attrs['long_name'].capitalize()
                        elif 'short_name'    in loc_attrs: str_xlabel = str_xlabel + loc_attrs['short_name']
                    else: str_xlabel = ax_xlabel
                    if ax_xunit is None:
                        if 'units' in loc_attrs: str_xlabel = str_xlabel+' / '+loc_attrs['units']
                    else: str_xlabel = str_xlabel+' / '+ ax_xunit
                        
                    if   'descript'      in loc_attrs: str_llabel = str_llabel + loc_attrs['descript']
                    if   'boxname'       in loc_attrs: str_blabel = str_blabel + loc_attrs['boxname']
                    if   'transect_name' in loc_attrs: str_blabel = str_blabel + loc_attrs['transect_name']
                    str_blabel = str_blabel.replace('MOC','').replace('_','')
                    if bi==0: list_strdatalabel.append(str_llabel)
                    if jj==0: list_strboxlabel.append(str_blabel)
                    
                    #_______________________________________________________________
                    # plot lines 
                    optline.update({'color':cmap.colors[cnt,:]})
                    optline.update(plt_opt)
                    if allinone and nrow*ncol==1: optline.update({'linestyle':list_lstyle[bi]})
                    optmark.update({'color':cmap.colors[cnt,:]})
                    optmark.update(mark_opt)
                    
                    # plot black underlying shadow slightly wider than line on top 
                    if do_shdw:
                        optline2 = optline.copy()
                        optline2.update({'color':'k', 'linewidth':optline['linewidth']*1.2, 'zorder':1})
                        hax_ii.plot(data_x, data_y, **optline2)
                        
                    h0 = hax_ii.plot(data_x, data_y, label=str_llabel, **optline)
                    hp.append(h0)
                    
                    # plot mean value with left triangle 
                    if do_mean: 
                        auxx, auxy = data_x.copy(), data_y.copy()
                        auxy[np.isnan(auxx)]=0
                        auxx[np.isnan(auxx)]=0
                        hax_ii.plot(np.nansum(auxx*auxy)/np.nansum(auxy), ymax+(data_y[-1]-data_y[0])*0.00, marker='v', **optmark)
                        del(auxx,auxy)
                    #_______________________________________________________________
                    cnt      = cnt+1
                    cnt_cycl = cnt_cycl+1
                    if n_cycl is not None:
                        if cnt_cycl>= n_cycl: cnt_cycl=0
                        
                #___________________________________________________________________
                if hax_ii.do_xlabel: 
                    # wrap xlabel string when they are to long
                    # Estimate the width of the axes dynamically
                    fig_width, fig_height = hfig.get_size_inches()
                    fig_dpi = hfig.get_dpi()
                    axes_width_px = hax_ii.get_position().width * fig_width * fig_dpi
                    
                    # Estimate the width of the axes in terms of characters
                    # font_size = plt.rcParams['font.size']
                    font_size = hax_ii.xaxis.get_label().get_size()
                    max_chars_per_line = int(axes_width_px / (font_size*0.6))  # Empirical factor for font size to character width ratio
                    str_xlabel = '\n'.join(textwrap.wrap(str_xlabel, width=max_chars_per_line))
                    hax_ii.set_xlabel(str_xlabel)
                    
                if hax_ii.do_ylabel: 
                    hax_ii.set_ylabel(str_ylabel)
            
            
            #___________________________________________________________________
            # add grids lines 
            ax_ylim0 = ax_ylim
            if ax_ylim is None: ax_ylim0=[ymin,ymax]
            h0 = do_plt_gridlines(hax_ii, do_grid, None, None, data_x=None, 
                                  data_y=data_y, xlim=ax_xlim, ylim=ax_ylim0, 
                                  grid_opt=grid_opt, do_rescale=do_rescale)
            hgrd.append(h0)
            
            #_______________________________________________________________________
            # do legend and legend positioning 
            # -->bbox_to_anchor=(1.0, 1.0), loc='upper left' this should make sure 
            #    that the legend is always outside in the upper right corner 
            if hax_ii.coli==ncol-1 and hax_ii.rowi==0 : 
                if allinone and nbox>1:
                    # make data legend:
                    legend1 = plt.legend([hp[i][0] for i in range(0,ndat,1)], list_strdatalabel, 
                                         frameon=True, fancybox=True, shadow=True, fontsize=10, ncol=1,
                                         labelspacing=0.5, bbox_to_anchor=(1.0, 0.0), loc='lower right')
                    hax_ii.add_artist(legend1)
                    
                    # make box list legend:
                    import copy as cp
                    lines = [cp.copy(hp[i][0]) for i in range(0,ndat*nbox, ndat)]
                    # make legend box lines always black, just show linestyle 
                    for li in range(0,len(lines)): lines[li].set(color='k')
                    legend2 = plt.legend(lines, list_strboxlabel, 
                                         frameon=True, fancybox=True, shadow=True, fontsize=10, ncol=1,
                                         labelspacing=0.5, bbox_to_anchor=(0.0, 1.0), loc='upper left')
                    hax_ii.add_artist(legend2)
                else:
                    # make data legend:
                    hax_ii.legend(frameon=True, fancybox=True, shadow=True, fontsize=10, ncol=1,
                                labelspacing=0.5, bbox_to_anchor=(1.0, 1.0), loc='upper left') #bbox_to_anchor=(1.5, 1.5))
                    # box label becomes here axes title
            #___________________________________________________________________
            # add title and axes labels
            if ax_title is not None: 
                if isinstance(ax_title,list): hax_ii.set_title(ax_title[ii], fontsize=hax_ii.fs_label)
                else                        : 
                    if not allinone and nbox>1: hax_ii.set_title(str_blabel, fontsize=hax_ii.fs_label)
         
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, hfig, dpi=save_dpi, save_opt=save_opt)
    
    #___________________________________________________________________________
    list_argout=[]
    if len(nargout)>0:
        for stri in nargout:
            try   : list_argout.append(eval(stri))
            except: print(f" -warning-> variable {stri} was not found, could not be added as output argument") 
            
        return(list_argout)
    else:
        return
    


#
#
#_______________________________________________________________________________
# --> do plot timeseries of mean index
def plot_tline(data, 
                box, 
                box_idx    = None      ,
                box_label  = None      , 
                boxl_opt   = dict()    , # option for box label string 
                nrow       = 1         , # number of row in figures panel
                ncol       = 1         , # number of column in figure panel
                proj       = 'index+time',
                n_cycl     = None      , 
                do_allcycl = False     , 
                do_concat  = False     , 
                do_shdw    = True      ,
                do_mean    = True      ,
                do_std     = False      ,
                #--- data -----------
                plt_opt    = dict()    ,
                mark_opt   = dict()    ,
                #--- gridlines ------
                do_grid    = True      , 
                grid_opt   = dict()    ,
                #--- axes -----------
                ax_title   = 'descript',
                ax_xlabel  = None      ,
                ax_ylabel  = None      ,
                ax_xunit   = None      ,
                ax_yunit   = None      , 
                ax_opt     = dict()    ,
                axl_opt    = dict()    ,
                ax_xlim    = None      ,
                ax_ylim    = None      ,
                #--- enumerate axes -
                do_enum    = False     ,
                enum_opt   = dict()    , 
                enum_str   = []        , 
                enum_x     = [0.005]   , 
                enum_y     = [1.00]    ,
                enum_dir   = 'lr'      ,# prescribed list of enumeration strings
                #--- save figure ----
                do_save    = None      , 
                save_dpi   = 300       ,
                save_opt   = dict()    ,
                #--- set output -----
                nargout=['hfig', 'hax'],
                ):    
    """
    --> do plotting of mean indices over time (e.g. time-series)
    
    __________________________________________________
    
    Parameters:

        :data:      xarray dataset object, or list of xarray dataset object

        :box:       None, list (default: None)  list with regional box limitation for index computation box can be: 

                    - ['global']   ... compute global index 
                    - [shp.Reader] ... index region defined by shapefile 
                    - [ [lonmin,lonmax,latmin, latmax], boxname] index region defined by rect box 
                    - [ [ [px1,px2...], [py1,py2,...]], boxname] index region defined by polygon
                    - [ np.array(2 x npts) boxname] index region defined by polygon

        :box_idx:   int, (default: None) index in boxlist 

        :box_label: str  (default: None) overwrites boxname string 

        :boxl_opt:  dict (default: dict() additional options for boxlabel string (fontsize...)
        
        :nrow:      int, (default: 1) number of columns when plotting multiple data panels

        :ncol:      int, (default: 1) number of rows when plotting multiple data panels

        :proj:      str, (default: 'index+xy')
                        
        :n_cycl:    int, (default: None) How many spinup cycles are contained
                    in the data_list. Info is needed when do_allcycl=True,

        :do_allcycl: bool, (default: False) plot all spinupcycles based on colormap value 

        :do_concat: bool, (default: False) attache time-series of the the same spinup
                    cycle behind each other. Create one long spinup time-series

        :do_shdw:   bool, (default: True) give lines a black outline

        :do_mean:   bool, (default: False) plot triangle on the left side that indicates 
                    the mean value

        :do_rescale: bool, str, np.array (defaul: False) do scaling of colorbar
                    - False      ... scale data automatically scientifically by 10^x, for data data larger 10^3 and smaller 10^-3
                    - log10      ... do logaritmic scaling
                    - slog10     ... do symetric logarithmic scaling
        
        ___plot data parameters_____________________________

        :plt_opt:   dict, (default: dict()) additional options that are given to 
                    line plot routine via the kwarg argument

        :mark_opt:  dict, (default: dict()) additional options that are given to 
                    control the line markers via the kwarg argument

        ___plot gridlines___________________________________

        :do_grid:   bool, (default: True) plot cartopy grid lines

        :grid_opt:  dict, (default: dict()) additional options that are given to 
                    the cartopy gridline plotting via kwarg
                    

        ___axes______________________________________

        :ax_title:  str, (default: 'descript') If 'descript' use descript attribute 
                    in data to title label axes, If 'str' use this string to label axes

        :ax_xlabel: str, (default: None) overwrites default xlabel

        :ax_ylabel: str, (default: None) overwrites default ylabel

        :ax_xunit:  str, (default: None) overwrites default xlabel unit

        :ax_yunit:  str, (default: None) overwrites default ylabel unit

        :ax_opt:    dict, (default: dict()) set option for axes arangement see subroutine do_axes_arrange

        :ax_xlim:   list (default: None) overright data related xlimits

        :ax_ylim:   list (default: None) overright data related ylimits  

        ___enumerate_________________________________

        :do_enum:   bool, (default: False) do enumeration of axes with a), b), c) ...

        :enum_opt:  dict, (default: dict()) direct option for enumeration strings via kwarg 

        :enum_str:  list, (default: []) overwrite default enumeration strings

        :enum_x:    float, (default: 0.005)  x position of enumeration string in axes coordinates

        :enum_y:    float, (default: 1.000)  y position of enumeration string in axes coordinates

        :enum_dir:  str, (default: 'lr')  direction of numbering, 'lr' from left to right, 'ud' from up to down

        ___save figure________________________________

        :do_save:   str, (default: None) if None figure will by not saved, if string figure 
                    will be saved, strings must give directory and filename  where to save.

        :save_dpi:  int, (default: 300) dpi resolution at which the figure is saved

        :save_opt:  dict, (default: dict()) direct option for saving via kwarg

        ___set output_________________________________

        :nargout:   list, (default: ['hfig', 'hax', 'hcb']) list of variables that are given 
                    out from the routine. 
                    Default: 
                    
                    - hfig ... figure handle
                    - hax  ... list of axes handle 
                    - (every variable that is defined in this subroutine can become output parameter)
    
    __________________________________________________
    
    Returns:
    
        :hfig: returns figure handle 

        :hax: returns list with axes handle 
    
    ____________________________________________________________________________
    """
    import matplotlib.patheffects as path_effects
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator

    #___________________________________________________________________________
    # --> check if input data is a list
    if not isinstance(data, list): data = [data]
    ndat = len(data)
    nbox = len(data[0])
    #print(ndat, nbox)
    
    #___________________________________________________________________________
    # check vertical plotting mode if index+depth+xy, zmoc, dmoc
    if proj is None:
        if isinstance(data[0], xr.Dataset):
            if 'proj' in data[0].attrs: proj=data[0].attrs['proj']
        elif isinstance(data[0], list):
            if isinstance(data[0][box_idx], xr.Dataset):
                if 'proj' in data[0][box_idx].attrs: proj=data[0][box_idx].attrs['proj']
    
    #___________________________________________________________________________
    # --> create projection
    proj_to = None
    # proj is string 
    if   isinstance(proj, str): 
        proj_to, box = do_projection(None, proj, box)
    # proj is cartopy projection object
    elif isinstance(proj, ccrs.CRS): 
        proj_to = proj
    # proj is list of cartopy projection objects or string
    elif isinstance(proj, list): 
        proj_to = list()
        for proj_ii in proj:
            if   isinstance(proj_ii, str): 
                proj_dum, box = do_projection(None, proj_ii, box)
                proj_to.append(proj_dum)                
            elif isinstance(proj_ii, ccrs.CRS):    
                proj_to.append(proj_ii)    

    #___________________________________________________________________________
    # --> pre-arange axes
    ax_optdefault=dict({'projection': proj_to, 'ax_sharey':False,})
    ax_optdefault.update(ax_opt)
    hfig, hax, hcb, cb_plt_idx = do_axes_arrange(ncol, nrow, **ax_optdefault)
    cb_plt_idx=cb_plt_idx[:ndat]
    
    #___________________________________________________________________________
    # --> axes enumeration 
    do_axes_enum(hax[:ndat], do_enum, nrow, ncol, enum_opt=enum_opt, enum_str=enum_str, 
                       enum_x=enum_x, enum_y=enum_y, enum_dir=enum_dir)

    #___________________________________________________________________________
    # setup colormap
    if do_allcycl: 
        if n_cycl is not None:
            cmap = categorical_cmap(np.int32(ndat/n_cycl), n_cycl, cmap="tab10")
        else:
            cmap = categorical_cmap(ndat, 1, cmap="tab10")
    else:
        if do_concat: do_concat=False
        cmap = categorical_cmap(ndat, 1, cmap="tab10")
    
    #___________________________________________________________________________
    # --> loop over axes
    hp, hbot, hmsh, hlsm, hgrd = list(), list(), list(), list(), list()
    cnt, xmin, xmax, ymin, ymax = 0, np.inf, -np.inf, np.inf, -np.inf
    list_strdatalabel,list_strboxlabel = list(), list()
    for ii, (hax_ii, hcb_ii) in enumerate(zip(hax, hcb)):
        # if there are no ddatra to fill axes, make it invisible 
        if ii>=nbox: 
            hax_ii.axis('off')
        # axis is normally fillt with data    
        else: 
            ii_valid=ii
            #___________________________________________________________________
            # prepare regular gridded data for plotting
            optline = dict({'linewidth':1.5, 'marker':'None', 'markerfacecolor':'w', 'markersize':5, 'zorder':2})
            optmark = dict({'markersize':8, 'markeredgecolor':'k', 'markeredgewidth':0.5,
                            'clip_on':False, 'zorder':3}) #'clip_box':False, 
            list_lstyle=['solid', 'dashed', 'dotted', 'dashdot', 'dashdotdotted']
            
            #___________________________________________________________________
            allinone = False
            if box_idx is None:
                if   nrow*ncol == 1:
                    if   nbox == 1: box_idx = [0]
                    elif nbox >  1: 
                        box_idx, allinone = list(range(0,nbox)), True
                        
                elif nrow*ncol > 1 and nrow*ncol >= nbox:   
                    box_idx, allinone = [ii], False
            xmin, xmax, ymin, ymax = np.inf, -np.inf, np.inf, -np.inf

            #___________________________________________________________________
            # If nrow*ncol=1    && nbox > 1: plot all box lines in one figure panel
            #    nrow*ncol=nbox && nbox > 1: plot each box lines in separate figure panel
            if not isinstance(box_idx, list): box_idx = [box_idx]
            for bi in box_idx:
                #___________________________________________________________________
                # prepare regular gridded data for plotting
                cnt, cnt_cycl = 0, 0
                xmax_list=list()
                if not allinone: xmin, xmax, ymin, ymax = np.inf, -np.inf, np.inf, -np.inf
                for jj in range(0,ndat):
                    #_______________________________________________________________
                    vname  = list(data[jj][bi].data_vars)[0]
                    data_y = data[jj][bi][vname].data.copy()
                    ymin, ymax = np.min([ymin, data_y.min()]), np.max([ymax, data_y.max()])
                    
                    #_______________________________________________________________
                    data_x = data[jj][bi]['time'].copy()
                    # total number of days per year considers leap years
                    if len(np.unique(data_x.dt.month))==1:
                        data_x = data_x.dt.year
                    else:
                        # data contain only mean seasonal cycle 
                        if len(np.unique(data_x.dt.year))==1:
                            data_x = data_x.dt.month
                        else:    
                            dperyr = np.where(data_x.dt.is_leap_year, 366, 365)
                            # time vector in units of years
                            data_x = data_x.dt.year + (data_x.dt.dayofyear-data_x.dt.day[0])/dperyr   
                    if cnt_cycl>0 and do_concat: 
                        data_x = data_x + (data_x[-1]-data_x[0]+1)*cnt_cycl# + (data_x[1]-data_x[0])
                        data_x = np.hstack((data_x0[-1], data_x))
                        data_y = np.hstack((data_y0[-1], data_y))
                        
                        
                    #data_x = data_x.values
                    xmin, xmax = np.min([xmin, data_x.min()]), np.max([xmax, data_x.max()])
                    data_x0, data_y0 = data_x, data_y
                    xmax_list.append(xmax)
                    
                    #_______________________________________________________________
                    # set labels
                    loc_attrs= data[jj][bi][vname].attrs
                    str_xlabel, str_ylabel, str_llabel, str_blabel = '', '', '', ''
                    
                    if ax_xlabel is None:
                        str_xlabel = 'Time'
                        if   'add2xlabel' in loc_attrs: str_xlabel = str_xlabel+' '+loc_attrs['add2xlabel']
                        if ax_xunit is None: str_xlabel = str_xlabel + ' / years'
                        else               : str_xlabel = str_xlabel + ' / ' + ax_xunit 
                    else:
                        str_xlabel = ax_xlabel
                    
                    if ax_ylabel is None:
                        if   'long_name'  in loc_attrs: str_ylabel = str_ylabel+loc_attrs['long_name']
                        elif 'short_name' in loc_attrs: str_ylabel = str_ylabel+loc_attrs['short_name']
                        if   'add2ylabel' in loc_attrs: str_ylabel = str_ylabel+' '  +loc_attrs['add2ylabel']
                        
                        if ax_yunit is None:
                            if   'units'      in loc_attrs: str_ylabel = str_ylabel+' / '+loc_attrs['units']
                        else: str_ylabel = str_ylabel+' / '+ ax_yunit
                    else:
                        str_ylabel = ax_ylabel
                    
                    if   'descript'       in loc_attrs: str_llabel = str_llabel +loc_attrs['descript']
                    if   'boxname'        in loc_attrs: str_blabel = str_blabel +loc_attrs['boxname']
                    if   'transect_name'  in loc_attrs: str_blabel = str_blabel +loc_attrs['transect_name']
                    str_blabel = str_blabel.replace('MOC','').replace('_','')
                    if bi==0: list_strdatalabel.append(str_llabel)
                    if jj==0: list_strboxlabel.append(str_blabel)
                    
                    #_______________________________________________________________
                    # plot lines 
                    optline.update({'color':cmap.colors[cnt,:]})
                    optline.update(plt_opt)
                    if allinone and nrow*ncol==1: optline.update({'linestyle':list_lstyle[bi]})
                    optmark.update({'color':cmap.colors[cnt,:]})
                    optmark.update(mark_opt)
                    
                    # plot black underlying shadow slightly wider than line on top 
                    if do_shdw:
                        optline2 = optline.copy()
                        optline2.update({'color':'k', 'linewidth':optline['linewidth']*1.2, 'zorder':1})
                        hax_ii.plot(data_x, data_y, **optline2)
                    
                    # plot actual time series with color 
                    h0 = hax_ii.plot(data_x, data_y, label=str_llabel, **optline)
                    
                    # plot mean value with left triangle 
                    if do_mean: 
                        hax_ii.plot(xmin-(data_x[-1]-data_x[0])*0.00, data_y.mean(), marker='<',  **optmark)
                    
                    # plot std. range with up/dwn triangle 
                    if do_std:
                        warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in multiply")
                        hax_ii.plot(xmin-(data_x[-1]-data_x[0])*0.0, data_y.mean()+data_y.std(), marker='^', **optmark)
                        hax_ii.plot(xmin-(data_x[-1]-data_x[0])*0.0, data_y.mean()-data_y.std(), marker='v', **optmark)
                        warnings.resetwarnings()
                        
                    hp.append(h0)
                    
                    #___________________________________________________________
                    cnt      = cnt+1
                    cnt_cycl = cnt_cycl+1
                    if n_cycl is not None:
                        if cnt_cycl>= n_cycl: cnt_cycl=0
                        
                #_______________________________________________________________
                if hax_ii.do_xlabel: 
                    hax_ii.set_xlabel(str_xlabel)
                    
                if hax_ii.do_ylabel:
                     # wrap xlabel string when they are to long
                    # Estimate the width of the axes dynamically
                    fig_width, fig_height = hfig.get_size_inches()
                    fig_dpi = hfig.get_dpi()
                    axes_width_px = hax_ii.get_position().width * fig_width * fig_dpi
                    
                    # Estimate the width of the axes in terms of characters
                    # font_size = plt.rcParams['font.size']
                    font_size = hax_ii.xaxis.get_label().get_size()
                    max_chars_per_line = int(axes_width_px / (font_size*0.6))  # Empirical factor for font size to character width ratio
                    str_ylabel = '\n'.join(textwrap.wrap(str_ylabel, width=max_chars_per_line))
                    hax_ii.set_ylabel(str_ylabel)
            
            #___________________________________________________________________
            # add grids lines 
            ax_xlim0, ax_ylim0 = ax_xlim, ax_ylim 
            if ax_xlim is None: ax_xlim0=[xmin,xmax]
            if ax_ylim is None: ax_ylim0=[ymin,ymax]
            h0 = do_plt_gridlines(hax_ii, do_grid, None, None, data_x=None, 
                                  data_y=data_y, xlim=ax_xlim0, ylim=ax_ylim0, 
                                  grid_opt=grid_opt)
            hgrd.append(h0)  
            
            #_______________________________________________________________
            if do_concat:
                aux_ylim=hax_ii.get_ylim()
                hax_ii.vlines(np.unique(xmax_list), aux_ylim[0], aux_ylim[1], 'k', linewidth=1.5, zorder=1)
                hax_ii.set_ylim(aux_ylim)
            
            #_______________________________________________________________________
            # do legend and legend positioning 
            # -->bbox_to_anchor=(1.0, 1.0), loc='upper left' this should make sure 
            #    that the legend is always outside in the upper right corner 
            if hax_ii.coli==ncol-1 and hax_ii.rowi==0 : 
                if allinone and nbox>1:
                    # make data legend:
                    legend1 = plt.legend([hp[i][0] for i in range(0,ndat,1)], list_strdatalabel, 
                                         frameon=True, fancybox=True, shadow=True, fontsize=10, ncol=1,
                                         labelspacing=0.5, bbox_to_anchor=(1.0, 0.0), loc='lower right')
                    hax_ii.add_artist(legend1)
                    
                    # make box list legend:
                    import copy as cp
                    lines = [cp.copy(hp[i][0]) for i in range(0,ndat*nbox, ndat)]
                    # make legend box lines always black, just show linestyle 
                    for li in range(0,len(lines)): lines[li].set(color='k')
                    legend2 = plt.legend(lines, list_strboxlabel, 
                                         frameon=True, fancybox=True, shadow=True, fontsize=10, ncol=1,
                                         labelspacing=0.5, bbox_to_anchor=(0.0, 1.0), loc='upper left')
                    hax_ii.add_artist(legend2)
                else:
                    # make data legend:
                    hax_ii.legend(frameon=True, fancybox=True, shadow=True, fontsize=10, ncol=1,
                                labelspacing=0.5, bbox_to_anchor=(1.0, 1.0), loc='upper left') #bbox_to_anchor=(1.5, 1.5))
                    # box label becomes here axes title
             #___________________________________________________________________
            # add title and axes labels
            if ax_title is not None: 
                if isinstance(ax_title,list): hax_ii.set_title(ax_title[ii], fontsize=hax_ii.fs_label)
                else                        : 
                    if not allinone and nbox>1: hax_ii.set_title(str_blabel, fontsize=hax_ii.fs_label)
           
         
    #___________________________________________________________________________
    # save figure based on do_save contains either None or pathname
    do_savefigure(do_save, hfig, dpi=save_dpi, save_opt=save_opt)
    
    #___________________________________________________________________________
    list_argout=[]
    if len(nargout)>0:
        for stri in nargout:
            try   : list_argout.append(eval(stri))
            except: print(f" -warning-> variable {stri} was not found, could not be added as output argument") 
            
        return(list_argout)
    else:
        return


    
#
#
#_______________________________________________________________________________
# --> setup cartopy projections     
def do_projection(mesh, proj, box):
    """
    --> set cartopy target projection 
    
    Parameters:
    
        :mesh:      fesom2 mesh object,  with all mesh information 

        :proj:      str, (default: 'pc') which projection should be used, 
                    - pc     ... PlateCarree         (box=[lonmin, lonmax, latmin, latmax])
                    - merc   ... Mercator            (box=[lonmin, lonmax, latmin, latmax])
                    - rob    ... Robinson            (box=[lonmin, lonmax, latmin, latmax])
                    - eqearth... EqualEarth          (box=[lonmin, lonmax, latmin, latmax])
                    - mol    ... Mollweide           (box=[lonmin, lonmax, latmin, latmax])
                    - nps    ... NorthPolarStereo    (box=[-180, 180, >0, latmax])
                    - sps    ... SouthPolarStereo    (box=[-180, 180, latmin, <0])
                    - ortho  ... Orthographic        (box=[loncenter, latcenter]) 
                    - nears  ... NearsidePerspective (box=[loncenter, latcenter, zoom]) 
                    - channel... PlateCaree

        :box:       None, list (default: None) regional limitation of plot. For 
                    ortho box = [lonc, latc], nears [lonc, latc, zoom], for all
                    others box = [lonmin, lonmax, latmin, latmax]
    
    Returns:
    
        :proj_to:   cartopy projection object 

        :box:       return projection adapted box list

    ____________________________________________________________________________
    """ 
    
    #___Horizontal Projection___________________________________________________
    if   proj=='pc'     :
        if len(box)!=4: raise ValueError( 'For PlateCarree projection: box=[lon_min, lon_max, lat_min, lat_max]')
        proj_to = ccrs.PlateCarree()     
    
    elif proj=='merc'   : 
        if len(box)!=4: raise ValueError( 'For Mercator projection: box=[lon_min, lon_max, lat_min, lat_max]')
        proj_to = ccrs.Mercator()
        box[0], box[1] = box[0]+0.25, box[1]-0.25
        box[2], box[3] = max(box[2], -85), min(box[3], 85)
        
    elif proj=='rob'    : 
        if len(box)!=4: raise ValueError( 'For Robinson projection: box=[lon_min, lon_max, lat_min, lat_max]')
        proj_to = ccrs.Robinson()  
        #box[0], box[1] = box[0]+0.25, box[1]-0.25
    
    elif proj=='eqearth': 
        if len(box)!=4: raise ValueError( 'For EqualEarth projection: box=[lon_min, lon_max, lat_min, lat_max]')
        proj_to = ccrs.EqualEarth(central_longitude=mesh.focus)
        box[0], box[1] = box[0]+0.25, box[1]-0.25
    
    elif proj=='mol': 
        if len(box)!=4: raise ValueError( 'For Mollweide projection: box=[lon_min, lon_max, lat_min, lat_max]')
        proj_to = ccrs.Mollweide(central_longitude=mesh.focus)
        box[0], box[1] = box[0]+0.25, box[1]-0.25
    
    elif proj=='nps'    : 
        if len(box)!=4: raise ValueError( 'For NorthPolarStereo projection: box=[-180, 180, lat_min, lat_max]')
        proj_to = ccrs.NorthPolarStereo()
        if box[2]<0: box[2]=0
    
    elif proj=='sps'    : 
        if len(box)!=4: raise ValueError( 'For SouthPolarStereo projection: box=[-180, 180, lat_min, lat_max]')
        proj_to = ccrs.SouthPolarStereo()
        if box[3]>0: box[3]=0
    
    elif proj=='ortho'   : 
        if len(box)!=2: raise ValueError( 'For Orthographic projection: box=[lonc, latc]')
        proj_to = ccrs.Orthographic(central_longitude=box[0], central_latitude=box[1], globe=None)        
    
    elif proj=='nears'   : 
        if len(box)!=3: raise ValueError( 'For NearsidePerspective projection: box=[lonc, latc, zoom]')
        proj_to = ccrs.NearsidePerspective(central_longitude=box[0], central_latitude=box[1], satellite_height=35785831.0/box[2])        
    
    elif proj=='channel':
        proj_to = ccrs.PlateCarree()
        if box is None or box=="None": box = [np.hstack((mesh.n_x,mesh.n_xa)).min(), np.hstack((mesh.n_x,mesh.n_xa)).max(), np.hstack((mesh.n_y,mesh.n_ya)).min(), np.hstack((mesh.n_y,mesh.n_ya)).max()]
    
        print(proj, box)
    #___Vertical "Projection"___________________________________________________
    elif  proj == 'index+depth+xy'   : proj_to = 'index+depth+xy'
    elif  proj == 'index+depth+time' : proj_to = 'index+depth+time'
    elif  proj == 'index+depth'      : proj_to = 'index+depth'
    elif  proj == 'index+time'       : proj_to = 'index+time'
    elif  proj == 'index+xy'         : proj_to = 'index+xy'
    elif  proj == 'zmoc'             : proj_to = 'zmoc'
    elif  proj == 'dmoc'             : proj_to = 'dmoc'
    elif  proj == 'dmoc+depth'       : proj_to = 'dmoc+depth'
    elif  proj == 'dmoc+dens'        : proj_to = 'dmoc+dens'
    else: 
        raise ValueError('The projection {} is not supporrted!'.format(proj))
        
    #___________________________________________________________________________    
    return(proj_to, box)



#
#
#_______________________________________________________________________________
# --> do triangulation
def do_triangulation(hax, mesh, proj_to, box, proj_from=ccrs.PlateCarree(), 
                     do_triorig=False, do_narea=True, do_earea=False):
    """
    --> create matplotlib triangulation object
    
    Parameters:
    
        :hax:       handle of current axes

        :mesh:      fesom2 mesh object,  with all mesh information 

        :proj_to:   cartopy destination projection object   

        :proj_from: cartopy source projection object (default: ccrs.PlateCarree())                  

        :do_triorig: bool, (default=False) save original vertices coordinate in lon,lat space 

        :do_narea:  bool (default=True), drag vertices area with you is needed for normalisation

        :do_earea:  bool (default=True), drag element area with you is needed for normalisation
    
    Returns:
    
        :tri:       matplotlib.tri triangulation object
    
    ____________________________________________________________________________
    """
    tri = Triangulation(np.hstack((mesh.n_x,mesh.n_xa)),
                        np.hstack((mesh.n_y,mesh.n_ya)),
                        np.vstack((mesh.e_i[mesh.e_pbnd_0,:],mesh.e_ia)))
    
    #___________________________________________________________________________
    # --> limit mesh triangulation to projection box
    e_box_mask = np.ones(tri.triangles.shape[0], dtype=bool)
    if isinstance(proj_to, (ccrs.NorthPolarStereo, ccrs.SouthPolarStereo, ccrs.Robinson, ccrs.EqualEarth, ccrs.Mollweide, ccrs.Mercator) ):
        e_box_mask = grid_cutbox_e(tri.x, tri.y, tri.triangles, box, which='soft')
    
    elif isinstance(proj_to, (ccrs.NearsidePerspective)):
        xpts, ypts = proj_to.transform_points(proj_from, tri.x[tri.triangles].sum(axis=1)/3, tri.y[tri.triangles].sum(axis=1)/3)[:,0:2].T
        e_box_mask = (np.isnan(xpts)==False) & (np.isnan(ypts)==False)
        del(xpts, ypts)
    
    elif not isinstance(proj_to, (ccrs.Orthographic)):    
        xpts, ypts = proj_from.transform_points(proj_to, tri.x[tri.triangles].sum(axis=1)/3, tri.y[tri.triangles].sum(axis=1)/3)[:,0:2].T
        if isinstance(hax, list):
            fig_pts    = hax[0].transData.transform(list(zip(xpts,ypts)))
            ax_pts     = hax[0].transAxes.inverted().transform(fig_pts)
        else:    
            fig_pts    = hax.transData.transform(list(zip(xpts,ypts)))
            ax_pts     = hax.transAxes.inverted().transform(fig_pts)
        e_box_mask = (ax_pts[:,0]>=-0.05) & (ax_pts[:,0]<=1.05) & (ax_pts[:,1]>=-0.05) & (ax_pts[:,1]<=1.05)
        del(xpts, ypts, fig_pts, ax_pts)
        
    # --> reindex vertices and elements --> ensure smallest triangulation object 
    # hopefully saves some memory when going to very large meshes 
    n_box_mask = None
    if any(e_box_mask==False):
        tri, n_box_mask = do_reindex_vert_and_elem(tri, e_box_mask)
        
    #___________________________________________________________________________
    # --> do the mapping transformation outside of tricontourf is absolutely 
    #     faster than let doing cartopy doing it within 
    # when you transpose a 2D array with two columns, it effectively swaps the 
    # rows and columns, allowing you to use tuple unpacking to assign each 
    # column to a separate variable.
    
    if do_triorig: tri.xorig, tri.yorig = tri.x, tri.y
    tri.x, tri.y = proj_to.transform_points(proj_from, tri.x, tri.y)[:,0:2].T
    
    #___________________________________________________________________________
    # add some more varaibles i need
    tri.mask_e_box     = e_box_mask
    tri.mask_n_box     = n_box_mask
    del(e_box_mask, n_box_mask)
    tri.n2dn, tri.n2de = mesh.n2dn, mesh.n2de
    
    #___________________________________________________________________________
    if do_narea:
        if tri.mask_n_box is not None: 
            tri.narea = np.hstack((mesh.n_area[0,:],mesh.n_area[0,mesh.n_pbnd_a]))
            tri.narea = tri.narea[tri.mask_n_box]
        else: 
            tri.narea = mesh.n_area[0,:]
    
    #___________________________________________________________________________
    if do_earea:
        if any(tri.mask_e_box==False): 
            tri.earea = np.hstack((mesh.e_area[mesh.e_pbnd_0],mesh.e_area[mesh.e_pbnd_a]))
            tri.earea = tri.earea[tri.mask_e_box]
        else: 
            tri.earea = mesh.e_area
            
    #___________________________________________________________________________
    return(tri)



#
#
#_______________________________________________________________________________
def do_reindex_vert_and_elem(tri, e_box_mask):
    """
    --> reindex element index in case of exluded triangles and unused vertices
    
    Parameters:
    
        :tri:       matplotlib.tri triangulation object
        
        :e_box_mask: bool np.array with element masking from regional box definition

    Returns:
    
        :tri:       matplotlib.tri triangulation object where unreferenced triangles 
                    are kickout and the indices of the remaining vertices are 
                    adapted accordingly in the element list

        :n_box_mask: bool np.array with vertices masking from regional box selection
    
    ____________________________________________________________________________
    """     
    # Identify used vertices
    n_box_mask = np.unique(tri.triangles[e_box_mask,:].flatten())
    
    # Create mapping dictionary between old and new vertex indices
    vert_map  = {old_index: new_index for new_index, old_index in enumerate(n_box_mask)}

    # Update vertices and elements using NumPy indexing
    tri       = Triangulation(tri.x[n_box_mask], 
                              tri.y[n_box_mask], 
                              np.vectorize(vert_map.get)(tri.triangles[e_box_mask,:]))
    
    #___________________________________________________________________________
    return(tri, n_box_mask)



#
#
#_______________________________________________________________________________
# --> this is based on work of Nils Bruegemann see 
# https://gitlab.dkrz.de/m300602/pyicon/-/blob/master/pyicon/pyicon_plotting.py
# i needed this to originally unify the ploting between icon and fesom for model 
# comparison paper
def do_axes_arrange(nx, ny,
                    #---LABEL OPTION--------------------------------------------
                    xlabel = '', ylabel = '', tlabel = '', 
                     # font sizes of labels, titles, ticks
                    fs_label = 10., fs_title = 10., fs_ticks = 10.,
                    # font size increasing factor
                    fs_fac = 1,
                    
                    #---AXES OPTIONS--------------------------------------------
                    ax_sharex   = True ,    # all subplot share x-axes
                    ax_sharey   = True,    # all subplot share y-axes
                    ax_optdict  = dict(),   # additional axes option: fontssize, ...
                    ax_asp      = 1.   ,    # aspect ratio of axes
                    ax_w ='auto', ax_h =4., # width and height of axes
                    # space aroung axes (left, right, top, bottom) 
                    ax_dl =0.6, ax_dr =0.6, ax_dt =0.6, ax_db =0.6, 
                    # factors to increase spaces (axes)
                    ax_fdl=1.0, ax_fdr=1.0, ax_fdt=1.0, ax_fdb=1.0,
                    ax_fw =1.0, ax_fh =1.0, 
                    
                    #---FIGURE OPTION-------------------------------------------
                    fig_optdict = dict() ,  # additional figure option: fontssize, ...
                    fig_sizefac = 1.,       # factor to resize figures
                    # extra figure spaces (left, right, top, bottom)
                    fig_dl =0.0, fig_dr =0.0, fig_dt =0.0, fig_db =0.0,
                    # factors to increase spaces (figures)
                    fig_fdl=1.0, fig_fdr=1.0, fig_fdt=1.0, fig_fdb=1.0,
                
                    #---COLORBAR OPTION-----------------------------------------
                    cb_plt        = True,
                    cb_plt_single = True, 
                    cb_pos        = 'vertical', 
                    # space around colorbars (left, right, top, bottom) 
                    cb_dl = 0.6, cb_dr =3.0, cb_dt =0.6, cb_db =0.6,
                    # factors to increase spaces (colorbars)
                    cb_fdl= 1.0, cb_fdr=1.0, cb_fdt=1.0, cb_fdb=1.0,
                    # width and height of colorbars
                    cb_w  = 0.5, cb_h  = 'auto',
                    # factors to increase widths and heights of axes and colorbars
                    cb_fw = 1.0, cb_fh = 1.0,
                 
                    #-----------------------------------------------------------
                    projection = None, # projection (e.g. for cartopy)
                    proj       = None,
                    box        = None, # define regional box needed for aspect ratio
                    #-----------------------------------------------------------
                    nargout=['hfig', 'hax', 'hcb', 'cb_plt_idx'],
                    **kwargs,
                ):
    """
    --> do multipanel axes arangement 

    __________________________________________________
    
    Parameters:
        
        ___LABEL OPTION____________________
        
        :xlabel:        str (default: '') provide prescribed xlabel string 

        :ylabel:        str (default: '') provide prescribed ylabel string 

        :tlabel:        str (default: '') provide prescribed title string 

        :fs_label:      int (default: 10) prescribed fontsize for labels

        :fs_title:      int (default: 10) prescribed fontsize for title

        :fs_ticks:      int (default: 10) prescribed fontsize for ticklabels

        :fs_fac:        int (default: 1)  factor to generally increase fontsize

        ___AXES OPTIONS____________________

        :ax_sharex:     bool (default: True) all subplot share x-axes

        :ax_sharey:     bool (default: True) all subplot share y-axes

        :ax_optdict:    dict (default: dict()) additional axes option: fontssize, ...

        :ax_asp:        float (default: 1.) aspect ratio of axes
        
        :ax_w:          float, int (default: 'auto') if 'auto' width is defined based on
                        aspect ration ax_asp and height ax_h in cm, if float value is 
                        used to define width in cm

        :ax_h:          float, int (default: 4) if 'auto' height is defined based on
                        aspect ration ax_asp and width ax_w in cm, if float value is 
                        used to define height in cm

        :ax_fw:         float (default: 1.0 factor to increase width spacing

        :ax_fh:         float (default: 1.0 factor to increase height spacing

        :ax_dl:         float (default: 0.6) left spacing around axes in cm

        :ax_dr:         float (default: 0.6) right spacing around axes in cm

        :ax_dt:         float (default: 0.6) top spacing around axes in cm

        :ax_db:         float (default: 0.6) bottom spacing around axes in cm

        :ax_fdl:        float (default: 1.0) factor to increase left axes spacing

        :ax_fdr:        float (default: 1.0) factor to increase right axes spacing

        :ax_fdt:        float (default: 1.0) factor to increase top axes spacing

        :ax_fdb:        float (default: 1.0) factor to increase bottom axes spacing

        ___FIGURE OPTION___________________

        :fig_optdict:   dict (default: dict()) additional figure option: fontssize, ...

        :fig_sizefac:   float (default: 1.) factor to resize figures
                        
        :fig_dl:        float (default: 0.6) left spacing around figure in cm

        :fig_dr:        float (default: 0.6) right spacing around figure in cm

        :fig_dt:        float (default: 0.6) top spacing around figure in cm

        :fig_db:        float (default: 0.6) bottom spacing around figure in cm

        :fig_fdl:       float (default: 1.0) factor to increase left figure spacing

        :fig_fdr:       float (default: 1.0) factor to increase right figure spacing

        :fig_fdt:       float (default: 1.0) factor to increase top figure spacing

        :fig_fdb:       float (default: 1.0) factor to increase bottom figure spacing

        ___COLORBAR OPTION_________________

        :cb_plt:        bool, list (default: True) If True colorbar is plotted to all 
                        axes (cb_plt_single=False) or just one colorbar is plotted for 
                        all axes (cb_plt_single=False), If list with 0 and 1 [0,1,0,1...], 
                        which axes should have colorbar, if number higher than 1 in 
                        list it is assumed there is more than one independed colorbar

        :cb_plt_single: bool (default: False) true/false if there is just one colorbar 
                        for all the axes or if each axes should get a colorbar

        :cb_pos:        str (default: 'vertical') orientation of colorbar, either 
                        vertical or horizontal
                        
        :cb_dl:         float (default: 0.6) left spacing around colorbar in cm

        :cb_dr:         float (default: 3.0) right spacing around colorbar in cm

        :cb_dt:         float (default: 1.0) top spacing around colorbar in cm

        :cb_db:         float (default: 0.6) bottom spacing around colorbar in cm

        :cb_fdl:        float (default: 1.0) factor to increase left colorbar spacing

        :cb_fdr:        float (default: 1.0) factor to increase right colorbar spacing

        :cb_fdt:        float (default: 1.0) factor to increase top colorbar spacing

        :cb_fdb:        float (default: 1.0) factor to increase bottom colorbar spacing
        
        :cb_w:          float, str (default: 0.5) if float it is used as width in cm,
                        if 'auto' width is defined automatically based on width of axes
                        in case of horizontal colorbar

        :cb_h:          float, str (default: 0.5) if float it is used as height in cm,
                        if 'auto' height is defined automatically based on height of axes
                        in case of vertical colorbar

        :cb_fw:         float (default: 1.0) factor to increase width of colorbar

        :cb_fh:         float (default: 1.0) factor to increase height colorbar

        ___PROJECTION_______________________

        :projection:    (default: None) provide single cartopy projection object to use 
                        for all axes, or provide list of cartopy projection objects

        :box:           None, list (default: None) regional limitation of plot. For 
                        ortho: box = [lonc, latc], nears: [lonc, latc, zoom], for all
                        others box = [lonmin, lonmax, latmin, latmax]

        ___OUTPUT___________________________

        :nargout:       list, (default: ['hfig', 'hax', 'hcb', 'cb_plt_idx']) list of variables that
                        are given out from the routine. 
                        Default: 
                        - hfig      ... figure handle
                        - hax       ... list of axes handle 
                        - hcb       ... list of colorbar handles
                        - cb_plt_idx... list that contains index of independent
                        - colorbars
                        (every variable that is defined in this subroutine can become 
                        output parameter)
    
    __________________________________________________
    
    Returns:
    
        :hfig:          returns figure handle 

        :hax:           returns list with axes handle 

        :hcb:           returns colorbar handle

        :cb_plt_idx:    list that contains index of independent colorbars
    
    ____________________________________________________________________________
    """        
    #___________________________________________________________________________
    # factor to convert cm into inch
    cm2inch = 0.3937

    #___________________________________________________________________________
    # make list of projections if it is not a list
    if not isinstance(projection, list): projection = [projection]*nx*ny
    
    #___________________________________________________________________________
    # determine ax_w in case it is auto
    #if isinstance(ax_w, str) and ax_w=='auto': ax_w = ax_h/ax_asp
    if isinstance(ax_w, str) and ax_w=='auto': 
        #if box is not None and ax_asp==1.0:
        if ax_asp==1.0:        
            #___________________________________________________________________
            # projection[0] is an arbitrary cartopy-projection object
            if isinstance(projection[0], ccrs.CRS) and proj!='channel':
                if isinstance(projection[0], (ccrs.NorthPolarStereo, ccrs.SouthPolarStereo, ccrs.Orthographic, ccrs.NearsidePerspective) ):
                    ax_asp = 1.0
                    
                elif isinstance(projection[0], (ccrs.Robinson, ccrs.EqualEarth, ccrs.Mollweide) ):
                    poly_x, poly_y = np.array([box[0], box[1], box[1], box[0] ]), np.array([box[3], box[3], box[2], box[2] ]),
                    ax_asp = (poly_x.max()-poly_x.min())/(poly_y.max()-poly_y.min())
                    
                else:
                    poly_x, poly_y = [box[0], box[1], box[1], box[0] ], [box[3], box[3], box[2], box[2] ]
                    points = projection[0].transform_points(ccrs.PlateCarree(), np.array(poly_x), np.array(poly_y))
                    ax_asp = ( (points[:,0].max()-points[:,0].min())/(points[:,1].max()-points[:,1].min()) )
            
            #___________________________________________________________________
            # channel 
            elif isinstance(projection[0], ccrs.CRS) and proj=='channel': ax_asp = 2.0
                
            #___________________________________________________________________
            # projection is vertical section
            elif projection[0]=='index+depth+xy'  : ax_asp = 2.0
            elif projection[0]=='index+depth+time': ax_asp = 2.0
            elif projection[0]=='index+depth'     : ax_asp = 0.75
            elif projection[0]=='index+time'      : ax_asp = 2.5
            elif projection[0]=='index+xy'        : ax_asp = 1.5
            elif projection[0]=='zmoc'            : ax_asp = 2.0   
            elif projection[0]=='dmoc'            : ax_asp = 2.0
            elif projection[0]=='dmoc+depth'      : ax_asp = 2.0   
            elif projection[0]=='dmoc+dens'       : ax_asp = 2.0   
            else                                  : ax_asp = 1.0    
        #print('ax_asp=', ax_asp)
        ax_w = ax_h*ax_asp
        #print('ax_w=', ax_w, ', ax_h=', ax_h)

    # rename horizontal->bottom and vertical->right
    if   cb_pos in ['horizontal', 'horiz', 'bottom', 'bot']: 
        if cb_h =='auto': cb_h = 0.5*fig_sizefac
        if isinstance(cb_w, str) and cb_w=='auto': cb_w = ax_w
        cb_pos = 'bottom'
        if nx>1: cb_h = cb_h*fig_sizefac
    elif cb_pos in ['vertical', 'vert', 'right']: 
        if cb_w =='auto': cb_w = 0.5
        if isinstance(cb_h, str) and cb_h=='auto': cb_h = ax_h
        cb_pos = 'right'
        if ny>1: cb_w = cb_w*fig_sizefac
    #print(ax_w, ax_h)
    #___________________________________________________________________________
    # apply fig_size_fac
    ax_fh *= fig_sizefac
    ax_fw *= fig_sizefac
    if cb_pos=='right' : cb_fh *= fig_sizefac
    if cb_pos=='bottom': cb_fw *= fig_sizefac
    #fs_fac *= fig_sizefac
    
    ## factors to increase spaces (figure)
    #fig_fdl *= fig_sizefac
    #fig_fdr *= fig_sizefac
    #fig_fdt *= fig_sizefac
    #fig_fdb *= fig_sizefac
    ## factors to increase spaces (axes)
    #ax_fdl  *= fig_sizefac
    #ax_fdr  *= fig_sizefac
    #ax_fdt  *= fig_sizefac
    #ax_fdb  *= fig_sizefac
    ## factors to increase spaces (colorbars)
    #cb_fdl  *= fig_sizefac
    #cb_fdr  *= fig_sizefac
    #cb_fdt  *= fig_sizefac
    #cb_fdb  *= fig_sizefac
  
    # apply font size factor
    if fig_sizefac!=1.0 and fs_fac==1.0:
        fs_label *= fig_sizefac**(1/fig_sizefac)
        fs_title *= fig_sizefac**(1/fig_sizefac)
        fs_ticks *= fig_sizefac**(1/fig_sizefac)
    else:    
        fs_label *= fs_fac
        fs_title *= fs_fac
        fs_ticks *= fs_fac
    #print(fs_label, fs_title, fs_ticks)
    
    #___________________________________________________________________________
    # make vector of plot_cb if it has been true or false before
    # --> 1: plot cb,  0: do not plot cb    # 
    if isinstance(cb_plt, bool):
        if   cb_plt==True: 
            # only one colorbar for the entire panels
            if cb_plt_single:
                cb_plt = np.zeros((nx,ny))  
                cb_plt[-1,-1] = 1
            # each panel has a colorbar    
            else:    
                cb_plt = np.ones((nx,ny))  
        
        # no colorbars et all
        else: cb_plt = np.zeros((nx,ny))  
        cb_plt_idx = np.ones((nx,ny)).flatten()
        
    # only distinct panels have colorbar
    elif isinstance(cb_plt, list):
        cb_plt = np.array(cb_plt)
        if cb_plt.size!=nx*ny    : raise ValueError('Vector cb_plt has wrong length!')
        if   cb_plt.shape[0]==nx*ny: cb_plt = cb_plt.reshape(ny,nx).transpose()
        elif cb_plt.shape[0]==ny   : cb_plt = cb_plt.transpose()
        
        if np.sum(cb_plt!=0)>1: cb_plt_single=False
        else                  : cb_plt_single=True
        
        # check if there should be more than one independent colorbar
        cb_plt_idx = cb_plt.T.flatten()
        if any(cb_plt_idx>1) : 
            cb_plt_new = cb_plt_idx.copy()
            ncb = np.unique(cb_plt_idx)
            for ii in ncb:
                idx = np.where(cb_plt_idx==ii)[0]
                cb_plt_new[idx[:-1]]=0
        
            cb_plt = cb_plt_new.reshape(ny,nx).transpose()

    else:
        raise ValueError(' the format of cb_plt is not supported')
    cb_plt_idx = cb_plt_idx.astype(np.int16)
    
    #___________________________________________________________________________
    # create multiple of horizontal axes, colorbar properties
    ax_dl = np.array([ax_dl]*nx) *ax_fdl
    ax_dr = np.array([ax_dr]*nx) *ax_fdr
    cb_dl = np.array([cb_dl]*nx) *cb_fdl
    cb_dr = np.array([cb_dr]*nx) *cb_fdr
    ax_w  = np.array([ax_w ]*nx) *ax_fw
    cb_w  = np.array([cb_w ]*nx) *cb_fw
    
    # create multiple of vertical axes, colorbar properties
    ax_dt = np.array([ax_dt]*ny) *ax_fdt
    ax_db = np.array([ax_db]*ny) *ax_fdb
    cb_dt = np.array([cb_dt]*ny) *cb_fdt
    cb_db = np.array([cb_db]*ny) *cb_fdb
    ax_h  = np.array([ax_h ]*ny) *ax_fh
    cb_h  = np.array([cb_h ]*ny) *cb_fh
    
    #print('ax_w=',ax_w,', ax_dl=',ax_dl, ', ax_dr=',ax_dr)
    #print('ax_h=',ax_h,', ax_dt=',ax_dt, ', ax_db=',ax_db)
    #print('cb_w=',cb_w,', cb_dl=',cb_dl, ', cb_dr=',cb_dr)
    #print('cb_h=',cb_h,', cb_dt=',cb_dt, ', cb_db=',cb_db)
    
    #___________________________________________________________________________
    # adjust for shared axes
    if ax_sharex: ax_db[:-1] = 0 # only last row has x-axes label
    if ax_sharey: ax_dl[ 1:] = 0 # only first column has y-axes label

    #___________________________________________________________________________
    # adjust for one colorbar at the right or bottom
    if cb_pos=='right':
        ax_dr_s = ax_dr[0]
        cb_dl_s = cb_dl[0]
        cb_dr_s = cb_dr[0]
        cb_w_s  = cb_w[ 0]
        cb_h_s  = cb_h[ 0]
        fig_dr += cb_dl_s+cb_w_s+0.*cb_dr_s+ax_dl[0]
        
        # adjust for columns without colorbar
        delete_cb_space = cb_plt.sum(axis=1)==0
        cb_dl[delete_cb_space] = 0.0
        cb_dr[delete_cb_space] = 0.0
        cb_w[ delete_cb_space] = 0.0
        ax_dr[delete_cb_space==False] = 0.0
        
    if cb_pos=='bottom':
        ax_db_s = ax_db[-1]
        cb_h_s  = cb_h[0]
        cb_w_s  = ax_w[0]
        cb_db_s = cb_db[0]+ax_db[-1]
        cb_dt_s = cb_dt[0]
        #hcb_s  = hcb[0]
        fig_db += cb_db_s+cb_h_s+cb_dt_s
  
        # adjust for columns without colorbar
        delete_cb_space = cb_plt.sum(axis=0)==0
        cb_dt[delete_cb_space] = 0.0
        cb_db[delete_cb_space] = 0.0
        cb_h[ delete_cb_space] = 0.0
    
    #___________________________________________________________________________
    # determine ax position and fig dimensions in centimeters!!!
    x0 , y0  = fig_dl, -(fig_dt)
    x00, y00 = x0, y0
    pos_axcm = np.zeros((nx*ny,4))
    pos_cbcm = np.zeros((nx*ny,4))
    nn  = -1
    for jj in range(ny):
        #_______________________________________________________________________
        if cb_pos == 'right':
            y0 = y0 - (ax_dt[jj]+ax_h[jj])
            x0 = x00
            for ii in  range(nx):
                nn = nn+1
                x0 = x0 + ax_dl[ii]
                pos_axcm[nn,:] = [x0, y0, ax_w[ii], ax_h[jj]]
                x0 = x0 + ax_w[ii] + ax_dr[ii] + cb_dl[ii]
                pos_cbcm[nn,:] = [x0, y0, cb_w[ii], cb_h[jj]]
                x0 = x0 + cb_w[ii] +cb_dr[ii]
            y0 = y0 - (ax_db[jj])
        #_______________________________________________________________________
        elif cb_pos == 'bottom':
            y0 = y0 - (ax_dt[jj]+ax_h[jj])
            x0 = x00
            for ii in  range(nx):
                nn = nn+1
                x0 = x0 + ax_dl[ii]
                pos_axcm[nn,:] = [x0, y0, ax_w[ii], ax_h[jj]]
                pos_cbcm[nn,:] = [x0, y0-cb_dt[jj]-cb_h[jj], cb_w[ii], cb_h[jj]]
                x0 = x0 + ax_w[ii] + ax_dr[ii]
            y0 = y0 - cb_dt[jj]-cb_h[jj]-cb_db[jj]
    #___________________________________________________________________________        
    fig_w = x0 + fig_dr
    fig_h = y0 - fig_db
    #print('fig_w=',fig_w,' ,fig_h=', fig_h)
    
    #___________________________________________________________________________
    # transform from negative y axis to positive y axis
    fig_h = -fig_h
    pos_axcm[:,1] += fig_h
    pos_cbcm[:,1] += fig_h
    
    #___________________________________________________________________________
    # now convert axis and colorbar position from centimeter to figcoords
    cm2fig_x, cm2fig_y = 1./fig_w, 1./fig_h
    
    pos_ax = 1. * pos_axcm
    pos_cb = 1. * pos_cbcm
    
    pos_ax[:,0] = pos_axcm[:,0]*cm2fig_x
    pos_ax[:,1] = pos_axcm[:,1]*cm2fig_y
    pos_ax[:,2] = pos_axcm[:,2]*cm2fig_x
    pos_ax[:,3] = pos_axcm[:,3]*cm2fig_y
    
    pos_cb[:,0] = pos_cbcm[:,0]*cm2fig_x
    pos_cb[:,1] = pos_cbcm[:,1]*cm2fig_y
    pos_cb[:,2] = pos_cbcm[:,2]*cm2fig_x
    pos_cb[:,3] = pos_cbcm[:,3]*cm2fig_y
    
    #___________________________________________________________________________
    # make figure
    #print(fig_w, fig_h)
    hfig = plt.figure(figsize=(fig_w*cm2inch, fig_h*cm2inch), facecolor='white')
  
    #___________________________________________________________________________
    # make axes and there positioning 
    hax = [0]*(nx*ny)
    hcb = [0]*(nx*ny)
    nn = -1
    for jj in range(ny):
        for ii in range(nx):
            nn+=1
            #___________________________________________________________________
            # axes
            if isinstance(projection[nn], ccrs.CRS) and proj=='channel':
                hax[nn] = hfig.add_subplot(position=pos_ax[nn,:], projection=projection[nn], aspect='auto' )
            elif isinstance(projection[nn], ccrs.CRS):
                hax[nn] = hfig.add_subplot(position=pos_ax[nn,:], projection=projection[nn])    
            else:   
                if not ax_sharex and not ax_sharey:
                    hax[nn] = hfig.add_subplot(position=pos_ax[nn,:])
                else:
                    if nn==0:
                        hax[nn] = hfig.add_subplot(position=pos_ax[nn,:])
                    else:
                        x_share, y_share = None, None
                        if ax_sharex: x_share=hax[0]
                        if ax_sharey: y_share=hax[0]
                        hax[nn] = hfig.add_subplot(position=pos_ax[nn,:], sharex=x_share, sharey=y_share)
                    
                #hax[nn] = hfig.add_subplot(position=pos_ax[nn,:])    
                hax[nn].projection = projection[nn]                           
                hax[nn].sharex     = ax_sharex
                hax[nn].sharey     = ax_sharey
            # set position of axes
            hax[nn].set_position(pos_ax[nn,:])
            
            #if box is not None: hax[nn].set_extent(box, crs=projection[nn])
            if box is not None and isinstance(projection[nn], ccrs.CRS): 
                if  not isinstance(projection[nn], (ccrs.Orthographic, ccrs.NearsidePerspective ) ): #ccrs.NorthPolarStereo, ccrs.SouthPolarStereo,
                    hax[nn].set_extent(box, crs=ccrs.PlateCarree())
            
            #___________________________________________________________________
            # label
            hax[nn].set_xlabel(xlabel, fontsize=fs_label)
            hax[nn].set_ylabel(ylabel, fontsize=fs_label)
            #hax[nn].set_title('', fontsize=fs_title)
            matplotlib.rcParams['axes.titlesize'] = fs_title
            hax[nn].tick_params(labelsize=fs_ticks)
            hax[nn].fs_label = fs_label
            hax[nn].fs_ticks = fs_ticks
            hax[nn].fs_title = fs_title
            hax[nn].fig_dpi = hfig.dpi
            hax[nn].fig_width, hax[nn].fig_height = hfig.get_size_inches()
            
            #___________________________________________________________________
            # colorbar
            if cb_plt[ii,jj] != 0 and projection[0]!='index+depth' and projection[0]!='index+time' and projection[0]!='index+xy':
                hcb[nn] = hfig.add_subplot(position=pos_cb[nn,:])
                hcb[nn].set_position(pos_cb[nn,:])
                hcb[nn].tick_params(labelsize=fs_ticks)
                hcb[nn].fs_label=fs_label
                hcb[nn].fs_ticks=fs_ticks
                hcb[nn].fs_title=fs_title
                
            #___________________________________________________________________
            # ticks for axes
            # delete labels for shared axes
            hax[nn].henum=None 
            hax[nn].do_xlabel=True
            if ax_sharex and jj!=ny-1:
                hax[nn].ticklabel_format(axis='x',style='plain',useOffset=False)
                hax[nn].tick_params(labelbottom=False)
                hax[nn].set_xlabel('')
                hax[nn].do_xlabel=False
            
            hax[nn].do_ylabel=True
            if ax_sharey and ii!=0:
                hax[nn].ticklabel_format(axis='y',style='plain',useOffset=False)
                hax[nn].tick_params(labelleft=False)
                hax[nn].set_ylabel('')
                hax[nn].do_ylabel=False
            
            #___________________________________________________________________
            # add more variables to axes handle 
            hax[nn].ncol, hax[nn].nrow = nx, ny
            hax[nn].coli, hax[nn].rowi = ii, jj
            
            #___________________________________________________________________
            # ticks for colorbar 
            if hcb[nn] != 0:
                if cb_pos=='right':
                    hcb[nn].set_xticks([])
                    hcb[nn].yaxis.tick_right()
                    hcb[nn].yaxis.set_label_position("right")
                    hcb[nn].do_orient = 'vertical'
                elif cb_pos=='bottom':
                    hcb[nn].set_yticks([])
                    hcb[nn].do_orient = 'horizontal'
    #___________________________________________________________________________
    # if there is a single colorbar for the entire pannel, than stretch out the 
    # width/height of the colorbar over the size of the pannel
    # ther eis one single colorbar for the entire panel
    if cb_plt_single and hcb[-1] != 0:
        #_______________________________________________________________________
        # find axes center (!= figure center) --> make sure that singular colorbar
        # is centered over all the pannels
        x_ax_cent = pos_axcm[ 0,0] + 0.5*(pos_axcm[-1,0]+pos_axcm[-1,2]-pos_axcm[0,0])
        y_ax_cent = pos_axcm[-1,1] + 0.5*(pos_axcm[ 0,1]+pos_axcm[ 0,3]-pos_axcm[-1,1])
        
        #_______________________________________________________________________
        # case of vertical colorbar
        if cb_pos=='right':
            w0, h0 = cb_w_s, (pos_axcm[0,1]+pos_axcm[0,3]-pos_axcm[-1,1]) #h0 = cb_h_s
            x0, y0 = (pos_cbcm[-1,0]), (y_ax_cent-0.5*h0)
            pos_cb = np.array([x0*cm2fig_x, y0*cm2fig_y, w0*cm2fig_x, h0*cm2fig_y ])
            
            nn = -1
            hcb[nn].set_position(pos_cb)
            hcb[nn].tick_params(labelsize=fs_ticks)
            hcb[nn].set_xticks([])
            hcb[nn].yaxis.tick_right()
            hcb[nn].yaxis.set_label_position("right")
            
        #_______________________________________________________________________
        # case of horizontal colorbar
        elif cb_pos=='bottom':
            w0, h0 = (pos_axcm[-1,0]+pos_axcm[-1,2]-pos_axcm[0,0]), cb_h_s
            x0, y0 = (x_ax_cent-0.5*w0), (pos_cbcm[-1,1])
            pos_cb = np.array([ x0*cm2fig_x, y0*cm2fig_y, w0*cm2fig_x, h0*cm2fig_y ])
            
            nn = -1
            hcb[nn].set_position(pos_cb)
            hcb[nn].tick_params(labelsize=fs_ticks)
            hcb[nn].set_yticks([])
            
    # there is more than independent colorbar within the figure panel        
    elif not cb_plt_single and any(cb_plt.flatten()>1):
        
        nn = -1
        for jj in range(ny):
            for ii in range(nx):
                nn+=1
                if hcb[nn] != 0:
                    
                    auxidx = np.where(cb_plt_idx[nn]==cb_plt_idx)[0]
                    print(auxidx)
                    
                    #_______________________________________________________________________
                    # case of vertical colorbar
                    if cb_pos=='right':
                        ymax   = np.max(pos_axcm[auxidx,1])
                        yd     = np.max(pos_axcm[auxidx,3])
                        ymin   = np.min(pos_axcm[auxidx,1])
                        y_ax_cent = ymin + 0.5*(ymax+yd-ymin)
                        
                        w0, h0 = cb_w_s, (ymax+yd-ymin)
                        x0, y0 = (pos_cbcm[nn,0]), (y_ax_cent-0.5*h0)
                        pos_cb = np.array([x0*cm2fig_x, y0*cm2fig_y, w0*cm2fig_x, h0*cm2fig_y ])
                        
                        hcb[nn].set_position(pos_cb)
                        hcb[nn].tick_params(labelsize=fs_ticks)
                        hcb[nn].set_xticks([])
                        hcb[nn].yaxis.tick_right()
                        hcb[nn].yaxis.set_label_position("right")
                        
                    #_______________________________________________________________________
                    # case of horizontal colorbar
                    elif cb_pos=='bottom':
                        xmax   = np.max(pos_axcm[auxidx,0])
                        xd     = np.max(pos_axcm[auxidx,2])
                        xmin   = np.min(pos_axcm[auxidx,0])
                        x_ax_cent = ymin + 0.5*(ymax+yd-ymin)
                        
                        w0, h0 = (xmax+xd-xmin), cb_h_s
                        x0, y0 = (x_ax_cent-0.5*w0), (pos_cbcm[nn,1])
                        pos_cb = np.array([ x0*cm2fig_x, y0*cm2fig_y, w0*cm2fig_x, h0*cm2fig_y ])
                        
                        hcb[nn].set_position(pos_cb)
                        hcb[nn].tick_params(labelsize=fs_ticks)
                        hcb[nn].set_yticks([])
        
    #___________________________________________________________________________
    list_argout=[]
    if len(nargout)>0:
        for stri in nargout:
            try:
                list_argout.append(eval(stri))
            except:
                print(f" -warning-> variable {stri} was not found, could not be added as output argument") 
            
        return(list_argout)
    else:
        return



#
#
#_______________________________________________________________________________
def do_axes_enum(hax, do_enum, nrow, ncol, enum_dir='lr', enum_str=[], enum_x=[0.005], enum_y=[1.00], enum_opt=dict()):
    """
    --> do enumeration of axes
    
    Parameters: 
    
        :hax:       list, list of all axes handles

        :do_enum:   bool, switch for using enumeration 

        :nrow:      int, number of rows in multi panel plot

        :ncol :     int, number of column in multi panel plot

        :enum_dir:  str, (default: 'lr')  direction of numbering, 'lr' from left to 
                    right, 'ud' from up to down

        :enum_str:  list, (default: []) overwrite default enumeration strings        , 

        :enum_x:    float, (default: 0.005)  x position of enumeration string in 
                    axes coordinates

        :enum_y:    float, (default: 1.000)  y position of enumeration string in 
                    axes coordinates

        :enum_opt:  dict, (default: dict()) direct option for enumeration strings via kwarg 
    
    Returns:     
    
    ____________________________________________________________________________
    """
    enumn_optdefault = dict({'horizontalalignment':'right', 'verticalalignment':'bottom', 'fontsize':hax[0].fs_label})
    enumn_optdefault.update(enum_opt)
    if do_enum:
        #_______________________________________________________________________
        # make list that looks like [ '(a)', '(b)', '(c)', ... ]
        if len(enum_str)==0:
            #lett = "abcdefghijklmnopqrstuvwxyz"
            lett  = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
            lett += ["a2","b2","c2","d2","e2","f2","g2","h2","i2","j2","k2","l2","m2","n2","o2","p2","q2","r2","s2","t2","u2","v2","w2","x2","y2","z2"]
            lett = lett[0:len(hax)]
            enum_str = [None]*len(hax)
            
            # direction of numbering: left-->right
            if   enum_dir=='lr':
                for nn, ax in enumerate(hax): 
                    enum_str[nn] = "(%s)" % (lett[nn])
            # direction of numbering: up-->down
            elif enum_dir=='ud': 
                for nn, ax in enumerate(hax): 
                    aux_nn = np.floor(nn/ncol).astype(np.int32) + np.mod(nn,ncol)*nrow
                    enum_str[nn] = "(%s)" % (lett[aux_nn])
    
        #_______________________________________________________________________
        if len(enum_x)==1: enum_x = enum_x*len(hax)
        if len(enum_y)==1: enum_y = enum_y*len(hax)
    
        #_______________________________________________________________________
        # draw text
        for nn, ax in enumerate(hax):
            ht = hax[nn].text(enum_x[nn], enum_y[nn], enum_str[nn], transform = hax[nn].transAxes, 
                            **enumn_optdefault)
            # add text handle to axes to give possibility of changing text properties later
            # e.g. by hca[nn].axlab.set_fontsize(8)
            hax[nn].henum = ht
    #___________________________________________________________________________
    #return(hax)


    
#
#
#_______________________________________________________________________________
def do_data_prepare_unstruct(mesh, tri, data_plot, do_ie2n):
    """
    --> prepare data for plotting, augment periodic boundaries, interpolate from elements
        to nodes, kick out nan values from plotting 
    
    Parameters: 
    
        :mesh:      fesom2 mesh object,  with all mesh information 
        
        :data_plot: np.array of unstructured data 
        
        :tri:       matplotlib.tri triangulation object
                    - tri.mask_e_box ... bool np.array with element masking from regional box definition
                    - tri.mask_n_box ... bool np.array with vertices masking from regional box selection
    
    Returns:
    
        :data_plot: np.array of unstructured data, augmented with periodic boundary,
                    limited to regional box
                    
        :tri:       matplotlib.tri triangulation object
    
    ____________________________________________________________________________
    """  
    is_onvert = True
    
    #___________________________________________________________________________
    # data are on vertices
    if   data_plot.size==mesh.n2dn:
        is_onvert = True
        data_plot = np.hstack((data_plot,data_plot[mesh.n_pbnd_a]))
        
        # reindex vertices array to box limits
        if tri.mask_n_box is not None: data_plot = data_plot[tri.mask_n_box]
        
    #___________________________________________________________________________
    # data are on elements
    elif data_plot.size==mesh.n2de:
            
        # interpolate from elements to vertices --> cartopy plotting is faster
        if do_ie2n:
            is_onvert = True
            data_plot = grid_interp_e2n(mesh,data_plot)
            data_plot = np.hstack((data_plot,data_plot[mesh.n_pbnd_a]))
           
            # reindex vertices array to box limits
            if tri.mask_n_box is not None: data_plot = data_plot[tri.mask_n_box]
            
        # plot the data on elements    
        else:    
            is_onvert = False
            data_plot = np.hstack((data_plot[mesh.e_pbnd_0],data_plot[mesh.e_pbnd_a]))
            
            # reindex element array to box limits
            data_plot = data_plot[tri.mask_e_box]
    
    #___________________________________________________________________________
    # kick out triangles with Nan cut elements to box size        
    isnan   = np.isnan(data_plot)
    if not is_onvert:
        tri.mask_e_ok = isnan==False
        data_plot     = data_plot[tri.mask_e_ok]
    else:
        tri.mask_e_ok = np.any(isnan[tri.triangles], axis=1)==False
    del(isnan) 
    
    #___________________________________________________________________________
    return(data_plot, tri)    


#
#
#_______________________________________________________________________________
def do_data_prepare_vslice(hax_ii, data_ii, box_idx):
    """
    --> prepare data for plotting, augment periodic boundaries, interpolate from elements
        to nodes, kick out nan values from plotting 

    Parameters:
    
        :hax:       handle of current axes
    
        :data_ii:   xarray dataset object of axes ii, can contains the info 
                    data_ii[box_idx] of several defined boxes which can be selected
                    via box_idx index 

        :box_idx:   index of box selection in data_ii[box_idx]
    
    Returns:
    
        :data_x:    np.array, data for x-axis  

        :data_y:    np.array, data fox y-axes

        :data_plot: np.array, data to plot on regular vertical grid 
    
    ____________________________________________________________________________
    """  
    #___________________________________________________________________________
    # prepare vertical regular gridded data for plotting --> here data are index 
    # list --> defined by box index box_idx
    if box_idx is not None:
        vname = list(data_ii[box_idx].keys())[0]
        data_plot = data_ii[box_idx][vname].data.copy()
        data_y, str_ylabel = np.abs(data_ii[box_idx]['depth'].values) , 'Depth / m'
        #_______________________________________________________________________
        # data must be a transect 
        if 'dst' in list(data_ii[box_idx].variables):
            auxlat, auxlon = data_ii[box_idx]['lat'].values[[1,-2]], data_ii[box_idx]['lon'].values[[1,-2]]
            auxlat, auxlon = np.abs(np.diff(auxlat)), np.abs(np.diff(auxlon))
            auxlat, auxlon = auxlat/np.sqrt(auxlat**2+auxlon**2), auxlon/np.sqrt(auxlat**2+auxlon**2)
            angle = np.abs(-np.arctan2(auxlat, auxlon)*180/np.pi)
            if   angle > 80: data_x, str_xlabel = data_ii[box_idx]['lat'].values  , 'Latitude / deg'
            elif angle < 10: data_x, str_xlabel = data_ii[box_idx]['lon'].values  , 'Longitude / deg'
            else           : data_x, str_xlabel = data_ii[box_idx]['dst'].values , 'Distance / km'   
            del(auxlat, auxlon, angle)
            
        #_______________________________________________________________________
        # data can be any other vertical index
        else:
            if   'lat'  in list(data_ii[box_idx].coords): 
                data_x, str_xlabel = data_ii[box_idx]['lat'].values , 'Latitude / deg'
            elif 'lon'  in list(data_ii[box_idx].coords): 
                data_x, str_xlabel = data_ii[box_idx]['lon'].values , 'Longitude / deg'
            elif 'time'  in list(data_ii[box_idx].coords):     
                data_x, str_xlabel = data_ii[box_idx]['time'] , 'Time / year'
                # recompute xarray time vector into units of year
                totdayperyear = np.where(data_x.dt.is_leap_year, 366, 365)
                data_x = data_x.dt.year + (data_x.dt.dayofyear-data_x.dt.day[0])/totdayperyear
                data_plot = data_plot.transpose()
                del(totdayperyear)
             
    # data is a direct vertical profile and not a list of index profiles e,g MOC         
    else:
        vname = list(data_ii.data_vars)[0]
        data_plot = data_ii[vname].data.copy()
        
        #___Y-axis variables____________________________________________________
        # 
        if   'depth' in list(data_ii.coords):
            data_y, str_ylabel = np.abs(data_ii['depth'].values) , 'Depth / m'
        
        # dmoc y-varaible for zcoord remapping 
        elif 'ndens_zfh' in list(data_ii.coords): 
            data_y, str_ylabel = data_ii['ndens_zfh'].values , 'Depth / m'
             
        elif 'nz_rho'    in list(data_ii.coords): 
            #data_x, data_y, data_v, dum = do_ztransform_martin(mesh, data[ii])
            #data_x, data_y, data_v = do_ztransform_mom6(mesh, data[ii])
            data_x, data_y, data_v = do_ztransform_hydrography(mesh, data_ii)
            data_y = -data_y
            
        elif 'ndens_z'    in list(data_ii.coords): 
            data_x, data_y = do_ztransform(data_plot)
            data_plot = data_plot.copy()
            data_plot = data_plot[1:-1,:]
        
        # dmoc in density coordinates         
        elif 'dens'    in list(data_ii.coords): 
            data_y, str_ylabel = data_ii['dens'].values, '${\\sigma}_{2}$ pot. Density / kg${\\cdot}$m$^{-3}$'
            data_y, data_plot = data_y[1:-1], data_plot[1:-1,:]
        
        #___X-axis variables____________________________________________________
        # data must be a transect 
        if 'dst' in  list(data_ii.variables):
            auxlat, auxlon = data_ii['lat'].values[[0,-1]], data_ii['lon'].values[[0,1]]
            auxlat, auxlon = np.abs(np.diff(auxlat)), np.abs(np.diff(auxlon))
            auxlat, auxlon = auxlat/np.sqrt(auxlat**2+auxlon**2), auxlon/np.sqrt(auxlat**2+auxlon**2)
            angle = np.abs(-np.arctan2(auxlat, auxlon)*180/np.pi)
            if   angle > 80: data_x, str_xlabel = data_ii['lat'].values  , 'Latitude / deg'
            elif angle < 10: data_x, str_xlabel = data_ii['lon'].values  , 'Longitude / deg'
            else           : data_x, str_xlabel = data_ii['dst'].values , 'Distance / km'   
            del(auxlat, auxlon, angle)
            
        # data can be any other vertical index    
        else:     
            if   'lat'  in list(data_ii.coords): 
                data_x, str_xlabel = data_ii['lat'].values , 'Latitude / deg'
            elif 'lon'  in list(data_ii.coords): 
                data_x, str_xlabel = data_ii['lon'].values , 'Longitude / deg'
            elif 'time'  in list(data_ii[box_idx].coords):     
                data_x, str_xlabel = data_ii[box_idx]['time'] , 'Time / year'
                # recompute xarray time vector into units of year
                totdayperyear = np.where(data_x.dt.is_leap_year, 366, 365)
                data_x = data_x.dt.year + (data_x.dt.dayofyear-data_x.dt.day[0])/totdayperyear  
                data_plot = data_plot.transpose()
                
            if 'ndens_zfh' in list(data_ii.coords): 
                data_x = np.ones(data_y.shape)*data_x

    #___________________________________________________________________________
    if hax_ii.do_xlabel: hax_ii.set_xlabel(str_xlabel)
    if hax_ii.do_ylabel: hax_ii.set_ylabel(str_ylabel)
    
    #___________________________________________________________________________
    return(data_x, data_y, data_plot)



#
#
#_______________________________________________________________________________
def do_data_norm(cinfo, do_rescale):
    """
    --> prepare renormation object, for log10 or slog10  
    
    Parameters:
    
        :cinfo:     None, dict() (default: None), dictionary with colorbar information. 
                    Information that are given are used, others are computed. cinfo dictionary 
                    entries can be,
                    
                    - cinfo['cmin'], cinfo['cmax'], cinfo['cref'] ... scalar min, max, reference value
                    - cinfo['crange'] ... list with [cmin, cmax, cref] overrides scalar values 
                    - cinfo['cnum']   ... minimum number of colors
                    - cinfo['cstr']   ... name of colormap see in sub_colormap_c2c.py
                    - cinfo['cmap']   ... colormap object ('wbgyr', 'blue2red, 'jet' ...)
                    - cinfo['clevel'] ... color level array
    
        :do_rescale: bool, str, np.array (defaul: False) do scaling of colorbar
                    - False      ... scale data automatically scientifically by 10^x, for data data larger 10^3 and smaller 10^-3
                    - log10      ... do logaritmic scaling (mcolors.LogNorm)
                    - slog10     ... do symetric logarithmic scaling (mcolors.SymLogNorm)
                    - np.array() ... scale colorbar stepwise according to values in np.array allows also for non-linear colortick steps (mcolors.BoundaryNorm)
    
    Returns:
        :which_norm:  None or renormation object
    
    ____________________________________________________________________________
    """      
    #___________________________________________________________________________
    which_norm = None
    if isinstance(do_rescale, str):
        if   do_rescale =='log10':
                which_norm = mcolors.LogNorm(vmin=cinfo['clevel'][0], vmax=cinfo['clevel'][-1])
                
        elif do_rescale =='slog10':    
                print(np.min(np.abs(cinfo['clevel'][cinfo['clevel']!=0])))
                which_norm = mcolors.SymLogNorm(np.min(np.abs(cinfo['clevel'][cinfo['clevel']!=0])),
                                                linscale=1.0, 
                                                vmin=cinfo['clevel'][0], vmax=cinfo['clevel'][-1], 
                                                clip=True)
                
    elif isinstance(do_rescale, np.ndarray):
            which_norm = mcolors.BoundaryNorm(do_rescale, len(do_rescale)-1, clip=True)

    #else:  
        #which_norm = mcolors.NoNorm(vmin=cinfo['clevel'][0], vmax=cinfo['clevel'][ -1], clip=False)
    #___________________________________________________________________________
    return(which_norm)



#
#
#_______________________________________________________________________________
def do_plt_data(hax_ii, do_plt, tri, data_plot, cinfo_plot, which_norm_plot,
                plt_opt  =dict(), 
                plt_contb=False, pltcb_opt=dict(), 
                plt_contf=False, pltcf_opt=dict(),
                plt_contr=False, pltcr_opt=dict(),
                plt_contl=False, pltcl_opt=dict()):
    """
    --> plot triangular data based on tripcolor or tricontourf
    
    Parameters: 
    
        :hax_ii:        handle of axes ii

        :do_plt:        str, (default: tpc)
                        - tpc ... make pseudocolor plot (tripcolor)
                        - tcf ... make contourf coor plot (tricontourf)  , # tpc:tripcolor, tcf:tricontourf    

        :tri:           matplotlib.tri triangulation object
                        - tri.mask_e_ok...provide mask with nan values, that describe the bottom limited to regional box

        :data_plot:     np.array of unstructured data, augmented with periodic boundary,

        :cinfo_plot:    None, dict() (default: None), dictionary with colorbar information. 
                        Information that are given are used, others are computed. cinfo dictionary 
                        entries can be,
                        
                        - cinfo['cmin'], cinfo['cmax'], cinfo['cref'] ... scalar min, max, reference value
                        - cinfo['crange'] ... list with [cmin, cmax, cref] overrides scalar values 
                        - cinfo['cnum']   ... minimum number of colors
                        - cinfo['cstr']   ... name of colormap see in sub_colormap_c2c.py
                        - cinfo['cmap']   ... colormap object ('wbgyr', 'blue2red, 'jet' ...)
                        - cinfo['clevel'] ... color level array
    
        :which_norm_plot:     None or renormation object

        :plt_opt:       dict, (default: dict()) additional options that are given to tripcolor 
                        or tricontourf via the kwarg argument

        :plt_contb:     bool, (default: False) overlay thin contour lines of all colorbar steps (background)

        :pltcb_opt:     dict, (default: dict()) background contour line option

        :plt_contf:     bool, (default: False) overlay thicker contour lines of the main colorbar steps (foreground)

        :pltcf_opt:     dict, (default: dict()) foreground contour line option

        :plt_contr:     bool, (default: False) overlay thick contour lines of reference color steps (reference)

        :pltcr_opt:     dict, (default: dict()) reference contour line option

        :plt_contl:     bool, (default: False) label overlayed  contour linec plot

        :pltcl_opt:     dict, (default: dict()) additional options that are given to clabel via the kwarg argument

    Returns:
    
        :h0:            return matplotlib handle of plot
    
    ____________________________________________________________________________
    """      
    h0=None
    if np.sum(tri.mask_e_ok)==0: return(h0)
    #___________________________________________________________________________
    # plot tripcolor
    if   do_plt in ['tpc','pc'] or (do_plt in ['tcf','cf'] and not tri.x.size==data_plot.size):
        plt_optdefault = dict({'shading':'gouraud', 'zorder':1})
        plt_optdefault.update(plt_opt)
        
        # pcolor plot in combination with shading :gouraud and orthographic projection
        # leads to an blow up of the plotting therefor change to flat shading 
        if tri.x.size!=data_plot.size or isinstance(hax_ii.projection, (ccrs.Orthographic, ccrs.NearsidePerspective)): 
            plt_optdefault.update({'shading':'flat'})
        
        # if which_normplot is specified like in case of log10 and slog10 scaling
        # vmin and vmax argumetns are not allows
        cminmax=dict()
        if which_norm_plot is None:cminmax.update({'vmin':cinfo_plot['clevel'][0], 'vmax':cinfo_plot['clevel'][-1]})
        
        h0 = hax_ii.tripcolor(tri.x, tri.y, tri.triangles[tri.mask_e_ok,:], data_plot,
                              cmap=cinfo_plot['cmap'], norm = which_norm_plot,
                              **cminmax, **plt_optdefault)

    #___________________________________________________________________________
    # plot tricontour 
    elif do_plt in ['tcf','cf']: 
        plt_optdefault = dict({'zorder':1})
        plt_optdefault.update(plt_opt)
    
        # supress warning message when compared with nan
        with np.errstate(invalid='ignore'):
            data_plot[data_plot<cinfo_plot['clevel'][ 0]] = cinfo_plot['clevel'][ 0]
            data_plot[data_plot>cinfo_plot['clevel'][-1]] = cinfo_plot['clevel'][-1]
                
        h0 = hax_ii.tricontourf(tri.x, tri.y, tri.triangles[tri.mask_e_ok,:], data_plot,
                                levels=cinfo_plot['clevel'], cmap=cinfo_plot['cmap'], extend='both',
                                norm=which_norm_plot, **plt_optdefault) 
        
    else: 
        raise ValueError(' --> this do_plt={:s} value is not valid'.format(do_plt))            
    
    #___________________________________________________________________________
    # overlay background contour lines, very thin lines 
    if plt_contb and tri.x.size==data_plot.size:
        pltcb_optdefault=dict({'colors':'k', 'linestyles':'solid', 'linewidths':0.1, 'zorder':2})
        pltcb_optdefault.update(pltcb_opt)
        h0cb = hax_ii.tricontour(tri.x, tri.y, tri.triangles[tri.mask_e_ok,:], data_plot,
                                levels=cinfo_plot['clevel'], **pltcb_optdefault) 
    
    #___________________________________________________________________________
    # overlay foreground contour lines, of colorbar steps thicker line 
    if plt_contf and tri.x.size==data_plot.size:
        pltcf_optdefault=dict({'colors':'k', 'linestyles':'solid', 'linewidths':0.5, 'zorder':2})
        pltcf_optdefault.update(pltcf_opt)
        h0cf = hax_ii.tricontour(tri.x, tri.y, tri.triangles[tri.mask_e_ok,:], data_plot,
                                levels=cinfo_plot['clab'], **pltcf_optdefault) 
        
        #_______________________________________________________________________
        if plt_contl:
            pltcl_optdefault=dict({'inline':1, 'inline_spacing':1, 'fontsize':6, 'fmt':'%1.2f', 'zorder':3})
            pltcl_optdefault.update(pltcl_opt)
            h0cft = hax_ii.clabel(h0cf, h0cf.levels, **pltcl_optdefault)
    
    #___________________________________________________________________________
    # overlay reference contour lines, of colorbar reference center value
    if plt_contr and tri.x.size==data_plot.size:
        pltcr_optdefault=dict({'colors':'k', 'linestyles':'solid', 'linewidths':1.5, 'zorder':2})
        pltcr_optdefault.update(pltcr_opt)
        h0cr = hax_ii.tricontour(tri.x, tri.y, tri.triangles[tri.mask_e_ok,:], data_plot,
                                levels=[cinfo_plot['cref']], **pltcr_optdefault) 
        
        #_______________________________________________________________________
        if plt_contl:
            pltcl_optdefault=dict({'inline':1, 'inline_spacing':1, 'fontsize':6, 'fmt':'%1.2f', 'zorder':3})
            pltcl_optdefault.update(pltcl_opt)
            h0crt= hax_ii.clabel(h0cr, h0cr.levels, **pltcl_optdefault)
    
    #___________________________________________________________________________
    return(h0)



#
#
#_______________________________________________________________________________
def do_plt_datareg(hax_ii, do_plt, data_x, data_y, data_plot, cinfo_plot, which_norm_plot, 
                plt_opt=dict(), which_transf=None, 
                plt_contb=False, pltcb_opt=dict(), 
                plt_contf=False, pltcf_opt=dict(),
                plt_contr=False, pltcr_opt=dict(),
                plt_contl=False, pltcl_opt=dict()):
    """
    --> plot regular gridded data (binned, coarse grained data) via pcolormesh and contourf

    Parameters: 
    
        :hax_ii:        handle of axes ii

        :do_plt:        str, (default: tpc)
                        - tpc ... make pseudocolor plot (tripcolor)
                        - tcf ... make contourf coor plot (tricontourf)

        :data_x:        regular longitude array
        
        :data_y:        regular latitude array
        
        :data_plot:     np.array of regular gridded data

        :cinfo_plot:    None, dict() (default: None), dictionary with colorbar information. 
                        Information that are given are used, others are computed. cinfo dictionary 
                        entries can be,
                        
                        - cinfo['cmin'], cinfo['cmax'], cinfo['cref'] ... scalar min, max, reference value
                        - cinfo['crange'] ... list with [cmin, cmax, cref] overrides scalar values 
                        - cinfo['cnum']   ... minimum number of colors
                        - cinfo['cstr']   ... name of colormap see in sub_colormap_c2c.py
                        - cinfo['cmap']   ... colormap object ('wbgyr', 'blue2red, 'jet' ...)
                        - cinfo['clevel'] ... color level array
    
        :which_norm:    None or renormation object

        :plt_opt:       dict, (default: dict()) additional options that are given to tripcolor 
                        or tricontourf via the kwarg argument

        :plt_contb:     bool, (default: False) overlay thin contour lines of all colorbar steps (background)

        :pltcb_opt:     dict, (default: dict()) background contour line option

        :plt_contf:     bool, (default: False) overlay thicker contour lines of the main colorbar steps (foreground)

        :pltcf_opt:     dict, (default: dict()) foreground contour line option

        :plt_contr:     bool, (default: False) overlay thick contour lines of reference color steps (reference)

        :pltcr_opt:     dict, (default: dict()) reference contour line option

        :plt_contl:     bool, (default: False) label overlayed  contour linec plot

        :pltcl_opt:     dict, (default: dict()) additional options that are given to clabel via the kwarg argument

    Returns:
    
        h0:             return matplotlib handle of plot
    
    ____________________________________________________________________________
    """      
    h0=None
    #___________________________________________________________________________
    # plot pcolor
    if   do_plt in ['tpc','pc']:
        #plt_optdefault = dict({'shading':'gouraud'})
        plt_optdefault = dict({'shading':'nearest', 'zorder':1})
        plt_optdefault.update(plt_opt)
        
        if 'shading' in plt_optdefault:
            if plt_optdefault['shading']=='flat':
                data_plot = (data_plot[1:,1:] + data_plot[:-1,:-1])*0.5
        
        # if which_normplot is specified like in case of log10 and slog10 scaling
        # vmin and vmax argumetns are not allows
        cminmax=dict()
        if which_norm_plot is None:cminmax.update({'vmin':cinfo_plot['clevel'][0], 'vmax':cinfo_plot['clevel'][-1]})
        
        # the transform=which_transf options in combination with pcolormesh seems 
        # only to work in case of horizontal cartopy plot not in vertical slice 
        # plot even when which_transf=None
        if isinstance(hax_ii.projection, ccrs.CRS): plt_optdefault.update({'transform':which_transf})
        h0 = hax_ii.pcolormesh(data_x, data_y, data_plot, 
                               cmap=cinfo_plot['cmap'], norm=which_norm_plot, **cminmax, **plt_optdefault)
        
        #0 = hax_ii.pcolormesh(data_x, data_y, data_plot,
                              #cmap=cinfo_plot['cmap'],
                              #norm=which_norm_plot, transform=which_transf, **cminmax, **plt_optdefault)
        
    #___________________________________________________________________________
    # plot contourf 
    elif do_plt in ['tcf','cf']: 
        plt_optdefault = dict({'zorder':1})
        plt_optdefault.update(plt_opt)
    
        # supress warning message when compared with nan
        with np.errstate(invalid='ignore'):
            data_plot[data_plot<cinfo_plot['clevel'][ 0]] = cinfo_plot['clevel'][ 0]
            data_plot[data_plot>cinfo_plot['clevel'][-1]] = cinfo_plot['clevel'][-1]
                
        h0 = hax_ii.contourf(data_x, data_y, data_plot,
                                levels=cinfo_plot['clevel'], cmap=cinfo_plot['cmap'], extend='both',
                                norm=which_norm_plot, transform=which_transf, **plt_optdefault) 
        
    else: 
        raise ValueError(' --> this do_plt={:s} value is not valid'.format(do_plt))            
    
    #___________________________________________________________________________
    # solve problem between  using lat_bnd & lon_bnd and lat & lon
    if   np.ndim(data_x) == 1:
        if   data_plot.shape[1] == data_x.shape[0]  : data_x0=data_x
        elif data_plot.shape[1] == data_x.shape[0]-1: data_x0=(data_x[1:] + data_x[:-1])*0.5
        if   data_plot.shape[0] == data_y.shape[0]  : data_y0=data_y
        elif data_plot.shape[0] == data_y.shape[0]-1: data_y0=(data_y[1:] + data_y[:-1])*0.5
    elif np.ndim(data_x) == 2: data_x0, data_y0=data_x, data_y
        
    #___________________________________________________________________________
    # overlay background contour lines, very thin lines 
    if plt_contb:
        pltcb_optdefault=dict({'colors':'k', 'linestyles':'solid', 'linewidths':0.1, 'zorder':2})
        pltcb_optdefault.update(pltcb_opt)
        h0cb = hax_ii.contour(data_x0, data_y0, data_plot,
                                levels=cinfo_plot['clevel'], transform=which_transf, **pltcb_optdefault) 
    
    #___________________________________________________________________________
    # overlay foreground contour lines, of colorbar steps thicker line 
    if plt_contf:    
        pltcf_optdefault=dict({'colors':'k', 'linestyles':'solid', 'linewidths':0.5, 'zorder':2})
        pltcf_optdefault.update(pltcf_opt)
        h0cf = hax_ii.contour(data_x0, data_y0, data_plot,
                                levels=cinfo_plot['clab'], transform=which_transf, **pltcf_optdefault) 
        #_______________________________________________________________________
        if plt_contl:
            pltcl_optdefault=dict({'inline':1, 'inline_spacing':1, 'fontsize':6, 'fmt':'%1.2f', 'zorder':3})
            pltcl_optdefault.update(pltcl_opt)
            hax_ii.clabel(h0cf, h0cf.levels, **pltcl_optdefault)
    
    #___________________________________________________________________________
    # overlay reference contour lines, of colorbar reference center value
    if plt_contr:    
        pltcr_optdefault=dict({'colors':'k', 'linestyles':'solid', 'linewidths':1.5, 'zorder':2})
        pltcr_optdefault.update(pltcr_opt)
        h0cr = hax_ii.contour(data_x0, data_y0, data_plot,
                                levels=[cinfo_plot['cref']], transform=which_transf, **pltcr_optdefault) 
        #_______________________________________________________________________
        if plt_contl:
            pltcl_optdefault=dict({'inline':1, 'inline_spacing':1, 'fontsize':6, 'fmt':'%1.2f', 'zorder':3})
            pltcl_optdefault.update(pltcl_opt)
            hax_ii.clabel(h0cr, h0cr.levels, **pltcl_optdefault)
            
    #___________________________________________________________________________
    return(h0)



#
#
#_______________________________________________________________________________

def do_plt_quiver(hax_ii, do_quiv, tri, data_plot_u, data_plot_v, 
                  cinfo_plot, norm_plot, quiv_scalfac=1, quiv_arrwidth=0.25, quiv_dens=0.4, 
                  quiv_smax=10, quiv_shiftL=2, quiv_smooth=2, 
                  quiv_opt=dict()):
    """
    --> plot triangular data as quiver plot 
    
    Parameters:
    
        :hax_ii:        handle of axes ii

        :do_quiv:       bool, do cartopy quiver plot
        
        :tri:           matplotlib.tri triangulation object
                        - tri.mask_e_ok...provide mask with nan values, that describe the bottom limited to regional box

        :data_plot_u:   np.array of unstructured zonal vector component

        :data_plot_v:   np.array of unstructured meridional vector component

        :cinfo_plot:    None, dict() (default: None), dictionary with colorbar information. 
                        Information that are given are used, others are computed. cinfo dictionary 
                        entries can be,
                        
                        - cinfo['cmin'], cinfo['cmax'], cinfo['cref'] ... scalar min, max, reference value
                        - cinfo['crange'] ... list with [cmin, cmax, cref] overrides scalar values 
                        - cinfo['cnum']   ... minimum number of colors
                        - cinfo['cstr']   ... name of colormap see in sub_colormap_c2c.py
                        - cinfo['cmap']   ... colormap object ('wbgyr', 'blue2red, 'jet' ...)
                        - cinfo['clevel'] ... color level array

        :norm_plot:     None or renormation object
        
        :quiv_scalfac:  float, (default: 1.0)  bigger means larger arrows

        :quiv_arrwidth: float, (default: 0.25) scale arrow width

        :quiv_dens:     float, (default: 0.5)  larger mean more excluded arrows

        :quiv_smax:     float, (default: 10) small arrow are scaled strong with factor smax, its off when smax=1

        :quiv_shiftL:   float, (default: 2) shift smothing function to the left

        :quiv_smooth:   float, (default: 2) slope of transitions zone, smaller value steeper transition

        :quiv_opt:      dict, (default: dict()) additional options that are given to quiver plot routine

    Returns:

        :h0:   return handle of quiver plot
    
    ____________________________________________________________________________
    """      
    h0=None
    if do_quiv: 
        #_______________________________________________________________________
        # prepare quiver data
        data_plot_n              = np.sqrt(data_plot_u**2 + data_plot_v**2)
        data_plot_u, data_plot_v = data_plot_u/data_plot_n, data_plot_v/data_plot_n
        data_plot_n[data_plot_n<cinfo_plot['clevel'][0]]  = cinfo_plot['clevel'][0] #+np.finfo(np.float32).eps
        data_plot_n[data_plot_n>cinfo_plot['clevel'][-1]] = cinfo_plot['clevel'][-1]#-np.finfo(np.float32).eps
        data_plot_u, data_plot_v = data_plot_u*data_plot_n, data_plot_v*data_plot_n
        
        nmax  = np.nanmax(data_plot_n)
        data_plot_u, data_plot_v = data_plot_u/nmax, data_plot_v/nmax

        # scale up weaker flow vectors stronger so that also weaker flows become more visible
        # if quiv_scal=1 this scaling is switched off 
        fac   = (1.0 - np.tanh(((data_plot_n/nmax*np.pi*4)-np.pi*2 + 2*np.pi/quiv_shiftL )/quiv_smooth) )/2.0
        fac   = fac - np.nanmin(fac)
        fac   = fac/np.nanmax(fac)
        fac   = fac*(quiv_smax-1.0) + 1.0
        data_plot_u, data_plot_v = data_plot_u*fac, data_plot_v*fac
        
                
        # convert into cartopy projection frame 
        if data_plot_u.size == tri.xorig.size: 
            isonvert=True
            tri0x, tri0y = tri.x, tri.y
            data_plot_u, data_plot_v = hax_ii.projection.transform_vectors(ccrs.PlateCarree(), 
                                                                tri.xorig, tri.yorig, 
                                                                data_plot_u, data_plot_v)
        else:
            isonvert=False
            triangles = tri.triangles[tri.mask_e_ok,:]
            tri0x    , tri0y     = tri.x[    triangles].sum(axis=1)/3.0, tri.y[    triangles].sum(axis=1)/3.0
            tri0xorig, tri0yorig = tri.xorig[triangles].sum(axis=1)/3.0, tri.yorig[triangles].sum(axis=1)/3.0,
            data_plot_u, data_plot_v = hax_ii.projection.transform_vectors(ccrs.PlateCarree(), 
                                                                tri0xorig, tri0yorig, 
                                                                data_plot_u, data_plot_v)
            del(triangles, tri0xorig, tri0yorig)
            
            
        # kick out nan values from quiver coordinates 
        mask_nan = np.isnan(data_plot_u) == False
        tri0x, tri0y = tri0x[mask_nan], tri0y[mask_nan]
        data_plot_u, data_plot_v, data_plot_n = data_plot_u[mask_nan], data_plot_v[mask_nan], data_plot_n[mask_nan]
        
        ## kick out to small arrows
        #mean, std   = np.nanmean(data_plot_n), np.nanstd(data_plot_n)
        #mask_quiv   = data_plot_n>mean-std*quiv_excl
        #tri0x       = tri0x[mask_quiv], 
        #tri0y       = tri0y[mask_quiv],                         
        #data_plot_u = data_plot_u[mask_quiv], 
        #data_plot_v = data_plot_v[mask_quiv], 
        #data_plot_n = data_plot_n[mask_quiv]
        
        # kick out arrows based on density 
        if quiv_dens is not None and tri.narea is not None:
            if isonvert:
                r0      = 1/(np.sqrt(tri.narea[mask_nan]))
            else:
                aux_earea = tri.earea[tri.mask_e_ok]
                r0      = 1/(np.sqrt(aux_earea[mask_nan]))
                del(aux_earea)
                
            mask_quiv   = np.random.rand(tri0x.size)>r0/np.max(r0)*quiv_dens #1.5
            #mask_quiv = np.logical_and(isok,mask_quiv)
            tri0x       = tri0x[mask_quiv], 
            tri0y       = tri0y[mask_quiv],                         
            data_plot_u = data_plot_u[mask_quiv], 
            data_plot_v = data_plot_v[mask_quiv], 
            data_plot_n = data_plot_n[mask_quiv]
            
        #_______________________________________________________________________
        # try to do scaling projection space dependent
        # Define the geographic coordinates bounding the area of interest
        min_x, max_x = hax_ii.get_xlim()
        min_y, max_y = hax_ii.get_ylim()
        ddx  , ddy   = max_x-min_x , max_y-min_y
          
        min_x += ddx*0.025  
        min_y += ddy*0.025
        max_x -= ddx*0.025  
        max_y -= ddy*0.025
        
        # Transform the minimum and maximum points
        min_lon, dum = ccrs.PlateCarree().transform_point(min_x, (min_y+max_y)/2, src_crs=hax_ii.projection)
        max_lon, dum = ccrs.PlateCarree().transform_point(max_x, (min_y+max_y)/2, src_crs=hax_ii.projection)
        dum, min_lat = ccrs.PlateCarree().transform_point((min_x+max_x)/2, min_y, src_crs=hax_ii.projection)
        dum, max_lat = ccrs.PlateCarree().transform_point((min_x+max_x)/2, max_y, src_crs=hax_ii.projection)
        
        # Calculate the distance in kilometers using the scale factor
        dlon = np.abs(max_lon - min_lon)  # Convert meters to kilometers
        dlat = np.abs(max_lat - min_lat)  # Convert meters to kilometers
        
        dy   = dlat*np.pi*6371/180
        dx   = dlon*np.pi*6371/180*np.cos(np.deg2rad( (min_lat+max_lat)/2 ))
        
        #_______________________________________________________________________
        # add quiver plot 
        max_dim = np.min([dx,dy])*10
        if quiv_scalfac is not None: quiv_scalfac = 1/max_dim/quiv_scalfac
        if quiv_arrwidth is not None: quiv_arrwidth = max_dim*quiv_arrwidth
        
        quiv_optdefault=dict({'edgecolor':'k', 'linewidth':0.10, 'width': quiv_arrwidth , 'units':'xy', \
                              'scale_units':'xy', 'angles':'xy', 'scale': quiv_scalfac}) 
        quiv_optdefault.update(quiv_opt)
        
        h0=hax_ii.quiver(tri0x, tri0y, 
                        data_plot_u, data_plot_v, 
                        data_plot_n,
                        cmap = cinfo_plot['cmap'],                    
                        norm = norm_plot,
                        zorder=10,
                        **quiv_optdefault, 
                        )
        
        h0.set_clim([cinfo_plot['clevel'][0],cinfo_plot['clevel'][-1]])
        del(tri0x, tri0y)    
    return(h0)



#
#
#_______________________________________________________________________________
def do_plt_bot(hax_ii, do_bot, tri=None, data_x=None, data_y=None, data_plot=None, ylim=None, bot_opt=dict()):
    """
    --> plot bottom mask

    Parameters: 
    
        :hax_ii:        handle of axes ii

        :do_bot:        bool, (default: True), overlay topographic bottom mask
    
        :tri:           matplotlib.tri triangulation object (default=None)
                        - tri.mask_e_ok...provide mask with nan values, that describe the bottom limited to regional box

        :data_x:        regular longitude array (default=None)

        :data_y:        regular latitude array (default=None)

        :data_plot:     np.array of regular gridded data (default=None)

        :ylim:          list, (default=None), overwrite limit of yaxis

        :bot_opt:       dict, (default: dict()) additional options that are given to 
                        the bottom mask plotting via kwarg
    
    Returns: 
    
        :h0:            return handle of bottom plot
    
    ____________________________________________________________________________
    """      
    from matplotlib.colors import ListedColormap
    h0=None
    
    # plot bottom mask for cartopy plot
    if isinstance(hax_ii.projection, ccrs.CRS) and tri is not None:
        if do_bot and np.any(tri.mask_e_ok==False):
            
            bot_optdefault = dict({'facecolors': [0.8, 0.8, 0.8], 'linewidth':0.1, 'zorder':4})
            bot_optdefault.update(bot_opt)
            
            # create single color colormap when options like 'facecolor', 'facecolors', 
            # 'color', 'colors' are present
            cmap = None
            rmv  = []
            for ii in bot_optdefault.keys():
                if ii in ['facecolor', 'facecolors', 'color', 'colors']:
                    cmap = ListedColormap(bot_optdefault[ii])
                    rmv.append(ii)
            if cmap is not None: bot_optdefault.update({'cmap':cmap})
            
            # remove facecolor string from dictionary since its doesnt exist for tripcolor
            if len(rmv)!=0:
                for ii in rmv:
                    del(bot_optdefault[ii])
            
            #h0 = hax_ii.triplot(tri.x, tri.y, tri.triangles[e_ok_mask==False,:], **bot_optdefault)
            h0 = hax_ii.tripcolor(tri.x, tri.y, tri.triangles[tri.mask_e_ok==False,:], np.ones(np.sum(tri.mask_e_ok==False)), **bot_optdefault)
    
    # plot bottom mask for index+depth+xy
    elif hax_ii.projection=='index+depth+xy':
        
        bot_optdefault = dict({'color':[0.5, 0.5, 0.5], 'edgecolor':'k', 'linewidth':1.0, 'zorder':4})
        bot_optdefault.update(bot_opt)
            
        if data_x is None or data_plot is None:
            raise ValueError(' cant plot bottom mask for index+depth+xy without data_x and data_plot!')
        
        # compute bottom line based on NaN values
        aux = np.isnan(data_plot)==False
        aux = aux.sum(axis=0)
        aux[aux!=0]=aux[aux!=0]-1
        bottom = data_y[aux]
        del(aux)
        
        # smooth bottom patch
        #filt=np.array([1,2,3,2,1]) #np.array([1,2,1])
        #filt=filt/np.sum(filt)
        #aux = np.concatenate( (np.ones((filt.size,))*bottom[0],bottom,np.ones((filt.size,))*bottom[-1] ) )
        #aux = np.convolve(aux,filt,mode='same')
        #bottom = aux[filt.size:-filt.size]
        #del(filt, aux)
        
        h0 = hax_ii.fill_between(data_x, bottom, data_y[-1], **bot_optdefault)#,alpha=0.95)
    
    # plot bottom mask for zmoc
    elif 'zmoc' in hax_ii.projection or 'dmoc+depth' in hax_ii.projection:
        bot_optdefault = dict({'color':[0.5, 0.5, 0.5], 'edgecolor':'k', 'linewidth':1.0, 'zorder':4})
        bot_optdefault.update(bot_opt)
            
        if ylim==None: maxbot=np.nanmax(data_y)
        else        : maxbot=ylim[-1]
            
        bottom = data_plot
        h0 = hax_ii.fill_between(data_x, bottom, maxbot, **bot_optdefault)#,alpha=0.95)
    
    return(h0)
    
#
#
#_______________________________________________________________________________
def do_plt_topo(hax_ii, do_topo, data_topo, mesh, tri, 
                plt_opt=dict(), 
                plt_contb=True,  pltcb_opt=dict(), 
                plt_contl=False, pltcl_opt=dict()):
    """
    --> plot topography contour or pcolor

    Parameters: 
    
        :hax_ii:    handle of axes ii

        :do_topo:   bool, (default: True), overlay model topography in quiver plots

        :data_topo: np.array with unstructured data of model topogrpahy 

        :mesh:      fesom2 mesh object,  with all mesh information 

        :tri:       matplotlib.tri triangulation object
                    - tri.mask_e_box ... bool np.array with element masking from regional box definition
                    - tri.mask_n_box ... bool np.array with vertices masking from regional box selection
    
        :plt_opt:   dict, (default: dict()) additional options that are given to tripcolor 
                    or tricontourf via the kwarg argument

        :plt_contb: bool, (default: False) overlay thin contour lines of all colorbar steps (background)

        :pltcb_opt: dict, (default: dict()) background contour line option

        :plt_contl: bool, (default: False) label overlayed  contour linec plot

        :pltcl_opt: dict, (default: dict()) additional options that are given to clabel via the kwarg argument

    Returns:
    
        :h0:        return handle of plot
    
    ____________________________________________________________________________
    """ 
    h0=None
    if do_topo in ['tpc', 'tcf']:
        data_topo, tri0   = do_data_prepare_unstruct(mesh, tri, data_topo, False)
                
        levels = np.hstack((25, 50, 100, 150, 200, 250, np.arange(500,6000+1,500)))
        N = len(levels)
        vals = np.ones((N, 4))
        vals[:, 0] = np.linspace(0.2, 0.95, N)
        vals[:, 1] = np.linspace(0.2, 0.95, N)
        vals[:, 2] = np.linspace(0.2, 0.95, N)
        vals = np.flipud(vals)
        topocmp = ListedColormap(vals)
        cinfo_topo = dict({'clevel':levels, 'cmap':topocmp})
                
        h0 = do_plt_data(hax_ii, do_topo, tri0, data_topo, 
                         cinfo_topo, None,    plt_opt  =plt_opt, 
                         plt_contb=plt_contb, pltcb_opt=pltcb_opt, 
                         plt_contf=False    , pltcf_opt=dict(),
                         plt_contr=False    , pltcr_opt=dict(),
                         plt_contl=plt_contl, pltcl_opt=pltcl_opt)
        del(tri0)
    return(h0)



#
#
#_______________________________________________________________________________
def do_plt_mesh(hax_ii, do_mesh, tri, mesh_opt=dict()):
    """
    --> plot overlaying triangular mesh

    Parameters: 
    
        :hax_ii:    handle of axes ii

        :do_mesh:   bool, (default: True), overlay FESOM grid over dataplot
    
        :tri:       matplotlib.tri triangulation object
                    - tri.mask_e_ok...provide mask with nan values, that describe the bottom limited to regional box
  
        :mesh_opt:  dict, (default: dict()) additional options that are given to 
                    the mesh plotting via kwarg
    
    Returns:
    
        :h0:        return handle of plot
    
    ____________________________________________________________________________
    """      
    h0=None
    if do_mesh: 
        mesh_optdefault = dict({'color':'k', 'linewidth':0.1, 'alpha':0.75})
        mesh_optdefault.update(mesh_opt)
        #h0 = hax_ii.triplot(tri.x, tri.y, tri.triangles[e_ok_mask,:], zorder=5, **mesh_optdefault)
        h0 = hax_ii.triplot(tri.x, tri.y, tri.triangles, zorder=5, **mesh_optdefault)
    return(h0)



#
#
#_______________________________________________________________________________
def do_plt_lsmask(hax_ii, do_lsm, mesh, lsm_opt=dict(), resolution='low'):
    """
    --> plot fesom mesh inverted land sea mask

    Parameters: 
    
        :hax_ii:        handle of axes ii
        
        :do_lsm:        str, (default: 'fesom'), overlay FESOM grid inverted land sea mask
                        option are here:
                    
                        - fesom      ... grey fesom landsea mask
                        - stock      ... uses cartopy stock image
                        - bluemarble ... uses bluemarble image in folder tripyview/background/
                        - etopo      ... uses etopo image in folder tripyview/background/

        :mesh:          fesom2 mesh object,  with all mesh information 
        
        :lsm_opt:       dict, (default: dict()) additional options that are given to 
                        the landsea mask plotting via kwarg
                    
        :resolution:    str, (default: 'low') switch resolution of background image 
                        for bluemarble and etopo between 'high' and 'low'
    
    Returns:
    
        :h0:            return handle of plot·
    
    ____________________________________________________________________________
    """  
    #___________________________________________________________________________
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="cartopy")
    
    lsm_optdefault = dict({'facecolor':[0.6, 0.6, 0.6], 'edgecolor':'k', 'linewidth':0.5, 'zorder':4})
    lsm_optdefault.update(lsm_opt)
    
    #___________________________________________________________________________
    if do_lsm in ['bluemarble', 'etopo']:
        import tripyview as tpv
        bckgrndir = os.path.join(tpv.__path__[0], 'backgrounds/')
    
    #___________________________________________________________________________
    # add mesh land-sea mask
    h0 = None
    if   do_lsm is None or do_lsm==False: 
        return()

    elif do_lsm=='fesom':
        h0=hax_ii.add_geometries(mesh.lsmask_p, crs=ccrs.PlateCarree(), **lsm_optdefault)
        
    elif do_lsm=='stock':  
        h01=hax_ii.stock_img()
        lsm_optdefault.update({'facecolor':'None'})
        h02=hax_ii.add_geometries(mesh.lsmask_p, crs=ccrs.PlateCarree(), **lsm_optdefault)
        h0 = [h01,h02]    
        
    elif do_lsm=='bluemarble': 
        # --> see original idea at http://earthpy.org/cartopy_backgroung.html#disqus_thread and 
        # https://stackoverflow.com/questions/67508054/improve-resolution-of-cartopy-map
        os.environ["CARTOPY_USER_BACKGROUNDS"] = bckgrndir
        h01=hax_ii.background_img(name=do_lsm, resolution=resolution)
        lsm_optdefault.update({'facecolor':'None'})
        h02=hax_ii.add_geometries(mesh.lsmask_p, crs=ccrs.PlateCarree(), **lsm_optdefault)
        h0 = [h01,h02]  
        
    elif do_lsm=='etopo':   
        # --> see original idea at http://earthpy.org/cartopy_backgroung.html#disqus_thread and 
        # https://stackoverflow.com/questions/67508054/improve-resolution-of-cartopy-map
        os.environ["CARTOPY_USER_BACKGROUNDS"] = bckgrndir
        h01=hax_ii.background_img(name=do_lsm, resolution=resolution)    
        lsm_optdefault.update({'facecolor':'None'})
        h02=hax_ii.add_geometries(mesh.lsmask_p, crs=ccrs.PlateCarree(), **lsm_optdefault)
        h0 = [h01,h02]
        
    else:
        raise ValueError(" > the do_lsm={} is not supported, must be either 'fesom', 'stock', 'bluemarble' or 'etopo'! ")
        
    #___________________________________________________________________________
    return(h0)



#
#
#_______________________________________________________________________________
def do_plt_gridlines(hax_ii, do_grid, box, ndat, 
                     data_x=None, data_y=None, xlim=None, ylim=None, grid_opt=dict(), 
                     proj=None, do_rescale=None):
    """
    --> do plot cartopy gridline and general gridlines together with the limit
        scaling of the axis (see non-linear option of x and y axis)
        
    Parameters:
    
        :hax_ii:    handle of one axes
        
        :do_grid:   bool, (default: True) plot cartopy grid lines
        
        :box:       None, or list with box definitions 
    
        :ndat:      int, total length of data list
        
        :data_x:    regular longitude array
        
        :data_y:    regular latitude array
    
        :xlim:      list, (default=None), overwrite limit of xaxis

        :ylim:      list, (default=None), overwrite limit of yaxis

        :grid_opt:  dict, (default: dict()) additional options that are given to 
                    the cartopy gridline plotting via kwarg
                    
        :proj:      None, cartopy projection object or string (e.g. index+depth+time...)
        
        :do_rescale: bool, str, np.array (defaul: False) do scaling of colorbar
                    - False      ... scale data automatically scientifically by 10^x, for data data larger 10^3 and smaller 10^-3
                    - log10      ... do logaritmic scaling
                    - slog10     ... do symetric logarithmic scaling
                    - np.array() ... scale colorbar stepwise according to values in np.array allows also for non-linear colortick steps
    
    Returns:
    
        :h0:        None or handle
    
    ____________________________________________________________________________
    """  
    #___________________________________________________________________________
    h0=None
    if do_grid:
        #_______________________________________________________________________
        if proj=='channel':
            grid_optdefault = dict({'color':'black', 'linestyle':'-', 'draw_labels':False, 'alpha':0.25, 'zorder':5})
            grid_optdefault.update(grid_opt)
            #___________________________________________________________________
            h0=hax_ii.gridlines(**grid_optdefault )
            if hax_ii.do_ylabel: h0.left_labels   = True
            if hax_ii.do_xlabel: h0.bottom_labels = True

        #_______________________________________________________________________
        elif isinstance(hax_ii.projection, ccrs.CRS):
            #___________________________________________________________________
            grid_optdefault = dict({'color':'black', 'linestyle':'-', 'draw_labels':False, 'alpha':0.25, 'zorder':5})
            grid_optdefault.update(grid_opt)
            
            #___________________________________________________________________
            h0=hax_ii.gridlines(**grid_optdefault )
            
            # ensure circular boundary for stereographic projection
            if isinstance(hax_ii.projection, (ccrs.NorthPolarStereo, ccrs.SouthPolarStereo) ):
                # give stereographic plot a circular boundary
                theta  = np.linspace(0, 2*np.pi, 100)
                center, radius = [0.5, 0.5], 0.5
                verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
                circle = mpath.Path(verts * radius + center)
                hax_ii.set_boundary(circle, transform=hax_ii.transAxes)
                del(theta, center, verts, circle)
            
            elif isinstance(hax_ii.projection, (ccrs.Orthographic, ccrs.NearsidePerspective)):
                hax_ii.set_global()
            
            elif isinstance(hax_ii.projection, (ccrs.Mollweide, ccrs.EqualEarth, ccrs.Robinson)):
                if box[1]-box[0]>359 and box[3]-box[2]>89 : hax_ii.set_global()
                
            if not isinstance(hax_ii.projection, (ccrs.NearsidePerspective, ccrs.Orthographic, ccrs.NorthPolarStereo, ccrs.SouthPolarStereo) ):
                #if hax_ii.do_ylabel: h0.ylabels_left   = True
                #if hax_ii.do_xlabel: h0.xlabels_bottom = True
                if hax_ii.do_ylabel: h0.left_labels   = True
                if hax_ii.do_xlabel: h0.bottom_labels = True
            # rotate xlabel for PlateCarree so that they dont overlay with 
            # neighboring plots
            if isinstance(hax_ii.projection, (ccrs.PlateCarree) ) and ndat>1 and hax_ii.ncol>1:
                h0.xlabel_style = {'rotation': 25}    
                
            h0.xlabel_style = {'fontsize': hax_ii.fs_ticks}       
            h0.ylabel_style = {'fontsize': hax_ii.fs_ticks} 
        #_______________________________________________________________________
        # grid options for index vs. depth vs. xy plot
        elif hax_ii.projection in ['index+depth+xy', 'index+depth+time', 
                                   'index+depth', 'index+time', 'index+xy', 
                                   'zmoc', 'dmoc', 'dmoc+dens', 'dmoc+depth']:
            #___________________________________________________________________
            grid_optdefault = dict({'color':'black', 'linestyle':'-', 'linewidth':0.25, 'alpha':1.0, 'zorder':-1})
            grid_optdefault.update(grid_opt)
            
            #___________________________________________________________________
            # default do_yexp, do_ysig settings
            do_yexp         = False  # do nonlinear/ exponential depth sscaling for vertical plots
            yexp_majorticks = np.array([10,100,250,500,1000,2000,4000,6000])
            yexp_fac        = 2.0
            
            do_ysig         = False  # do nonlinear sigma sscaling for dmoc plot
            ysig_majorticks = np.array([30.00, 36.00, 36.65, 36.92, 37.05])
            ysig_minorticks = np.sort(np.unique(np.hstack([0.00, 
                                np.arange(30.00, 35.99, 1.00), np.arange(36.00, 36.64, 0.20),# 0.15
                                np.arange(36.65, 36.91, 0.05), np.arange(36.92, 37.04, 0.02),
                                np.arange(37.05, 38.50, 0.25), 40.00])))
            
            do_yinv         = False  # invert y-axis
            
            #___________________________________________________________________
            # check if do_yscaling (do_yexp, do_ysig) related string exists in 
            # grid_opt dictionary
            rmv  = []
            for ii in grid_optdefault.keys():
                if ii in ['do_yexp', 'yexp', 'do_yexponential', 'yexponential'] :
                    # for MOC keep depth axes linear
                    if not 'zmoc' in hax_ii.projection and not 'dmoc' in hax_ii.projection : do_yexp = grid_optdefault[ii]
                    rmv.append(ii)
                elif ii in ['yexp_majorticks'] :
                    yexp_majorticks = grid_optdefault[ii]
                    rmv.append(ii)
                elif ii in ['yexp_fac'] :
                    yexp_fac = grid_optdefault[ii]
                    rmv.append(ii)       
                elif ii in ['do_yinv', 'yinv', 'do_yinvert', 'yinvert'] :
                    # for MOC keep depth axes linear
                    do_yinv = grid_optdefault[ii]
                    rmv.append(ii)    
                elif ii in ['do_ysig', 'ysig', 'ysigma', 'do_ysigma'] :    
                    if 'dmoc+dens' in hax_ii.projection : do_ysig = grid_optdefault[ii]
                    rmv.append(ii) 
                elif ii in ['ysig_majorticks'] :
                    yexp_majorticks = grid_optdefault[ii]
                    rmv.append(ii) 
                elif ii in ['ysig_minorticks'] :
                    ysig_minorticks = grid_optdefault[ii]
                    rmv.append(ii)     
            
            # remove do_ylog, do_yinv, do_ysig string from dictionary since its 
            # doesnt exist for tripcolor
            if len(rmv)!=0:
                for ii in rmv:
                    del(grid_optdefault[ii])
            
            #___________________________________________________________________
            # do exponential depth axis scaling
            if do_yexp and hax_ii.projection not in ['dmoc+dens']: 
                # Define your custom transformation function
                def forward_yexp(y, *args):
                    return np.abs(y)**(1.0/args[0])
                
                def inverse_yexp(y, *args):
                    return np.abs(y)**(args[0])
                
                hax_ii.set_yscale( 'function', functions=(lambda y: forward_yexp(y, yexp_fac), lambda y: inverse_yexp(y, yexp_fac)) )
                hax_ii.set_yticks(yexp_majorticks)
                         
            #___________________________________________________________________
            # do nonlinear sigma2 scaling for dmoc in density space
            elif do_ysig and hax_ii.projection in ['dmoc+dens']:
                # Define your custom transformation function
                def forward_ysig(y, *args):
                    reg = np.linspace(0, len(args[0]), len(args[0]))[::-1]
                    dens2reg = interp1d(args[0], reg, kind='linear')
                    y1       = dens2reg(y)
                    y1[y1<0] = 0
                    y1[y1>np.max(reg)] = np.max(reg)
                    return y1
                
                def inverse_ysig(y, *args):
                    reg = np.linspace(0, len(args[0]), len(args[0]))[::-1]
                    reg2dens = interp1d(reg, args[0], kind='linear')
                    y[y<0]   = 0
                    y[y>np.max(reg)] = np.max(reg)
                    y1       = reg2dens(y)
                    return y1 
                
                hax_ii.set_yscale( 'function', functions=(lambda y: forward_ysig(y, ysig_minorticks), lambda y: inverse_ysig(y, ysig_minorticks)) )
                
                # do custom sigma y-labels
                # do major ticklabels in larger font
                hax_ii.set_yticks( ysig_majorticks, minor=False ) 
                ylabelmayjor_list_fmt=list()
                for num in ysig_majorticks: ylabelmayjor_list_fmt.append('{:2.2f}'.format(num))
                hax_ii.set_yticklabels(ylabelmayjor_list_fmt, minor=False, size=10)
                    
                # do minor ticklabels in small font
                yminorticks = np.setdiff1d(ysig_minorticks[1:-1], ysig_majorticks)
                hax_ii.set_yticks(yminorticks, minor=True  )
                ylabelminor_list_fmt=list()
                for num in yminorticks: ylabelminor_list_fmt.append('{:2.2f}'.format(num))
                hax_ii.set_yticklabels(ylabelminor_list_fmt, minor=True, size = 6)
                
                if not hax_ii.do_ylabel: # bugfix in case of shared yaxes
                    hax_ii.tick_params(axis='y', which='minor', left=True, right=False, labelleft=False, labelright=False)
            
            #___________________________________________________________________
            # rescale x-axhes in case of vline index+depth plot
            if   hax_ii.projection in ['index+depth'] and isinstance(do_rescale,str): 
                if   do_rescale in ['log10' , 'slog10']: 
                    if   do_rescale=='log10' : hax_ii.set_xscale('log')
                    elif do_rescale=='slog10': hax_ii.set_xscale('symlog')
                    hax_ii.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, ), numticks=10))
                    
                    xtlabels = hax_ii.get_xticklabels()
                    if hax_ii.xaxis.get_tick_space()+2<len(xtlabels):
                        for ll in range(0,len(xtlabels),2):
                                xtlabels[ll] = ''
                    hax_ii.set_xticklabels(xtlabels)
                    hax_ii.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=10))
                else: hax_ii.get_xaxis().set_minor_locator(AutoMinorLocator())
            else: hax_ii.get_xaxis().set_minor_locator(AutoMinorLocator())
               
            #___________________________________________________________________
            # set x/y limits
            if data_y is not None: 
                if np.ndim(data_y)==1 : hax_ii.set_ylim(data_y[0],data_y[-1])
                if np.ndim(data_y)==2 : hax_ii.set_ylim(np.nanmin(data_y),np.nanmax(data_y))
            
            if ylim is not None: 
                if do_yexp: hax_ii.set_ylim(np.max([ylim[0], data_y[0]]) ,np.min([ylim[1], data_y[-1]]))
                # bug fix for invert y-axis when doing moc 
                elif hax_ii.projection not in ['zmoc', 'dmoc+depth']: 
                    hax_ii.set_ylim(ylim[0]-(ylim[1]-ylim[0])*0.025  ,ylim[-1]+(ylim[1]-ylim[0])*0.025)
            
            if xlim is not None: 
                hax_ii.set_xlim(xlim[0]  ,xlim[-1])
            else: 
                hax_ii.set_xlim(data_x[0], data_x[-1])
            
            #___________________________________________________________________
            # invert y-axis
            if isinstance(hax_ii.projection, str):
                if   hax_ii.projection in ['index+depth+xy', 'index+depth+time', 'index+depth']: 
                    hax_ii.invert_yaxis()
                
                elif hax_ii.projection in ['zmoc', 'dmoc+depth']:    
                    if ylim is not None: hax_ii.set_ylim(ylim[0]  ,ylim[-1])
                    hax_ii.invert_yaxis()
                
                else:
                    if do_yinv: hax_ii.invert_yaxis()
            
            #___________________________________________________________________
            # set grid options 
            hax_ii.get_yaxis().set_major_formatter(ScalarFormatter())
            hax_ii.grid(True,which='major')
            hax_ii.grid(**grid_optdefault)
            
    #___________________________________________________________________________
    return(h0)



#
#
#_______________________________________________________________________________
def do_cbar(hcb_ii, hax_ii, hp, data, cinfo, do_rescale, cb_label, cb_lunit, cb_ltime, cb_ldep,
            box_idx=None, norm=None, cb_opt=dict(), cbl_opt=dict(), cbtl_opt=dict()):
    """
    --> plot colorbars (tripyview allows also to have more than one colorbar within the 
        multipanel plot)
    
    Parameters:
    
        :hcb_ii:    actual colorbar handle 

        :hax_ii:    actual axes handle 

        :hp:        actual plot handle 

        :data:      xarray dataset object with all attributes (needed for the default colorbar labels)

        :cinfo:     None, dict() (default: None), dictionary with colorbar information. 
                    Information that are given are used, others are computed. cinfo dictionary 
                    entries can be,
                    
                    - cinfo['cmin'], cinfo['cmax'], cinfo['cref'] ... scalar min, max, reference value
                    - cinfo['crange'] ... list with [cmin, cmax, cref] overrides scalar values 
                    - cinfo['cnum']   ... minimum number of colors
                    - cinfo['cstr']   ... name of colormap see in sub_colormap_c2c.py
                    - cinfo['cmap']   ... colormap object ('wbgyr', 'blue2red, 'jet' ...)
                    - cinfo['clevel'] ... color level array

        :do_rescale: bool, str, np.array (defaul: False) do scaling of colorbar
                    - False      ... scale data automatically scientifically by 10^x, for data data larger 10^3 and smaller 10^-3
                    - log10      ... do logaritmic scaling
                    - slog10     ... do symetric logarithmic scaling
                    - np.array() ... scale colorbar stepwise according to values in np.array allows also for non-linear colortick steps


        :cb_label:  str, (default: None) if string its used as colorbar label, otherwise 
                    information from data ('long_name, short_name) are used

        :cb_lunit:  str, (default: None) if string its used as colorbar unit label, 
                    otherwise info from data are used

        :cb_ltime:  str, (default: None) if string its used as colorbar time label, 
                    otherwise info from data are used

        :cb_ldep:   str, (default: None) if string its used as colorbar depth label, 
                    otherwise info from data are used                

        :box_idx:   None or index of box selection in data_ii[box_idx]
        
        :norm:      None or renormation object
        
        :cb_opt:    dict, (default: dict()) direct option for colorbar via kwarg

        :cbl_opt:   dict, (default: dict()) direct option for colorbar labels (fontsize, fontweight, ...) via kwarg

        :cbtl_opt:  dict, (default: dict()) direct option for colorbar tick labels (fontsize, fontweight, ...) via kwarg

    Returns:
    
        :hcb_ii:    actual colorbar handle 
    
    ____________________________________________________________________________
    """  
    which_orient = hcb_ii.do_orient
    #___________________________________________________________________________
    cb_optdefault = dict({'extendrect':False, 'extendfrac':None, 'drawedges':True})
    cb_optdefault.update(cb_opt)
    #___________________________________________________________________________
    hcb_ii = plt.colorbar(mappable=hp[-1], ax=hax_ii, cax=hcb_ii, orientation=which_orient,
                          ticks=cinfo['clevel'], boundaries=cinfo['clevel'], #norm=norm, 
                          **cb_optdefault)
    
    #___________________________________________________________________________
    # do formating of colorbar tick labels
    hcb_ii = do_cbar_formatting(hcb_ii, do_rescale, cinfo, cbtl_opt=cbtl_opt)
            
    #___________________________________________________________________________
    loc_attrs = dict()
    if isinstance(data,list) and box_idx is not None:
        vname    = list(data[box_idx].keys())[0]        
        loc_attrs= data[box_idx][vname].attrs
    elif isinstance(data, xr.Dataset) :
        vname    = list(data.keys())[0]        
        loc_attrs= data[vname].attrs
            
    if cb_label is None:
        cb_label = ''
        if  'long_name' in loc_attrs:
            cb_label = cb_label+loc_attrs['long_name'].capitalize()
        elif 'short_name' in loc_attrs:
            c_label = cb_label+loc_attrs['short_name'].capitalize()
        
        if cb_lunit  is None:
            if 'units' in loc_attrs: cb_label = cb_label+' / '+loc_attrs['units']
        else:                cb_label = cb_label+' / '+cb_lunit
            
        if 'str_ltim' in loc_attrs: cb_label = cb_label+'\n'+loc_attrs['str_ltim']
        if 'str_ldep' in loc_attrs: cb_label = cb_label+loc_attrs['str_ldep']
        
    else:
        if cb_lunit  is None: cb_label = cb_label+' / '+loc_attrs['units']
        else:                 cb_label = cb_label+' / '+cb_lunit
        if cb_ltime is None:
            if 'str_ltim' in loc_attrs: cb_label = cb_label+'\n'+loc_attrs['str_ltim']
        else: cb_label = cb_label+'\n'+cb_ltime
        if cb_ldep is None:    
            if 'str_ldep' in loc_attrs: cb_label = cb_label+loc_attrs['str_ldep']
        else: cb_label = cb_label+'\n'+cb_ldep
        
    if   which_orient=='vertical'  : fsize =  hcb_ii.ax.get_yticklabels()[0].get_fontsize()
    elif which_orient=='horizontal': fsize =  hcb_ii.ax.get_xticklabels()[0].get_fontsize()
    
    cbl_optdefault = dict({'fontsize':fsize})
    cbl_optdefault.update(cbl_opt)
    
    ## wrap xlabel string when they are to long
    ## Estimate the width of the axes dynamically
    #axes_width_px = hcb_ii.ax.get_position().height * hax_ii.fig_height * hax_ii.fig_dpi
    
    ## Estimate the width of the axes in terms of characters
    ## font_size = plt.rcParams['font.size']
    #font_size = hcb_ii.ax.yaxis.get_label().get_size()
    #max_chars_per_line = int(axes_width_px / (font_size))  # Empirical factor for font size to character width ratio
    #cb_label = '\n'.join(textwrap.wrap(cb_label, width=max_chars_per_line))
    
    #___________________________________________________________________________
    
    hcb_ii.set_label(cb_label, **cbl_optdefault)
    #___________________________________________________________________________
    
    return(hcb_ii)



#
# 
#_______________________________________________________________________________
def do_cbar_formatting(cbar, do_rescale, cinfo, pw_lim=[-3,4], cbtl_opt=dict()):
    """
    --> do formating of colorbar for logarithmic data and exponential data

    Parameters: 
    
        :cbar:      actual colorbar handle  
    
        :do_rescale: bool, str, np.array (defaul: False) do scaling of colorbar
                    - False      ... scale data automatically scientifically by 10^x, for data data larger 10^3 and smaller 10^-3
                    - log10      ... do logaritmic scaling
                    - slog10     ... do symetric logarithmic scaling
                    - np.array() ... scale colorbar stepwise according to values in np.array allows also for non-linear colortick steps

        :cinfo:     dict(), dictionary with colorbar information. 
                    Information that are given are used, others are computed. cinfo dictionary 
                    entries can be,
                    
                    - cinfo['cmin'], cinfo['cmax'], cinfo['cref'] ... scalar min, max, reference value
                    - cinfo['crange'] ... list with [cmin, cmax, cref] overrides scalar values 
                    - cinfo['cnum']   ... minimum number of colors
                    - cinfo['cstr']   ... name of colormap see in sub_colormap_c2c.py
                    - cinfo['cmap']   ... colormap object ('wbgyr', 'blue2red, 'jet' ...)
                    - cinfo['clevel'] ... color level array

        :pw_lim:    list, in which decimal limits matplot will rescale the colorbar with 10^x

        :cbtl_opt:  dict, (default: dict()) direct option for colorbar tick labels (fontsize, fontweight, ...) via kwarg

    Returns:
    
        :cbar:      actual colorbar handle  
    
    ____________________________________________________________________________
    """
    if len(cinfo['clevel'])>=48: cbar.dividers.set_color('None')
    
    cbtl_optdefault = dict()
    cbtl_optdefault.update(cbtl_opt)
    cbar.ax.tick_params(**cbtl_optdefault)
    
    # formating for log and symlog colorbar axis   
    # do_rescale == 'log10', do_rescale == 'slog10':
    if  isinstance(do_rescale,str):
        #cbar.set_ticks(cinfo['clevel'][np.mod(np.log10(np.abs(cinfo['clevel'])),1)==0.0])
        cbar.update_ticks()
        cbar.ax.minorticks_off()
        
        #cbar.ax.yaxis.set_major_locator(cinfo['clevel'][np.mod(np.log10(np.abs(cinfo['clevel'])),1)==0.0])
        #cbar.ax.yaxis.set_major_locator(mticker.SymmetricalLogLocator(cinfo['clevel'][np.mod(np.log10(np.abs(cinfo['clevel'])),1)==0.0])), 
                                                                      #base=10))
        if cbar.orientation=='vertical':
            cbar.ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
            cbar.ax.yaxis.get_offset_text().set(horizontalalignment='right')
        elif cbar.orientation=='horizontal':
            cbar.ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())
            cbar.ax.xaxis.get_offset_text().set(horizontalalignment='center')
        #cbar.update_ticks()
    
    # do_rescale = np.array(...), do_rescale = None, do_rescale = False:
    else: 
        # set new cticks based on clab
        if 'clab' in cinfo:
            cbar.set_ticks(cinfo['clab'])
            cbar.update_ticks()    
        else:
            cbar.locator  = mticker.FixedLocator(cinfo['clevel'], nbins=cinfo['cnlab'])
            cbar.update_ticks()    
            
        formatter     = mticker.ScalarFormatter(useOffset=True, useMathText=True, useLocale=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((pw_lim[0], pw_lim[-1]))      
        cbar.formatter= formatter
        #cbar.ax.yaxis.get_offset_text().set(size=fontsize, horizontalalignment='center')
        cbar.ax.yaxis.get_offset_text().set(horizontalalignment='center')
        cbar.update_ticks()
        
    #cbar.ax.tick_params(labelsize=fontsize)
    #STOP
    
    #___________________________________________________________________________
    return(cbar)



#
#
# ___DO COLORMAP INFO__________________________________________________________
def do_setupcinfo(cinfo, data, do_rescale, mesh=None, tri=None, do_vec=False, 
                  do_index=False, do_moc=False, do_dmoc=None, do_hbstf=False, box_idx=0):
    """
    --> build up colormap dictionary  

    Parameters:
        
        :cinfo:     dict(), dictionary with colorbar information. 
                    Information that are given are used, others are computed. cinfo dictionary 
                    entries can be,
                    
                    - cinfo['cmin'], cinfo['cmax'], cinfo['cref'] ... scalar min, max, reference value
                    - cinfo['crange'] ... list with [cmin, cmax, cref] overrides scalar values 
                    - cinfo['cnum']   ... minimum number of colors
                    - cinfo['cstr']   ... name of colormap see in sub_colormap_c2c.py
                    - cinfo['cmap']   ... colormap object ('wbgyr', 'blue2red, 'jet' ...)
                    - cinfo['clevel'] ... color level array

        :data:      xarray dataset object                                      

        :do_rescale: bool, str, np.array (defaul: False) do scaling of colorbar
                    - False      ... scale data automatically scientifically by 10^x, for data data larger 10^3 and smaller 10^-3
                    - log10      ... do logaritmic scaling
                    - slog10     ... do symetric logarithmic scaling
                    - np.array() ... scale colorbar stepwise according to values in np.array allows also for non-linear colortick steps


        :mesh:      None or fesom2 mesh object,                                        

        :tri:       None or matplotlib.tri triangulation object

        :do_vec:    bool (default: False) flag,input data are vector data

        :do_index:  bool (default: False) flag,input data are index data

        :do_moc:    bool (default: False) flag,input data are zmoc data

        :do_dmoc:   str  (default: None ) str, input data are dmoc data 
                    ('inner', 'dmoc', 'srf')
                    
        :do_hbstf:  bool (default: False)

        :box_idx:   in case input data are list of regional shapefile boxes, 
                    this is the index of a specific box
    
    Returns:
    
        :cinfo:     None, dict() (default: None), dictionary with colorbar info
    
    ____________________________________________________________________________
    """  
    #___________________________________________________________________________
    # set up color info 
    if cinfo is None: cinfo=dict()
    else            : cinfo=cinfo.copy()
    do_cweights=None
    #___________________________________________________________________________
    # check if dictionary keys exist, if they do not exist fill them up 
    cfac = 1
    if 'cfac'       in cinfo.keys(): cfac = cinfo['cfac' ]
    if 'chist'  not in cinfo.keys(): cinfo['chist' ] = True
    if 'ctresh' not in cinfo.keys(): cinfo['ctresh'] = 0.995
    if 'cnlab'  not in cinfo.keys(): cinfo['cnlab' ] = 8
    
    if (('cmin' not in cinfo.keys()) or ('cmax' not in cinfo.keys())) and ('crange' not in cinfo.keys()):
        #_______________________________________________________________________
        # loop over all the input data --> find out total cmin/cmax value
        cmin, cmax = np.inf, -np.inf
        for data_ii in data:
            
            if isinstance(data_ii, np.ndarray):
                data_plot = data_ii.copy()
            else:    
                if do_index: vname = list(data_ii[box_idx].keys())
                else       : vname = list(data_ii.keys())
                
                #_______________________________________________________________
                # --> consider scalar data
                if do_vec==False:
                    if   do_index: data_plot = data_ii[box_idx][ vname[0] ].data.copy()
                    elif do_moc  : data_plot = data_ii['zmoc'].isel(nz=np.abs(data_ii['depth'])>=500).values.copy()
                    elif do_dmoc : data_plot = data_ii[ vname[0] ].data.copy()
                    #elif do_dmoc is not None  : 
                        #if   do_dmoc=='dmoc'  : data_plot = data_ii['dmoc'].data.copy()
                        #elif do_dmoc=='srf'   : data_plot = -(data_ii['dmoc_fh'].data.copy()+data_ii['dmoc_fw'].data.copy()+data_ii['dmoc_fr'].data.copy())
                        #elif do_dmoc=='inner' : data_plot = data_ii['dmoc'].data.copy() + \
                                                            #(data_ii['dmoc_fh'].data.copy()+data_ii['dmoc_fw'].data.copy()+data_ii['dmoc_fr'].data.copy())
                    elif do_hbstf: data_plot = data_ii[ vname[0] ].data.copy() 
                    else         : 
                        data_plot   = data_ii[ vname[0] ].data.copy()
                        if cinfo['chist'] and 'w_A' in list(data_ii.coords.keys()): 
                            do_cweights = data_ii['w_A'].data.copy()
                
                #_______________________________________________________________
                # --> consider vector norm data
                else:
                    # compute norm when vector data
                    data_plot = np.sqrt(data_ii[ vname[0] ].data.copy()**2 + data_ii[ vname[1] ].data.copy()**2)
                    if cinfo['chist']: do_cweights = data_ii['w_A'].data.copy()
            
            #___________________________________________________________________    
            # for logarythmic rescaling cmin or cmax can not be zero
            if isinstance(do_rescale, str):
                if do_rescale=='log10' or do_rescale=='slog10': 
                    data_plot[np.abs(data_plot)==0]=np.nan
                    data_plot[np.abs(data_plot)<=1e-15]=np.nan
            
            #___________________________________________________________________
            # Consider regular griddet data or index data
            if tri is None or do_index:
                #cmin = np.min([cmin,np.nanmin(data_plot) ])
                #cmax = np.max([cmax,np.nanmax(data_plot) ])
                #print('cmin, cmax = ', cmin, cmax)
                if cinfo['chist']:
                    auxcmin,auxcmax = do_climit_hist(data_plot.flatten(),ctresh=cinfo['ctresh'], cweights=do_cweights)
                    cmin, cmax = np.min([cmin,auxcmin]), np.max([cmax,auxcmax])
                    print('--> histo: cmin, cmax = ', cmin, cmax)
                else:    
                    cmin = np.min([cmin,np.nanmin(data_plot) ])
                    cmax = np.max([cmax,np.nanmax(data_plot) ])
                    print('cmin, cmax = ', cmin, cmax)
            
            #___________________________________________________________________
            # Consider unstructured griddet data --> tri is not None
            else:
                # data_plot is on vertices --> add augmentation arrays --> now 
                # size mesh.n2dna
                if   data_plot.size == mesh.n2dn: 
                    data_plot = np.hstack((data_plot, data_plot[mesh.n_pbnd_a]))
                    if tri.mask_n_box is not None: data_plot = data_plot[tri.mask_n_box]
                    
                # data_plot is on elements --> add augmentation arrays --> now 
                # size mesh.n2dea
                elif data_plot.size == mesh.n2de: 
                    data_plot = np.hstack((data_plot[mesh.e_pbnd_0], data_plot[mesh.e_pbnd_a]))
                    if any(tri.mask_e_box==False): data_plot = data_plot[tri.mask_e_box]
                    
                # compute min/max value range by histogram, computation of cumulativ 
                # distribution function at certain cutoff treshold allow to kick out 
                # outlier values from the cmin/cmax value range
                if cinfo['chist']:
                    if do_cweights is not None: 
                        if   do_cweights.size == mesh.n2dn: 
                            do_cweights = np.hstack((do_cweights, do_cweights[mesh.n_pbnd_a]))
                            if tri.mask_n_box is not None: do_cweights = do_cweights[tri.mask_n_box]
                            
                        elif do_cweights.size == mesh.n2de: 
                            do_cweights = np.hstack((do_cweights[mesh.e_pbnd_0], do_cweights[mesh.e_pbnd_a]))
                            if any(tri.mask_e_box==False): do_cweights = do_cweights[tri.mask_e_box]
                    
                    histcmin,histcmax = do_climit_hist(data_plot, ctresh=cinfo['ctresh'], cweights=do_cweights)                    
                    cmin, cmax = np.min([cmin,histcmin]), np.max([cmax,histcmax])
                    print('--> cmin/cmax: norm: {:f}/{:f}, hist: {:f}/{:f}, fin: {:f}/{:f}'.format(np.nanmin(data_plot), np.nanmax(data_plot), histcmin, histcmax, cmin, cmax))
                
                # just take global or box reduced local cmin and cmax
                else:    
                    cmin, cmax = np.min([ cmin, np.nanmin(data_plot) ]), np.max([ cmax, np.nanmax(data_plot) ])
                    print('--> cmin/cmax: fin: {:f}/{:f}'.format(cmin,cmax))
                    
            # increase/decrease cmin/cmax limit by pre-defined factor 
            cmin, cmax = cmin*cfac, cmax*cfac
            
        #_______________________________________________________________________
        if 'climit' in cinfo.keys():
           cmin = np.max([cmin, cinfo['climit'][0]])
           cmax = np.min([cmax, cinfo['climit'][-1]])
        
        #_______________________________________________________________________
        # dezimal rounding of cmin and cmax
        # if not do_rescale=='log10' and not do_rescale=='slog10':
        if not isinstance(do_rescale,str) or not isinstance(do_rescale, np.ndarray):
            cdmin, cdmax = 0.0, 0.0
            if np.abs(np.mod(np.abs(cmin),1))!=0: cdmin = np.floor(np.log10(np.abs(np.mod(np.abs(cmin),1))))
            if np.abs(np.mod(np.abs(cmax),1))!=0: cdmax = np.floor(np.log10(np.abs(np.mod(np.abs(cmax),1))))
            cdez        = np.min([cdmin,cdmax])
            cmin, cmax  = np.around(cmin, -np.int32(cdez-1)), np.around(cmax, -np.int32(cdez-1))
            
        #_______________________________________________________________________
        # write cmin/cmax too cinfo dictionary
        if 'cmin' not in cinfo.keys(): cinfo['cmin'] = cmin
        if 'cmax' not in cinfo.keys(): cinfo['cmax'] = cmax  
        
    #___________________________________________________________________________    
    if 'crange' in cinfo.keys():
        cinfo['cmin'], cinfo['cmax'], cinfo['cref'] = cinfo['crange'][0], cinfo['crange'][1], cinfo['crange'][2]
        if do_rescale=='slog10': cinfo['cref'] = np.abs(cinfo['cref']) 
    else:
        if (cinfo['cmin'] == cinfo['cmax'] ): raise ValueError (' --> can\'t plot! data are everywhere: {}'.format(str(cinfo['cmin'])))
        cref = cinfo['cmin'] + (cinfo['cmax']-cinfo['cmin'])/2
        if 'cref' not in cinfo.keys(): 
            if isinstance(do_rescale, str):
                if do_rescale=='log10':
                    # compute cref in decimal units and tranfer back to normal units 
                    # afterwards
                    cdmin = np.floor(np.log10(np.abs(cinfo['cmin'])))
                    cdmax = np.floor(np.log10(np.abs(cinfo['cmax'])))
                    cref = cdmin + (cdmax-cdmin)/2
                    cinfo['cref'] = np.around(cref, -np.int32(np.floor(np.log10(np.abs(cref)))-1) )
                    cinfo['cref'] = np.power(10.0,cinfo['cref'])
                elif do_rescale=='slog10':    
                    # cref becomes cutoff value for logarithmic to liner transition in 
                    # case of symetric log10
                    cinfo['cref'] = np.power(10.0,-6)
            else:
                dez = 0
                if cref==0.0: 
                    cinfo['cref'] = cref
                else:
                    while True:
                        new_cref = np.around(cref, -np.int32(np.floor(np.log10(np.abs(cref)))-dez) )
                        #print(cref, new_cref, cinfo['cmin'], cinfo['cmax'])
                        if new_cref>cinfo['cmin'] and new_cref<cinfo['cmax']:
                            break
                        else: 
                            dez=dez+1
                    cinfo['cref'] = new_cref
        
    #___________________________________________________________________________    
    if 'cnum' not in cinfo.keys(): cinfo['cnum'] = 20
    if 'cstr' not in cinfo.keys(): cinfo['cstr'] = 'wbgyr'
    
    #___________________________________________________________________________    
    # compute clevels and cmap
    if isinstance(do_rescale, str):
        if do_rescale=='log10':
            # transfer cmin, cmax, cref into decimal units
            cdmin = np.floor(np.log10(np.abs(cinfo['cmin'])))
            cdmax = np.floor(np.log10(np.abs(cinfo['cmax'])))
            cdref = np.floor(np.log10(np.abs(cinfo['cref'])))
            
            #print(cinfo)
            print(cdmin,cdmax,cdref)
            #compute levels in decimal units
            cinfo['cmap'],cinfo['clevel'],cinfo['cref'] = colormap_c2c(cdmin,cdmax,cdref,cinfo['cnum'],cinfo['cstr'])
            
            # transfer levels back to normal units
            cinfo['clevel'] = np.power(10.0,cinfo['clevel'])
            cinfo['cref']   = np.power(10.0,cinfo['cref'])
            
        elif do_rescale=='slog10':
            # transfer cmin, cmax, cref into decimal units
            cdmin = np.floor(np.log10(np.abs(cinfo['cmin'])))
            cdmax = np.floor(np.log10(np.abs(cinfo['cmax'])))
            cdref = np.floor(np.log10(np.abs(cinfo['cref'])))
            
            ddcmin, ddcmax = -(cdmin-cdref), (cdmax-cdref)
            cinfo['cmap'],cinfo['clevel'],cinfo['cref'] = colormap_c2c(ddcmin,ddcmax,0.0,cinfo['cnum'],cinfo['cstr'], do_slog=True)
            
            # rescale clevels towards symetric logarithm
            isneg = cinfo['clevel']<0
            ispos = cinfo['clevel']>0
            cinfo['clevel'][isneg] = (np.abs(cinfo['clevel'][isneg]) + cdref)
            cinfo['clevel'][ispos] = (np.abs(cinfo['clevel'][ispos]) + cdref)
            cinfo['clevel'][isneg] = -np.power(10.0, cinfo['clevel'][isneg])
            cinfo['clevel'][ispos] = np.power(10.0, cinfo['clevel'][ispos])
    
    # define custom non-linear colorsteps via numpy array
    elif isinstance(do_rescale, np.ndarray):
        rescal_ref=None
        if any(do_rescale==0.0) and do_rescale[0]!=0.0 and cinfo['cref']==0.0: rescal_ref=cinfo['cref']
        nrscal = len(do_rescale)-1
        print(do_rescale, rescal_ref)
        cinfo['cmap'],cinfo['clevel'],cinfo['cref'] = colormap_c2c(0, nrscal, np.int16(nrscal/2), nrscal, cinfo['cstr'], 
                                                                   cstep=1 ,do_slog=False, do_rescal=do_rescale, rescal_ref=rescal_ref)
        cinfo['clevel'] = do_rescale
        cinfo['cref'] = do_rescale[cinfo['cref']]
        
    else:    
        if cinfo['cref'] == 0.0:
            if cinfo['cref'] > cinfo['cmax']: cinfo['cmax'] = cinfo['cref']+np.finfo(np.float32).eps
            if cinfo['cref'] < cinfo['cmin']: cinfo['cmin'] = cinfo['cref']-np.finfo(np.float32).eps
        cinfo['cmap'],cinfo['clevel'],cinfo['cref'] = colormap_c2c(cinfo['cmin'],cinfo['cmax'],cinfo['cref'],cinfo['cnum'],cinfo['cstr'])
    
    #___________________________________________________________________________
    if 'cmap0' in list(cinfo.keys()): cinfo['cmap'] = cinfo['cmap0'].resampled(cinfo['clevel'].size-1)
        
    #___________________________________________________________________________
    # colorbar tick labels
    if isinstance(do_rescale, np.ndarray):
        cinfo['clab'] = cinfo['clevel'][1:-1]
    else:    
        nclev    = len(cinfo['clevel'])
        idx_cref = np.where(cinfo['clevel']==cinfo['cref'])[0]
        idx_cref = idx_cref.item()
        
        nstep    = nclev/cinfo['cnlab']
        nstep    = np.max([np.int32(np.ceil(nstep)),1])
        
        idx      = np.arange(0, nclev, 1)
        idxb     = np.ones(nclev, dtype=bool)                
        idxb[idx_cref::nstep]  = False
        idxb[idx_cref::-nstep] = False
        if do_rescale == 'log10' or do_rescale == 'slog10':
            idxb[cinfo['clevel']==0.0]=True
        
        idx_not  = idx[idxb==True]
        idx_yes  = idx[idxb==False]
        
        
        cinfo['clab'] = cinfo['clevel'][idx_yes]
        del(idx_not, idx_yes, idx, idxb, idx_cref)
    #___________________________________________________________________________
    print(cinfo)
    return(cinfo)    



#
#
#_______________________________________________________________________________
def do_climit_hist(data_in, ctresh=0.99, cbin=1000, cweights=None):
    """
    --> compute min/max value range by histogram, computation of cumulativ distribution
    function at certain cutoff treshold
    
    Parameters:

        :data_in:   np.array with data to plot 

        :ctresh:    cover 99% of value range, means that extreme outliers are cutted of

        :cbin:      number of bin that are used for the value range histogram

        :cweight:   provide are weights for the histogramm
    
    Returns:
    
        :cmin:      return minimum value of value range

        :cmax:      return minimum value of value range
    
    ____________________________________________________________________________
    """      
    if cweights is None:
        hist, bin_e = np.histogram(data_in[~np.isnan(data_in)], bins=cbin, density=False,) #weights=mesh.n_area[isnotnan]/np.sum(mesh.n_area[isnotnan]), )
    else:
        hist, bin_e = np.histogram(data_in[~np.isnan(data_in)], bins=cbin, weights=cweights[~np.isnan(data_in)], density=True,) #weights=mesh.n_area[isnotnan]/np.sum(mesh.n_area[isnotnan]), )
    
    hist        = hist/hist.sum()
    bin_m       = bin_e[:-1]+(bin_e[:-1]-bin_e[1:])/2
    cmin        = bin_m[np.where(np.cumsum(hist[::-1])[::-1]>=ctresh)[0][-1]]
    cmax        = bin_m[np.where(np.cumsum(hist)            >=ctresh)[0][ 0]]
    if cmin==cmax: 
        cmax        = bin_m[np.where(np.cumsum(hist)        >=ctresh)[0][1]]
    return(cmin, cmax)



#
#
#_______________________________________________________________________________
def set_cinfo(cstr, cnum, crange, cmin, cmax, cref, cfac, climit, chist, ctresh):
    """
    --> initialise cinfo dictionary
    
    Parameters: 
    
        :cstr:      provide colormap string, can be either own defined colormap 
                    see sub_colormap.py ('blue2red', 'wbgyr',...), it can be also 
                    a matplotlib colormap ('matplotlib.viridis', 'matplotlib.coolwarm', 
                    ...), or a cmocean colormap ('cmocean.dens', 'cmocean.thermal'....)

        :cnum:      minimum number of colors to use 

        :crange:    list of min/max/ref colorrange [cmin, cmax, cref]

        :cmin:      set min value of colorrange

        :cmax:      set max value of colorrange

        :cref:      set reference value (center value of colorrange)

        :cfac:      factor to multiply cmin and cmax

        :climit:    provide list with cmin cmax value [cmin, cmax], reference value 
                    is determined autonom    

        :chist:     True/Fals if colorrange should be limited by histogram to exclude
                    extreme outliers

        :ctresh:    how much percent of the colorrange should be covered, default is 99%
    
    Returns:

        :cinfo:     None, dict() (default: None), dictionary with colorbar info
    
    ____________________________________________________________________________
    """  
    cinfo=dict()
    if cstr      is not None: cinfo['cstr'     ]=cstr
    if cnum      is not None: cinfo['cnum'     ]=cnum
    if crange    is not None: cinfo['crange'   ]=crange
    if cmin      is not None: cinfo['cmin'     ]=cmin
    if cmax      is not None: cinfo['cmax'     ]=cmax
    if cref      is not None: cinfo['cref'     ]=cref
    if cfac      is not None: cinfo['cfac'     ]=cfac
    if climit    is not None: cinfo['climit'   ]=climit    
    if chist     is not None: cinfo['chist'    ]=chist
    if ctresh    is not None: cinfo['ctresh'   ]=ctresh
    return(cinfo)
  
  
  
#
#
#_______________________________________________________________________________
def do_savefigure(do_save, hfig, dpi=300, do_info=True, save_opt=dict()):
    """
    --> save figure to file
    
    Parameters: 
    
        :do_save:   None, str (default:None) if None figure will by not saved,
                    if string figure will be saved, strings must give         
                    directory and filename  where to save.                    
        
        :hfig:      figure handle                     
        
        :dpi:       int, (default:600), resolution of image    
        
        :save_opt:  dict, (default: dict()) direct option for saving via kwarg
    
    Returns: 
    
    ____________________________________________________________________________
    """  
    if do_save is not None:
        save_optdefault=dict({'bbox_inches':'tight', 'pad_inches':0.1, 'transparent':True})
        save_optdefault.update(save_opt)
        if isinstance(do_save, str):
            #___________________________________________________________________
            # extract filename from do_save
            sfname = os.path.basename(do_save)
            
            # extract file extensions
            sfformat = os.path.splitext(sfname)[1][1:]
            
            # extract dirname from do_save --> create directory if not exist
            sdname = os.path.dirname(do_save)
            sdname = os.path.expanduser(sdname)
            if do_info: print(' > save figure: {}'.format(os.path.join(sdname,sfname)))
            
            #___________________________________________________________________
            if not os.path.isdir(sdname): os.makedirs(sdname)
            
            #___________________________________________________________________
            hfig.savefig(os.path.join(sdname,sfname), format=sfformat, dpi=dpi, 
                        **save_optdefault)
        elif isinstance(do_save, list):
            #___________________________________________________________________
            # extract filename from do_save
            for do_save0 in do_save:
                sfname = os.path.basename(do_save0)
                
                # extract file extensions
                sfformat = os.path.splitext(sfname)[1][1:]
                
                # extract dirname from do_save --> create directory if not exist
                sdname = os.path.dirname(do_save0)
                sdname = os.path.expanduser(sdname)
                if do_info: print(' > save figure: {}'.format(os.path.join(sdname,sfname)))
                
                #___________________________________________________________________
                if not os.path.isdir(sdname): os.makedirs(sdname)
                
                #___________________________________________________________________
                hfig.savefig(os.path.join(sdname,sfname), format=sfformat, dpi=dpi, 
                            **save_optdefault)



##
##
##_______________________________________________________________________________
## --> this based on work of Nils Bruegemann see 
## https://gitlab.dkrz.de/m300602/pyicon/-/blob/master/pyicon/pyicon_plotting.py
## i needed this to unify the ploting between icon and fesom for model comparison
## paper
#def arrange_axes(nx, ny,
                 #sharex = True, sharey = False,
                 #xlabel = '', ylabel = '',
                 ## labeling axes with e.g. (a), (b), (c)
                 #do_axes_labels = True,
                 #axlab_kw = dict(),
                 ## colorbar
                 #plot_cb = True,
                 ## projection (e.g. for cartopy)
                 #projection = None,
                 ## aspect ratio of axes
                 #asp = 1.,
                 #sasp = 0.,  # for compability with older version of arrange_axes
                 ## width and height of axes
                 #wax = 'auto', hax = 4.,
                 ## extra figure spaces (left, right, top, bottom)
                 #dfigl= 0.0, dfigr=0.0, dfigt=0.0, dfigb=0.0,
                 ## space aroung axes (left, right, top, bottom) 
                 #daxl = 1.8, daxr =0.8, daxt =0.8, daxb =1.2, 
                 ## space around colorbars (left, right, top, bottom) 
                 #dcbl =-0.5, dcbr =1.4, dcbt =0.0, dcbb =0.5,
                 ## width and height of colorbars
                 #wcb = 0.5, hcb = 'auto',
                 ## factors to increase widths and heights of axes and colorbars
                 #fig_size_fac = 1.,
                 #f_wax  =1., f_hax  =1., f_wcb  =1., f_hcb  =1.,
                 ## factors to increase spaces (figure)
                 #f_dfigl=1., f_dfigr=1., f_dfigt=1., f_dfigb=1.,
                 ## factors to increase spaces (axes)
                 #f_daxl =1., f_daxr =1., f_daxt =1., f_daxb =1.,
                 ## factors to increase spaces (colorbars)
                 #f_dcbl =1., f_dcbr =1., f_dcbt =1., f_dcbb =1.,
                 ## font sizes of labels, titles, ticks
                 #fs_label = 10., fs_title = 12., fs_ticks = 10.,
                 ## font size increasing factor
                 #f_fs = 1,
                 #reverse_order = False,
                 #nargout=['fig', 'hca', 'hcb'],
                #):

    ## factor to convert cm into inch
    #cm2inch = 0.3937

    #if sasp!=0:
        #print('::: Warning: You are using keyword ``sasp`` for setting the aspect ratio but you should switch to use ``asp`` instead.:::')
        #asp = 1.*sasp

    ## --- set hcb in case it is auto
    #if isinstance(wax, str) and wax=='auto': wax = hax/asp

    ## --- set hcb in case it is auto
    #if isinstance(hcb, str) and hcb=='auto': hcb = hax

    ## --- rename horizontal->bottom and vertical->right
    #if isinstance(plot_cb, str) and plot_cb=='horizontal': plot_cb = 'bottom'
    #if isinstance(plot_cb, str) and plot_cb=='vertical'  : plot_cb = 'right'
  
    ## --- apply fig_size_fac
    ## font sizes
    ##f_fs *= fig_size_fac
    ## factors to increase widths and heights of axes and colorbars
    #f_wax *= fig_size_fac
    #f_hax *= fig_size_fac
    ##f_wcb *= fig_size_fac
    #f_hcb *= fig_size_fac
    ### factors to increase spaces (figure)
    ##f_dfigl *= fig_size_fac
    ##f_dfigr *= fig_size_fac
    ##f_dfigt *= fig_size_fac
    ##f_dfigb *= fig_size_fac
    ### factors to increase spaces (axes)
    ##f_daxl *= fig_size_fac
    ##f_daxr *= fig_size_fac
    ##f_daxt *= fig_size_fac
    ##f_daxb *= fig_size_fac
    ### factors to increase spaces (colorbars)
    ##f_dcbl *= fig_size_fac
    ##f_dcbr *= fig_size_fac
    ##f_dcbt *= fig_size_fac
    ##f_dcbb *= fig_size_fac
  
    ## --- apply font size factor
    #fs_label *= f_fs
    #fs_title *= f_fs
    #fs_ticks *= f_fs

    ## make vector of plot_cb if it has been true or false before
    ## plot_cb can have values [{1}, 0] 
    ## with meanings:
    ##   1: plot cb; 
    ##   0: do not plot cb
    #plot_cb_right  = False
    #plot_cb_bottom = False
    #if isinstance(plot_cb, bool) and (plot_cb==True):
        #plot_cb = np.ones((nx,ny))  
    #elif isinstance(plot_cb, bool) and (plot_cb==False):
        #plot_cb = np.zeros((nx,ny))
    #elif isinstance(plot_cb, str) and plot_cb=='right':
        #plot_cb = np.zeros((nx,ny))
        #plot_cb_right = True
    #elif isinstance(plot_cb, str) and plot_cb=='bottom':
        #plot_cb = np.zeros((nx,ny))
        #plot_cb_bottom = True
    #else:
        #plot_cb = np.array(plot_cb)
        #if plot_cb.size!=nx*ny    : raise ValueError('Vector plot_cb has wrong length!')
        #if plot_cb.shape[0]==nx*ny: plot_cb = plot_cb.reshape(ny,nx).transpose()
        #elif plot_cb.shape[0]==ny : plot_cb = plot_cb.transpose()
  
    ## --- make list of projections if it is not a list
    #if not isinstance(projection, list): projection = [projection]*nx*ny
    
    ## --- make arrays and multiply by f_*
    #daxl = np.array([daxl]*nx)*f_daxl
    #daxr = np.array([daxr]*nx)*f_daxr
    #dcbl = np.array([dcbl]*nx)*f_dcbl
    #dcbr = np.array([dcbr]*nx)*f_dcbr
    
    #wax  = np.array([wax]*nx)*f_wax
    #wcb  = np.array([wcb]*nx)*f_wcb
    
    #daxt = np.array([daxt]*ny)*f_daxt
    #daxb = np.array([daxb]*ny)*f_daxb
    #dcbt = np.array([dcbt]*ny)*f_dcbt
    #dcbb = np.array([dcbb]*ny)*f_dcbb
    
    #hax  = np.array([hax]*ny)*f_hax
    #hcb  = np.array([hcb]*ny)*f_hcb
  
    ## --- adjust for shared axes
    #if sharex: daxb[:-1] = 0.
    
    #if sharey: daxl[1:] = 0.

    ## --- adjust for one colorbar at the right or bottom
    #if plot_cb_right:
        #daxr_s = daxr[0]
        #dcbl_s = dcbl[0]
        #dcbr_s = dcbr[0]
        #wcb_s  = wcb[0]
        #hcb_s  = hcb[0]
        #dfigr += dcbl_s+wcb_s+0.*dcbr_s+daxl[0]
    #if plot_cb_bottom:
        #hcb_s  = wcb[0]
        #wcb_s  = wax[0]
        #dcbb_s = dcbb[0]+daxb[-1]
        #dcbt_s = dcbt[0]
        ##hcb_s  = hcb[0]
        #dfigb += dcbb_s+hcb_s+dcbt_s
  
    ## --- adjust for columns without colorbar
    #delete_cb_space = plot_cb.sum(axis=1)==0
    #dcbl[delete_cb_space] = 0.0
    #dcbr[delete_cb_space] = 0.0
    #wcb[delete_cb_space]  = 0.0
    
    ## --- determine ax position and fig dimensions
    #x0 =   dfigl
    #y0 = -(dfigt)
    
    #pos_axcm = np.zeros((nx*ny,4))
    #pos_cbcm = np.zeros((nx*ny,4))
    #nn = -1
    #y00 = y0
    #x00 = x0
    #for jj in range(ny):
        #y0 += -(daxt[jj]+hax[jj])
        #x0 = x00
        #for ii in range(nx):
            #nn += 1
            #x0   += daxl[ii]
            #pos_axcm[nn,:] = [x0, y0, wax[ii], hax[jj]]
            #pos_cbcm[nn,:] = [x0+wax[ii]+daxr[ii]+dcbl[ii], y0, wcb[ii], hcb[jj]]
            #x0   += wax[ii]+daxr[ii]+dcbl[ii]+wcb[ii]+dcbr[ii]
        #y0   += -(daxb[jj])
    #wfig = x0+dfigr
    #hfig = y0-dfigb
  
    ## --- transform from negative y axis to positive y axis
    #hfig = -hfig
    #pos_axcm[:,1] += hfig
    #pos_cbcm[:,1] += hfig
    
    ## --- convert to fig coords
    #cm2fig_x = 1./wfig
    #cm2fig_y = 1./hfig
    
    #pos_ax = 1.*pos_axcm
    #pos_cb = 1.*pos_cbcm
    
    #pos_ax[:,0] = pos_axcm[:,0]*cm2fig_x
    #pos_ax[:,2] = pos_axcm[:,2]*cm2fig_x
    #pos_ax[:,1] = pos_axcm[:,1]*cm2fig_y
    #pos_ax[:,3] = pos_axcm[:,3]*cm2fig_y
    
    #pos_cb[:,0] = pos_cbcm[:,0]*cm2fig_x
    #pos_cb[:,2] = pos_cbcm[:,2]*cm2fig_x
    #pos_cb[:,1] = pos_cbcm[:,1]*cm2fig_y
    #pos_cb[:,3] = pos_cbcm[:,3]*cm2fig_y

    ## --- find axes center (!= figure center)
    #x_ax_cent = pos_axcm[0,0] +0.5*(pos_axcm[-1,0]+pos_axcm[-1,2]-pos_axcm[0,0])
    #y_ax_cent = pos_axcm[-1,1]+0.5*(pos_axcm[0,1] +pos_axcm[0,3] -pos_axcm[-1,1])
    
    ## --- make figure and axes
    #fig = plt.figure(figsize=(wfig*cm2inch, hfig*cm2inch), facecolor='white')
  
    #hca = [0]*(nx*ny)
    #hcb = [0]*(nx*ny)
    #nn = -1
    #for jj in range(ny):
        #for ii in range(nx):
            #nn+=1
            
            ## --- axes
            #hca[nn] = fig.add_subplot(position=pos_ax[nn,:], projection=projection[nn])
            #hca[nn].set_position(pos_ax[nn,:])
            
            ## --- colorbar
            #if plot_cb[ii,jj] == 1:
                #hcb[nn] = fig.add_subplot(position=pos_cb[nn,:])
                #hcb[nn].set_position(pos_cb[nn,:])
            #ax  = hca[nn]
            #cax = hcb[nn] 
            
            ## --- label
            #ax.set_xlabel(xlabel, fontsize=fs_label)
            #ax.set_ylabel(ylabel, fontsize=fs_label)
            ##ax.set_title('', fontsize=fs_title)
            #matplotlib.rcParams['axes.titlesize'] = fs_title
            #ax.tick_params(labelsize=fs_ticks)
            #if plot_cb[ii,jj] == 1:
                #hcb[nn].tick_params(labelsize=fs_ticks)
            
            ##ax.tick_params(pad=-10.0)
            ##ax.xaxis.labelpad = 0
            ##ax._set_title_offset_trans(float(-20))
            
            ## --- axes ticks
            ## delete labels for shared axes
            #if sharex and jj!=ny-1:
                #hca[nn].ticklabel_format(axis='x',style='plain',useOffset=False)
                #hca[nn].tick_params(labelbottom=False)
                #hca[nn].set_xlabel('')
            
            #if sharey and ii!=0:
                #hca[nn].ticklabel_format(axis='y',style='plain',useOffset=False)
                #hca[nn].tick_params(labelleft=False)
                #hca[nn].set_ylabel('')
            
            ## ticks for colorbar 
            #if plot_cb[ii,jj] == 1:
                #hcb[nn].set_xticks([])
                #hcb[nn].yaxis.tick_right()
                #hcb[nn].yaxis.set_label_position("right")

    ##--- needs to converted to fig coords (not cm)
    #if plot_cb_right:
        #nn = -1
        ##pos_cb = np.array([(wfig-(dfigr+dcbr_s+wcb_s))*cm2fig_x, (y_ax_cent-0.5*hcb_s)*cm2fig_y, wcb_s*cm2fig_x, hcb_s*cm2fig_y])
        #pos_cb = np.array([ (pos_axcm[-1,0]+pos_axcm[-1,2]+daxr_s+dcbl_s)*cm2fig_x, 
                            #(y_ax_cent-0.5*hcb_s)*cm2fig_y, 
                            #(wcb_s)*cm2fig_x, 
                            #(hcb_s)*cm2fig_y 
                        #])
        #hcb[nn] = fig.add_subplot(position=pos_cb)
        #hcb[nn].tick_params(labelsize=fs_ticks)
        #hcb[nn].set_position(pos_cb)
        #hcb[nn].set_xticks([])
        #hcb[nn].yaxis.tick_right()
        #hcb[nn].yaxis.set_label_position("right")

    #if plot_cb_bottom:
        #nn = -1
        #pos_cb = np.array([ (x_ax_cent-0.5*wcb_s)*cm2fig_x, 
                            #(dcbb_s)*cm2fig_y, 
                            #(wcb_s)*cm2fig_x, 
                            #(hcb_s)*cm2fig_y
                        #])
        #hcb[nn] = fig.add_subplot(position=pos_cb)
        #hcb[nn].set_position(pos_cb)
        #hcb[nn].tick_params(labelsize=fs_ticks)
        #hcb[nn].set_yticks([])

    #if reverse_order:
        #isort = np.arange(nx*ny, dtype=int).reshape((ny,nx)).transpose().flatten()
        #hca = list(np.array(hca)[isort]) 
        #hcb = list(np.array(hcb)[isort])

    ## add letters for subplots
    #if (do_axes_labels) and (axlab_kw is not None):
        #hca = axlab(hca, fontdict=axlab_kw)

    ##___________________________________________________________________________
    #list_argout=[]
    #if len(nargout)>0:
        #for stri in nargout:
            #try:
                #list_argout.append(eval(stri))
            #except:
                #print(f" -warning-> variable {stri} was not found, could not be added as output argument") 
            
        #return(list_argout)
    #else:
        #return
    ##return fig, hca, hcb



##
##
##_______________________________________________________________________________
## --> this based on work of Nils Bruegemann see 
## https://gitlab.dkrz.de/m300602/pyicon/-/blob/master/pyicon/pyicon_plotting.py
## i needed this to unify the ploting between icon and fesom for model comparison
## paper    
#def axlab(hca, figstr=[], posx=[-0.00], posy=[1.05], fontdict=None):
  #"""
#input:
#----------
  #hca:      list with axes handles
  #figstr:   list with strings that label the subplots
  #posx:     list with length 1 or len(hca) that gives the x-coordinate in ax-space
  #posy:     list with length 1 or len(hca) that gives the y-coordinate in ax-space
#last change:
#----------
#2015-07-21
  #"""

  ## make list that looks like [ '(a)', '(b)', '(c)', ... ]
  #if len(figstr)==0:
    ##lett = "abcdefghijklmnopqrstuvwxyz"
    #lett  = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
    #lett += ["a2","b2","c2","d2","e2","f2","g2","h2","i2","j2","k2","l2","m2","n2","o2","p2","q2","r2","s2","t2","u2","v2","w2","x2","y2","z2"]
    #lett = lett[0:len(hca)]
    #figstr = ["z"]*len(hca)
    #for nn, ax in enumerate(hca):
      #figstr[nn] = "(%s)" % (lett[nn])
  
  #if len(posx)==1:
    #posx = posx*len(hca)
  #if len(posy)==1:
    #posy = posy*len(hca)
  
  ## draw text
  #for nn, ax in enumerate(hca):
    #ht = hca[nn].text(posx[nn], posy[nn], figstr[nn], 
                      #transform = hca[nn].transAxes, 
                      #horizontalalignment = 'right',
                      #fontdict=fontdict)
    ## add text handle to axes to give possibility of changing text properties later
    ## e.g. by hca[nn].axlab.set_fontsize(8)
    #hca[nn].axlab = ht
##  for nn, ax in enumerate(hca):
##    #ax.set_title(figstr[nn]+'\n', loc='left', fontsize=10)
##    ax.set_title(figstr[nn], loc='left', fontsize=10)
  #return hca
