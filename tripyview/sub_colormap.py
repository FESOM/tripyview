# Patrick Scholz, 14.12.2017

# ___CREATE INDIVIDUAL COLORMAP________________________________________________
#|                                                                             |
#|                                                                             |
#|                                                                             |
#|_____________________________________________________________________________|
def colormap_c2c(cmin, cmax, cref, cnumb, cname, cstep=None, do_slog=False, 
                 do_rescal=None, rescal_ref=None, cmap_arr=None):
    """
    --> create reference value centered colormap with distinct colorsteps based 
        on an rgb matrix
        
    Parameters: 
    
        :cmin:      float, value of minimum color
        
        :cmax:      float, value of maximum color
        
        :cref:      float, value of reference color
        
        :cnumb:     int, minimum number of colors that should be at least in 
                    the  colormap
                    
                    
        :cname:     str, name of colormap ---> see colormap definition, this can 
                    be extended by everyone. If you have have nice new colormap 
                    definition, come back to me and lets add it to the default!
                    
                    - blue2red, red2blue
                    - dblue2dred, dred2dblue
                    - green2orange, orange2green
                    - grads
                    - rainbow
                    - heat
                    - jet
                    - jetw (white reference color)
                    - hsv
                    - gnuplot
                    - arc
                    - wbgyr, rygbw (white, blue, green ,yellow,red)
                    - odv
                    - odvw (white reference color)
                    - wvt
                    - wvt1
                    - wvt2
                    - wvt3
                    - seaice
                    - precip
                    - drought
                    - ars
                    
                    If you put a "_i" behind the color name string, the colormap will 
                    be inverted
                    
        :cstep:     float, (default=None) provide fixed color step instead of computing it 
                    based on the covered value range and the minimum number of colors    
        
        :do_slog:   bool, (default=False) provide colormap for symmetric logarithmic 
                    scaling
        
        :do_rescal: np.array, (default=None) provide np.array with non linear colorstep
                    values
                    
        :rescal_ref: float, (default=None) provide the reference value in the non-linear
                     colorstep array
                     
        :cmap_arr:  np.array, (default=None) provide a (nc x 3) rgb value colormap 
                    definition array from the outside beside the predefined ones. 
                    nc must be an odd number. The index of np.ceil(nc/2) will represent 
                    the color of the reference value 
                    
        Returns:
        
            :cmap:  matplotlib.colors.ListedColormap computed based on the input 
                    parameters
            
            :clevel:np.array() with chosen clevel values, needed for contours and 
                    colorbar definition
            
            :cref:  float, returns used cref value 

    #___________________________________________________________________________
    """
    # get colormap matrix from predifined cmaps
    # --> import matplotlib.pyplot as plt
    # --> cmap = plt.get_cmap('rainbow',9)
    # --> cmaplist = [cmap(i) for i in range(cmap.N)]

    import numpy                as np
    from   matplotlib.colors    import ListedColormap
    from   matplotlib.pyplot    import get_cmap # for python ver>=3.11
    from   scipy                import interpolate
    import cmocean 
    
    #___________________________________________________________________________
    # calculate optimal color step size
    if cstep is None:
        cdelta   = (cmax-cmin)/cnumb
        cstep_all= np.array([0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 25.0, 50.0])
        cstep_all=cstep_all*(10**np.floor(np.log10( np.abs(cdelta) )))
        cstep_i  = np.squeeze(np.where(cstep_all<=cdelta))
        cstep_i  = cstep_i[-1] 
        cstep    = cstep_all[cstep_i]
    
    #___________________________________________________________________________
    # calculate colormap levels
    clevel   = np.concatenate((np.sort(np.arange(cref-cstep,cmin-cstep,-cstep)),np.arange(cref,cmax+cstep,cstep)))
    if np.abs(clevel.min())>1.0e-15:
        #clevel   = np.around(clevel, -np.int32(np.floor(np.log10(np.abs( clevel.min() ))-2) ) )
        clevel   = np.around(clevel, -np.int32(np.floor(np.log10(np.abs( cstep ))-2) ) )
        cref     = np.around(cref  , -np.int32(np.floor(np.log10(np.abs( cstep ))-2) ) )
    clevel   = np.unique(clevel)
    #print(clevel)
    
    # rescale the interpolation frojm linear to more exponential, but only for 
    # the color interpolation
    if do_slog==True:
        exp=2.0
        dum_clev = np.arange(cref,cmax+cstep,cstep)
        dum_clev = np.interp(x=np.linspace(0,1,dum_clev.size)**exp, xp=np.linspace(0,1,dum_clev.size), fp=dum_clev)
        dum_clev = np.concatenate((-dum_clev[1:], dum_clev))
        clevel_slg10 = np.sort(np.unique(dum_clev))
        
    #___________________________________________________________________________
    
    if do_rescal is not None and rescal_ref is not None:
        # number of colors below ref value
        cnmb_bref = sum(do_rescal<rescal_ref)
        # number of color above ref value
        cnmb_aref = sum(do_rescal>rescal_ref)
        cref = cnmb_bref
    else:
        # number of colors below ref value
        cnmb_bref = sum(clevel<cref)
        # number of color above ref value
        cnmb_aref = sum(clevel>cref)
    #print(cnmb_aref, cnmb_bref)
    
    #___________________________________________________________________________
    # if the cmap_arr is not already predifined from the outside
    if cmap_arr is None:
        if   'matplotlib' in cname:
            dum, cstr = cname.rsplit('.')
            if '_i' in cname: cstr, dum = cstr.rsplit('_i')
            cmap_arr = get_cmap(cstr)(np.linspace(0,1,11))
            #cmap_arr = get_cmap(cstr)(np.linspace(0,1,cnumb+1))
            #cmap_arr = get_cmap(cstr)(np.linspace(0,1,len(clevel)+1))
            cmap_arr = cmap_arr[:,:-1]
            
        elif 'cmocean' in cname:
            dum, cstr = cname.rsplit('.')
            if '_i' in cname: cstr, dum = cstr.rsplit('_i')
            cmap = eval("cmocean.cm.{}".format(cstr))
            cmap_arr = cmap(np.linspace(0,1,11))
            #cmap_arr = cmap(np.linspace(0,1,cnumb))
            cmap_arr = cmap_arr[:,:-1]    
        
        else:
            #___________________________________________________________________
            # MY different colormap definitions
            #-------------------------------------------------------------------
            if   cname in ['blue2red', 'red2blue']: 
                cmap_arr = np.array([[0.0, 0.19, 1.0], # blue
                                    [0.0, 0.72, 1.0],
                                    [1.0, 1.0 , 1.0], # white 
                                    [1.0, 0.6 , 0.0],
                                    [1.0, 0.19, 0.0]])# red
                if cname == 'red2blue': cmap_arr = np.flipud(cmap_arr)
            #-------------------------------------------------------------------
            elif cname in ['dblue2dred', 'dred2dblue']:
                cmap_arr = np.array([[0.0, 0.19, 0.5],
                                    [0.0, 0.19, 1.0],
                                    [0.0, 0.72, 1.0],
                                    [1.0, 1.0 , 1.0],
                                    [1.0, 0.6 , 0.0],
                                    [1.0, 0.19, 0.0],
                                    [0.5, 0.19, 0.0]])        
                if cname == 'dred2dblue': cmap_arr = np.flipud(cmap_arr)
            #-------------------------------------------------------------------
            elif cname in ['green2orange', 'orange2green']:    
                cmap_arr = np.array([
                                    [0.6039,    0.8039,      0.0],
                                    [0.8000,    1.0000,      0.0],
                                    [1.0000,    1.0000,   1.0000],
                                    [1.0000,    0.6000,      0.0],
                                    [0.6000,    0.2000,      0.0]]) #[0.2196,    0.4196,      0.0],
                if cname == 'orange2green': cmap_arr = np.flipud(cmap_arr)
                
            #-------------------------------------------------------------------
            elif cname in ['grads']:    
                cmap_arr = np.array([[0.6275, 0.0   , 0.7843],
                                    [0.1176, 0.2353, 1.0000],
                                    [0.0   , 0.6275, 1.0000],
                                    [0.0   , 0.8627, 0.0   ],
                                    [1.0   , 1.0   , 1.0   ],
                                    [0.9020, 0.8627, 0.1961],
                                    [0.9412, 0.5098, 0.1569],
                                    [0.9804, 0.2353, 0.2353],
                                    [0.9412, 0.0   , 0.5098]])
                
            #-------------------------------------------------------------------
            elif cname in ['rainbow']:  
                cmap_arr = np.array([[0.5   , 0.0   , 1.0   ],
                                    [0.25  , 0.3826, 0.9807],
                                    [0.0   , 0.7071, 0.9238],
                                    [0.25  , 0.9238, 0.8314],
                                    [0.5   , 1.0   , 0.7071],
                                    [0.75  , 0.9238, 0.5555],
                                    [1.0   , 0.7071, 0.3826],
                                    [1.0   , 0.3826, 0.1950],
                                    [1.0   , 0.0   , 0.0   ]])
            #-------------------------------------------------------------------
            elif   cname in ['curl']: 
                cmap_arr = np.array([[  0, 200,  50], # green
                                     [  0, 200, 167],
                                     [255, 255, 255], # white 
                                     [200,   0, 140], 
                                     [120,   0, 200]])# purple
                                    
            #-------------------------------------------------------------------
            elif cname in ['heat']:  
                cmap_arr = np.array([[1.0   , 1.0   , 1.0],
                                    [1.0   , 0.75  , 0.5],
                                    [1.0   , 0.5   , 0.0],
                                    [0.9375, 0.25  , 0.0],
                                    [0.75  , 0.0   , 0.0],
                                    [0.5625, 0.0   , 0.0],
                                    [0.375 , 0.0   , 0.0],
                                    [0.1875, 0.0   , 0.0],
                                    [0.0   , 0.0   , 0.0]])
                
            #-------------------------------------------------------------------
            elif cname in ['jet']:    
                cmap_arr = np.array([[0.0   , 0.0   , 0.5   ],
                                    [0.0   , 0.0   , 1.0   ],
                                    [0.0   , 0.5   , 1.0   ],
                                    [0.0806, 1.0   , 0.8870],
                                    [0.4838, 1.0   , 0.4838],
                                    [0.8870, 1.0   , 0.0806],
                                    [1.0   , 0.5925, 0.0   ],
                                    [1.0   , 0.1296, 0.0   ],
                                    [0.5   , 0.0   , 0.0  ]])
            
            #-------------------------------------------------------------------
            elif cname in ['jetw']:    
                cmap_arr = np.array([[0.0   , 0.0   , 0.5   ],
                                    [0.0   , 0.0   , 1.0   ],
                                    [0.0   , 0.5   , 1.0   ],
                                    [0.0806, 1.0   , 0.8870],
                                    [1.0   , 1.0   , 1.0   ],
                                    [0.8870, 1.0   , 0.0806],
                                    [1.0   , 0.5925, 0.0   ],
                                    [1.0   , 0.1296, 0.0   ],
                                    [0.5   , 0.0   , 0.0  ]])
            
            #-------------------------------------------------------------------
            elif cname in ['hsv']:    
                cmap_arr = np.array([[1.0   , 0.0   , 0.0   ],
                                    [1.0   , 0.7382, 0.0   ],
                                    [0.5236, 1.0   , 0.0   ],
                                    [0.0   , 1.0   , 0.2148],
                                    [0.0   , 1.0   , 0.9531],
                                    [0.0   , 0.3085, 1.0   ],
                                    [0.4291, 0.0   , 1.0   ],
                                    [1.0   , 0.0   , 0.8320],
                                    [1.0   , 0.0   , 0.0937]])
            
            #-------------------------------------------------------------------
            elif cname in ['gnuplot']:    
                cmap_arr = np.array([[1.0   , 1.0   , 1.0    ],
                                    [1.0   , 1.0   , 0.0    ],
                                    [0.9354, 0.6699, 0.0    ],
                                    [0.8660, 0.4218, 0.0    ],
                                    [0.7905, 0.2441, 0.0    ],
                                    [0.7071, 0.125 , 0.0    ],
                                    [0.6123, 0.0527, 0.7071 ],
                                    [0.5   , 0.0156, 1.0    ],
                                    [0.3535, 0.0019, 0.7071]])
            
            #-------------------------------------------------------------------
            elif cname in ['arc']:    
                cmap_arr = np.array([[1.0000,    1.0000,    1.0000],
                                    [0.6035,    0.8614,    0.7691],
                                    [0.2462,    0.7346,    0.4610],
                                    [0.2980,    0.7399,    0.2196],
                                    [0.7569,    0.8776,    0.0754],
                                    [0.9991,    0.9390,    0.0017],
                                    [0.9830,    0.7386,    0.0353],
                                    [0.9451,    0.2963,    0.1098],
                                    [0.9603,    0.4562,    0.5268]])
            
            #-------------------------------------------------------------------
            elif cname in ['wbgyr', 'rygbw']:
                cmap_arr = np.array([[1.0000,    1.0000,    1.0000],
                                    [0.2000,    0.6000,    1.0000],
                                    [0.0   ,    1.0000,    0.6000],
                                    [1.0000,    1.0000,       0.0],
                                    [1.0000,       0.0,       0.0]])
                if cname in ['rygbw']: cmap_arr = np.flipud(cmap_arr)
                
            #-------------------------------------------------------------------
            elif cname in ['odv']:    
                cmap_arr = np.array([[0.9373,    0.7765,    0.9373],
                                    [0.7804,    0.3647,    0.7490],
                                    [0.1922,    0.2235,    1.0000],
                                    [0.4824,    0.9686,    0.8706],
                                    [0.4980,    1.0000,    0.4980],
                                    [1.0000,    0.7843,    0.1373],
                                    [1.0000,       0.0,       0.0],
                                    [0.8392,    0.0627,    0.1922],
                                    [1.0000,    0.7765,    0.5804]])
            
            #-------------------------------------------------------------------
            elif cname in ['odvw']:    
                cmap_arr = np.array([[0.9373,    0.7765,    0.9373],
                                    [0.7804,    0.3647,    0.7490],
                                    [0.1922,    0.2235,    1.0000],
                                    [0.4824,    0.9686,    0.8706],
                                    [1.0   ,    1.0000,    1.0   ],
                                    [1.0000,    0.7843,    0.1373],
                                    [1.0000,       0.0,       0.0],
                                    [0.8392,    0.0627,    0.1922],
                                    [1.0000,    0.7765,    0.5804]])
            
            #-------------------------------------------------------------------
            elif cname in ['wvt']:    
                cmap_arr = np.array([[255, 255, 255],
                                    [255, 255, 153],
                                    [255, 204,  51],
                                    [255, 177, 100],
                                    [255, 102, 102],
                                    [255,  51,  51],
                                    [153,   0,  51]])
                
            #-------------------------------------------------------------------
            elif cname in ['seaice']:    
                cmap_arr = np.array([[153,   0,  51],
                                    [204,   0,   0],
                                    [255, 102, 102],
                                    [255, 153, 153],
                                    [255, 255, 255],
                                    [153, 255, 255],
                                    [  0, 153, 204],
                                    [  0,  51, 204],
                                    [  0,  51, 153]])
            
            #-------------------------------------------------------------------
            elif cname in ['precip']:
                cmap_arr = np.array([[ 64,   0,  75],
                                    [118,  42, 131],
                                    [153, 112, 171],
                                    [194, 165, 207],
                                    [231, 212, 232],
                                    [247, 247, 247],
                                    [217, 240, 211],
                                    [166, 219, 160],
                                    [ 90, 174,  97],
                                    [ 27, 120,  55],
                                    [  0,  68,  27]])
            
            #-------------------------------------------------------------------
            elif cname in ['drought']:    
                cmap_arr = np.array([[165,   0,  38],
                                    [215,  48,  39],
                                    [244, 109,  67],
                                    [253, 174,  97],
                                    [254, 224, 139],
                                    [255, 255, 255],
                                    [217, 239, 139],
                                    [166, 217, 106],
                                    [102, 189,  99],
                                    [ 26, 152,  80],
                                    [  0, 104,  55]])
            
            #-------------------------------------------------------------------
            elif cname in ['wvt1']:    
                cmap_arr = np.array([[255, 255, 255],
                                    [161, 218, 180],
                                    [ 65, 182, 196],
                                    [ 44, 127, 184],
                                    [ 37,  52, 148],
                                    [194, 230, 120],
                                    [254, 204,  92],
                                    [253, 141,  60],
                                    [240,  59,  32],
                                    [189,   0,  38],
                                    [165,  15,  21]])
            
            #-------------------------------------------------------------------
            elif cname in ['wvt2']:    
                cmap_arr = np.array([[255, 255, 255],
                                    [236, 226, 240],
                                    [208, 209, 230],
                                    [166, 189, 219],
                                    [103, 169, 207],
                                    [ 54, 144, 192],
                                    [  2, 129, 138],
                                    [  1, 108,  89],
                                    [  1,  70,  54]])
            
            #-------------------------------------------------------------------
            elif cname in ['wvt3']:        
                cmap_arr = np.array([[247, 247, 247],
                                    [ 50, 136, 189],
                                    [102, 194, 165],
                                    [171, 221, 164],
                                    [230, 245, 152],
                                    [255, 255, 191],
                                    [254, 224, 139],
                                    [253, 174,  97],
                                    [244, 109,  67],
                                    [213,  62,  79],
                                    [158,   1,  66]]);
            
            #-------------------------------------------------------------------
            elif cname in ['ars']:        
                cmap_arr = np.array([[255, 255, 255],
                                    [ 25, 175, 255],
                                    [ 68, 202, 255],
                                    [106, 235, 225],
                                    [138, 236, 174],
                                    [205, 255, 162],
                                    [240, 236, 121],
                                    [255, 189,  87],
                                    [244, 117,  75],
                                    [255,  90,  90],
                                    [255, 158, 158],
                                    [255, 196, 196],
                                    [255, 235, 235]])
            
            #-------------------------------------------------------------------
            else: raise ValueError('this colormap name is not supported')    
        
        #_______________________________________________________________________
        # invert colormap if "_i" in cname
        if '_i' in cname: cmap_arr = np.flipud(cmap_arr)
        
    #___________________________________________________________________________
    if np.any(cmap_arr>1.0): cmap_arr = cmap_arr/255
    
    #___________________________________________________________________________
    # define RGBA interpolator  
    cmap_idx  = np.linspace(0,1,cmap_arr.shape[0])
    
    cint_idx0 = clevel[:-1]+(clevel[1:]-clevel[:-1])/2
    if do_slog: cint_idx0 = clevel_slg10[:-1]+(clevel_slg10[1:]-clevel_slg10[:-1])/2
    
    if cnmb_aref<=sum(cmap_idx>0.5):
        #print('case A')
        cint_idx = interpolate.interp1d([cint_idx0[0],  cref], [0.0, 0.5], fill_value='extrapolate')(cint_idx0)
    elif cnmb_bref<=sum(cmap_idx<0.5):
        #print('case B')
        cint_idx = interpolate.interp1d([cref, cint_idx0[-1]], [0.5, 1.0], fill_value='extrapolate')(cint_idx0) 
    else:    
        #print('case C')
        cint_idx = interpolate.interp1d([cint_idx0[0], cref, cint_idx0[-1]], [0.0, 0.5, 1.0], fill_value='extrapolate')(cint_idx0) 
    #print(cint_idx)
    
    #___________________________________________________________________________
    # define RGBA color matrix
    #print(cmap_arr)
    r    = np.interp(x=cint_idx, xp=cmap_idx, fp=cmap_arr[:,0])
    g    = np.interp(x=cint_idx, xp=cmap_idx, fp=cmap_arr[:,1])
    b    = np.interp(x=cint_idx, xp=cmap_idx, fp=cmap_arr[:,2])
    a    = np.ones(cint_idx.shape)
    rgba = np.vstack((r, g, b, a)).transpose()
    del(r, g, b, a)
    #print(rgba[:,:-1])
    
    #___________________________________________________________________________
    # define colormap 
    cmap =  ListedColormap(rgba, name=cname, N=cint_idx.size)
    cmap.set_under(rgba[ 0,:])
    cmap.set_over( rgba[-1,:])
    
    #___________________________________________________________________________
    return(cmap,clevel,cref)



#_______________________________________________________________________________
# please see at:
# --> https://stackoverflow.com/questions/47222585/matplotlib-generic-colormap-from-tab10
# also https://www.py4u.net/discuss/222050
#_______________________________________________________________________________
def categorical_cmap(nc, nsc, cmap="tab10", cmap2='nipy_spectral', continuous=False, light2dark=True):
    """
    --> build a categorical colormap, based on the predefined standard matplotlib
        colormap e.g. tab10 and split each color into a number subcolors from light 
        to dark 
        
    Parameters:
    
        :nc:        int, number of colors in colormap
        
        :nsc:       int, number of sub-colors in each color
    
        :cmap:      str, (default="tab10") name of matplotlib colormap
        
        :cmap2:     str, (default="nipy_spectral") replacement colormap when the 
                    number of colors nc exceed the number of colors in the actual colormap
        
        :continuous: bool, (default=False)
        
        :light2dark: bool, (default=True), True: do subcategorigal color interpolation from 
                     lighttgrey-->color, False: do subcategorigal color interpolation from 
                     color --> darkgrey
                     
    Returns:
        
        :cmap:      matplotlib.colors.ListedColormap computed based on the input 
                    parameters
    
    #___________________________________________________________________________
    """
    
    from   matplotlib.colors import ListedColormap, rgb_to_hsv, hsv_to_rgb
    import matplotlib.pylab as plt
    import numpy as np
    #if nc > plt.get_cmap(cmap).N: cmap = "hsv"
        #raise ValueError("Too many categories for colormap.")
        
    if continuous:
        if nc > plt.get_cmap(cmap).N:
            ccolors = ListedColormap(plt.cm.get_cmap(cmap2, nc)(np.linspace(0,1,nc))).colors
        else:    
            ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
            
    else:
        if nc > plt.get_cmap(cmap).N:
            ccolors = ListedColormap(plt.cm.get_cmap(cmap2, nc)(np.arange(nc, dtype=int))).colors
        else:
            ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
            
            
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv,nsc).reshape(nsc,3)
        if not chsv[0]==0.0 and not chsv[1]==0.0:
            arhsv[:,1] = np.linspace(chsv[1],0.2,nsc)
            if light2dark: arhsv[:,2] = np.linspace(chsv[2], 0.9, nsc)
            else         : arhsv[:,2] = np.linspace(0.1, chsv[2], nsc)[::-1]  
            
        else:
            if light2dark: arhsv[:,2] = np.linspace(chsv[2],0.9,nsc)
            else         : arhsv[:,2] = np.linspace(0.1, chsv[2], nsc)[::-1]
            
        arhsv      = np.flipud(arhsv)
        rgb = hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = rgb       
    cmap = ListedColormap(cols)
    return cmap
