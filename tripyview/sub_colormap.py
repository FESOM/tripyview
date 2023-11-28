# Patrick Scholz, 14.12.2017
def colormap_c2c(cmin, cmax, cref, cnumb, cname, cstep=[]):

    import numpy                as np
    from   matplotlib.colors    import ListedColormap
    from   matplotlib.cm        import get_cmap
    from   scipy                import interpolate
    import cmocean 
    #import cmocean
    # cmin ... value of minimum color
    # cmax ... value of maximum color
    # cref ... value of reference color
    # cnumb... minimum number of colors that should be in colormap
    # cname... name of colormap ---> see colormap definition

    # colormaps are:
    # --> blue2red
    # --> red2blue
    # --> grads
    # --> rainbow
    # --> heat
    # --> jet
    # --> jet_w
    # --> hsv
    # --> gnuplot
    # --> arc
    # --> wbgyr
    # --> odv
    # --> odv_w

    # get colormap matrix from predifined cmaps
    # --> import matplotlib.pyplot as plt
    # --> cmap = plt.get_cmap('rainbow',9)
    # --> cmaplist = [cmap(i) for i in range(cmap.N)]

    #___________________________________________________________________________
    # calculate optimal color step size
    if not cstep:
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
    
    #___________________________________________________________________________
    # number of colors below ref value
    cnmb_bref = sum(clevel<cref)
    # number of color above ref value
    cnmb_aref = sum(clevel>cref)
    
    #___________________________________________________________________________
    if   'matplotlib' in cname:
        dum, cstr = cname.rsplit('.')
        if '_i' in cname: cstr, dum = cstr.rsplit('_i')
        cmap_def = get_cmap(cstr)(np.linspace(0,1,11))
        #cmap_def = get_cmap(cstr)(np.linspace(0,1,cnumb+1))
        #cmap_def = get_cmap(cstr)(np.linspace(0,1,len(clevel)+1))
        cmap_def = cmap_def[:,:-1]
        if '_i' in cname: cmap_def = np.flipud(cmap_def)
        
    elif 'cmocean' in cname:
        dum, cstr = cname.rsplit('.')
        if '_i' in cname: cstr, dum = cstr.rsplit('_i')
        cmap = eval("cmocean.cm.{}".format(cstr))
        cmap_def = cmap(np.linspace(0,1,11))
        #cmap_def = cmap(np.linspace(0,1,cnumb))
        cmap_def = cmap_def[:,:-1]    
        if '_i' in cname: cmap_def = np.flipud(cmap_def)
    else:
        #___________________________________________________________________________
        # MY different colormap definitions
        #---------------------------------------------------------------------------
        if   cname in ['blue2red', 'red2blue']: 
            cmap_def = np.array([[0.0, 0.19, 1.0], # blue
                                [0.0, 0.72, 1.0],
                                [1.0, 1.0 , 1.0], # white 
                                [1.0, 0.6 , 0.0],
                                [1.0, 0.19, 0.0]])# red
            if cname == 'red2blue': cmap_def = np.flipud(cmap_def)
        #---------------------------------------------------------------------------    
        elif cname in ['dblue2dred', 'dred2dblue']:
            cmap_def = np.array([[0.0, 0.19, 0.5],
                                [0.0, 0.19, 1.0],
                                [0.0, 0.72, 1.0],
                                [1.0, 1.0 , 1.0],
                                [1.0, 0.6 , 0.0],
                                [1.0, 0.19, 0.0],
                                [0.5, 0.19, 0.0]])        
            if cname == 'dred2dblue': cmap_def = np.flipud(cmap_def)
        #---------------------------------------------------------------------------    
        elif cname in ['green2orange', 'orange2green']:    
            cmap_def = np.array([[0.2196,    0.4196,      0.0],
                                [0.6039,    0.8039,      0.0],
                                [0.8000,    1.0000,      0.0],
                                [1.0000,    1.0000,   1.0000],
                                [1.0000,    0.6000,      0.0],
                                [0.6000,    0.2000,      0.0]])
            if cname == 'orange2green': cmap_def = np.flipud(cmap_def)
        #---------------------------------------------------------------------------    
        elif cname in ['grads', 'grads_i']:    
            cmap_def = np.array([[0.6275, 0.0   , 0.7843],
                                [0.1176, 0.2353, 1.0000],
                                [0.0   , 0.6275, 1.0000],
                                [0.0   , 0.8627, 0.0   ],
                                [1.0   , 1.0   , 1.0   ],
                                [0.9020, 0.8627, 0.1961],
                                [0.9412, 0.5098, 0.1569],
                                [0.9804, 0.2353, 0.2353],
                                [0.9412, 0.0   , 0.5098]])
            if cname == 'grads_i': cmap_def = np.flipud(cmap_def)
        #---------------------------------------------------------------------------
        elif cname in ['rainbow', 'rainbow_i']:  
            cmap_def = np.array([[0.5   , 0.0   , 1.0   ],
                                [0.25  , 0.3826, 0.9807],
                                [0.0   , 0.7071, 0.9238],
                                [0.25  , 0.9238, 0.8314],
                                [0.5   , 1.0   , 0.7071],
                                [0.75  , 0.9238, 0.5555],
                                [1.0   , 0.7071, 0.3826],
                                [1.0   , 0.3826, 0.1950],
                                [1.0   , 0.0   , 0.0   ]])
            if cname == 'rainbow_i': cmap_def = np.flipud(cmap_def)
        #---------------------------------------------------------------------------
        elif cname in ['heat', 'heat_i']:  
            cmap_def = np.array([[1.0   , 1.0   , 1.0],
                                [1.0   , 0.75  , 0.5],
                                [1.0   , 0.5   , 0.0],
                                [0.9375, 0.25  , 0.0],
                                [0.75  , 0.0   , 0.0],
                                [0.5625, 0.0   , 0.0],
                                [0.375 , 0.0   , 0.0],
                                [0.1875, 0.0   , 0.0],
                                [0.0   , 0.0   , 0.0]])
            if cname == 'heat_i': cmap_def = np.flipud(cmap_def)
        #---------------------------------------------------------------------------
        elif cname in ['jet', 'jet_i']:    
            cmap_def = np.array([[0.0   , 0.0   , 0.5   ],
                                [0.0   , 0.0   , 1.0   ],
                                [0.0   , 0.5   , 1.0   ],
                                [0.0806, 1.0   , 0.8870],
                                [0.4838, 1.0   , 0.4838],
                                [0.8870, 1.0   , 0.0806],
                                [1.0   , 0.5925, 0.0   ],
                                [1.0   , 0.1296, 0.0   ],
                                [0.5   , 0.0   , 0.0  ]])
            if cname == 'jet_i': cmap_def = np.flipud(cmap_def)
        #---------------------------------------------------------------------------    
        elif cname in ['jetw', 'jetw_i']:    
            cmap_def = np.array([[0.0   , 0.0   , 0.5   ],
                                [0.0   , 0.0   , 1.0   ],
                                [0.0   , 0.5   , 1.0   ],
                                [0.0806, 1.0   , 0.8870],
                                [1.0   , 1.0   , 1.0   ],
                                [0.8870, 1.0   , 0.0806],
                                [1.0   , 0.5925, 0.0   ],
                                [1.0   , 0.1296, 0.0   ],
                                [0.5   , 0.0   , 0.0  ]])
            if cname == 'jetw_i': cmap_def = np.flipud(cmap_def)
        #---------------------------------------------------------------------------    
        elif cname in ['hsv', 'hsv_i']:    
            cmap_def = np.array([[1.0   , 0.0   , 0.0   ],
                                [1.0   , 0.7382, 0.0   ],
                                [0.5236, 1.0   , 0.0   ],
                                [0.0   , 1.0   , 0.2148],
                                [0.0   , 1.0   , 0.9531],
                                [0.0   , 0.3085, 1.0   ],
                                [0.4291, 0.0   , 1.0   ],
                                [1.0   , 0.0   , 0.8320],
                                [1.0   , 0.0   , 0.0937]])
            if cname == 'hsv_i': cmap_def = np.flipud(cmap_def)
        #---------------------------------------------------------------------------    
        elif cname in ['gnuplot', 'gnuplot_i']:    
            cmap_def = np.array([[1.0   , 1.0   , 1.0    ],
                                [1.0   , 1.0   , 0.0    ],
                                [0.9354, 0.6699, 0.0    ],
                                [0.8660, 0.4218, 0.0    ],
                                [0.7905, 0.2441, 0.0    ],
                                [0.7071, 0.125 , 0.0    ],
                                [0.6123, 0.0527, 0.7071 ],
                                [0.5   , 0.0156, 1.0    ],
                                [0.3535, 0.0019, 0.7071]])
            if cname == 'gnuplot_i': cmap_def = np.flipud(cmap_def)
        #---------------------------------------------------------------------------        
        elif cname in ['arc', 'arc_i']:    
            cmap_def = np.array([[1.0000,    1.0000,    1.0000],
                                [0.6035,    0.8614,    0.7691],
                                [0.2462,    0.7346,    0.4610],
                                [0.2980,    0.7399,    0.2196],
                                [0.7569,    0.8776,    0.0754],
                                [0.9991,    0.9390,    0.0017],
                                [0.9830,    0.7386,    0.0353],
                                [0.9451,    0.2963,    0.1098],
                                [0.9603,    0.4562,    0.5268]])
            if cname == 'arc_i': cmap_def = np.flipud(cmap_def)
        #---------------------------------------------------------------------------        
        elif cname in ['wbgyr', 'wbgyr_i', 'rygbw', 'rygbw_i']:
            cmap_def = np.array([[1.0000,    1.0000,    1.0000],
                                [0.2000,    0.6000,    1.0000],
                                [0.0   ,    1.0000,    0.6000],
                                [1.0000,    1.0000,       0.0],
                                [1.0000,       0.0,       0.0]])
            if cname in ['wbgyr_i', 'rygbw']: cmap_def = np.flipud(cmap_def)
        #---------------------------------------------------------------------------
        elif cname in ['odv', 'odv_i']:    
            cmap_def = np.array([[0.9373,    0.7765,    0.9373],
                                [0.7804,    0.3647,    0.7490],
                                [0.1922,    0.2235,    1.0000],
                                [0.4824,    0.9686,    0.8706],
                                [0.4980,    1.0000,    0.4980],
                                [1.0000,    0.7843,    0.1373],
                                [1.0000,       0.0,       0.0],
                                [0.8392,    0.0627,    0.1922],
                                [1.0000,    0.7765,    0.5804]])
            if cname in ['odv_i']: cmap_def = np.flipud(cmap_def)
        #---------------------------------------------------------------------------
        elif cname in ['odvw', 'odvw_i']:    
            cmap_def = np.array([[0.9373,    0.7765,    0.9373],
                                [0.7804,    0.3647,    0.7490],
                                [0.1922,    0.2235,    1.0000],
                                [0.4824,    0.9686,    0.8706],
                                [1.0   ,    1.0000,    1.0   ],
                                [1.0000,    0.7843,    0.1373],
                                [1.0000,       0.0,       0.0],
                                [0.8392,    0.0627,    0.1922],
                                [1.0000,    0.7765,    0.5804]])
            if cname in ['odvw_i']: cmap_def = np.flipud(cmap_def)    
        #---------------------------------------------------------------------------
        elif cname in ['wvt', 'wvt_i']:    
            cmap_def = np.array([[255, 255, 255],
                                [255, 255, 153],
                                [255, 204,  51],
                                [255, 177, 100],
                                [255, 102, 102],
                                [255,  51,  51],
                                [153,   0,  51]])
            
        #---------------------------------------------------------------------------
        elif cname in ['seaice', 'seaice_i']:    
            cmap_def = np.array([[153,   0,  51],
                                [204,   0,   0],
                                [255, 102, 102],
                                [255, 153, 153],
                                [255, 255, 255],
                                [153, 255, 255],
                                [  0, 153, 204],
                                [  0,  51, 204],
                                [  0,  51, 153]])
            if cname in ['seaice_i']: cmap_def = np.flipud(cmap_def)      
        #---------------------------------------------------------------------------
        elif cname in ['precip', 'precip_i']:
            cmap_def = np.array([[ 64,   0,  75],
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
            if cname in ['precip_i']: cmap_def = np.flipud(cmap_def)      
        #---------------------------------------------------------------------------
        elif cname in ['drought', 'drought_i']:    
            cmap_def = np.array([[165,   0,  38],
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
            if cname in ['drought_i']: cmap_def = np.flipud(cmap_def)      
        #---------------------------------------------------------------------------
        elif cname in ['wvt1', 'wvt1_i']:    
            cmap_def = np.array([[255, 255, 255],
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
            if cname in ['wvt1_i']: cmap_def = np.flipud(cmap_def)      
        #---------------------------------------------------------------------------
        elif cname in ['wvt2', 'wvt2_i']:    
            cmap_def = np.array([[255, 255, 255],
                                [236, 226, 240],
                                [208, 209, 230],
                                [166, 189, 219],
                                [103, 169, 207],
                                [ 54, 144, 192],
                                [  2, 129, 138],
                                [  1, 108,  89],
                                [  1,  70,  54]])
            if cname in ['wvt2_i']: cmap_def = np.flipud(cmap_def)      
        #---------------------------------------------------------------------------
        elif cname in ['wvt3', 'wvt3_i']:        
            cmap_def = np.array([[247, 247, 247],
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
            if cname in ['wvt3_i']: cmap_def = np.flipud(cmap_def)
        #---------------------------------------------------------------------------
        elif cname in ['ars', 'ars_i']:        
            cmap_def = np.array([[255, 255, 255],
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
            if cname in ['ars_i']: cmap_def = np.flipud(cmap_def)      
        
        #---------------------------------------------------------------------------
        else: raise ValueError('this colormap name is not supported')    
        
    #___________________________________________________________________________
    if np.any(cmap_def>1.0): cmap_def = cmap_def/255
    
    #___________________________________________________________________________
    # define RGBA interpolator  
    cmap_idx = np.linspace(0,1,cmap_def.shape[0])
    cint_idx0 = clevel[:-1]+(clevel[1:]-clevel[:-1])/2
    #cint_idx0 = clevel
    
    if cnmb_aref<=sum(cmap_idx>0.5):
        cint_idx = interpolate.interp1d([cint_idx0[0],  cref], [0.0, 0.5], fill_value='extrapolate')(cint_idx0)
    elif cnmb_bref<=sum(cmap_idx<0.5):
        cint_idx = interpolate.interp1d([cref, cint_idx0[-1]], [0.5, 1.0], fill_value='extrapolate')(cint_idx0) 
    else:    
        cint_idx = interpolate.interp1d([cint_idx0[0], cref, cint_idx0[-1]], [0.0, 0.5, 1.0], fill_value='extrapolate')(cint_idx0) 
    
    #___________________________________________________________________________
    # define RGBA color matrix
    #print(cmap_def)
    r    = np.interp(x=cint_idx, xp=cmap_idx, fp=cmap_def[:,0])
    g    = np.interp(x=cint_idx, xp=cmap_idx, fp=cmap_def[:,1])
    b    = np.interp(x=cint_idx, xp=cmap_idx, fp=cmap_def[:,2])
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
            else         : arhsv[:,2] = np.linspace(0.2, chsv[2], nsc)[::-1]  
            
        else:
            if light2dark: arhsv[:,2] = np.linspace(chsv[2],0.8,nsc)
            else         : arhsv[:,2] = np.linspace(0.8, chsv[2], nsc)[::-1]
            
        arhsv      = np.flipud(arhsv)
        rgb = hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = rgb       
    cmap = ListedColormap(cols)
    return cmap
