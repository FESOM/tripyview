import os
import xarray as xr
import pandas as pa
import numpy  as np

#
#
#_____________________________________________________________________________________________
def do_node_neighbour(mesh, do_nghbr_e=False):
    print(' --> compute node neighbourhood')
    #_________________________________________________________________________________________
    # compute  infrastructure
    # nmb_n_in_e ...number of elements with node
    # n_nghbr_e  ...element index with node   
    nmb_n_in_e = np.zeros((mesh.n2dn,), dtype=np.int32)
    n_nghbr_e     =  [ [] for _ in range(mesh.n2dn)] 
    for elemi in range(0,mesh.n2de):
        nmb_n_in_e[mesh.e_i[elemi,:]] = nmb_n_in_e[mesh.e_i[elemi,:]] + 1
        nodes = mesh.e_i[elemi,:]
        for ni in range(0,3): n_nghbr_e[nodes[ni]].append(elemi)  

    # n_nghbr_n   ...index of neighbouring nodes         
    n_nghbr_n     =  [ [] for _ in range(mesh.n2dn)] 
    for nodei in range(0,mesh.n2dn):

        # loop over neigbouring elements
        for nie in range(nmb_n_in_e[nodei]):
            elemi  = n_nghbr_e[nodei][nie]

            # loop over vertice indecies of element elemi 
            for ni in mesh.e_i[elemi,:]:
                if (ni==nodei) or (ni in n_nghbr_n[nodei]): continue
                n_nghbr_n[nodei].append(ni)    

        # sort indices (not realy necessary ?!)
        n_nghbr_n[nodei].sort()
    
    #_________________________________________________________________________________________
    if do_nghbr_e: 
        return(n_nghbr_n, n_nghbr_e)
    else:
        return(n_nghbr_n)

#
#
#_____________________________________________________________________________________________
def do_elem_neighbour(mesh):
    #_________________________________________________________________________________________
    # compute node neighbourhood
    # n_nghbr_n ... neighboring nodes
    # n_nghbr_e ... neighboring elem
    n_nghbr_n, n_nghbr_e = do_node_neighbour(mesh, do_nghbr_e=True)
    
    #_________________________________________________________________________________________
    print(' --> compute edge neighbourhood')
    # compute edge and neighbouring elements with respect to edge
    # edges     ... list with node indices that form edge
    # ed_nghbr_e... neighbouring elements with respect to edge 
    edges      = list()
    ed_nghbr_e = list()
    # loop over nodes
    for n in range(0,mesh.n2dn):
        # loop over the neighbouring nodes 
        for nghbr_n in n_nghbr_n[n]:
            
            # do not dublicate edges
            if nghbr_n<n: continue
            
            # loop over the neighbouring elements
            elems=[]
            for nghbr_e in n_nghbr_e[n]:
                elnodes = mesh.e_i[nghbr_e,:]
                
                # if neighbouring nodes to node n is found in neighbouring elements
                if nghbr_n in elnodes:
                    elems.append(nghbr_e)
                
                if len(elems)==2: break
                    
            # write node indices and elem indices that contribute to edge
            if len(elems)==1: elems.append(-999)
            edges.append([n,nghbr_n])
            ed_nghbr_e.append(elems)
            
    #_________________________________________________________________________________________
    print(' --> compute elem neighbourhood')        
    # compute edge indices with respect to element
    # e_nghbr_ed ... neighbouring edges with respect to element 
    n2ded = len(edges)   
    e_nghbr_ed =  [ [] for _ in range(mesh.n2de)]
    # loop over edges
    for edi in range(0,n2ded):
        # loop over neighbouring elements with respect to edge
        for nghbr_e in ed_nghbr_e[edi]:
            
            # its a boundary edge, has only one valid elem neighbour
            if nghbr_e<0: 
                continue 
            else:    
                e_nghbr_ed[nghbr_e].append(edi)
                
                
    # compute neighbouring elem indecies with respect to elem
    # e_nghbr_e ... neighbouring elements with respect to elem
    e_nghbr_e =  [ [] for _ in range(mesh.n2de)]
    for ei in range(0,mesh.n2de):
        for nghbr_ed in e_nghbr_ed[ei]:
            elems = ed_nghbr_e[nghbr_ed]     
            if elems[0]==ei: e_nghbr_e[ei].append(elems[1])
            else:            e_nghbr_e[ei].append(elems[0])
    
    #_________________________________________________________________________________________
    return(e_nghbr_e)

#
#
#_____________________________________________________________________________________________
def do_node_smoothing(mesh, data_orig, n_nghbr_n, weaksmth_boxlist, rel_cent_weight, num_iter):
    print(' --> compute node smoothing')
    #_________________________________________________________________________________________
    data_smooth = data_orig.copy()
    # compute smoothing
    for it in range(num_iter):
        print('     iter: {}'.format(str(it)))
        # loop over nodes
        data_done = data_smooth.copy()
        for nodei in range(0,mesh.n2dn):
            # smooth from down to top 
            idx = np.argmax(data_done)
            data_done[idx] = -99999.0

            # set coeff
            coeff1=1.0
            
            # include special region with less smoothing by increaing the 
            # weight of the center
            if( not weaksmth_boxlist == False): 
                for box in weaksmth_boxlist:
                    if (mesh.n_x[idx]>=box[0] and mesh.n_x[idx]<=box[1] and
                        mesh.n_y[idx]>=box[2] and mesh.n_y[idx]<=box[3] ): 
                        coeff1=box[4]
            
            # do convolution over neighbours (weight of neighbours = 1)
            # sum depth over neighbouring nodes 
            dsum=0.0
            nmb_n_nghbr=0
            for ni in n_nghbr_n[idx]: 
                dsum=dsum+data_smooth[ni]
                nmb_n_nghbr=nmb_n_nghbr+1

            # add contribution from center weight 
            dsum=dsum + np.real(coeff1*rel_cent_weight*(nmb_n_nghbr-1.0)-1.0)*data_smooth[idx]
            data_smooth[idx]=dsum/np.real( nmb_n_nghbr + coeff1*rel_cent_weight*(nmb_n_nghbr-1)-1.0 )
            #                                  |                       | 
            #                   sum over weight (=1)          center weight (depends
            #                   of neighbours                 on number of neighbours)
    #_________________________________________________________________________________________
    return(data_smooth)

#
#
#_____________________________________________________________________________________________
def do_elem_smoothing(mesh, data_orig, e_nghbr_e, weaksmth_boxlist, rel_cent_weight, num_iter):
    print(' --> compute elem smoothing')
    #_________________________________________________________________________________________
    data_smooth = data_orig.copy()
    # compute smoothing
    for it in range(num_iter):
        print('     iter: {}'.format(str(it)))
        # loop over nodes
        data_done = data_smooth.copy()
        for elemi in range(0,mesh.n2de):
            # smooth from down to top 
            idx = np.argmax(data_done)
            data_done[idx] = -99999.0

            # set coeff
            coeff1=1.0
            
            # include special region with less smoothing
            if( not weaksmth_boxlist == False): 
                for box in weaksmth_boxlist:
                    if (mesh.n_x[mesh.e_i[idx,:]].sum()/3 >=box[0] and mesh.n_x[mesh.e_i[idx,:]].sum()/3 <=box[1] and
                        mesh.n_y[mesh.e_i[idx,:]].sum()/3 >=box[2] and mesh.n_y[mesh.e_i[idx,:]].sum()/3 <=box[3] ): 
#                         coeff1=special_smth_coeff
                        coeff1=box[4]

            # sum depth over neighbouring elements 
            dsum=0.0
            nmb_e_nghbr = 0
            for ei in e_nghbr_e[idx]: 
                if ei<0: continue
                dsum=dsum+data_smooth[ei]
                nmb_e_nghbr = nmb_e_nghbr+1

            # add contribution from center weight 
            dsum=dsum + (coeff1*rel_cent_weight*(nmb_e_nghbr-1.0)-1.0)*data_smooth[idx]
            data_smooth[idx]=dsum/np.real( nmb_e_nghbr + coeff1*rel_cent_weight*(nmb_e_nghbr-1)-1.0 )
            #                                  |                        | 
            #                   sum over weight (=1)          center weight (depends
            #                   of neighbours                 on number of neighbours)
            
    #_________________________________________________________________________________________
    return(data_smooth)




import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual, HBox, VBox, Layout
import ipywidgets as widgets
from sub_index import do_boxmask
from matplotlib.tri import Triangulation
class select_scatterpts_depth(object):
    
    #
    #
    # ___INIT SELECT_SCATTER_POINTS DEPTH OBJECT______________________________________________________
    def __init__(self, mesh, data, box_list, do_elem=True, do_tri=True, clim=None, 
                 cname='terrain', cnum=20, seldeprange=[-2000, -100], seldepdefault=-680, 
                 figax=None, figsize=[10, 8], scatsize=500, zoom_fac=0.25, do_centersel=True,
                 showtxt=True, do_datacopy=True):
        
        #_____________________________________________________________________________________________
        # init varaibles for selection
        self.xs, self.ys          = [], []
        self.vs, self.vs_new      = [], []
        self.idx_box              = 0
        self.do_elem              = do_elem
        self.do_tri, self.tri     = do_tri, []
        self.htri                 = None
        
        # data from mouse/key press event
        self.xm, self.ym, self.bm = [], [], []
        self.xk, self.yk, self.bk = [], [], []
        self.idx_c = []
        
        # handle for: figure, axes, scatter, text slider
        self.fig, self.ax         = [], []
        self.hscat, self.hchos    = None, []
        self.htxt, self.hitxt     = None, []
        self.slider, self.seldep  = [], seldepdefault
        self.ssize, self.zoom_fac = scatsize, zoom_fac
        self.showtxt              = showtxt
        self.centersel            = do_centersel
        
        # handle of mouse key press event 
        self.cid_m, self.cid_k    = [], []
        
        # original xy limits
        self.xlim_o, self.ylim_o  = [], []
        self.dxlim_o, self.dylim_o= [], []
        self.cbar                 = []
        
        self.mesh                 = mesh
        if do_datacopy: self.data = data.copy()
        else          : self.data = data        
        self.box_list             = box_list
        self.idxd                 = [] # index to convert from local box to global data
        
        self.cmin, self.cmax      = [], []
        self.cstep, self.clevel   = 50, []
        self.cmap, self.cname     = [], cname
        self.cnum                 = cnum
        
        #_____________________________________________________________________________________________
        # set indices mask for vertices and element centroids in box 
        # create first round of scatter points from box_list with index 0 
        self._update_scatterpts_()
        
        #_____________________________________________________________________________________________
        if figax is not None:
            self.fig, self.ax = figax[0], figax[1]
        else:    
            self.fig, self.ax = plt.figure(figsize=figsize), plt.gca()
        
        #_____________________________________________________________________________________________
        # set cmin, cmax, clevel, cmap
        self.cmin, self.cmax = self.vs.min(), self.vs.max()
        self.cmin = round(self.cmin,-np.int(np.fix(np.log10(np.abs(self.cmin))-1)))
        self.cmax = round(self.cmax,-np.int(np.fix(np.log10(np.abs(self.cmax))-1)))
        self._sel_colorrange_([self.cmin,self.cmax])
        
        #_____________________________________________________________________________________________
        # scatter plot with vertices or elmental depth values #
        self.hscat=self.ax.scatter(self.xs, self.ys, self.ssize, self.vs, 
                              vmin=self.clevel[0], vmax=self.clevel[-1], 
                              cmap=self.cmap, edgecolor='k')  
                
        #_____________________________________________________________________________________________
        # write text string of depth in each scatter point
        #self._update_scattertxt_()
        
        #_____________________________________________________________________________________________
        self._update_scattertri_()
        
        #_____________________________________________________________________________________________
        # set initial axis limits from box 
        self._update_scatteraxes_()
        
        #_____________________________________________________________________________________________
        # do title string and colorbar string
        self._update_scattertitle_()
            
        if not self.cbar==True:    
            self.cbar = self.fig.colorbar(self.hscat, ax=self.ax, ticks=self.clevel, label='depth [m]')
        
        #_____________________________________________________________________________________________
        # create infoo text at bottom of axes
        self.hitxt = self.ax.annotate("default", [0.01,0.01], xycoords='figure fraction' , va="bottom", ha="left", zorder=10)
        
        # create another scatterplot outside of axes for the highlighting of the selection
        self.hchos = self.ax.scatter(-999, -999, s=self.ssize, c='w', edgecolor=[0.1,1,0.5], facecolor=None, linewidth=8, alpha=0.75)  
        
        #_____________________________________________________________________________________________
        # create slider to choose the depth that should be setted
        interact(self._sel_depth_      , depth = widgets.IntSlider(min=seldeprange[0], max=seldeprange[1], step=5, 
                                                 value=seldepdefault, description='select depth:', continuous_update=False,
                                                 layout=widgets.Layout(width='50%')))
        
        interact(self._sel_scattersize_, ssize = widgets.IntSlider(min=10, max=2000, step=50, value=self.ssize, 
                                                 description='scatpts size:', continuous_update=False,
                                                 layout=widgets.Layout(width='50%')))
        
        interact(self._sel_colorrange_ , crange= widgets.IntRangeSlider(value=[self.cmin, self.cmax], min=-6000, max=0, step=50,
                                                 description='select color:', disabled=False,
                                                 continuous_update=False,  orientation='horizontal',
                                                 readout=True, readout_format='d', layout=widgets.Layout(width='50%')))
        
        #interact(self._sel_showtxt_    , showtx= widgets.Checkbox(value=self.showtxt, description='show txt:',
                                                 #disabled=False, indent=False))
        
        interact(self._sel_centersel_  , csel  = widgets.Checkbox(value=self.centersel, description='center selection:',
                                                                   disabled=False, indent=False))
        
        #_____________________________________________________________________________________________
        # make interactive mpl connection for mouse and keybouad press 
        self.cid_m = self.fig.canvas.mpl_connect('button_press_event',self._mousepress_)
        self.cid_k = self.fig.canvas.mpl_connect('key_press_event'   ,self._keypress_  )
        
    #_________________________________________________________________________________________________
    # define mouse press events 
    def _mousepress_(self, event):
        self.xm, self.ym, self.bm = event.xdata, event.ydata, event.button
        # self.hitxt.set_text('xm={:f}, ym={:f}, bm={:d},'.format(self.xm, self.ym, self.bm))

        # left mouse button
        if self.bm==1 : 
            # make selection circle green
            self.hchos.set_edgecolor([0.1,1,0.5])
            
            # search closet scatter points with respect to mous position
            self.idx_c = np.argmin( np.sqrt((self.xs-self.xm)**2 + (self.ys-self.ym)**2) )
            
            # move around selection circle
            self.hchos.set_offsets( [self.xs[self.idx_c], self.ys[self.idx_c]] )
            
            # center window around selection circle
            if self.centersel: self._movecenter_()
            self.hitxt.set_text('---{{,_,"> [left]')

        # right mouse button --> zoom to original
        if self.bm==3 : 
            self.hitxt.set_text('---{{,_,"> [right]')
            self._zoomorig_()
            
    #__________________________________________________________________________________________________
    # define keyboard press events 
    def _keypress_(self, event):
        self.xk, self.yk, self.bk = event.xdata, event.ydata, event.key
        self.hitxt.set_text('xk={:f}, yk={:f}, bk={},'.format(self.xk, self.yk, str(self.bk)))
        
        # press c key --> choose new depth value
        if   self.bk=='c'    : self._keychoose_()
        
        # press e key --> exit and disconnect interactive selection
        elif self.bk=='e'    : self._disconnect_()
        
        # press . key --> choose previous box from box_list
        elif self.bk==','    : self._boxbefore_()
        
        # press - key --> choose next box from box_list
        elif self.bk=='.'    : self._boxnext_()
        
        # press + key --> zoom in 
        elif self.bk=='+'    : self._zoomin_()
        
        # press - key --> zoom in 
        elif self.bk=='-'    : self._zoomout_() 
        
        # press up key    --> move up 
        elif self.bk=='up'   : self._moveup_()
        
        # press down key  --> move down
        elif self.bk=='down' : self._movedown_()
        
        # press left key  --> move left 
        elif self.bk=='left' : self._moveleft_()
        
        # press right key --> move right
        elif self.bk=='right': self._moveright_()    
    
    #__________________________________________________________________________________________________
    # disconect interative mpl connection 
    def _keychoose_(self):
        # choose a point selection circle becomes yellow
        self.hchos.set_edgecolor('y')
        
        # set new depth value 
        self.vs_new[self.idx_c] = self.seldep
        
        # change local depth label and adapt color towards new depth value
        self.htxt[self.idx_c].set_text('{:.0f}m'.format(self.vs_new[self.idx_c]))
        self.hscat.set_array(self.vs_new)
        self.hitxt.set_text('[c]')
    
    #__________________________________________________________________________________________________
    # disconect interactive mpl connection 
    def _disconnect_(self):
        # when exit selection circle become red
        self.hchos.set_edgecolor('r')
        
        # actualize position of selection circle 
        self.hchos.set_offsets( [self.xs[self.idx_c], self.ys[self.idx_c]] )
        self.hitxt.set_text('[e]--> finish')
        
        # before exiting and disconnecting plot save changes from the actual one 
        self._savechanges_()
        
        # disconect interactive mouse, keyboard
        self.fig.canvas.mpl_disconnect(self.cid_m)
        self.fig.canvas.mpl_disconnect(self.cid_k)
        plt.show(block=False)
        return(self) 
    
    #__________________________________________________________________________________________________
    # zoom in, out, to original
    def _zoomin_(self):
        self.hitxt.set_text('[+]')
        aux_xlim, aux_ylim   = self.ax.get_xlim(), self.ax.get_ylim()
        aux_dxlim, aux_dylim = aux_xlim[1]-aux_xlim[0], aux_ylim[1]-aux_ylim[0]
        # shift xy axes limits
        self.ax.set_xlim(self.xs[self.idx_c]-aux_dxlim*self.zoom_fac, self.xs[self.idx_c]+aux_dxlim*self.zoom_fac)
        self.ax.set_ylim(self.ys[self.idx_c]-aux_dylim*self.zoom_fac, self.ys[self.idx_c]+aux_dylim*self.zoom_fac)
        
    def _zoomout_(self):
        self.hitxt.set_text('[-]')
        aux_xlim, aux_ylim   = self.ax.get_xlim(), self.ax.get_ylim()
        aux_dxlim, aux_dylim = aux_xlim[1]-aux_xlim[0], aux_ylim[1]-aux_ylim[0]
        # shift xy axes limits
        self.ax.set_xlim(aux_xlim[0]-aux_dxlim*self.zoom_fac, aux_xlim[1]+aux_dxlim*self.zoom_fac)
        self.ax.set_ylim(aux_ylim[0]-aux_dylim*self.zoom_fac, aux_ylim[1]+aux_dylim*self.zoom_fac) 
    
    def _zoomorig_(self):
        # shift xy axes limits
        self.ax.set_xlim(self.xlim_o)
        self.ax.set_ylim(self.ylim_o) 
    
    #__________________________________________________________________________________________________
    # move up, down, left, right
    def _moveup_(self):
        self.hitxt.set_text('[up]')
        aux_ylim  = self.ax.get_ylim()
        aux_dylim = aux_ylim[1]-aux_ylim[0]
        # shift y axes limits
        self.ax.set_ylim(aux_ylim[0]+aux_dylim*self.zoom_fac, aux_ylim[1]+aux_dylim*self.zoom_fac)
        
    def _movedown_(self):
        self.hitxt.set_text('[down]')
        aux_ylim  = self.ax.get_ylim()
        aux_dylim = aux_ylim[1]-aux_ylim[0]
        # shift y axes limits
        self.ax.set_ylim(aux_ylim[0]-aux_dylim*self.zoom_fac, aux_ylim[1]-aux_dylim*self.zoom_fac)
    
    def _moveleft_(self):
        self.hitxt.set_text('[left]')
        aux_xlim  = self.ax.get_xlim()
        aux_dxlim = aux_xlim[1]-aux_xlim[0]
        # shift x axes limits
        self.ax.set_xlim(aux_xlim[0]-aux_dxlim*self.zoom_fac, aux_xlim[1]-aux_dxlim*self.zoom_fac)
        
    def _moveright_(self):
        self.hitxt.set_text('[right]')
        aux_xlim  = self.ax.get_xlim()
        aux_dxlim = aux_xlim[1]-aux_xlim[0]
        # shift x axes limits
        self.ax.set_xlim(aux_xlim[0]+aux_dxlim*self.zoom_fac, aux_xlim[1]+aux_dxlim*self.zoom_fac)
    
    def _movecenter_(self):
        aux_xlim, aux_ylim   = self.ax.get_xlim(), self.ax.get_ylim()
        aux_dxlim, aux_dylim = aux_xlim[1]-aux_xlim[0], aux_ylim[1]-aux_ylim[0]
        # shift xy axes limits
        self.ax.set_xlim(self.xs[self.idx_c]-aux_dxlim*0.5, self.xs[self.idx_c]+aux_dxlim*0.5)
        self.ax.set_ylim(self.ys[self.idx_c]-aux_dylim*0.5, self.ys[self.idx_c]+aux_dylim*0.5)
        
    #___________________________________________________________________________________________________
    # update scatter points size from slider --> make points bigger or smaller
    def _sel_scattersize_(self, ssize):
        self.hscat.set_sizes(self.xs*0+ssize)
        self.hchos.set_sizes([ssize])
    
    # update selected depth values from slider 
    def _sel_depth_(self, depth):
        self.seldep = depth
        
    # update selected values from checkbox if depth labels are shown or not
    def _sel_showtxt_(self, showtx):
        self.showtxt = showtx    
        for itxt in self.htxt:
            if self.showtxt: itxt.set_alpha(1.0)
            else:            itxt.set_alpha(0.0)
            
    # update selected values from checkbox if selection circle should be 
    # always centered
    def _sel_centersel_(self, csel):
        self.centersel = csel
    
    # update colorange from slider
    def _sel_colorrange_(self, crange):
        # getting new cmin cmax values from slider
        self.cmin, self.cmax = crange[0], crange[1]
        
        # compute new colorstep increment, depending on minimum number of 
        # colors
        csteplist  = np.array([5, 10, 20, 25, 50, 100, 200, 250, 500 ])
        cidx       = np.argmin( np.abs( np.abs(self.cmax-self.cmin)/self.cnum - csteplist) )
        self.cstep = csteplist[cidx]
        self.clevel= np.arange(self.cmin, self.cmax+1, self.cstep)
        self.cmap  = plt.get_cmap(self.cname, self.clevel.size-1)
        
        # update colors of scatter plot and colorbar ticks and labels
        if self.hscat is not None:
            self.hscat.set_cmap(self.cmap)
            self.hscat.set_clim(vmin=self.clevel[0], vmax=self.clevel[-1])
            self.cbar.set_ticks(self.clevel, update_ticks=False)
            self.cbar.update_ticks()
        
    #___________________________________________________________________________________________________
    # go to next box in box_list and update entire scatter plot
    def _boxnext_(self):
        self.hitxt.set_text('[.]')
        
        # before going to next plot save changes from the actual one 
        self._savechanges_()
        
        # going to next plot 
        old_idx_box = self.idx_box
        self.idx_box = np.min([self.idx_box+1, len(self.box_list)])
        if self.idx_box != old_idx_box:
            # update entire plot to new box
            self._update_scatterpts_()
            self._update_scatterplot_()
            self._update_scatteraxes_()
            self._update_scattertxt_()
            self._update_scattertri_()
            self._update_scattertitle_()
            
            # adapt also colorbar when changing box area
            self.cmin, self.cmax = self.vs.min(), self.vs.max()
            self.cmin = round(self.cmin,-np.int(np.fix(np.log10(np.abs(self.cmin))-1)))
            self.cmax = round(self.cmax,-np.int(np.fix(np.log10(np.abs(self.cmax))-1)))
            self._sel_colorrange_([self.cmin,self.cmax])
            
    # go to previous box in box_list and update entire scatter plot
    def _boxbefore_(self):
        self.hitxt.set_text('[,]')
        
        # before going to previous plot save changes from the actual one 
        self._savechanges_()
        
        # goingto previous plot 
        old_idx_box = self.idx_box
        self.idx_box = np.max([self.idx_box-1, 0])
        if self.idx_box != old_idx_box:
            # update entire plot to new box
            self._update_scatterpts_()
            self._update_scatterplot_()
            self._update_scatteraxes_()
            self._update_scattertxt_()
            self._update_scattertri_()
            self._update_scattertitle_()
            
            # adapt also colorbar when changing box area
            self.cmin, self.cmax = self.vs.min(), self.vs.max()
            self.cmin = round(self.cmin,-np.int(np.fix(np.log10(np.abs(self.cmin))-1)))
            self.cmax = round(self.cmax,-np.int(np.fix(np.log10(np.abs(self.cmax))-1)))
            self._sel_colorrange_([self.cmin,self.cmax])
            
    #___________________________________________________________________________________________________
    # update position and data of scatter points according to selected box
    def _update_scatterpts_(self):
        # set indices mask for vertices and element centroids in box 
        # create first round of scatter points from box_list with index 0 
        idxe = do_boxmask(self.mesh, self.box_list[self.idx_box][0], do_elem=True)
        if not self.do_elem: 
            idxn      = do_boxmask(self.mesh, self.box_list[self.idx_box][0], do_elem=False)
            self.xs   = self.mesh.n_x[idxn]
            self.ys   = self.mesh.n_y[idxn]
            self.vs   = self.data[idxn]
            self.idxd = idxn
        else:     
            self.xs   = self.mesh.n_x[self.mesh.e_i[idxe,:]].sum(axis=1)/3.0
            self.ys   = self.mesh.n_y[self.mesh.e_i[idxe,:]].sum(axis=1)/3.0
            self.vs   = self.data[idxe]
            self.idxd = idxe
        self.vs_new = self.vs.copy()    
        if self.do_tri: self.tri = Triangulation(self.mesh.n_x, self.mesh.n_y,self.mesh.e_i[idxe,:])
    
    # update scatter plot itself
    def _update_scatterplot_(self):
        self.hscat.set_offsets(np.vstack((self.xs, self.ys)).transpose())
        self.hscat.set_array(self.vs)
    
    # update axes of scatter plot according to box
    def _update_scatteraxes_(self):
        self.ax.set_xlim(self.box_list[self.idx_box][0][0], self.box_list[self.idx_box][0][1])
        self.ax.set_ylim(self.box_list[self.idx_box][0][2], self.box_list[self.idx_box][0][3])
        self.xlim_o,  self.ylim_o  = self.ax.get_xlim(), self.ax.get_ylim()
        self.dxlim_o, self.dylim_o = self.xlim_o[1] - self.xlim_o[0], self.ylim_o[1] - self.ylim_o[0]
    
    # update text for data points    
    def _update_scattertxt_(self):
        # 1st. remove text if it exist
        if self.htxt is not None:
            for htxt in self.htxt:
                htxt.remove()
        
        # 2nd. redo depth text labels
        self.htxt=list()
        for ii in range(0,len(self.xs)):
            # do here annotation instead of text because anotation can become
            # transparent --> i can hide them with text checkbox
            htxt = self.ax.annotate('{:.0f}m'.format(self.vs[ii]), [self.xs[ii], self.ys[ii]], 
                                    xycoords='data' , va="center", ha="center", fontsize=6)
            self.htxt.append(htxt)
    
    # update title of scatter plot
    def _update_scattertitle_(self):
        self.ax.set_title(self.box_list[self.idx_box][1]+' sill depth')  
    
    # update triangular mesh in scatter plot
    def _update_scattertri_(self):
        if self.do_tri:
            if self.htri is not None: 
               for line in self.htri :
                line.remove()
            self.htri=self.ax.triplot(self.tri, linewidth=0.25, color= 'k', zorder=0)

    #___________________________________________________________________________________________________
    # save all depth changes that are made to self.data
    def _savechanges_(self):
        if any(self.vs-self.vs_new != 0.0):
            self.data[self.idxd]=self.vs_new
