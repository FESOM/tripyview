# Overview of all colormaps that can be used in tripyview via the cname argument
# of tpv.colormap_c2c (and so of all tripyview plotting routines):
#
#   - the hard-coded tripyview colormaps          cname='blue2red'
#   - the cmocean colormaps                       cname='cmocean.<name>'
#   - the matplotlib colormaps                    cname='matplotlib.<name>'
#
# Every colormap is rendered through tpv.colormap_c2c itself, so the figure
# shows them exactly as tripyview uses them (cmocean/matplotlib maps are sampled
# by colormap_c2c at 11 colours and interpolated in between). The colormaps are
# grouped into linear (sequential, incl. rainbow-like and cyclic) and diverging
# ones, the label next to every bar is the cname string to use. Append '_i' to
# any cname to invert the colormap.
#
# Layout, the rows flow from column to column, DIVERGING starts a new column:
#
#   +------------+------------+------------+------------+
#   | LINEAR     | (LINEAR)   | (LINEAR)   | DIVERGING  |
#   |  tripyview |  ...       |  ...       |  tripyview |
#   |  cmocean   |            |            |  cmocean   |
#   |  matplotl. |            |            |  matplotl. |
#   +------------+------------+------------+------------+
#
# Run (no compute node needed):  python plot_colormaps.py
# Output: plot_colormaps.png next to this script
import os
os.environ.setdefault('TRIPYVIEW_WITHOUT_VTK', '1')   # no pyvista needed here
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tripyview as tpv

#_______________________________________________________________________________
# hard-coded tripyview colormaps, (name, flipped twin or None); classified by
# looking at their colour definitions in sub_colormap.colormap_c2c
tpv_linear    = [('heat', None), ('cool', None), ('wvt', None), ('wvt1', None), ('wvt2', None),
                 ('wvt3', None), ('gnuplot', None), ('arc', None), ('wbgyr', 'rygbw'), ('ars', None),
                 ('rainbow', None), ('jet', None), ('odv', None), ('hsv', None)]
tpv_diverging = [('blue2red', 'red2blue'), ('dblue2dred', 'dred2dblue'), ('green2orange', 'orange2green'),
                 ('purple2green', 'green2purple'), ('gold2magenta', 'magenta2gold'),
                 ('teal2magenta', 'magenta2teal'), ('cool2heat', None), ('curl', None), ('grads', None),
                 ('jetw', None), ('odvw', None), ('seaice', None), ('precip', None), ('drought', None)]

# cmocean colormaps, classification as in the cmocean documentation (phase is
# cyclic, topo is the land/sea map)
cmo_linear    = ['thermal', 'haline', 'solar', 'ice', 'gray', 'oxy', 'deep', 'dense', 'algae',
                 'matter', 'turbid', 'speed', 'amp', 'tempo', 'rain', 'phase']
cmo_cyclic    = ['phase']
cmo_diverging = ['balance', 'delta', 'curl', 'diff', 'tarn', 'topo']

# matplotlib colormaps, categories as in the matplotlib colormap reference.
# The qualitative ones (tab10, Set1, ...) are left out: colormap_c2c samples
# every map at 11 points and interpolates, which makes no sense for them.
mpl_linear    = [('perceptually uniform', ['viridis', 'plasma', 'inferno', 'magma', 'cividis']),
                 ('sequential',           ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                                           'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu',
                                           'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
                 ('sequential (2)',       ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                                           'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                                           'hot', 'afmhot', 'gist_heat', 'copper']),
                 ('cyclic',               ['twilight', 'twilight_shifted', 'hsv']),
                 ('miscellaneous',        ['flag', 'prism', 'ocean', 'gist_earth', 'terrain',
                                           'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix',
                                           'brg', 'gist_rainbow', 'rainbow', 'jet', 'turbo',
                                           'nipy_spectral', 'gist_ncar'])]
mpl_diverging = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral',
                 'coolwarm', 'bwr', 'seismic']

#_______________________________________________________________________________
# one flat list of rows: ('group', title) bold group title, ('header', text)
# source sub-title, ('cmap', label, cname) one colour bar
def tpv_rows(tpv_list):
    out = [('header', 'tripyview, hard-coded')]
    for name, twin in tpv_list:
        label = "'{}'".format(name) if twin is None else "'{}' / '{}'".format(name, twin)
        out.append(('cmap', label, name))
    return out

def lib_rows(title, prefix, names, cyclic=()):
    out = [('header', title)]
    for name in names:
        cname = '{}.{}'.format(prefix, name)
        out.append(('cmap', "'{}'".format(cname) + (' (cyclic)' if name in cyclic else ''), cname))
    return out

linear    = [('group', 'LINEAR')] + tpv_rows(tpv_linear) \
          + lib_rows('cmocean', 'cmocean', cmo_linear, cyclic=cmo_cyclic)
for title, names in mpl_linear:
    linear += lib_rows('matplotlib, ' + title, 'matplotlib', names)
diverging = [('group', 'DIVERGING')] + tpv_rows(tpv_diverging) \
          + lib_rows('cmocean', 'cmocean', cmo_diverging) \
          + lib_rows('matplotlib', 'matplotlib', mpl_diverging)

#_______________________________________________________________________________
# flow the rows into columns of at most nrow_max rows, DIVERGING starts a new
# column, a sub-title is never left alone at the bottom of a column
def flow(rows, nrow_max):
    cols, col = list(), list()
    for ii, row in enumerate(rows):
        orphan = row[0] in ('group', 'header') and len(col) >= nrow_max-1
        if len(col) >= nrow_max or orphan:
            cols.append(col); col = list()
        col.append(row)
    cols.append(col)
    return cols

nrow_max = int(np.ceil(len(diverging)/1.0))           # diverging fits one column
columns  = flow(linear, nrow_max) + flow(diverging, nrow_max)
nrow     = max(len(c) for c in columns)
ncol     = len(columns)

#_______________________________________________________________________________
# draw: one thin axes per colormap, symmetric range so the full map is shown
row_h    = 0.30                                        # inch per row
col_w    = 1.0/ncol                                    # figure fraction per column
fig      = plt.figure(figsize=(6.0*ncol, (nrow+1)*row_h))
gradient = np.linspace(-1, 1, 400)[None, :]
for ci, rows in enumerate(columns):
    x_bar = ci*col_w + 0.47*col_w                      # left edge of the colour bars
    for ri, row in enumerate(rows):
        y = 1 - (ri+0.6)/(nrow+1)
        if row[0] == 'group':
            fig.text(ci*col_w+0.02*col_w, y, row[1], fontsize=14, fontweight='bold', va='center')
        elif row[0] == 'header':
            fig.text(ci*col_w+0.02*col_w, y, row[1], fontsize=10, fontstyle='italic', color='0.35', va='center')
        else:
            _, label, cname = row
            cmap, clevel, cref = tpv.colormap_c2c(-1.0, 1.0, 0.0, 40, cname)
            ax = fig.add_axes([x_bar, y-0.35/(nrow+1), 0.50*col_w, 0.7/(nrow+1)])
            ax.imshow(gradient, aspect='auto', cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
            ax.set_axis_off()
            fig.text(x_bar-0.01*col_w, y, label, fontsize=9, ha='right', va='center', family='monospace')

fig.text(0.5, 0.0, "label = cname string; rendered with tpv.colormap_c2c(cmin=-1, cmax=1, cref=0, "
         "cnumb=40, cname); append '_i' to any cname to invert the colormap",
         ha='center', va='top', fontsize=9, color='0.4')
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_colormaps.png'),
            dpi=150, bbox_inches='tight')
print('saved')
