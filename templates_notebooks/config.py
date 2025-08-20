############################
# Module loading         #
############################

#Misc
import os
import sys
import warnings
from tqdm import tqdm
import logging
import joblib
import dask
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import random as rd
import time
import copy as cp
import subprocess


#Data access and structures
import pyfesom2 as pf
import xarray as xr
from cdo import *   
cdo = Cdo(cdo=os.path.join(sys.prefix, 'bin')+'/cdo')
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from collections import OrderedDict
import csv
from bg_routines.update_status import update_status

#Plotting
import math as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.ticker import Locator
from matplotlib import ticker
from matplotlib import cm
import seaborn as sns
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from mpl_toolkits.basemap import Basemap
import cmocean as cmo
from cmocean import cm as cmof
import matplotlib.pylab as pylab
import matplotlib.patches as Polygon
import matplotlib.ticker as mticker


#Science
import math
from math import sqrt
from sklearn.metrics import mean_squared_error
from eofs.standard import Eof
from eofs.examples import example_data_path
import shapely
from scipy import signal
from scipy.stats import linregress
from scipy.spatial import cKDTree
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator


#Fesom related routines
from bg_routines.set_inputarray  import *
from bg_routines.sub_fesom_mesh  import * 
from bg_routines.sub_fesom_data  import * 
from bg_routines.sub_fesom_moc   import *
from bg_routines.colormap_c2c    import *


############################
# Simulation Configuration #
############################

#Name of model release
model_version  = 'AWI-CM-v3.4'
oasis_oifs_grid_name = 'A096'

#Spinup
spinup_path    = '/work/ab0246/a270092/runtime/awicm3-v3.3/SPIN/outdata/'
spinup_name    = model_version+'_spinup'
spinup_start   = 1350
spinup_end     = 1849

#Preindustrial Control
pi_ctrl_path   = '/work/ab0246/a270092/runtime/awicm3-v3.3/PI/outdata/'
pi_ctrl_name   = model_version+'_pi-control'
pi_ctrl_start  = 1850
pi_ctrl_end    = 2014

#Historic
historic_path  = '/work/ab0246/a270092/runtime/awicm3-v3.3/HIST/outdata/'
historic_name  = model_version+'_historic'
historic_start = 1850
historic_end   = 2014

#Modelled time
#model_path  = '/work/ab0995/a270275/experiments/awicm3test011/outdata'
#model_name  = model_version + '_simulated'
#model_start = 1990
#model_end   = 1991

#Misc
reanalysis             = 'ERA5'
remap_resolution       = '512x256'
dpi                    = 300
historic_last25y_start = historic_end-25
historic_last25y_end   = historic_end
status_csv             = "log/status.csv"

#Mesh
mesh_name      = 'CORE2'
grid_name      = 'TCo95'
meshpath       = '/work/ab0246/a270092/input/fesom2/core2/'
mesh_file      = 'mesh.nc'
griddes_file   = 'mesh.nc'
abg            = [0, 0, 0]
reference_path = '/work/ab0246/a270092/postprocessing/climatologies/fdiag/'
reference_name = 'clim'
reference_years= 1958
accumulation_period = 1

observation_path = '/work/ab0246/a270092/obs/'

tool_path      = os.getcwd()
out_path       = tool_path+'/output/'+model_version+'/'
os.makedirs(out_path, exist_ok=True)
mesh = pf.load_mesh(meshpath)
data = xr.open_dataset(meshpath+'/fesom.mesh.diag.nc')


