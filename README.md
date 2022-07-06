# <img src="tripyview/tripyview_icon_NH.png" width=15% align="left" > ----------------- Tripyview ----------------- <img src="tripyview/tripyview_icon_SH.png" width=15% align="right" >
<br />
<br />
<br />
<br />
Triangular plotting and diagnostics for FESOM2 + command line utility similar to 
fdiag using plotting suite of Patrick.Scholz@awi.de (former fesom2/view_pscholz folder). 
Data reading got updated to xarray, plotting got updated to cartopy, actually works well for python3.7
<br />

## Installation

In moment only possible by source files using pip:

1st. clone respository: 
```bash
git clone https://github.com/patrickscholz/tripyview
```

2nd. install using pip: 
```bash
cd tripyview
pip install -e .
```

## Different diagnostics
### see ./template_notebook/:

#### General plotting: 
* `hslice`       - plot horizontal slices for ocean variable (compute depth and time averages)
* `hslice_np`    - plot horizontal slices for ocean variable north polar stereographic projection
* `hslice_sp`    - plot horizontal slices for ocean variable south polar stereographic projection
* `hslice_clim`  - plot horizontal slices of temperature, salinity and density climatology (compute depth and time averages)
* `hslice_clim_np` - plot horizontal slices of temperature, salinity and density climatology in north polar stereographic projection
* `hslice_clim_sp` - plot horizontal slices of temperature, salinity and density climatology in south polar stereographic projection

* `hovm`         - plot hovmöller diagramms for 3d ocean varaible over depth and time averaged over
specific region that defined by shapefile (see. tripyview/shapefiles/...)
* `hovm_clim`    - plot hovmöller diagramms for  temperature, salinity and density climatology 
over depth and time averaged over specific region that defined by shapefile (see. tripyview/shapefiles/...)

* `transect`         - plot point wise transects of 3d ocean varíables
* `transect_clim`    - plot point wise transects of 3d temperature, salinity and density climatology
* `transect_transp`  - plot volume transport through arbitrary transects (model accurate transport)
* `transect_transp_t`- plot time-series of volume transport through arbitrary transects (model accurate transport)
* `transect_zmean`   - plot zonally averaged transects of 3d ocean varíables
* `transect_zmean_clim`- plot zonally averaged transects of 3d temperature, salinity and density climatology

* `vprofile`     - plot lines for averaged vertical profile of 3d ocean varíables
* `vprofile_clim`- plot lines for averaged vertical profile of 3d temperature, salinity and density climatology

* `var_t`        - plots time series of of globally averaged 2d and 3d variables 

#### Transports/Streamfunctions: 
* `zmoc`         - plot meridional overturning circulation from vertical velocity (based on latitudinal binning), Globally, 
for the: Atlantic(amoc), Atlantic+Arctic(aamoc), Pacific(pmoc), Pacific+Indian Ocean(ipmoc) and Indian Ocean(imoc)

<p align="left" width="100%">
  <img width=32% src="tripyview/tripyview_zAMOC.png">
  <img width=32% src="tripyview/tripyview_zIPMOC.png">
  <img width=32% src="tripyview/tripyview_zGMOC.png">
</p>

* `zmoc_t`       - plot timeseries of atlantic meridional overturning circulation at specific latitudes 
(default: which_lats: [26, 40, 'max'])

* `dmoc`         - plot meridional overturning circulation in density  coordinates using FESOM2 density MOC diagnostics (graditues goes to **Dmitry.Sidorenko@awi.de**), Globally, 
for the: Atlantic(amoc), Atlantic+Arctic(aamoc), Pacific(pmoc), Pacific+Indian Ocean(ipmoc) and Indian Ocean(imoc)
* `dmoc_srf`     - plot surface transformation of meridional overturning circulation in density coordinates
* `dmoc_inner`   - plot inner transformation of meridional overturning circulation in density coordinates

<p align="left" width="100%">
  <img width=32% src="tripyview/tripyview_dAMOC.png">
  <img width=32% src="tripyview/tripyview_dAMOC_srf.png">
  <img width=32% src="tripyview/tripyview_dAMOC_inner.png">
</p>

* `dmoc_z`, `dmoc_srf_z`, `dmoc_inner_z` - plot meridional overturning circulation in density coordinates back into z-coordinate space 

<p align="left" width="100%">
  <img width=32% src="tripyview/tripyview_dAMOC_z.png">
  <img width=32% src="tripyview/tripyview_dIPMOC_z.png">
  <img width=32% src="tripyview/tripyview_dGMOC_z.png">
</p>

* `dmoc_wdiap`   - plot diapycnal velocity for specific water-class

* `dmoc_srfcbflx`- plot surface buoyancy flux transformation for specific water-class

* `dmoc_t`       - plot timeseries of atlantic meridional overturning circulation in density coordinates at specific latitudes 
(default: which_lats: [26, 55, 'max'])

* `hbarstreamf`  - plot horizontal barotropic streamfunction 

* `ghflx`        - plots global meridional heat flux computed from surface heatfluxes fh
* `mhflx`        - plots global meridional heat flux computed from the the turbulent fluxes temp*v, temp*u

#### 3D plotting: 
* `3dsphere`     - plot 3d model of FESOM grid with variables using pyvista interface
<p align="center" width="100%">
  <img width=75% src="tripyview/tripyview_icon_NH.png">
</p>

* `make 3D animations`
<p align="center" width="100%">
<a href="https://www.youtube.com/watch?v=Gokzel60GFk" title="Movie core2 bathymetry"><img width=45% src="https://img.youtube.com/vi/Gokzel60GFk/hqdefault.jpg" alt="Movie core2 bathymetry"/></a> <a href="https://www.youtube.com/watch?v=_8onDgsbLiQ" title="Movie core2 resolution"><img width=45% src="https://img.youtube.com/vi/_8onDgsbLiQ/hqdefault.jpg" alt="Movie core2 resolution"/></a>
</p>

### command line batch diagnostics:
`diagrun <workflowfile.yml>` - command line diagnostics similar to fdiag (fdiag, served as a 
template, graditues go to Nikolay.Koldunov@awi.de), creates html files with linked plots

## Credits:
If you use the package please give the package or the author some credit.

