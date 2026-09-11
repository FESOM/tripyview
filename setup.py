#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The setup script.'''

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    #___________________________________________________________________________
    # numerical libraries 
    'numpy',
    'xarray',
    'dask',
    'distributed',
    'pandas',
    'geopandas',
    'scipy',
    'numba',
    #___________________________________________________________________________
    # jupyter stuff 
    'ipython',
    'jupyter',
    'jupyterlab<4.0',
    'jupyter_lsp',
    #___________________________________________________________________________
    # plotting 
    'matplotlib',
    'cartopy',
    'cmocean',
    'bokeh!=3.0.*,>=2.4.2',
    'shapely',
    #___________________________________________________________________________
    # file reading 
    'netCDF4',
    'libnetcdf',
    'h5netcdf',
    'hdf5plugin', 
    "pickle5; python_version<'3.9'",
    'joblib',
    #___________________________________________________________________________
    # ocean properties
    'seawater',
    'gsw',
    #___________________________________________________________________________
    # tripyrun functionaallity
    'papermill', 
    'black',
    'jinja2',
    'pyyaml',
    #___________________________________________________________________________
    # 3d stuff 
    'pyvista[all]', #,jupyter,trame]',
    'vtk',
    'ipyvtklink',
    'imageio[ffmpeg]', 
    'ipympl',
    'ffmpeg-python',
    #___________________________________________________________________________
    #'pyfesom2',
    #'pyresample',
    'pytest',
    
]

setup_requirements = ['pytest-runner']

test_requirements = ['pytest']

setup(
    author='FESOM team',
    author_email='Patrick.Scholz@awi.de',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    entry_points={
        'console_scripts': [
            'tripyrun=tripyview.sub_tripyrun:tripyrun',  # command=package.module:function
        ]
    },
    description='FESOM2 tools',
    install_requires=requirements,
    license='MIT license',
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='tripyview',
    name='tripyview',
    # the notebook/html templates live at the repo root, map them into the
    # installed package so that non-editable installs (e.g. the Dockerfile's
    # `pip install .`) ship them too
    packages=['tripyview', 'tripyview.templates_notebooks', 'tripyview.templates_html'],
    package_dir={'tripyview'                    : 'tripyview',
                 'tripyview.templates_notebooks': 'templates_notebooks',
                 'tripyview.templates_html'     : 'templates_html'},
    # only shapefile components one directory level deep (shapefiles/<category>/<name>.*),
    # which matches the repository layout and keeps large local-only data out of builds
    package_data={'tripyview'                    : ['shapefiles/*.geojson',
                                                    'shapefiles/*/*.shp', 'shapefiles/*/*.shx',
                                                    'shapefiles/*/*.dbf', 'shapefiles/*/*.prj',
                                                    'shapefiles/*/*.cpg', 'shapefiles/*/*.cst',
                                                    'backgrounds/*'],
                  'tripyview.templates_notebooks': ['template_*.ipynb'],
                  'tripyview.templates_html'     : ['*.html', '*.png']},
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/patrickscholz/tripyview',
    version='0.3.0',
    zip_safe=False,
)




# conda install -c conda-forge cartopy cmocean dask ipython joblib jupyter matplotlib pickle5 netCDF4 numba numpy pandas geopandas scipy seawater shapely  xarray  pyfesom2 pyresample pytest papermill  jinja2  pyyaml pyvista vtk ipyvtklink
