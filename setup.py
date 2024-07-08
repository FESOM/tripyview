#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The setup script.'''

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'cartopy',
    'bokeh!=3.0.*,>=2.4.2',
    'cmocean',
    'dask',
    'distributed',
    'ipython',
    'joblib',
    'jupyter',
    'jupyterlab<4.0',
    'jupyter_lsp',
    'matplotlib',
    "pickle5; python_version<'3.9'",
    'libnetcdf',
    'hdf5plugin', 
    'netCDF4',
    'numba',
    'numpy',
    'pandas',
    'geopandas',
    'scipy',
    'seawater',
    'gsw',
    'shapely',
    'xarray',
    'pyfesom2',
    'pyresample',
    'pytest',
    'papermill', 
    'black',
    'jinja2',
    'pyyaml',
    'pyvista[all]', #,jupyter,trame]',
    'vtk',
    'ipyvtklink',
    'imageio[ffmpeg]', 
    'ipympl',
    'ffmpeg-python',
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
    packages=['tripyview'],
    package_dir={'tripyview': 'tripyview'},
    #package_data={'': ['*.shp',]},
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/patrickscholz/tripyview',
    version='0.3.0',
    zip_safe=False,
)




# conda install -c conda-forge cartopy cmocean dask ipython joblib jupyter matplotlib pickle5 netCDF4 numba numpy pandas geopandas scipy seawater shapely  xarray  pyfesom2 pyresample pytest papermill  jinja2  pyyaml pyvista vtk ipyvtklink
