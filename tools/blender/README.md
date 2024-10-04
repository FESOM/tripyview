 
How to install tripyview in blender: 

1) download and unpack Blender src code archive e.g. blender-4.2.2-linux-x64.tar.xz

2) unzip folder

3) go to directory /blender-4.2.2-linux-x64/4.2/python/bin/ and execute

/blender-4.2.2-linux-x64/4.2/python/bin/python3.11 -m pip -e install /path_to_tripyview/

--> This should install tripyview in the python version that blender is using

4) test if tripyview installation works:

/blender-4.2.2-linux-x64/4.2/python/bin/python3.11 -c "import tripyview"

6) Open Blender 

7) go to the Scripting tab and load the script do_blenderplanet_fesom2.py 
make sure all the path are correct

8) run that script!!!

9) There might occure the error message where the blender interface colides with pyvista: 

Python: Traceback (most recent call last):
  File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 940, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/home/pscholz/Python/tripyview/tripyview/__init__.py", line 25, in <module>
    from .sub_3dsphere          import *
  File "/home/pscholz/Python/tripyview/tripyview/sub_3dsphere.py", line 3, in <module>
    import vtk
  File "/home/pscholz/Software/blender-4.2.2-linux-x64/4.2/python/lib/python3.11/site-packages/vtk.py", line 47, in <module>
    from vtkmodules.vtkRenderingMatplotlib import *
  ImportError: /home/pscholz/Software/blender-4.2.2-linux-x64/4.2/python/lib/python3.11/site-packages/vtkmodules/libvtkPythonInterpreter-9.3.so: undefined symbol: Py_RunMain

simply open the file /home/pscholz/Software/blender-4.2.2-linux-x64/4.2/python/lib/python3.11/site-packages/vtk.py
and comment line 47!
