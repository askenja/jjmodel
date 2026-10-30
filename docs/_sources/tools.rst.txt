.. jjmodel documentation master file, created by
   sphinx-quickstart on Thu Mar 24 16:00:37 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



Useful tools 
======================================

Reading parameters 
------------------------------

.. autofunction:: jjmodel.tools.read_parameters
   
.. autofunction:: jjmodel.tools.resave_parameters
   

Smoothing distributions
------------------------

.. autofunction:: jjmodel.tools.gauss_weights
   
.. autofunction:: jjmodel.tools.convolve1d_gauss
 
.. autofunction:: jjmodel.tools.convolve2d_gauss
 
.. autofunction:: jjmodel.tools.cumhist_smooth_savgol
 

Interpolation tools
--------------------

.. autoclass:: jjmodel.tools.ConvertAxes
   :members:
 
.. autofunction:: jjmodel.tools.rebin_histogram
 

Retrieving status
-------------------

.. autoclass:: jjmodel.tools.Timer
   :members:

.. autoclass:: jjmodel.tools.LogFiles
   :members:


Other 
----------------

.. autofunction:: jjmodel.tools.reduce_table
 

  
   


