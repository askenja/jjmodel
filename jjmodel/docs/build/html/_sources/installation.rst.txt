.. jjmodel documentation master file, created by
   sphinx-quickstart on Thu Mar 24 16:00:37 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Installation
===============


Standard installation 
-----------------------

To install the **jjmodel** code as *anaconda* package, do in terminal:

.. code-block:: bash

   $ pip install git+https://github.com/askenja/jjmodel.git@main
   
To install a specific version (see releases on github), use the corresponding tag instead of *main*, e.g. *v.1.0.1*. 

Note that **jjmodel** requires python 3.8 or higher. Also, it depends on the following packages: 

- `astropy (v>=4.2) <http://www.astropy.org/>`_
- `fast_histogram (v>=0.9) <https://github.com/astrofrog/fast-histogram>`_
- gdown (v=6.4.2)
- healpy
- `matplotlib (v>=3.2.1) <http://matplotlib.sourceforge.net/>`_
- `numpy (v>=1.18.1) <https://numpy.org/>`_
- pandas
- `scipy (v>=1.5.0) <http://www.scipy.org/>`_

All of these packages will be automatically installed during the **jjmodel** installation process. 


Installation without admin rights
----------------------------------

You can install **jjmodel** into a folder where you have write access. 
Open there a terminal and run the following command:

.. code-block:: bash

   $ pip install --install-option='--prefix=~/extra_package/' git+https://github.com/askenja/jjmodel.git@main

Then you have to add the ``site-packages/`` folder which will be one of the newly created subfolders in ``extra_package/`` into the PYTHONPATH variable, e.g.:

.. code-block:: bash

   $ export PYTHONPATH=~/extra_package/lib/python3.8/site-packages/:$PYTHONPATH

If you want this to be permanent, you can add the last line to your ``.bashrc`` file.


Getting isochrones
-------------------------

In order to work with mock stellar populations, the **jjmodel** code needs an isochrone grid. 

We provide the following isochrone packages compatible with the JJ model. 

I. Stellar volution up to TP-AGB phase:

- `PARSEC (Padova) <http://stev.oapd.inaf.it/cgi-bin/cmd>`_
- `MIST <https://waps.cfa.harvard.edu/MIST/interp_isos.html>`_
- `BaSTI <http://basti-iac.oa-abruzzo.inaf.it/isocs.html>`_

II. WD isochrones (generated with initial-final mass relation from Cummings et al. 2018): 

- `LPCODE <https://evolgroup.fcaglp.unlp.edu.ar/modelos.html>`_
- `BaSTI <http://basti-iac.oa-abruzzo.inaf.it/wdmodels.html>`_
- `Montréal <http://www.astro.umontreal.ca/~bergeron/CoolingModels>`_

After you have installed **jjmodel** to your *anaconda*, you can automatically download any 
of the provided isochrone sets (or all of them). Run the following code in a python script or jupyther notebook: 

.. code-block:: python 
   
   import jjmodel as jjm
   from jjmodel.iso_loader import isochrone_loader 
   
   isochrone_loader(iso_type='main',model='Padova') # or 'MIST', 'BaSTI'
   isochrone_loader(iso_type='wd',model='LPCODE') # or 'BaSTI', 'Montreal'
   
This will save and unpack all three isochrone grids to your local **jjmodel** folder, e.g. 
``/home/user/anaconda2/envs/py3/lib/python_version/site-packages/jjmodel/input/isochrones`` 
(you can use command ``jjm.__file__`` to see the path to the **jjmodel** code on your computer). 

Note that some isochonre grids are heavy (5-8 Gb), so you need to have enough space. Also, downloading process can take a while. 

After you have downloaded isochrones, the installation is complete, and you can start using the model 
(e.g. see :doc:`tutorials`). 



