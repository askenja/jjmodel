.. jjmodel documentation master file, created by
   sphinx-quickstart on Thu Mar 24 16:00:37 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Getting started
======================================

Parameter files
-------------------------

First of all, you need to create and fill parameter file(s) whose content will define properties of your 
model realisation. You can understand parameter files' structure by looking at them in tutorials (see below). 
Also, you can use our `manual <https://github.com/askenja/jjmodel/tree/main/jjmodel/docs/mydoc>`_ which includes description of all JJ-model parameters. It is also stored in your local ``jjmodel/docs/mydoc`` folder in *anaconda* (to find your ``jjmodel`` folder, execute ``print(jjmodel.__file__)`` in your python compiler). 

Tutorials
-------------------------

The code **jjmodel** includes several tutorials which should help user to familiarize himself with the code 
syntax. After the installation, jupyter notebooks can be found in the folder ``jjmodel/tutorials``. 
Tutorials can be also viewed on github. 

- `Tutorial 1 <https://github.com/askenja/jjmodel/tree/main/jjmodel/tutorials/tutorial1>`_ . An example of basic usage of the code, modeling of the Solar neighbourhood. 

- `Tutorial 2 <https://github.com/askenja/jjmodel/tree/main/jjmodel/tutorials/tutorial2>`_. Radially extended JJ model (general case of Tutorial 1). 
 
- `Tutorial 3 <https://github.com/askenja/jjmodel/tree/main/jjmodel/tutorials/tutorial3>`_. More detailed modeling of the Solar neighbourhood: stars in the local volume (sphere), selection of the data-motivated sample (cut on absolute or aparent magnitudes). 

- `Tutorial 4 <https://github.com/askenja/jjmodel/tree/main/jjmodel/tutorials/tutorial4>`_. Advanced level. Modeling of the Red Clump sample across the disk and reconstruction of the age-metallicity relation based on the APOGEE data (analysis presented in Sysoliatina and Just 2022). Note that this tutorial has been developed for the earlier version of the code and may be not runnable after year 2026. The rest of tutorials are compatible with the latest code version. 

- `Tutorial 5 <https://github.com/askenja/jjmodel/tree/main/jjmodel/tutorials/tutorial5>`_. Advanced level. Prediction of the colour-magnitude diagram including white dwarfs and a custom initial-final mass relation (analysis presented in Sysoliatina and Vani in prep.). 






