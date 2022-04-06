# jjmodel

A semi-analytic Just-Jahreiß (JJ) model of the Milky Way disk. The model 
has been recently calibrated against the stellar populations selected from the Gaia DR2 in the Solar neighbourhood, and is now generalised for a wide range of Galactocentric distances, R > 4 kpc (avoiding bulge region). The generalised JJ model is complemented by a set of Padova and MIST isochrones, such that users can synthesise stellar populations and select the observable samples on the colour-magnitude diagram, as well as apply custom cuts on such quantities as ages, metallicities, logg or effective temperatures. 

An extended documentation will be added soon. 

## Installation

```
pip install git+https://github.com/askenja/jjmodel.git@main
```
The jjmodel package requires python 3.8. Also, it depends on the following packages: [Numpy (v1.18.1)](https://numpy.org/), [SciPy (v1.5.0)](http://www.scipy.org/), [Astropy (v4.2)](http://www.astropy.org/), [matplotlib (v3.2.1)](http://matplotlib.sourceforge.net/) and [fast_histogram (v0.9)](https://github.com/astrofrog/fast-histogram). 
All of them can be installed using pip.


### Installation without admin rights:
You can install *jjmodel* into a folder where you have write access:

```
pip install --install-option='--prefix=~/extra_package/' git+https://github.com/askenja/jjmodel.git@main
```

Then you have to add the `site-packages/` folder which will be one of the newly created subfolders in `extra_package/` into the ```PYTHONPATH``` variable, e.g.:

```
export PYTHONPATH=~/extra_package/lib/python3.8/site-packages/:$PYTHONPATH
```

If you want this to be permanent, you can add the last line to your `.bashrc`.


## Authors
- Kseniia Sysoliatina (ARI, Sysoliatina@uni-heidelberg.de)

## Collaborators
- Andreas Just (ARI)



## Getting started

### Introduction to the model

The JJ model describes an axisymmetric (no spiral arms) Galactic disk in a steady state. 
The model consists of the exponential thin, thick and two gaseous disks, as well as spherical or flattened halo and dark matter (DM) components. The thin disk, that is assumed to evolve parallely with the thick disk, is described by input functions given in analytic form: the initial mass function (IMF), star formation rate (SFR), age-velocity dispersion relation (AVR), and age-metallicity relation (AMR). At each radial zone R, the JJ model calculates a self-consistent pair of the vertical density profile and gravitation potential by iterative solving of the Poisson-Boltzmann equation. The AVR parameters are adapted at each R in such a way, that the overall thickness of the thin disk is kept constant. However, the flaring option is also available in the model. Motivated by our Gaia DR2-based finding of the recent star formation (SF) bursts in the thin-disk SFR, we allow the thin-disk SFR to have any number of Gaussian peaks on top of its monotonously declining continuum. It is also possible to decouple the kinematics of the stellar populations associated with the SF-excess in the peaks from the kinematics of the underlying disk populations, as given by the AVR. 


### Building the model

To start using the code, you need to find the package folder on your computer. It should be something like /anaconda2/lib/python3.8/site-packages/jjmodel. Most of the model parameters are listed in the file parameters. In case if any Gaussian peaks need to be added to the thin-disk SFR, peaks-related parameters should be listed in the file sfrd_peaks_parameters. All output data, including figures, will be stored in jjmodel/output directory. 

The jupyter [tutorial](https://github.com/askenja/jjmodel/blob/main/tutorials/model_disk.ipynb) explains how to run the code when the parameter files are filled, and also shows plotting options for the output. 

### Working with isochrones

Isochrone loader will be here. 

## Acknowledgements
Please cite the [paper_in_preparation](link) when using the jjmodel code.
Optionally cite some of the following papers:

[Just and Jahreiß (2010)](https://ui.adsabs.harvard.edu/abs/2010MNRAS.402..461J/abstract), Towards a fully consistent Milky Way disc model - I. The local model based on kinematic and photometric data

[Just et al. (2011)](https://ui.adsabs.harvard.edu/abs/2011MNRAS.411.2586J/abstract), Towards a fully consistent Milky Way disc model - II. The local disc model and SDSS data of the NGP region

[Rybizki and Just (2015)](https://ui.adsabs.harvard.edu/abs/2015MNRAS.447.3880R/abstract), Towards a fully consistent Milky Way disc model - III. Constraining the initial mass function

[Sysoliatina et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018A%26A...620A..71S/abstract), Local disc model in view of Gaia DR1 and RAVE data

[Sysoliatina and Just (2021)](https://ui.adsabs.harvard.edu/abs/2021arXiv210209311S/abstract), Towards a fully consistent Milky Way disk model -- IV. The impact of Gaia DR2 and APOGEE


