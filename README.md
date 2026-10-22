
# jjmodel

A semi-analytic Just-Jahreiß (JJ) model of the Milky Way disk. The model has been calibrated against stellar populations in the Solar neighbourhood and generalised to a range of intermediate Galactocentric distances, approximately \(4 < R < 14\) kpc - excluding the bulge region and the warped disk. It is complemented by a set of alternative stellar isochrone grids - PARSEC, MIST, and BaSTI for the evolutionary stages up to TP-AGB. This allows users to synthesise stellar populations and select observable samples in the colour–magnitude diagram, as well as apply custom cuts on quantities such as age, metallicity, surface gravity, effective temperature, etc.

The generalised JJ model is described in [Sysoliatina & Just (2022)](https://ui.adsabs.harvard.edu/abs/2022A%26A...666A.130S/abstract).

## CALOMERA: Stellar population synthesis including white dwarfs

<p align="center">
  <img src="jjmodel/docs/images/CalomeraLogo.png" alt="CALOMERA logo" width="300">
</p>

**CALOMERA** (*Chemo-dynAmicaL mOdel for Milky way Evolution and stellaR populAtion synthesis*) is the updated stellar population synthesis framework built on the JJ model, extending it to include white dwarfs (WDs) in a self-consistent description of the Galactic stellar population. Named after the fast-moving tiger beetle genus *Calomera*, it reflects the framework's emphasis on computational speed and efficiency.

CALOMERA combines the JJ model's semi-analytic chemo-dynamical description of the Galactic disk with modern WD cooling sequences, metallicity-dependent progenitor lifetimes, and synthetic photometry. Its calibration jointly constrains the initial mass function, Galactic star formation history, WD initial–final mass relation using a *Gaia* DR3 sample of nearly 8 million stars within 1 kpc, including approximately 110,000 WDs.

The calibrated model reproduces star counts across colour–magnitude diagram regions to within 3% and the total number count to within 0.5%. It also enables predictions of Galactic WD distributions and luminosity functions, with an extension to the substellar regime using ATMO models to study L, T, and Y dwarfs.

CALOMERA is the WD-inclusive stellar population synthesis version of the JJ model; the underlying JJ model remains unchanged.

The new framework is presented in Sysoliatina & Vani, *The full Gaia colour–magnitude diagram with white dwarfs reveals a declining Milky Way star formation history* (in prep).

## Documentation

Extended documentation on code installation and usage, as well as further information about the JJ model, is available [here](https://jjmodel.readthedocs.io/en/latest/index.html).
