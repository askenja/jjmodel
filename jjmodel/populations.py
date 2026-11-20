"""
Created on Mon Feb 20 18:17:18 2017

@author: Skevja
"""
import os 
import h5py
import glob
import numpy as np 
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from astropy.table import Table

from .constants import tp, tr
from .tools import gauss_weights
from . import localpath


def _starmap_with_kwargs_(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(_apply_args_and_kwargs_, args_for_starmap)

def _apply_args_and_kwargs_(fn, args, kwargs):
    return fn(*args, **kwargs)


def find_ind(value,table):
    ind = np.where(np.abs(np.subtract(table,value))==\
                        np.amin(np.abs(np.subtract(table,value))))[0][0]
    return ind


def spread_metallicity(times, metallicities, n_spread, sigma):
    """Construct time-metallicity pairs with AMR-spread weights."""
    if n_spread == 1:
        return [times, metallicities, np.ones(len(times))]

    t_spread, met_spread, weights_spread = [], [], []

    for t, met in zip(times, metallicities):
        met_grid = np.linspace(met - 3*sigma, met + 3*sigma, n_spread)
        weights = gauss_weights(met_grid, met, sigma)

        t_spread.extend([t] * len(weights))
        met_spread.extend(met_grid)
        weights_spread.extend(weights)

    return [np.array(t_spread), np.array(met_spread), np.array(weights_spread)] 


def mass_per_population(p,pop,t,sigma,weights):

    indt = np.array(np.array(t)//tr,dtype=int)

    # Thin disk
    if (pop == 'd') or (pop == 't'):
        mass = sigma[indt]*tr*weights
    elif pop == 'sh':
        g_grid = np.load(os.path.join(localpath,'input','mass_loss','g_grid.npy'))
        fe0, dfe = -2.0, 0.02 # parameters of mass loss grid
        indmet = p.FeHsh//dfe - fe0//dfe + 1
        mass = sigma/g_grid[int(indmet)][0]*weights
    else:
        mass = None
        print('Cannot calculate population mass, population name not recognized!')
    return mass


def _fast_imf_weights(imf, mass_edges):
    """
    Calculates IMF bin weights using the precomputed IMF arrays.

    This is a faster replacement for repeatedly calling ``imf(m1, m2)`` in a
    Python loop. If the input IMF does not expose the expected internal arrays,
    the function returns ``None`` and the caller should fall back to the old method.
    """
    imf_obj = getattr(imf, "__self__", None)

    required = ("mlow", "mup", "mres", "m_lin", "Nmdm")
    if imf_obj is None or not all(hasattr(imf_obj, attr) for attr in required):
        return None

    mass1 = np.asarray(mass_edges[:-1], dtype=float)
    mass2 = np.asarray(mass_edges[1:], dtype=float)

    mass1 = np.maximum(mass1, imf_obj.mlow)
    mass2 = np.minimum(mass2, imf_obj.mup)

    m1_ind = ((mass1 - imf_obj.mlow) // imf_obj.mres).astype(int)
    m2_ind = ((mass2 - imf_obj.mlow) // imf_obj.mres).astype(int)

    close = (m1_ind == m2_ind) | ((m2_ind - m1_ind) == 1)
    m2_eff = m2_ind.copy()
    m2_eff[close] = m1_ind[close] + 2

    m1_ind = np.clip(m1_ind, 0, len(imf_obj.Nmdm) - 1)
    m2_eff = np.clip(m2_eff, 0, len(imf_obj.Nmdm))

    csum = np.concatenate(([0.0], np.cumsum(imf_obj.Nmdm)))
    weights = csum[m2_eff] - csum[m1_ind]

    interval_weight = np.ones_like(weights)
    interval_weight[close] = (
        (mass2[close] - mass1[close]) /
        (imf_obj.m_lin[m2_eff[close]] - imf_obj.m_lin[m1_ind[close]])
    )

    return weights * interval_weight


def apply_IMF(imf,iso_masses,mass):   
    r"""
    Applies IMF to the isochrone mass column.
    
    :param imf: IMF PDF function returning the probability to form a 
        star with a mass between *mass1* and *mass2*. 
    :type imf: *function(mass1,mass2)* 
    :param iso_masses: Mass column from the isochrone table. 
    :type iso_masses: array-like
    :param mass: Total mass that was converted into stars of the chosen 
        metallicity and age (this isochrone), :math:`\mathrm{M}_\odot`.
    :type mass: scalar  
    
    :return: New column with the present-day surface number densities of the semi-(metallicity-age-mass)
            'stellar assemblies',  :math:`\mathrm{number \ pc}^{-2}`. 
    :rtype: 1d-array            
    """ 
    
    lenm = len(iso_masses)
    m_centers = np.zeros((lenm+1))
    m_centers[0], m_centers[-1] = iso_masses[0], iso_masses[-1]
    m_centers[1:-1] = [np.mean([iso_masses[i],iso_masses[i+1]]) for i in np.arange(lenm-1)]
    
    fast_weights = _fast_imf_weights(imf, m_centers)
    if fast_weights is not None:
        return fast_weights * mass
    
    return np.array([
        imf(m_centers[k], m_centers[k + 1]) * mass
        for k in np.arange(lenm)
    ])


def _imf_to_feh_age_pop_(
        mode,
        phot,
        filenames,
        table_age,
        age_column,
        FeH_column,
        M_column,
        imf,
        f_da=0.8,
        **kwargs
        ):

    res_parts = []

    for mode_key in mode.keys():

        f1 = h5py.File(filenames[mode_key],"r")   

        for age,feh,mass in zip(age_column,FeH_column,M_column):

            # Convert formation time to stellar age
            ind_age = find_ind(age,table_age)
            age_best = table_age[ind_age]

            # Read the requested columns from this age group
            if mode_key == 'main':

                age_name = f'age_{age_best:.2f}'

                iso_age = StellarAssemblies.read_hdf5_isochrone(f1, age_name, phot,**kwargs)
                    
                # Add number density column (apply IMF)
                iso_age['N'] = apply_IMF(imf,iso_age['Mini'],mass)

                n_pops = len(iso_age['Mini'])
                # Add placeholder phase column for basti
                if mode[mode_key]=='BaSTI':
                    iso_age['phase'] = np.ones(n_pops, dtype=int) # MS to AGB are 1

                if 'wd' in mode.keys():
                    iso_age['t_cool'] = np.full(n_pops, np.nan)

            elif mode_key == 'wd':

                age_name = f'age_{age_best:.2f}'
                iso_age_da = StellarAssemblies.read_hdf5_isochrone(
                                    f1, age_name, phot, model='DA',**kwargs)
                iso_age_db = StellarAssemblies.read_hdf5_isochrone(
                                    f1, age_name, phot,model='DB',**kwargs)

                # Add number density column (apply IMF)
                iso_age_da['N'] = apply_IMF(imf,iso_age_da['Mini'],mass*f_da)
                iso_age_db['N'] = apply_IMF(imf,iso_age_db['Mini'],mass*(1-f_da))

                # Add phase 
                iso_age_da['phase'] = np.full(len(iso_age_da['N']),10)
                iso_age_db['phase'] = np.full(len(iso_age_db['N']),11)

                iso_age = {key: np.concatenate((iso_age_da[key], iso_age_db[key])) 
                           for key in iso_age_da}

            elif mode_key == 'bd':

                age_name = f'age_{age_best:.3f}'

                bd_model = mode[mode_key].split('ATMO_')[-1]

                iso_age = StellarAssemblies.read_hdf5_isochrone(f1, age_name, phot, model=bd_model, **kwargs)
                                    
                # Add number density column (apply IMF)
                iso_age['N'] = apply_IMF(imf,iso_age['Mini'],mass)

                n_pops = len(iso_age['Mini'])

                # Add phase
                iso_age['phase'] = np.full(len(iso_age['N']),20)

                                        
            # Add age and FeH columns
            n_pops = len(iso_age['Mini'])
            iso_age['age'] = np.full(n_pops, age)
            iso_age['FeH'] = np.full(n_pops, feh)

            res_parts.append(iso_age)

    res = {
        key: np.concatenate([part[key] for part in res_parts])
        for key in res_parts[0]
    }

    return res



class StellarAssemblies:

    def __init__(self,p,a,mode,photometric_system):
        self.p = p 
        self.a = a 
        self.iso_dir = os.path.join(localpath,'input','isochrones')
        self.mwcomponent_labels = ['d','t','sh']
        self.mode = mode
        self.phot = photometric_system

    @staticmethod
    def get_available_ages(h5):
        """Return the ages available in one metallicity file."""
        return np.array(sorted(
            float(name.removeprefix('age_'))
            for name in h5.keys()
            if name.startswith('age_')
        ))

    @staticmethod
    def read_hdf5_isochrone(h5,age_name,photometric_system,model=None,**kwargs):
        """
        Read the requested physical columns from one age group.

        Assumes each age group contains an 'Astro' dataset and one or
        more photometric datasets, each with a 'columns' attribute.
        """

        if model:
            category = h5[model]
            group = category[age_name]
        else:
            group = h5[age_name]

        # Public photometric-system names -> HDF5 dataset names.
        category_map = {
            'GaiaDR3': 'GDR3',
            'UBVRIplus': 'UBVRIJHK',
            'MKO_WISE_IRAC': 'MKO_WISE_IRAC',
            'NIRCAM': 'NIRCAM_modA'
            # update when necessary
        }

        systems = photometric_system.split('+')
        datasets = ['Astro'] + [category_map[system] for system in systems]

        available_columns = {}

        for dataset_name in datasets:
            dataset = group[dataset_name]
            values = np.asarray(dataset)
            names = dataset.attrs['columns']

            # HDF5 string attributes may be stored as bytes.
            names = [
                name.decode() if isinstance(name, bytes) else str(name)
                for name in names
            ]

            for j, name in enumerate(names):
                available_columns[name] = values[:, j]

        if 'requested' in kwargs:
            missing = set(kwargs['requested']) - available_columns.keys()
            if missing:
                raise KeyError(
                    f"Missing columns in {age_name}: {sorted(missing)}"
                )
            return {name: available_columns[name] for name in kwargs['requested']}
        else:
            return available_columns

        
    def create_pops(self,R,amrd,amrt,sfrd,sfrt,sigmash,imf,**kwargs):

        # Check if brown dwarfs are modelled 
        # ---------------------------------------------------------------
        pop_tabs = {}

        if 'bd' in self.mode.keys():

            if len(self.mode.keys()) > 1:
                raise KeyError("Modelling brown drawfs together with MS+giant or WD populations is currently not supported!") 

            # Only one metallicity model is available for BD -> 
            # no metallicity spread 

            # ---------------------------------------------------------------
            # 1. Construct time-metallicity populations
            # ---------------------------------------------------------------

            # Stellar-halo population
            t_birth_sh = 0    # Gyr
            wsh = 1 
            popsh = [np.array([t_birth_sh]), np.array([self.p.FeHsh]), np.array([wsh])]

            # Disk populations
            n_FeHdt_bd = 1 
            dFeHdt_bd = 0 
            popd = spread_metallicity(self.a.t, amrd, n_FeHdt_bd, dFeHdt_bd)
            popt = spread_metallicity(self.a.t[:self.a.jt], amrt[:self.a.jt],n_FeHdt_bd, dFeHdt_bd)

            # ---------------------------------------------------------------
            # 2. Calculate the mass reservoir of each subpopulation
            # ---------------------------------------------------------------

            md = mass_per_population(self.p,'d',popd[0],sfrd,popd[2])
            mt = mass_per_population(self.p,'t',popt[0],sfrt[:self.a.jt],popt[2])
            msh = mass_per_population(self.p,'sh',popsh[0],sigmash,popsh[2])

            # Available ages - for BD resolution is 0.025 Gyr
            table_age = np.round(np.arange(0.025,13.025,0.025),3)

            pop_type = list(self.mode.keys())[0]
            iso_model = list(self.mode.values())[0].split('_')[0]

        else:
            # ---------------------------------------------------------------
            # 1. Construct time-metallicity populations
            # ---------------------------------------------------------------

            # Stellar-halo population
            amrsh_spread = np.linspace(
                self.p.FeHsh - 3*self.p.dFeHsh,
                self.p.FeHsh + 3*self.p.dFeHsh,
                self.p.n_FeHsh
            )
            wsh = gauss_weights(amrsh_spread, self.p.FeHsh, self.p.dFeHsh)
            popsh = [np.zeros(self.p.n_FeHsh), amrsh_spread, wsh]

            # Disk populations
            popd = spread_metallicity(self.a.t, amrd, self.p.n_FeHdt, self.p.dFeHdt)
            popt = spread_metallicity(self.a.t[:self.a.jt], amrt[:self.a.jt],self.p.n_FeHdt, self.p.dFeHdt)

            # ---------------------------------------------------------------
            # 2. Calculate the mass reservoir of each subpopulation
            # ---------------------------------------------------------------

            md = mass_per_population(self.p,'d',popd[0],sfrd,popd[2])
            mt = mass_per_population(self.p,'t',popt[0],sfrt[:self.a.jt],popt[2])
            msh = mass_per_population(self.p,'sh',popsh[0],sigmash,popsh[2])

            # Available ages - for BD resolution is 0.025 Gyr
            table_age = np.round(np.arange(0.05,13.05,0.05),2)

            pop_type = list(self.mode.keys())[0]
            iso_model = list(self.mode.values())[0]

        # Combine all populations.
        # Each row will represent one population:
        # [formation_time, FeH, spread_weight, mass]
        times = {'d':popd[0],'t':popt[0], 'sh':popsh[0]}
        metallicities = {'d':popd[1], 't':popt[1], 'sh':popsh[1]}
        masses = {'d':md,'t':mt,'sh':msh}

        # ---------------------------------------------------------------
        # 3. Discover available metallicities from HDF5 filenames
        # ---------------------------------------------------------------

        iso_dir_main = os.path.join(self.iso_dir,pop_type,iso_model)
        iso_files = glob.glob(os.path.join(iso_dir_main, 'iso_feh*.h5'))

        # Available metallicities
        table_met = np.sort([float(fn.split('iso_feh')[1].split('.h5')[0]) for fn in iso_files])

        for i,label in enumerate(self.mwcomponent_labels):

            age_column = tp - times[label]
            FeH_column = metallicities[label]
            M_column = masses[label]

            # Then for all populations find closest values in the metallicity grid (indices)
            
            ind_feh = np.array([find_ind(val,table_met) for val in FeH_column])

            # Get their unique values (in case if there are duplications) 
            # These uniques values will be files to open
            ind_feh_unique = np.unique(ind_feh)

            ind_pop = [np.flatnonzero(ind_feh==ind) for ind in ind_feh_unique]

            # ---------------------------------------------------------------
            # 4. Process each metallicity file only once
            # ---------------------------------------------------------------
            filenames = {}
            for mode_key in self.mode.keys():
                iso_model = self.mode[mode_key]
                if mode_key == 'bd':
                    iso_model = iso_model.split('_')[0]
                iso_dir = os.path.join(self.iso_dir,mode_key,iso_model)
                filenames[mode_key] = [os.path.join(iso_dir,f'iso_feh{table_met[ind]}.h5') 
                                       for ind in ind_feh_unique]

            kwargs_list = repeat(kwargs)   
            argument_list = [
                (
                self.mode,
                self.phot,
                {key:filenames[key][k] for key in self.mode.keys()},
                table_age,
                age_column[ind_pop[k]],
                FeH_column[ind_pop[k]],
                M_column[ind_pop[k]],
                imf
                )
                for k in np.arange(len(ind_feh_unique))
                ]
            
            if 'f_da' in kwargs:
                func = partial(_imf_to_feh_age_pop_, f_da=kwargs['f_da'])
            else:
                func = _imf_to_feh_age_pop_
                
            pool = Pool(processes=self.p.nprocess)
            result = _starmap_with_kwargs_(
                pool,
                func,
                argument_list,
                kwargs_list
                )
            pool.close()
            pool.join()

            # ---------------------------------------------------------------
            # 5. Combine the individual population tables
            # ---------------------------------------------------------------
            # Create output lists
            all_columns = list(result[0].keys())
            all_columns += ['label']

            ncols = len(all_columns)
            output = [[] for _ in range(ncols)]

            for k in range(len(result)):
                for m in range(ncols-1):
                    output[m].extend(result[k][all_columns[m]])
                output[-1].extend(np.repeat(i,len(result[k]['age'])))
                    
            out_tab = Table()
            for k in range(ncols):
                out_tab[all_columns[k]] = output[k]
                
            # Save the table        
            mode_label = '-'.join(self.mode.values())
            out_tab.write(os.path.join(self.a.T['poptab'],''.join(('SSP_R',
                                    str(R),'_',label,'_',mode_label,'.csv'))),overwrite=True) 
            pop_tabs[label] = out_tab

        return pop_tabs

