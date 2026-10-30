
import os
import sys
import copy
import warnings
import numpy as np 
from scipy import optimize
from scipy.stats import norm
from astropy.table import Table
from fast_histogram import histogram1d, histogram2d
from scipy.interpolate import RegularGridInterpolator

import matplotlib as mpl
import matplotlib.pyplot as plt

from .constants import tp
from .constants import KM, tr
from .geometry import Volume
from .funcs import IMF, SFR
from .analysis import _extend_mag_
from .tools import convolve1d_gauss, convolve2d_gauss
from .populations import StellarAssemblies
from . import localpath



class ParHandler:
    r"""Class for manipulating the parameter objects for MCMC runs
    and several helper functions.
    """
    def __init__(self,param_names):
        r"""Initialization. 

        Parameters:
        -----------
        - param_names: dict
            Dictionary with parameter names, structured in a hierarchical way. 
            Check 'mcmc_run_pop.py' script for details on building this dictionary 
            (called par_optim there).
        """
        param_struct = {}

        # Build the parameter structure and initialize everything with nan values
        for key in param_names.keys():
            if key != 'sfr':
                param_struct[key] = {name:np.nan for name in param_names[key]}
            else:
                param_struct[key] = {}
                for sfr_key in param_names[key].keys():
                    param_struct[key][sfr_key] = {name:np.nan for name in param_names[key][sfr_key]}

        self.param_struct = param_struct


    def get_flat_param_list(self):
        r""" 
        Flattenes the hierarchical parameter structure into a list of parameter names. 

        Parameters:
        -----------
        None

        Returns:
        --------
        - list of strings
            List of parameter names, in the same order 
            as they are given in the hierarchical parameter structure
            used for the class initialization.
        """
        flat_param_list = []
        for par_class in self.param_struct.keys():
            #print(par_class)
            if par_class != 'sfr':
                flat_param_list.extend(list(self.param_struct[par_class].keys()))
            else:
                flat_param_list.extend(list(self.param_struct[par_class]['d'].keys()))
                if 't' in self.param_struct['sfr'].keys():
                    flat_param_list.extend(list(self.param_struct[par_class]['t'].keys()))
        
        self.flat_param_list = flat_param_list
        return self.flat_param_list


    def fill_param_struct(self,theta):
        """
        Fill the created upon class initialization hierarchical parameter dictionary 
        with values from an input list.

        Parameters:
        -----------
        - theta: list of floats
            List of parameter values, in the same order as the parameter names 
            in the flat parameter list obtained with the method get_flat_param_list().

        Returns:
        --------
        - dict
            The hierarchical parameter dictionary, filled with the input parameter values.
        """
        if len(theta) != len(self.flat_param_list):
            print(theta)
            raise ValueError('Length of parameter list does not match the number of parameters in structured dict!\nlen(theta)=%d, len(flat_param_list)=%d' % (len(theta), len(self.flat_param_list)))

        i = 0
        for group, subdict in self.param_struct.items():
            # Case 1: simple subdict (e.g. ifmr, dcool, f_dadb, imf)
            if all(not isinstance(v, dict) for v in subdict.values()):
                for key in subdict.keys():
                    self.param_struct[group][key] = theta[i]
                    i += 1
            # Case 2: nested subdict (e.g. sfr -> d, t)
            else:
                for subgroup, subsubdict in subdict.items():
                    for key in subsubdict.keys():
                        self.param_struct[group][subgroup][key] = theta[i]
                        i += 1
        return self.param_struct
    


class HessConstructor:
    """ Class to generate Hess diagram from the model."""

    def __init__(self, p, a, mag_range, mag_step, 
                 color_scheme=['G_Gaia_DR3','G_RP_Gaia_DR3'],r_max=50, zmax=150):
        r""" 
        Initialization. 

        Parameters:
        -----------
        - p: namedtuple
            JJ-model parameter object from the model initialization. 
        - a: namedtuple
            JJ-model helpers array object from the model initialization.
        - mag_range: list of lists of floats
            Range of xy-magnitudes in the Hess diagram. 
            Built as [[color_min,color_max], [absmag_min, absmag_max]]. 
        - mag_step: list of floats  
            Step size in xy-magnitudes for the Hess diagram.
        - color_scheme: list of strings
            Gaia color to use. By default G-GRP, ['G_Gaia_DR3','G_RP_Gaia_DR3'], 
            but can b also ['G_BP_Gaia_DR3','G_RP_Gaia_DR3']
        - r_max: float
            Maximum radius in pc for the spherical volume around the Sun,
            for which the Hess diagram is constructed. By default 50 pc. 
            If volume is not a spher, this parameter is ignored. 
        - zmax: float
            Maximum z in pc for the custom volume, if the modeled volume is not a sphere.
            Default is 150 pc. 
        """
        self.p = p 
        self.a = a

        self.mag_range = mag_range
        self.mag_step = mag_step
        self.color_scheme = color_scheme

        # Define z limits for the volume. Only needed when the modeled volume is a sphere, 
        # otherwise these arrays are not used
        zlim = [0,r_max + 1.5*self.p.zsun]
        indz1, indz2 = int(abs(zlim[0])//self.p.dz), int(abs(zlim[1]//self.p.dz))
        self.indz1, self.indz2 = np.sort([indz1,indz2])

        # Define spherical grid 
        V = Volume(self.p,self.a)
        self.volume = V.local_sphere(0,r_max)[0][self.indz1:self.indz2]

        # If a custom valume is used, zmax should be given 
        self.ind_zmax = np.where(self.a.z - zmax >=0)[0][0] 

        # Boundaries for the CMD-defined populations
        # Values (A,B,C) are used as A*x + B*y + C >= 0 
        # x - color axis, y - magnitude axis
        self.pop_defs = {
            "wd":  [(-6, 1, -9.3)],
            "ums": [(1, 0, 1),   # dummy placeholder
                    (0, -1, 4.3),
                    (-35, 1, 15)],
            "g":   [(0, -1, 4.3),
                    (35, -1, -15)],
            "ms":  [(0, 1, -4.3),
                    (0, -1, 8.5),
                    (6, -1, 9.3)],
            "lms": [(0, 1, -8.5),
                    (6, -1, 9.3)]
        }
        # Edges of the Hess diagram bins 
        self.x_edges = np.arange(
            self.mag_range[0][0], 
            self.mag_range[0][1] + self.mag_step[0], 
            self.mag_step[0]
            )
        self.y_edges = np.arange(
            self.mag_range[1][0], 
            self.mag_range[1][1] + self.mag_step[1], 
            self.mag_step[1]
            )
        # Number of bins in x and y, and cell area in the CMD
        self.nx = len(self.x_edges)-1
        self.ny = len(self.y_edges)-1
        self.cell_area = mag_step[0] * mag_step[1]

        # Build masks for the CMD-defined populations, to be applied to the Hess diagram
        # Values in the masks are between 0 and 1, 
        # and represent the fraction of the Hess cell area that belongs to the population.
        # Approach with masks gives less accurate results than a direct sorting 
        # the stellar assemblies into the populations, but was tested in a hope to improve performance. 
        # Is not used in the final version of the likelihood. 
        self.masks = self.build_population_masks()


    def _clip_polygon_(self, poly, A, B, C):
        """Clip a polygon with a line defined by A*x + B*y + C = 0"""

        if len(poly) == 0:
            return poly

        def inside(p):
            return A*p[0] + B*p[1] + C >= 0

        def intersect(p1, p2):
            x1,y1 = p1
            x2,y2 = p2
            d1 = A*x1 + B*y1 + C
            d2 = A*x2 + B*y2 + C
            t = d1 / (d1 - d2)
            return (x1 + t*(x2-x1), y1 + t*(y2-y1))

        out = []
        for i in range(len(poly)):
            P = poly[i]
            Q = poly[(i+1)%len(poly)]
            if inside(P) and inside(Q):
                out.append(Q)
            elif inside(P) and not inside(Q):
                out.append(intersect(P,Q))
            elif not inside(P) and inside(Q):
                out.append(intersect(P,Q))
                out.append(Q)
        return out
    
    def _polygon_area_(self, poly):
        """Calculate the area of a polygon defined by a list of vertices (x,y)."""

        if len(poly) < 3:
            return 0.0
        x = np.array([p[0] for p in poly])
        y = np.array([p[1] for p in poly])
        return 0.5*np.abs(np.dot(x,np.roll(y,-1)) - np.dot(y,np.roll(x,-1)))
    

    def build_population_masks(self):
        """
        Build masks for the CMD-defined populations
        for selecting them from a pre-calculated Hess diagram. 
        """
        masks = {pop: np.zeros((self.ny,self.nx)) for pop in self.pop_defs}

        for iy in range(self.ny):
            y0,y1 = self.y_edges[iy], self.y_edges[iy+1]
            for ix in range(self.nx):
                x0,x1 = self.x_edges[ix], self.x_edges[ix+1]

                # Hess cell rectangle
                cell = [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]

                for pop,planes in self.pop_defs.items():
                    poly = cell
                    for A,B,C in planes:
                        poly = self._clip_polygon_(poly,A,B,C)
                        if len(poly)==0:
                            break
                    if len(poly)>0:
                        masks[pop][iy,ix] = self._polygon_area_(poly) / self.cell_area
        return masks


    def pops_in_volume(self,pop_tabs,indt,inp_tabs):
        r"""
        Calculates the number of stars in a _spherical_ volume.  
        For a custom volume use method pops_in_slice() instead.

        Parameters:
        -----------
        - pop_tabs: dict of pandas DataFrames or astropy tables
            Tables of stellar assemblies for the thin/thick disk and halo: 
            {'d':tab1, 't':tab2, 'sh':tab3}.
        - indt: 1d-array
            Time indices for the thin-disk table. For each stellar assembly (table row)
            gives its index in the JJ-model time array a.t.  
        - inp_tabs: dict
            Model predictions for potential, AVR, scale heights, etc. 
            Output of the call local_run(p,a,inp). 

        Returns:
        --------
        - dict
            Tables of stellar assemblies for the thin/thick disk and halo: 
            {'d':tab1, 't':tab2, 'sh':tab3} with an additional column 'Nz' 
            that gives the overall number of each type of stars in the modeled volume. 
        """
        # Read tables and helper arrays
        tabd, tabt, tabsh = pop_tabs['d'],pop_tabs['t'],pop_tabs['sh']
        
        volume = self.volume
        jd_array = self.a.jd_array

        Fi_sliced = inp_tabs['Fi'][self.indz1:self.indz2]
        Hd, Ht, Hsh = inp_tabs['Hd'], inp_tabs['Ht'], inp_tabs['Hsh']
        avr  = inp_tabs['AVR']

        # Pre-calculate the total weight term for the thick disk and halo 
        # Nz = N * weight, where N is the surface number density of the assembly
        # and Nz is the predited number of these stars in the modeled volume. 
        sigt = self.p.sigt
        sigsh = self.p.sigsh
        wt = 0.5 / Ht * np.sum(np.exp(-Fi_sliced / KM**2 / sigt**2)*volume)
        wsh = 0.5 / Hsh * np.sum(np.exp(-Fi_sliced / KM**2 / sigsh**2)*volume) 

        # For the thin disk, we precalculate only the part of the weight;
        # this exp term depends on age of the population. 
        exp_AVR = np.exp(-Fi_sliced[None,:] / KM**2 / avr[jd_array][:, None]**2)  # shape: (z, jd)
        
        # Get total weights for the thin disk
        if self.p.pkey==1:
            # Extra SFRd peaks included
            Fp = inp_tabs['Fp']
            Hdp = inp_tabs['Hdp']
            sigp = self.p.sigp
            fpr0 = 1 - np.sum(Fp,axis=0) 
            '''
            print("jd_array:", jd_array.shape)
            print("fpr0:", fpr0.shape)
            print("Hd:", Hd.shape)
            print("Fi_sliced:", Fi_sliced.shape)
            print("volume:", volume.shape)
            print("avr:", avr.shape)
            print("exp_AVR:", exp_AVR.shape)

            print("fpr0[jd_array]:", fpr0[jd_array].shape)
            print("Hd[jd_array]:", Hd[jd_array].shape)
            print("integral:", np.sum(exp_AVR * volume[None, :], axis=0).shape)
            ''';
            wd_total = (fpr0[jd_array] / (2 * Hd[jd_array])) * np.sum(exp_AVR * volume[None,:], axis=1)

            exp_sigp = np.exp(-Fi_sliced[None, :] / KM**2 / sigp[:, None]**2)  # shape: (z, npeak)
            peak_int = np.sum(exp_sigp * volume[None, :], axis=1)     # (n_peak,)
            
            wd_total += np.sum((Fp[:, jd_array] / (2 * Hdp[:, None]) * peak_int[:, None]), axis=0)
        else:
            # Only continuum SFRd
            wd_total =  0.5 / Hd[jd_array] * np.sum(exp_AVR * volume[None,:], axis=0)
        
        # Add the Nz column
        tabd['Nz'] = np.array(tabd['N']*wd_total[indt])
        tabt['Nz'] = np.array(tabt['N']*wt)
        tabsh['Nz'] = np.array(tabsh['N']*wsh)

        return {'d':tabd, 't':tabt, 'sh':tabsh}
    

    def pops_in_slice(self,pop_tabs,indt,inp_tabs,vz_mag):
        r"""
        Calculates the overall number of stars of each type in a custom volume.  

        Parameters:
        -----------
        - pop_tabs: dict of pandas DataFrames or astropy tables
            Tables of stellar assemblies for the thin/thick disk and halo: 
            {'d':tab1, 't':tab2, 'sh':tab3}.
        - indt: 1d-array
            Time indices for the thin-disk table. For each stellar assembly (table row)
            gives its index in the JJ-model time array a.t.  
        - inp_tabs: dict
            Model predictions for potential, AVR, scale heights, etc. 
            Output of the call local_run(p,a,inp).
        - vz_mag: dict of 2d-arrays
            Dictionary with arrays (n_pop, nz) giving the volume at each z 
            for each population of the input tables. Structure is 
            {'d':(n_pop_d, nz),'t':(n_pop_t, nz),'sh':(n_pop_sh, nz)}. 
            Arrays must be 2d because modeled volume depends on the absolute magnitude
            (approach as in our SJ21 paper). 

        Returns:
        --------
        - dict
            Tables of stellar assemblies for the thin/thick disk and halo: 
            {'d':tab1, 't':tab2, 'sh':tab3} with an additional column 'Nz' 
            that gives the overall number of each type of stars in the modeled volume. 
        """
        # Read tables and helper arrays
        tabd, tabt, tabsh = pop_tabs['d'],pop_tabs['t'],pop_tabs['sh']
        
        jd_array = self.a.jd_array

        avr  = inp_tabs['AVR']
        Fi_sliced = inp_tabs['Fi'][:self.ind_zmax] 
        Hd, Ht, Hsh = inp_tabs['Hd'], inp_tabs['Ht'], inp_tabs['Hsh']

        # Pre-calculate the total weight term for the thick disk and halo 
        # Nz = N * weight, where N is the surface number density of the assembly
        # and Nz is the predited number of these stars in the modeled volume. 
        
        sigt = self.p.sigt
        exp_t = np.exp(-Fi_sliced / KM**2 / sigt**2)   # shape: (nz,)
        wt = (0.5 / Ht) * np.sum(vz_mag['t'] * exp_t[None, :], axis=1)
        
        sigsh = self.p.sigsh
        exp_sh = np.exp(-Fi_sliced / KM**2 / sigsh**2)   # shape: (nz,)
        wsh = (0.5 / Hsh) * np.sum(vz_mag['sh'] * exp_sh[None, :], axis=1)

        # Same for the thin disk

        vol_d = vz_mag['d']
        AVR_row = avr[jd_array][indt] 
        exp_AVR = np.exp(-Fi_sliced[None,:] / KM**2 / AVR_row[:, None]**2)  # shape: (n_rows_d, z)

        if self.p.pkey==1:
            # Extra SFRd peaks included
            Fp = inp_tabs['Fp']
            Hdp = inp_tabs['Hdp']
            sigp = self.p.sigp
            fpr0 = 1 - np.sum(Fp,axis=0) 

            wd_total = (fpr0[jd_array][indt] / (2 * Hd[jd_array][indt])) * np.sum(exp_AVR * vol_d, axis=1)
            exp_sigp = np.exp(-Fi_sliced[None, :] / KM**2 / sigp[:, None]**2)  # (n_peak, z)
            peak_int = np.sum(exp_sigp[:, None, :] * vol_d[None, :, :], axis=2)
            
            wd_total += np.sum((Fp[:, jd_array][:, indt] / (2 * Hdp[:, None]) * peak_int), axis=0)
            tabd['Nz'] = np.array(tabd['N']*wd_total)
        else:
            # Only continuum SFRd
            wd_total =  0.5 / Hd[jd_array][indt] * np.sum(exp_AVR * vol_d, axis=1)
            tabd['Nz'] = np.array(tabd['N']*wd_total[indt])

        tabt['Nz'] = np.array(tabt['N']*wt)
        tabsh['Nz'] = np.array(tabsh['N']*wsh)

        return {'d':tabd, 't':tabt, 'sh':tabsh}
    

    @staticmethod
    def hess_from_data(
        x_column,
        y_column,
        mag_range,
        mag_step,
        mag_smooth=None,
        weights=None
        ):
        r""" 
        Simple calculation of the Hess diagram given the color and magnitude columns
        of the thin/thick disk or halo table. 

        Parameters:
        -----------
        - x_column: 1d-array
            Color column. 
        - y_column: 1d_array
            Magnitude column. 
        - mag_range: list of lists of floats
            Range of xy-magnitudes in the Hess diagram. 
            Built as [[color_min,color_max], [absmag_min, absmag_max]]. 
        - mag_step: list of floats  
            Step size in xy-magnitudes for the Hess diagram.
        - mag_smooth: list of floats
            Width of the smoothing windows in x and y in mag. By default None (no smoothing). 
        - weights: 1d-array
            Weights of each row in color and magnitude (predicted numbers of stars). 

        Returns:
        --------
        - 2d-array:
            Hess diagram. 
        """
                
        if weights is None:
            weights = np.ones_like(x_column)

        xlen = int(round(abs((mag_range[0][0]-mag_range[0][1])/mag_step[0]),0)) 
        ylen = int(round(abs((mag_range[1][0]-mag_range[1][1])/mag_step[1]),0))
        
        hess = np.zeros((xlen,ylen))
        hess = histogram2d(x_column,y_column,weights=weights,
                           bins=[xlen,ylen],range=mag_range)

        if mag_smooth is not None:                                        
            hess = convolve2d_gauss(hess,mag_smooth,mag_range)

        return hess
    
    @staticmethod
    def hessproj_from_data(
        column,
        range,
        step,
        smooth=None,
        weights=None,
        ind_lim=-1
        ):
        """
        Projection of a Hess diagram to one of its axes calculated 
        directly from the color or magnitude column (not pre-calculated Hess). 

        Parameters:
        -----------
        - column: 1d-array
            Color or magnitude column. 
        - range: list of lists of floats
            Range of xy-magnitudes in the Hess diagram. 
            Built as [[color_min,color_max], [absmag_min, absmag_max]]. 
        - step: list of floats  
            Step size in xy-magnitudes for the Hess diagram.
        - smooth: list of floats
            Width of the smoothing windows in x and y in mag. By default None (no smoothing). 
        - weights: 1d-array
            Weights of each row in color and magnitude (predicted numbers of stars). 
        - ind_lim: int
            Max index where the distribution will be cut off. 
            By default -1 (no cutting, the whole range is used). 

        Returns:
        --------
        - 1d-array:
            Color or magnitude distribution. No normalization is applied 
            (unless it's already built into provided weights).  
        """
        if weights is None:
            weights = np.ones_like(column)

        dlen = int(round(abs((range[0]-range[1])/step),0)) 
        
        distribution = np.zeros((dlen))
        distribution = histogram1d(column,weights=weights,bins=dlen,range=range)
        # numpy 
        #distribution = np.histogram(column,weights=weights,bins=dlen,range=range)[0]

        if smooth:                                        
            distribution = convolve1d_gauss(distribution,smooth,range)

        return distribution[:ind_lim]
    

    @staticmethod
    def hessproj_all_pops_from_data(
        column,
        mag_range,
        mag_step,
        weights,
        pop_id,
        mag_smooth=None
        ):
        """
        Projections of a Hess diagram to one of its axes calculated 
        directly from the color or magnitude column (not pre-calculated Hess). 
        Projections are calculated for several CMD-defined populations. 

        Parameters:
        -----------
        - column: 1d-array
            Color or magnitude column. 
        - mag_range: list of lists of floats
            Range of xy-magnitudes in the Hess diagram. 
            Built as [[color_min,color_max], [absmag_min, absmag_max]]. 
        - mag_step: list of floats  
            Step size in xy-magnitudes for the Hess diagram.
        - mag_smooth: list of floats
            Width of the smoothing windows in x and y in mag. By default None (no smoothing). 
        - weights: 1d-array
            Weights of each row in color and magnitude (predicted numbers of stars). 
        - pop_id: list of int
            Population index for each row. 

        Returns:
        --------
        - 2d-array:
            Color or magnitude distributions for defined populations. 
            No normalization is applied 
            (unless it's already built into provided weights).  
        """
        n_pops = np.max(pop_id)
        dlen = int(round(abs((mag_range[0]-mag_range[1])/mag_step),0))

        distribution = np.zeros((n_pops,dlen))
        for i in range(n_pops):
            weights_pop = weights * (pop_id == i + 1) # select population
            dist = np.histogram(column,bins=dlen,range=mag_range,weights=weights_pop)[0]
            if mag_smooth:                                        
                distribution[i,:] = convolve1d_gauss(dist,mag_smooth,mag_range)
            else:
                distribution[i,:] = dist
        return distribution
    

    def proj_from_hess(
        self,
        H, 
        range={'col':[-0.5,1.5],'mag':[-5,18.0]},
        smooth={'col':0.04,'mag':0.4},
        ind_lim={'col':-1,'mag':-1}
        ):
        """
        Projections of a Hess diagram to its axes calculated 
        from a not pre-calculated Hess diagram (not original color and magnitude columns). 

        Parameters:
        -----------
        - H: 2d-array
            Hess diagram.  
        - range: dict of lists of floats
            Range of xy-magnitudes in the Hess diagram. 
            Built as ['col':[color_min,color_max], 'mag':[absmag_min, absmag_max]}. 
        - smooth: dict of floats
            Width of the smoothing windows in x and y in mag.
            Built as ['col':col_smooth, 'mag':mag_smooth}. 
        - ind_lim: dict of int
            Max index where the distributions will be cut off. 
            Built as ['col':col_ind_cut, 'mag':mag_ind_cut}. By default -1 for both (no cut). 

        Returns:
        --------
        - dict of dict of floats:
            Color and magnitude distributions for the defined populations. 
            Structure: 
            {'pop_name':{
                'cdf':color_distribution,
                'mdf':magnitude_distribution,
                'ncdf':area_normalized_smoothed_color_distribution,
                'mcdf':area_normalized_smoothed_magnitude_distribution,
                'n_pop':number_of_stars_for_this_population
                },
            ...
            }
        """
        proj = {}

        for pop, mask in self.masks.items():
            Hp = H[:ind_lim['mag'],:ind_lim['col']] * mask[:ind_lim['mag'],:ind_lim['col']]
            
            # color distribution (x)
            cdf = Hp.sum(axis=0)   # shape (nx,)
            # magnitude distribution (y)
            mdf = Hp.sum(axis=1)   # shape (ny,)

            scdf = convolve1d_gauss(cdf,smooth['col'],range['col'])
            smdf = convolve1d_gauss(mdf,smooth['mag'],range['mag'])

            n_tot = Hp.sum()
            # same normalized on area
            nscdf = scdf/(n_tot*self.mag_step[0]) 
            nsmdf = smdf/(n_tot*self.mag_step[1])

            proj[pop] = {
                "cdf": scdf,
                "mdf": smdf,
                "ncdf": nscdf,
                "nmdf": nsmdf,
                "n_pop": n_tot
            }

        return proj


    def generate_hess(
            self,
            pop_tabs,
            indt,
            inp_tabs,
            mag_range,
            mag_step,
            mag_smooth,
            color_shift=0,
            vz_mag=None,
            volume='sphere',
            save_tabs=False
        ):
        r"""
        Total Hess diagram (thin/thick disk and halo) 
        calculated from the color and magnitude columns. 
        
        Parameters:
        -----------
        - pop_tabs: dict of pandas DataFrame objects or Astropy tables
            Tables of stellar assemblies for the thin/thick disk and stellar halo. 
            Built as {'d':table1,'t':table2,'sh':table3}. 
        - indt: 1d-array
            Time indices for the thin-disk table. 
        - inp_tabs: dict of arrays
            Step size in xy-magnitudes for the Hess diagram.
        - mag_range: list of lists of floats
            Range of xy-magnitudes in the Hess diagram. 
            Built as [[color_min,color_max], [absmag_min, absmag_max]]. 
        - mag_step: list of floats  
            Step size in xy-magnitudes for the Hess diagram.
        - mag_smooth: list of floats
            Width of the smoothing windows in x and y in mag.
        - color_shift: float
            Mag value to add to all colors. By default 0, no shift. 
        - vz_mag: dict of 2d-arrays
            Dictionary with arrays (n_pop, nz) giving the volume at each z 
            for each population of the input tables. Structure is 
            {'d':(n_pop_d, nz),'t':(n_pop_t, nz),'sh':(n_pop_sh, nz)}. 
            Arrays must be 2d because modeled volume depends on the absolute magnitude
            (approach as in our SJ21 paper). 
        - volume: string
            Type of modeled volume: 'sphere' for the local sphere with radius r_max 
            as can be specified during class initialization, 
            or 'slice' - a custom volume.

        Returns:
        --------
        Hess diagram, 2d-array
        """
        
        sum_Nz, mag = [], [[], [], []]
        
        if volume == 'sphere':
            pop_tabs = self.pops_in_volume(pop_tabs,indt,inp_tabs)
        if volume == 'slice':
            pop_tabs = self.pops_in_slice(pop_tabs,indt,inp_tabs,vz_mag)

        # Stack thin/thick disk and halo tables into one Nz-color-mag table 
        for table in pop_tabs.values():
            sum_Nz.extend(table['Nz'])
            mag = _extend_mag_(
                mag,table,
                ['G_Gaia_DR3',self.color_scheme[0],self.color_scheme[1]]       
                )

        sum_Nz = np.array(sum_Nz)
        color = np.subtract(mag[1],mag[2]) + color_shift
        magnitude = mag[0]

        # Create Hess diagram
        hess = self.hess_from_data(color,magnitude,mag_range,mag_step,
                                   weights=sum_Nz,mag_smooth=mag_smooth)
        
        if save_tabs:
            for pop, table in pop_tabs.items():
                pop_tabs[pop] = table.write(
                    os.path.join(self.a.T['poptab'],
                    ''.join(('SSP_R',str(self.p.Rsun),'_' + pop +'_nz.csv')))
                    )
        return hess.T
    

    def generate_hessproj(
        self,
        pop_tabs,
        indt,
        inp_tabs,
        mag_range,
        mag_step,
        mag_smooth,
        color_shift=0,
        vz_mag=None,
        volume='sphere',
        idx_pop=None,
        ind_lim={'col':-1,'mag':-1}
        ):
        r"""
        Projections of a Hess diagram its axes calculated 
        directly from the color and magnitude columns (not pre-calculated Hess). 
        Projections are calculated for several CMD-defined populations. 
        Alternative to hessproj_all_pops_from_data() method, but for both projections. 

        Parameters:
        -----------
        - pop_tabs: dict of pandas DataFrame objects or Astropy tables
            Tables of stellar assemblies for the thin/thick disk and stellar halo. 
            Built as {'d':table1,'t':table2,'sh':table3}. 
        - indt: 1d-array
            Time indices for the thin-disk table. 
        - inp_tabs: dict of arrays
            Step size in xy-magnitudes for the Hess diagram.
        - mag_range: list of lists of floats
            Range of xy-magnitudes in the Hess diagram. 
            Built as [[color_min,color_max], [absmag_min, absmag_max]]. 
        - mag_step: list of floats  
            Step size in xy-magnitudes for the Hess diagram.
        - mag_smooth: list of floats
            Width of the smoothing windows in x and y in mag.
        - color_shift: float
            Mag value to add to all colors. By default 0, no shift.  
        - vz_mag: dict of 2d-arrays
            Dictionary with arrays (n_pop, nz) giving the volume at each z 
            for each population of the input tables. Structure is 
            {'d':(n_pop_d, nz),'t':(n_pop_t, nz),'sh':(n_pop_sh, nz)}. 
            Arrays must be 2d because modeled volume depends on the absolute magnitude
            (approach as in our SJ21 paper). 
        - idx_pop: dict of list of int
            Indices of CMD-defined populations given for the total(!) d+t+sh stellar assemblies table. 
        - ind_lim: dict of int
            Max index where the distributions will be cut off. 
            Built as ['col':col_ind_cut, 'mag':mag_ind_cut}. By default -1 for both (no cut). 

        Returns:
        --------
        - dict of dict of floats:
            Color and magnitude distributions for the defined populations. 
            Structure: 
            {'pop_name':{
                'cdf':color_distribution,
                'mdf':magnitude_distribution,
                'ncdf':area_normalized_smoothed_color_distribution,
                'mcdf':area_normalized_smoothed_magnitude_distribution,
                'n_pop':number_of_stars_for_this_population
                },
            ...
            }        
        """

        sum_Nz, mag = [], [[], [], []]
        
        if volume == 'sphere':
            pop_tabs = self.pops_in_volume(pop_tabs,indt,inp_tabs)
        if volume == 'slice':
            pop_tabs = self.pops_in_slice(pop_tabs,indt,inp_tabs,vz_mag)

        # Stack thin-thick disk and halo into one Nz-and-mag-table 
        for label in pop_tabs.keys():
            table = pop_tabs[label]
            sum_Nz.extend(table['Nz'])
            mag = _extend_mag_(
                mag,table,
                ['G_Gaia_DR3',self.color_scheme[0],self.color_scheme[1]]
            )

        sum_Nz = np.array(sum_Nz)
        color = np.subtract(mag[1],mag[2]) + color_shift
        magnitude = np.array(mag[0])

        pops = list(idx_pop.keys())
        output = {pop:{} for pop in pops}

        for pop in pops:
            idx = idx_pop[pop] # select population
            # Color distribution
            cdf = self.hessproj_from_data(
                color[idx],
                mag_range[0],
                mag_step[0],
                smooth=mag_smooth[0],
                weights=sum_Nz[idx],
                ind_lim=ind_lim['col']
            )
            # Magnitude distribution
            mdf = self.hessproj_from_data(
                magnitude[idx],
                mag_range[1],
                mag_step[1],
                smooth=mag_smooth[1],
                weights=sum_Nz[idx],
                ind_lim=ind_lim['mag']
            )
            
            n_pop = np.sum(cdf)

            # Normalize on area
            ncdf = cdf/(n_pop*mag_step[0])
            nmdf = mdf/(n_pop*mag_step[1])

            output[pop] = {'cdf':cdf,'ncdf':ncdf,'mdf':mdf,'nmdf':nmdf,'n_pop':n_pop}       

        return output
    

class SFRHandler():
    """Class for SFR manipulation"""

    def __init__(self, p, a, inp):
        """Initialization
        
        Parameters:
        -----------
        - p: namedtuple
            JJ-model parameter object from the model initialization. 
        - a: namedtuple
            JJ-model helpers array object from the model initialization.
        - inp: dict of arrays 
            JJ-model input array aobject from the model initialization.
        """
        self.a = a
        self.p = p
        self.inp = inp
    
    def create_reference_sfr(self):
        r"""
        Parameters:
        -----------
        None 
        
        Returns:
        --------
        - dict
            The solar-radius thin- and thick-disk SFR arrays calculated 
            for the model time grid a.t and parameters from the (two) parameter files. 
            Built as {'d':sfrd, 't':sfrt}. 
        """
        return {'d':self.inp['SFRd0'],'t':self.inp['SFRt0']}
    

    def sfrd_mcmc(self,**kwargs):
        r""" 
        Calculate thin-disk SFR for the MCMC run
        (with some parameters updated). 

        Parameters:
        -----------
        - kwargs: dict
            Parameters to update with their new values. 

        Returns:
        --------
        - 1d-array, 1d-array, nd-array
            Output of SFR.sfrd_sj21_multipeak() method from the jjmodel.funcs: 
            SFRd0 - raw SFR values in Msun / pc**2
            NSFRd0 - same but normalized on area
            Fp0 - if p.pkey==1, contributions of n additional peaks in each time bin.  
        """

        # SFR parameter names
        par_names = ['dzeta','eta','td1','td2','sigmad',
                     'sigmap0','sigmap1','tpk0','tpk1','dtp']
        
        # Initialize with nans
        pars = {par:np.nan for par in par_names}

        # Fill either from namedtuple p (default values)
        # or from kwargs (custom values)
        for name in list(pars.keys()):
            if name in kwargs:
                pars[name] = kwargs[name]
            else:
                if name in ['dzeta','eta','td1','td2','sigmad','dtp']:
                    pars[name] = getattr(self.p,name)
                elif name == 'sigmap0':
                    pars[name] = getattr(self.p,'sigmap')[0]
                elif name == 'sigmap1':
                    pars[name] = getattr(self.p,'sigmap')[1]
                elif name == 'tpk0':
                    pars[name] = getattr(self.p,'tpk')[0]
                elif name == 'tpk1':
                    pars[name] = getattr(self.p,'tpk')[1]

        sfr = SFR()
        out = sfr.sfrd_sj21_multipeak(
            tp,
            tr,
            self.a.t,
            pars['dzeta'],
            pars['eta'],
            pars['td1'],
            pars['td2'],
            pars['sigmad'],
            np.array([pars['sigmap0'],pars['sigmap1']]),
            np.array([pars['tpk0'],pars['tpk1']]),
            pars['dtp'],
            g=self.inp['gd0']
            ) 
        
        SFRd0, NSFRd0, Fp0 = out[0], out[1], out[2]

        return (SFRd0, NSFRd0, Fp0)
    

    def sfrt_mcmc(self,**kwargs):
        r""" 
        Calculate thick-disk SFR for the MCMC run
        (with some parameters updated). 

        Parameters:
        -----------
        - kwargs: dict of floats
            Parameters to update with their new values. 

        Returns:
        --------
        - 1d-array, 1d-array
            Output of SFR.sfrt_sj21() method from the jjmodel.funcs: 
            SFRt0 - raw SFR values in Msun / pc**2
            NSFRt0 - same but normalized on area
        """

        par_names = ['gamma','beta','tt1','tt2','sigmat']
        pars = {par:np.nan for par in par_names}

        for name in list(pars.keys()):
            if name in kwargs:
                pars[name] = kwargs[name]
            else:
                pars[name] = getattr(self.p,name)

        sfr = SFR()
        out = sfr.sfrt_sj21(
            self.a.t[:self.a.jt],
            pars['gamma'],
            pars['beta'],
            pars['tt1'],
            pars['tt2'],
            pars['sigmat'],
            g=self.inp['gt']
        )
        
        return out


    def update_sfr(self, **kwargs):
        r""" 
        Update both thin- and thick-disk SFR given new parameter values. 

        Parameters:
        -----------
        - kwargs: dict of dict of floats
            Parameters to update with their new values. 
            Structured as {'d':dict_sfrd_params,'t':dict_sfrt_params}. 

        Returns:
        --------
        - dict:
            Dictionary with raw and normalized thin/thick-disk and total SFR. 
            This dict is created to match the corresponding tables
            of the inp dict generated upon the JJmodel initialization and 
            to be passed to local_run() function during each MCMC iteration.   
        - dict:
            New thin- and thick-disk SFR arrays under keys 'd' and 't', respectively.
        """

        # Update thin-disk SFR
        sfrd_out = self.sfrd_mcmc(**kwargs['d'])
        SFRd0, NSFRd0, Fp0 = sfrd_out[0], sfrd_out[1], sfrd_out[2]

        # Optional: update thick-disk SFR
        if 't' in kwargs:
            SFRt0, NSFRt0 = self.sfrt_mcmc(**kwargs['t'])
        else:
            SFRt0, NSFRt0 = self.inp['SFRt0'], self.inp['NSFRt0']

        # Update total SFR         
        SFRtot0 = np.concatenate((np.add(SFRd0[:self.a.jt],SFRt0),SFRd0[self.a.jt:]),axis=None)
        NSFRtot0 = SFRtot0/np.mean(SFRtot0)

        # Create output
        names = ['SFRd0', 'NSFRd0', 'SFRtot0', 'NSFRtot0', 'Fp0']
        tables = [SFRd0, NSFRd0, SFRtot0, NSFRtot0, Fp0]

        if 't' in kwargs:
            names += ['SFRt0','NSFRt0']
            tables += [SFRt0, NSFRt0]

        for name, table in zip(names,tables):
            self.inp[name] = table

        out = {'d':self.inp['SFRd0']}
        if 't' in kwargs:
            out['t'] = self.inp['SFRt0']
        
        return self.inp, out


class IMFHandler:
    """Class for IMF manipulation."""

    def __init__(self, p, mres=0.0001, m_step = 0.01):
        r""" 
        Initialization. 

        Parameters:
        -----------
        - p: namedtuple 
            JJ-model parameter object from the model initialization. 
        - mres: float
            Mass resolution, Msun. Default is 0.0001 Msun. 
            Must be small to have correct normalization factors for good continuity 
            across the different IMF slopes. 
        - mstep: float
            Actual step in mass, Msun, to be used for creation of the stellar assemblies. 
            Default is 0.01 Msun. 

        Returns:
        --------
        
        """
        self.p = p

        self.M_low, self.M_up = 0.08, 100      # Msun, lower and upper IMF mass limits
        self.mres = mres
        self.m_step = m_step

        # Linear grid for stellar assemblies
        mass_bins = np.arange(self.M_low,self.M_up + self.m_step, self.m_step)
        self.mass_binsc = mass_bins[:-1] + self.m_step/2


    def imf_mcmc(self,**kwargs):
        r""" 
        Calculate dn/dm values for the adoptem mass grid (m_step parameter). 

        Parameters:
        -----------
        - kwargs: dict
            Parameter names and values of the 4-slope SJ21 IMF to be updated. 

        Returns:
        --------
        - jjmodel.funcs.IMF object 
        """
        par_names = ['a0', 'a1', 'a2', 'a3', 'm0', 'm1', 'm2'] 
        pars = {par:np.nan for par in par_names}

        # Fill either from kwargs or from parameter file
        for name in list(pars.keys()):
            if name in kwargs:
                pars[name] = kwargs[name]
            else:
                pars[name] = getattr(self.p,name)

        # To avoid division by zero
        eps = 1e-5
        a0, a1, a2 = pars['a0'], pars['a1'], pars['a2']
        if pars['a0'] == 2:
            a0 = pars['a0'] + eps
        if pars['a1'] == 2:
            a1 = pars['a1'] + eps
        if pars['a2'] == 2:
            a2 = pars['a2'] + eps

        imf = IMF(self.M_low, self.M_up, mres = self.mres)  
        _, _ = imf.BPL_4slopes(
            a0,
            a1,
            a2,
            pars['a3'],
            pars['m0'],
            pars['m1'],
            pars['m2']
        )
        return imf


    def create_reference_imf(self):
        r""" 
        Calculate dn/dm values for the adopted mass grid (m_step parameter)
        for the 4-slope SJ21 IMF. 

        Parameters:
        -----------
        None

        Returns:
        --------
        - jjmodel.funcs.IMF object 
            IMF function. 
        - tuple (1d-array, 1d-array)
            Mass grid and rerefence dn/dm values
        """
        # Define the IMF 
        imf_ref = IMF(self.M_low,self.M_up,mres=self.mres)        
        _, _ = imf_ref.BPL_4slopes(self.p.a0,self.p.a1,self.p.a2,self.p.a3,
                                   self.p.m0,self.p.m1,self.p.m2)   
        # Calculate dn/dm values
        IMF_ref = [imf_ref.number_stars(mass - self.m_step/2, mass + self.m_step/2) for mass in self.mass_binsc] 
        return imf_ref, (self.mass_binsc, IMF_ref)
    

    def update_imf(self, **kwargs):
        r""" 
        Update IMF given the new parameters. Wrapper for create_reference_imf() method. 

        Parameters:
        -----------
        - kwargs: dict
            Parameter names and values to update. 

        Returns:
        --------
        - jjmodel.funcs.IMF object 
            IMF function. 
        - tuple (1d-array, 1d-array)
            Mass grid and rerefence dn/dm values
        """
        imf = self.imf_mcmc(**kwargs)
        IMF_new = [imf.number_stars(mass - self.m_step/2, mass + self.m_step/2) for mass in self.mass_binsc]

        return imf, (self.mass_binsc, IMF_new)


class IFMRHandler():
    """Class for IFMR manipulation."""

    def __init__(self,a,extend_mf_limits=False):
        r""" 
        Initialization. 

        Parameters:
        -----------
        - a: namedtuple
            JJ-model helper arrays from its initialization. 
        - extend_mf_limits: bool
            If True, IFMR can be used outside of its applicability range (not recommended). 
            By default False.
        """
        self.a = a 
        self.M_sun = 1 # all in solar units

        # Cummings+18 IFMR
        self.segments_cummings = {
            'padova':[ # Padova-calibrated
                {"range": [0.87, 2.80], "a": 0.0873, "a_err": 0.0190, "b": 0.476, "b_err": 0.033},
                {"range": [2.80, 3.65], "a": 0.181, "a_err": 0.041, "b": 0.210, "b_err": 0.131},
                {"range": [3.65, 8.20], "a": 0.0835, "a_err": 0.0144, "b": 0.565, "b_err": 0.073},
            ],
            'mist': [ # MIST-calibrated
                {"range": [0.83, 2.85], "a": 0.080, "a_err": 0.016, "b": 0.489, "b_err": 0.030},
                {"range": [2.85, 3.60], "a": 0.187, "a_err": 0.061, "b": 0.184, "b_err": 0.199},
                {"range": [3.60, 7.20], "a": 0.107, "a_err": 0.016, "b": 0.471, "b_err": 0.077},
            ]
        }

        # Cunningham+23 IFMR
        self.segments_cunningham = [
            {"range": [1.0, 2.50], "a": 0.086, "a_err": 0.003, "b": 0.469, "b_err": 0.004},
            {"range": [2.50, 3.4], "a": 0.1, "a_err": 0.01, "b": 0.43214, "b_err": 0.03},
            {"range": [3.4, 5.03], "a": 0.06, "a_err": 0.01, "b": 0.57, "b_err": 0.05},
            {"range": [5.03, 7.6], "a": 0.17, "a_err": 0.02, "b": 0.0144, "b_err": 0.08},
        ]

        # Miller+26 IFMR,  Full sample fit 1, Eqs. 6-9 
        self.segments_miller = [
            {"range": [0.84, 1.98], "a":0.16, "a_err":0.045,"b":0.394, "b_err":0.063},
            {"range": [1.98, 2.33], "a":-0.155, "a_err":0.171,"b":1.015, "b_err":1.644},
            {"range": [2.33, 3.98], "a":0.174, "a_err":0.026,"b":0.251, "b_err":0.183},
            {"range": [3.98, 8.39], "a":0.075, "a_err":0.011,"b":0.644, "b_err":0.048}
        ] 

        # ElBadry+18 IFMR, err values are placeholders
        self.segments_elbadry = [
            {"range": [0.95, 2.75], "a":0.094, "a_err":0.03,"b":0.412, "b_err":0.063},
            {"range": [2.75, 3.54], "a":0.175, "a_err":0.08,"b":0.191, "b_err":0.044},
            {"range": [3.54, 5.21], "a":0.06, "a_err":0.002,"b":0.597, "b_err":0.183},
            {"range": [5.21, 8.0], "a":0.163, "a_err":0.05,"b":0.066, "b_err":0.048}
        ] 

        # Calculate segments' ranges in Mf for the inverted IFMR
        self.segments_cummings = self._calc_cummings_range_r_(
            extend_mf_limits=extend_mf_limits
            )
        self.cunningham_segments = self._calc_cunningham_range_r_(
            extend_mf_limits=extend_mf_limits
            )
        self.miller_segments = self._calc_miller_range_r_(
            extend_mf_limits=extend_mf_limits
            )
        self.elbadry_segments = self._calc_elbadry_range_r_(
            extend_mf_limits=extend_mf_limits
            )

    def _calc_cummings_range_r_(self,extend_mf_limits=False):
        """Convert Cummings IFMR ranges from the final mass Mf 
        to the initial mass Mini space.
        Adds fields 'range_r' to self.segments_cummings.
        """
        segments_extended = {'padova':[],'mist':[]}

        for cal in ['padova','mist']:
            segments = self.segments_cummings[cal]
            for i, seg in enumerate(segments):
                vals, sigmas, _ = self.cummings(seg['range'],calibration=cal)
                if extend_mf_limits:
                    if i==0:
                        val_upper = vals[1]
                        val_lower = vals[0] - 3*sigmas[0]
                        if val_lower < 0:
                            val_lower = 0 
                    elif i==len(segments) - 1:
                        val_lower = vals[0]
                        val_upper = vals[1] + 3*sigmas[1]
                    else:
                        val_lower, val_upper = vals
                else:
                    val_lower, val_upper = vals
                
                seg['range_r'] = np.round([val_lower,val_upper],2)
                segments_extended[cal].append(seg)

        return segments_extended
    

    def _calc_cunningham_range_r_(self,extend_mf_limits=False):
        """Convert Cunningham IFMR ranges from the final mass Mf 
        to the initial mass Mini space.
        Adds fields 'range_r' to self.segments_cunningham.
        """
        segments = self.segments_cunningham
        for i, seg in enumerate(segments):
            vals, sigmas, _ = self.cunningham(seg['range'])
            if extend_mf_limits:
                if i==0:
                    val_upper = vals[1]
                    val_lower = vals[0] - 3*sigmas[0]
                    if val_lower < 0:
                        val_lower = 0 
                elif i==len(segments) - 1:
                    val_lower = vals[0]
                    val_upper = vals[1] + 3*sigmas[1]
                else:
                    val_lower, val_upper = vals
            else:
                val_lower, val_upper = vals
            
            seg['range_r'] = np.round([val_lower,val_upper],2)

        return segments
    
    def _calc_miller_range_r_(self,extend_mf_limits=False):
        """Convert Miller IFMR ranges from the final mass Mf 
        to the initial mass Mini space.
        Adds fields 'range_r' to self.segments_miller.
        """
        segments = self.segments_miller
        for i, seg in enumerate(segments):
            vals, sigmas, _ = self.miller(seg['range'])
            if extend_mf_limits:
                if i==0:
                    val_upper = vals[1]
                    val_lower = vals[0] - 3*sigmas[0]
                    if val_lower < 0:
                        val_lower = 0 
                elif i==len(segments) - 1:
                    val_lower = vals[0]
                    val_upper = vals[1] + 3*sigmas[1]
                else:
                    val_lower, val_upper = vals
            else:
                val_lower, val_upper = vals
            
            seg['range_r'] = np.round([val_lower,val_upper],2)

        return segments
    
    def _calc_elbadry_range_r_(self,extend_mf_limits=False):
        """Convert ElBadry IFMR ranges from the final mass Mf 
        to the initial mass Mini space.
        Adds fields 'range_r' to self.segments_elbadry.
        """
        segments = self.segments_elbadry
        for i, seg in enumerate(segments):
            vals, sigmas, _ = self.elbadry(seg['range'])
            if extend_mf_limits:
                if i==0:
                    val_upper = vals[1]
                    val_lower = vals[0] - 3*sigmas[0]
                    if val_lower < 0:
                        val_lower = 0 
                elif i==len(segments) - 1:
                    val_lower = vals[0]
                    val_upper = vals[1] + 3*sigmas[1]
                else:
                    val_lower, val_upper = vals
            else:
                val_lower, val_upper = vals
            
            seg['range_r'] = np.round([val_lower,val_upper],2)

        return segments
    

    def update_ifmr(self,calibration='padova',**kwargs):
        r""" 
        Calculates new Cummings-like IFMR (3-slope) given the new parameters. 

        Parameters:
        -----------
        - calibration: string
            Can be 'padova' or 'mist' - depending on which Cummings+18 IFMR option 
            we stars with. 
        - kwargs: dict
            Parameter names and values to be updated. 

        Returns:
        --------
        None
        """
        
        # Update parameters in self.segments_cummings
        for par in list(kwargs.keys()):
            if par == 'm_br1':
                self.segments_cummings[calibration][0]['range'][1] = kwargs[par]
                self.segments_cummings[calibration][1]['range'][0] = kwargs[par]
            if par == 'm_br2':
                self.segments_cummings[calibration][1]['range'][1] = kwargs[par]
                self.segments_cummings[calibration][2]['range'][0] = kwargs[par]
            if par == 'alpha1':
                self.segments_cummings[calibration][0]['a'] = kwargs[par]
            if par == 'alpha2':
                self.segments_cummings[calibration][1]['a'] = kwargs[par]
            if par == 'alpha3':
                self.segments_cummings[calibration][2]['a'] = kwargs[par]
            if par == 'b1':
                self.segments_cummings[calibration][0]['b'] = kwargs[par]

        # Update Mini ranges in case if Mf breakpoints were modified 
        # Calculate inverse ranges from the updated relation.
        for seg in self.segments_cummings[calibration]:
            values, _, _ = self.cummings(seg['range'], calibration=calibration)
            seg['range_r'] = np.round(np.asarray(values, dtype=float), 2)

        # Ensure continuity by updating offsets
        b1 = self.segments_cummings[calibration][0]['b']
        a1 = self.segments_cummings[calibration][0]['a']
        a2 = self.segments_cummings[calibration][1]['a']
        a3 = self.segments_cummings[calibration][2]['a']
        m_br1 = self.segments_cummings[calibration][0]['range'][1]
        m_br2 = self.segments_cummings[calibration][1]['range'][1]

        self.segments_cummings[calibration][1]['b'] = float(np.round(b1 - m_br1*(a2 - a1),3))
        self.segments_cummings[calibration][2]['b'] = float(np.round(b1 - m_br1*(a2 - a1) - m_br2*(a3 - a2),3))


    def _segmented_forward_(self, segment, Mini):
        r"""
        Apply IFMR forwards: WD final masses from progenitors' initial masses. 

        Parameters:
        -----------
        - segment: list of dicts
            IFMR parameters to use. Structure (also see self.segments_cunningham):
            [{'range':minmax_mini,
              'a':slope_value,
              'b':offset_value,
              'a_err':slope_value_1sigma,
              'b':offset_value_1sigma
              },
              ...
            ]
        - Mini: 1d-array or float
            Grid of initial masses (or a single mass value) of WD progenitors.  

        Returns:
        --------
        - 1d-array or float
            WD final mass(es). 
        - 1d-array or float
            Uncertainty in final mass(es). 
        - 1d-array or float
            Uncertainty in initial mass(es). 
        """
        x = Mini
        if np.isscalar(Mini):
            x = [Mini]

        Mf, sigma_Mf, sigma_Mini = [], [], []
        for mini in x:
            if mini < segment[0]['range'][0] or mini > segment[-1]['range'][-1]:
                Mf.append(np.nan)
                sigma_Mf.append(np.nan)
                sigma_Mini.append(np.nan)
            else:
                # Find the right segment
                for seg in segment:
                    if (seg['range'][0] <= mini) and (mini <= seg['range'][1]):
                        mf = seg["a"] * mini + seg["b"] * self.M_sun
                        sigma_mf = np.sqrt((mini * seg["a_err"])**2 + (self.M_sun * seg["b_err"])**2)
                        sigma_mini2 = (mini*seg['a_err']/seg['a'])**2 + (seg['b_err']*self.M_sun/seg['a'])**2
                        sigma_mini = np.sqrt(sigma_mini2)
                        Mf.append(mf)
                        sigma_Mf.append(sigma_mf)
                        sigma_Mini.append(sigma_mini)
                        break

        output = ((Mf[0], sigma_Mf[0], sigma_Mini[0]) if np.isscalar(Mini) else (np.array(Mf), np.array(sigma_Mf), np.array(sigma_Mini)))
        return output


    def _segmented_backward_(self, segment, Mf):
        r"""
        Apply IFMR backwards: Progenitors' initial masses from WD final masses. 

        Parameters:
        -----------
        - segment: list of dicts
            IFMR parameters to use. Structure (also see self.segments_cunningham):
            [{'range_r':minmax_mf,
              'a':slope_value,
              'b':offset_value,
              'a_err':slope_value_1sigma,
              'b':offset_value_1sigma
              },
              ...
            ]
        - Mf: 1d-array or float
            Grid of WD final masses (or a single mass value).  

        Returns:
        --------
        - 1d-array or float
            WD progenitors' initial mass(es). 
        - 1d-array or float
            Uncertainty in initial mass(es). 
        """
        x = Mf
        if np.isscalar(Mf):
            x = [Mf]
        
        Mini, sigma_Mini = [], []
        for mf in x:
            if mf < segment[0]['range_r'][0] or mf > segment[-1]['range_r'][-1]:
                Mini.append(np.nan)
                sigma_Mini.append(np.nan)
            else:
                for seg in segment:
                    if (seg['range_r'][0] <= mf) and (mf <= seg['range_r'][1]):
                        mini = (mf - seg["b"] * self.M_sun)/seg["a"]
                        sigma_mini = np.sqrt(((mf - seg["b"]*self.M_sun)/seg["a"] * seg["a_err"])**2 +\
                                            (self.M_sun * seg["b_err"])**2) / seg["a"]
                        
                        # Check for unphysical results due to extension of applicability range
                        if mini < 0.08:
                            mini = np.nan
                            sigma_mini = np.nan
                        Mini.append(mini)
                        sigma_Mini.append(sigma_mini)
                        break

        output = ((Mini[0], sigma_Mini[0]) if np.isscalar(Mf) else (np.array(Mini), np.array(sigma_Mini)))
        return output


    def cummings(self, Mini, calibration='padova'):
        """
        Forward application of Cummings-like (3-slope) IFMR
        """
        segment = self.segments_cummings[calibration]
        return self._segmented_forward_(segment,Mini)
    

    def cunningham(self, Mini):
        """
        Forward application of Cunningham+23 IFMR
        """
        segment = self.segments_cunningham
        return self._segmented_forward_(segment,Mini)

    def miller(self, Mini):
        """
        Forward application of Miller+26 IFMR
        """
        segment = self.segments_miller
        return self._segmented_forward_(segment,Mini)
    

    def elbadry(self, Mini):
        """
        Forward application of ElBadry+23 IFMR
        """
        segment = self.segments_elbadry
        return self._segmented_forward_(segment,Mini)
    

    def cummings_r(self, Mf, calibration='padova'):
        """
        Backward application of Cummings-like (3-slope) IFMR
        """
        segment = self.segments_cummings[calibration]
        return self._segmented_backward_(segment,Mf)
    

    def cunningham_r(self, Mf):
        """
        Backward application of Cunningham+23 IFMR
        """
        segment = self.segments_cunningham
        return self._segmented_backward_(segment,Mf)


    def miller_r(self, Mf):
        """
        Backward application of Miller+26 IFMR
        """
        segment = self.segments_miller
        return self._segmented_backward_(segment,Mf)
    

    def elbadry_r(self, Mf):
        """
        Backward application of ElBadry+23 IFMR
        """
        segment = self.segments_elbadry
        return self._segmented_backward_(segment,Mf)


    def apply_ifmr_scatter(self, N_ref, Mini, sigma_Mini):
        r"""
        Returns new stellar assemblies' number desity column N_new [Msun/pc**2]
        calculated with IFMR scatter taken ino account.

        Parameters:
        ----------- 
        - N_ref: 1d-array 
            Number desities corresponding to the deterministic IFMR. 
        - Mini: 1d-array
            Stellar assemblies' initial masses. 
        - sigma_Mini
            Stellar assemblies' initial mass uncertainties. 

        Returns:
        --------
        - 1d-array
            Number density column with IFMR scatter effect. 
        """
        
        n  = len(Mini)
        N_new = np.zeros_like(N_ref, dtype=float)

        # Precompute bin centers and widths
        Mini_edges = np.zeros((n + 1))
        Mini_edges[0], Mini_edges[-1] = Mini[0], Mini[-1]
        Mini_edges[1:-1] = [np.mean([Mini[i],Mini[i+1]]) for i in np.arange(n - 1)]

        # Precompute edges
        # compute integrated Gaussian probabilities for each destination bin
        # bin integral = CDF((b-mu)/sigma) - CDF((a-mu)/sigma)
        a_edges = Mini_edges[:-1]
        b_edges = Mini_edges[1:]

        # Compute CDF values in one go:
        cdf_a = norm.cdf((a_edges[None,:] - Mini[:,None]) / sigma_Mini[:,None])
        cdf_b = norm.cdf((b_edges[None,:] - Mini[:,None]) / sigma_Mini[:,None])
        bin_integrals = cdf_b - cdf_a
        
        # Apply scatter
        for j in range(n):
            mu = Mini[j]
            sigma = sigma_Mini[j]

            # destination bin indices within ±3σ
            nsig = 3
            lo = mu - nsig * sigma
            hi = mu + nsig * sigma
            k0 = max(np.searchsorted(Mini_edges, lo) - 1, 0)
            k1 = min(np.searchsorted(Mini_edges, hi), n)

            if k0 >= k1:
                continue

            # bin weights for row j
            bin_weights = bin_integrals[j, k0:k1]
            #bin_weights /= bin_weights.sum()

            # accumulate
            N_new[j] += np.sum(N_ref[k0:k1] * bin_weights)

        return N_new


    def add_sigma_mini_column(self,pop_tabs,ind_wds,calibration='padova'):
        r""" 
        Calculate the uncertainty in initial mass for the WD stellar assemblies. 

        Parameters:
        -----------
        - pop_tabs: dict of pandas.DataFrame objects or Astropy tables
            Stellar assemblies tables for thin/thick disk and halo:
            {'d':table1, 't':table2, 'sh':table3}. 
        - ind_wds: dict of dict of ints
            Indices of DA and DB WDs in pop_tabs. Structure: 
            {'d':{'da':da_wd_indices,'db':db_wd_indices,},...}. 
        - calibration: string
            Cummings+18 calibration to use: 'padova' or 'mist'. 
            Note that at this point definition of this IFMR might have already been changed
            and calibration refers to the dictionary with IFMR parameters you applied changes to. 

        Returns:
        --------
        - dict of pandas.DataFrame objects or Astropy tables
            Stellar assemblies tables with a new column 'sigma_Mini'. 
        """
                
        labels = list(pop_tabs.keys())

        for label in labels:

            tab = pop_tabs[label]
            ind_wd = ind_wds[label]

            tab['sigma_Mini'] = [np.nan for _ in np.arange(len(tab['Mini']))] 
            for ind in ind_wd:
                _, sigma_Mini = self.cummings_r(tab['Mf'][ind],calibration=calibration)                            
                tab['sigma_Mini'][ind] = sigma_Mini

        return pop_tabs
    

    def extract_mini_grid(self, pop_tabs, ind_wd):
        '''
        Get unique values of Mini and sigma_Mini for WD populations. 
        
        Parameters:
        -----------
        - pop_tabs: dict of pandas.DataFrame objects or Astropy tables
            Stellar assemblies tables for thin/thick disk and halo:
            {'d':table1, 't':table2, 'sh':table3}. 
        - ind_wd: dict of dict of ints
            Indices of DA and DB WDs in pop_tabs. Structure: 
            {'d':{'da':da_wd_indices,'db':db_wd_indices,},...}. 

        Returns:
        --------
        - 1d-array
            Array with unique values of WD progenitor masses Mini 
            from the total(!) table: d+t+sh. 
        - 1d-array
            Unique values of Mini uncertainties corresponding to the unique Mini array. 

        '''
        mini_all = []
        sigma_all = []

        for tab_key in pop_tabs:
            tab = pop_tabs[tab_key]
            inds_list = ind_wd[tab_key]
            for ind in inds_list:
                mini_all.append(tab['Mini'][ind])
                sigma_all.append(tab['sigma_Mini'][ind])

        mini_all = np.concatenate(mini_all)
        sigma_all = np.concatenate(sigma_all)

        # extract unique Mini and corresponding sigma by first occurrence
        mini_grid, idx = np.unique(mini_all, return_index=True)
        sigma_grid = sigma_all[idx]

        return mini_grid, sigma_grid
    

    def apply_ifmr_scatter_poptabs_exact(self,pop_tabs,ind_wd):
        r"""
        Apply IFMR scatter - exact calculation approach.

        Parameters:
        -----------
        - pop_tabs: dict of pandas.DataFrame objects or Astropy tables
            Stellar assemblies tables for thin/thick disk and halo:
            {'d':table1, 't':table2, 'sh':table3}. 
            Tables must already include column 'sigma_Mini', 
            see method self.add_sigma_mini_column().
        - ind_wd: dict of dict of ints
            Indices of DA and DB WDs in pop_tabs. Structure: 
            {'d':{'da':da_wd_indices,'db':db_wd_indices,},...}. 

        Returns:
        --------
        - dict of pandas.DataFrame objects or Astropy tables
            Updated tables with a new number density [Msun/pc**2] column N_sm
            calculated with accounting for IFMR scatter (affects only WD population). 
        """

        jd = self.a.jd  # number of thin-disk single-age populations
        dt = self.a.t[1] - self.a.t[0] # time/age resolution

        tables = pop_tabs.copy()

        # Find unique values of WD progenitors' initial masses and their uncertainties
        mini_grid, mini_sigma_grid = self.extract_mini_grid(pop_tabs,ind_wd)

        # Apply IFMR scatter effect
        #---------------------------------------
        # Iterate over thin/ thick disk and halo
        for tab,indices in zip(tables.values(),ind_wd.values()): 
            
            # Initialize a new number density column with deterministic values
            tab['N_sm'] = tab['N']

            # Iterate over ('DA', 'DB') WD types
            for ind in indices: 
                
                # Calculate age indices for selected WD type
                indt = np.array(np.round(tab['age'][ind],3)//dt,dtype=int)

                # Iterate over all ages
                for j in range(jd):
                    
                    # Select all rows with a given age
                    ind_age = np.where(indt==j)[0]
                    rows = ind[ind_age]
                    
                    # Sum populations for this age over the unique mass grid 
                    # (what N is predicted for this age-mass bin)
                    indm = np.searchsorted(mini_grid, tab['Mini'][rows])
                    N_sum = np.bincount(indm, weights=tab['N'][rows], minlength=len(mini_grid))

                    # Apply IFMR scatter
                    N_sum_sm = self.apply_ifmr_scatter(N_sum,mini_grid,mini_sigma_grid)

                    # Redistribute proportional to original N in that age bin
                    for i in range(len(mini_grid)):
                        mask = indm == i
                        if N_sum[i] > 0:
                            tab['N_sm'][rows[mask]] = N_sum_sm[i] * tab['N'][rows[mask]] / N_sum[i]
                        else:
                            tab['N_sm'][rows[mask]] = 0

                    # Renormalize to keep total number the same?
                    #if sum(tab['N_sm'][rows]) > 0:
                    #    norm_corr = sum(tab['N'][rows])/sum(tab['N_sm'][rows])
                    #    tab['N_sm'][rows] *= norm_corr

        return tables
    

    def apply_ifmr_scatter_poptabs_reduced(self,pop_tabs,ind_wd):
        r"""
        Application of the IFMR scatter -- reduced (faster) approach. 
        Tests showed that there is essenitally no difference between the exact and 
        reduced approaches, so use this method to make calculation faster. 

        Parameters:
        -----------
        - pop_tabs: dict of pandas.DataFrame objects or Astropy tables
            Stellar assemblies tables for thin/thick disk and halo:
            {'d':table1, 't':table2, 'sh':table3}. 
            Tables must already include column 'sigma_Mini', 
            see method self.add_sigma_mini_column().
        - ind_wd: dict of dict of ints
            Indices of DA and DB WDs in pop_tabs. Structure: 
            {'d':{'da':da_wd_indices,'db':db_wd_indices,},...}. 

        Returns:
        --------
        - dict of pandas.DataFrame objects or Astropy tables
            Updated tables with a new number density [Msun/pc**2] column N_sm
            calculated with accounting for IFMR scatter (affects only WD population). 
        """
            
        tables = pop_tabs.copy()

        # Find unique values of WD progenitors' initial masses and their uncertainties
        mini_grid, mini_sigma_grid = self.extract_mini_grid(pop_tabs,ind_wd)

        # Apply IFMR scatter effect
        #---------------------------------------
        # Iterate over thin/ thick disk and halo
        for tab,indices in zip(tables.values(),ind_wd.values()):  
            
            # Initialize a new number density column with deterministic values
            tab['N_sm'] = tab['N']

            # Iterate over ('DA', 'DB') WD types
            for ind in indices: 
                
                # Select rows with WDs 
                rows = ind
                
                # Sum all rows over the unique mass grid (and across different ages!)
                indm = np.searchsorted(mini_grid, tab['Mini'][rows])
                N_sum = np.bincount(indm, weights=tab['N'][rows], minlength=len(mini_grid))

                # Apply IFMR scatter
                N_sum_sm = self.apply_ifmr_scatter(N_sum,mini_grid,mini_sigma_grid)

                # Redistribute proportional to original N
                for i in range(len(mini_grid)):
                    mask = indm == i
                    if N_sum[i] > 0:
                        tab['N_sm'][rows[mask]] = N_sum_sm[i] * tab['N'][rows[mask]] / N_sum[i]
                    else:
                        tab['N_sm'][rows[mask]] = 0

                # Renormalize to keep total number the same?
                #if sum(tab['N_sm'][rows]) > 0:
                #    norm_corr = sum(tab['N'][rows])/sum(tab['N_sm'][rows])
                #    tab['N_sm'][rows] *= norm_corr

        return tables
    

class PopHandler:
    """Class for stellar assemblies manipulation."""

    def __init__(self, p, a, inp):
        """Initialization
        
        Parameters:
        -----------
        - p: namedtuple
            JJ-model parameter object from the model initialization. 
        - a: namedtuple
            JJ-model helpers array object from the model initialization.
        - inp: dict of arrays 
            JJ-model input array aobject from the model initialization.
        """
        self.a = a
        self.p = p
        self.inp = inp


    def get_age_mass_idx(self, pop_tabs, mass_binsc):
        r""" 
        Calculate time and initial mass indices of the input table rows
        in the model time and mass grids. 

        Parameters:
        -----------
        - pop_tabs: dict of pandas.DataFrame objects or Astropy tables
            Stellar assemblies tables for thin/thick disk and halo:
            {'d':table1, 't':table2, 'sh':table3}. 
        - mass_binc: 1d-array
            IMF mass grid (bin centers). 

        Returns:
        --------
        - dict of 1d-arrays
            Time indices corresponding to the model time grid a.t. 
            Structure: {'d':ind_d,'t':ind_t} (halo not added because it's a single age). 
        - dict of 1d-arrays
            Initial mass indices corresponding to the provided mass grid. 
            Structure: {'d':ind_d,'t':ind_t,'sh':ind_sh}. 
        """
        def nearest(grid, values):
            grid = np.asarray(grid)
            values = np.asarray(values)

            right = np.searchsorted(grid, values)
            right = np.clip(right, 0, len(grid) - 1)
            left = np.maximum(right - 1, 0)

            use_left = (np.abs(values - grid[left]) <= np.abs(values - grid[right]))

            indices = np.where(use_left, left, right)
            return np.where(np.isfinite(values), indices, 0)

        indt = {}
        indm = {}

        for label, grid in [('d', self.a.t), ('t', self.a.t[:self.a.jt])]:
            times = tp - np.asarray(pop_tabs[label]['age'])
            indt[label] = nearest(grid, times)

        for label in ['d', 't', 'sh']:
            indm[label] = nearest(mass_binsc, pop_tabs[label]['Mini'])

        return indt, indm


    def get_wd_idx(self, pop_tabs):
        r""" 
        Get positions of WDs in the stellar assemblies' tables. 

        Parameters:
        -----------
        - pop_tabs: dict of pandas.DataFrame objects or Astropy tables
            Stellar assemblies tables for thin/thick disk and halo:
            {'d':table1, 't':table2, 'sh':table3}. 

        Returns:
        --------
        - dict of 1d-arrays of int
            DA WD positions. Structure: {'d':wd_da_ind_d,'t':wd_da_ind_t,'sh':wd_da_ind_sh}
        - dict of 1d-arrays of int
            DB WD positions. Structure: {'d':wd_db_ind_d,'t':wd_db_ind_t,'sh':wd_db_ind_sh}
        """
        ind_da, ind_db = {}, {}
        for label in ['d','t','sh']:
            ind_da[label] = np.where(pop_tabs[label]['phase']==10)[0]
            ind_db[label] = np.where(pop_tabs[label]['phase']==11)[0]
        return (ind_da, ind_db)


    def display_wd_stats(self,mode_pop,ind_wd):
        r""" 
        Print basic statistics of the WD stellar assemblies. 
        - Number of DA and DB assemblies in thin/thick disk and halo tables. 

        Parameters:
        -----------
        - mode_pop: string
            Type of modeled populations. 
        - ind_wd: dict of 2d-array of int
            Structure: {'d':[wd_da_ind,wd_db_ind],'t':..,'sh':..}.     
        """
        print('\n')
        if mode_pop in ('wd','tot','pops_joined'):
            print('WD indices d:\t',len(ind_wd['d'][0]),len(ind_wd['d'][1]))
            print('WD indices t:\t',len(ind_wd['t'][0]),len(ind_wd['t'][1]))
            print('WD indices sh:\t',len(ind_wd['sh'][0]),len(ind_wd['sh'][1]))
    

    def make_wd_idx_dict(self, pop_tabs):
        r""" 
        Obtain WD indices in the dict format. 

        Parameters:
        -----------
        - pop_tabs: dict of pandas.DataFrame objects or Astropy tables
            Stellar assemblies tables for thin/thick disk and halo:
            {'d':table1, 't':table2, 'sh':table3}. 

        Returns:
        --------
        - dict of 2d-array of int
            Positions of DA and DB WDs in the stellar assemblies' tables. 
            Structure: {'d':[wd_da_ind,wd_db_ind],'t':..,'sh':..}. 
        """
        ind_da, ind_db = self.get_wd_idx(pop_tabs)
        ind_wd = {label: [ind_da[label],ind_db[label]] for label in pop_tabs.keys()}
        return ind_wd
    

    def separate_wd_ms_idx(self,pop_tabs,ind_wd):
        r""" 
        Get separate lists of indices for WD and other stars. 

        Parameters:
        -----------
        - pop_tabs: dict of pandas.DataFrame objects or Astropy tables
            Stellar assemblies tables for thin/thick disk and halo:
            {'d':table1, 't':table2, 'sh':table3}. 
        - ind_wd: dict of 2d-arrays of int
            Positions of WDs in the provided tables, 
            output of self.make_wd_idx_dict() method. 

        Returns:
        --------
        - dict of 1d-aaray of int
            Positions of all(DA+DB!) WDs. Same structure as of pop_tabs. 
        - dict of 1d-aaray of int
            Positions of all other populations. Same structure as of pop_tabs. 
        """
        idx_wd = {}
        idx_ms = {}

        for label in pop_tabs.keys():
            idx_wd[label] = np.concatenate(ind_wd[label]) 
            idx_ms[label] = np.array([i for i in range(len(pop_tabs[label])) if i not in idx_wd])

        return idx_wd, idx_ms
    

    def split_into_pops(self,pop_tabs):
        r""" 
        Splits stellar assemblies tables into CMD-defined populations. 
        Separation is done in the (G-GRP,M_G) space. 

        Parameters:
        -----------
        - pop_tabs: dict of pandas.DataFrame objects or Astropy tables
            Stellar assemblies tables for thin/thick disk and halo:
            {'d':table1, 't':table2, 'sh':table3}. 

        Returns:
        --------
        - dict of dicts of ints
            Indices of the populations in the provided tables. 
            Structure: {'pop_name':{'d':ind_pop_d,'t':ind_pop_t,'sh':ind_pop_sh},...}.
        """
        labels = list(pop_tabs.keys())
        pops = ['wd','g','ums','ms','lms']
        pop_tabs_split = {pop:{label:[] for label in labels} for pop in pops}

        for label in labels:
            y = pop_tabs[label]['G_Gaia_DR3']
            x = pop_tabs[label]['G_Gaia_DR3'] - pop_tabs[label]['G_RP_Gaia_DR3']

            pop_tabs_split['wd'][label] = np.where(y > 6*x + 9.3)[0]
            pop_tabs_split['ums'][label] = np.where((y < 4.3) & (y > 35*x - 15))[0]
            pop_tabs_split['g'][label] = np.where((y < 4.3) & (y <= 35*x - 15))[0]
            pop_tabs_split['ms'][label] = np.where((y >= 4.3) & (y < 8.5) & (y <= 6*x + 9.3))[0]
            pop_tabs_split['lms'][label] = np.where((y >= 8.5) & (y <= 6*x + 9.3))[0]

        return pop_tabs_split
    

    def combine_pops_indices(self, pop_tabs, idx_pop):
        r""" 
        Converts CMD-population indices from dict of dicts into dict format. 

        Parameters:
        -----------
        - pop_tabs: dict of pandas.DataFrame objects or Astropy tables
            Stellar assemblies tables for thin/thick disk and halo:
            {'d':table1, 't':table2, 'sh':table3}. 
        - idx_pop: dict of dicts of ints
            Indices of the populations in the provided tables. 
            Structure: {'pop_name':{'d':ind_pop_d,'t':ind_pop_t,'sh':ind_pop_sh},...}.

        Returns:
        --------
        - dict of ints
            Indices of the populations in the provided stacked(!) d+t+sh tables. 
            Structure: {'pop_name':ind_pop_dtsh,...}.
        """
        len_d = len(pop_tabs['d']['N'])
        len_t = len(pop_tabs['t']['N'])

        pops = idx_pop.keys()
        combined_idx = {pop:[] for pop in pops}

        for pop in pops:
            combined_idx[pop] = np.concatenate((
                idx_pop[pop]['d'],
                idx_pop[pop]['t'] + len_d,
                idx_pop[pop]['sh'] + len_d + len_t
            ))

        return combined_idx
    

    def create_pop_id_column(self,pop_tabs,idx_pop):
        r""" 
        Creates a single label column with an integer number specifying 
        to which of the CMD-defined population the current row belongs.  
        Populations and indices: 1 - WD, 2 - G, 3 - UMS, 4 - MS, 5 - LMS. 

        Parameters:
        -----------
        - pop_tabs: dict of pandas.DataFrame objects or Astropy tables
            Stellar assemblies tables for thin/thick disk and halo:
            {'d':table1, 't':table2, 'sh':table3}. 
        - idx_pop: dict of ints
            Indices of the populations in the provided stacked(!) d+t+sh tables. 
            Structure: {'pop_name':ind_pop_dtsh,...}.

        Returns:
        --------
        - 1d-array of ints
            Column with the CDM-population index corresponding to stacked(!) d+t+sh 
            input tables. 
        """
        ids = {'wd':1,'g':2,'ums':3,'ms':4,'lms':5}

        n = len(pop_tabs['d']['N']) + len(pop_tabs['t']['N']) + len(pop_tabs['sh']['N'])
        pop_id = np.zeros(n,dtype=int)
        
        for pop in idx_pop.keys():
            pop_id[idx_pop[pop]] = ids[pop]

        return pop_id


    def get_gmag_idx(self, pop_tabs, mag_bins):
        r""" 
        Get an index column corresponding to positions of the input tables' rows
        in the given absolute magnitude grid.  

        Parameters:
        -----------
        - pop_tabs: dict of pandas.DataFrame objects or Astropy tables
            Stellar assemblies tables for thin/thick disk and halo:
            {'d':table1, 't':table2, 'sh':table3}. 
        - mag_bins: 1d-array of floats
            M_G-grid, mag. 

        Returns:
        --------
        - dict of 1d-array of ints
            Structure: {'d':ind_mg_d,'t':ind_mg_t,'sh':ind_mg_sh}. 
        """    
        mag_step = mag_bins[1] - mag_bins[0]
        ind_mag = {}

        for label in pop_tabs.keys():
            gmag = pop_tabs[label]['G_Gaia_DR3']
            ind_mag[label] = np.array((gmag - mag_bins[0])//mag_step,dtype=int)

        return ind_mag


    def get_dlim_idx(self, pop_tabs, mag_bins, d_low_bins, d_up_bins):
        r""" 
        Find min and max distance each row in stellar assemblies tables correspond to
        based on its ansolute G magnitude. Distance limits must be precalculated 
        in advance (from the data-driven min-max apparent G cut, see SJ21)
        for the same magnitude grid as provided here. 

        Parameters:
        -----------
        - pop_tabs: dict of pandas.DataFrame objects or Astropy tables
            Stellar assemblies tables for thin/thick disk and halo:
            {'d':table1, 't':table2, 'sh':table3}. 
        - mag_bins: 1d-array of floats
            M_G-grid, mag. Binning used for calculating min and max distances. 
        - d_low_bins: 1d-array of floats
            Mimimal distances where stars of M_G from the corresponding mag_bins grid
            are fainter than some assumed G_min, pc. 
        - d_up_bins: 1d-array of floats
            Maximal distances where stars of M_G from the corresponding mag_bins grid
            are brighter than some assumed G_max, pc. 

        Returns:
        --------
        - dict of 2d-arrays of floats
            Structure: {'d':[dmin_column,dmax_column],...}.
        """
        # Find which M_G bin each row in tabs corresponds to
        ind_mag  = self.get_gmag_idx(pop_tabs, mag_bins)

        ind_dlim = {}
        for label in pop_tabs.keys():
            ind_dlim[label] = np.zeros((len(pop_tabs[label]['N']),2),dtype=int)
            # Select corresponding mg-row from min-max distance grid 
            ind_dlim[label][:,0] = np.array(d_low_bins, dtype=int)[ind_mag[label]]
            ind_dlim[label][:,1] = np.array(d_up_bins, dtype=int)[ind_mag[label]]

        return ind_dlim


    def get_vz_mag(self, pop_tabs, d_mg_lim, vz_grid):
        r"""
        Calculate Volume at each z for each magnitude bin. 
        Magnitude dependence models Gaia completeness via distance limits.
        
        Parameters:
        -----------
        - pop_tabs: dict of pandas.DataFrame objects or Astropy tables
            Stellar assemblies tables for thin/thick disk and halo:
            {'d':table1, 't':table2, 'sh':table3}. 
        - d_mg_lim: 2d-array
            Table with information on completeness. 
            Columns: 
            -- M_G-bin lower boundary, mag  
            -- M_G-bin upper boundary, mag
            -- Distance-bin lower boundary (stars become not too bright), 
            -- Distance-bin upper boundary (stars not too faint yet).
        - vz_grid: 2d-array
            Precalculated volume grid, pc**3. 
            Local sphere of radius d_max + cut RSun - dR < R < RSun + dR, dR = 150 pc, 
            Rows: bin centers of z_grid = (1,150) pc, step 5 pc; 
            Columns: d_max = (1, 1000) pc, step 1 pc
            values in column-row cells give volume of of that z-slice given 
            the min-max distance limitations and the adopted volume geometry. 

        Returns:
        --------
        vz_mag: dict of 1d-arrays of floats
            For each thin disk, thick disk and halo stellar assembly: 
            Volume accupied by the assembly at each z corresponsing based on 
            the adopted volume geometry and completeness limitations 
            (i.e. its absolute magnitude). 
        """

        # Define magnitude bins from completeness table (take first column)
        mag_bins = d_mg_lim[0]

        # Extract distance limits for magnitude grid 
        # Cut at 1 kpc as we don't model beyond that
        d_low_bins = [int(d) if d <= 1000 else 1000 for d in d_mg_lim[2]]
        d_up_bins = [int(d) if d <= 1000 else 1000 for d in d_mg_lim[3]]

        # Find limiting distances for each population in pop_tabs
        # These values can be used as indices in vz_grid because 
        # distance step in vz_grid is 1 pc
        ind_dlim_mag = self.get_dlim_idx(pop_tabs, mag_bins, d_low_bins, d_up_bins)

        # Add a column with zeros to use lmiting distances as indices 
        # (vz grid was calculed for d from 1 pc, not from 0). 
        vz_grid = np.concatenate((np.zeros((1,vz_grid.shape[1])), vz_grid), axis=0)

        # Calculate volume for each MW component and each population
        vz_mag = {}
        for label in ind_dlim_mag.keys():
            n_pops = ind_dlim_mag[label].shape[0]
            vz_mag[label] = np.zeros((n_pops, vz_grid.shape[1]))

            for i in range(n_pops):
                # Subtract volume at d_min (inner hole due to bright limit)
                vz_mag[label][i,:] = vz_grid[ind_dlim_mag[label][i,1],:] - vz_grid[ind_dlim_mag[label][i,0],:]
        
        return vz_mag   


    def create_reference_pop_tabs(self, imf_ref, mode_iso, phot, phot_cut=True, **kwargs):
        r""" 
        Generate tables of the stellar assemblies for thin/thick disk and halo
        for the solar neighborhood. 

        Parameters:
        -----------
        - imf_ref: jjmodel.IMF object
            IMF function. 
        - mode_iso: dict of strings
            Isochrone set to use, 'main' - 'Padova', 'MIST', or 'BaSTI',
            'wd' - 'LPCODE', 'Montreal', or 'BaSTI'
        - phot: str
            Photometric system. 
        - phot_cut: bool
            If True, only stars with M_G < 25 and G - GRP < 1.65 are selected. 
            By default is True. 
        - kwargs: dict
            Optional keyword arguments as for jjmodel.populations.StellarAssemblies.stellar_assemblies_r(). 

        Returns:
        --------
            - dict of astropy Tables
            Structure: {'d':table1, 't':table2, 'sh':table3}. 
        """
        # Create and save stellar assemblies
        sa = StellarAssemblies(self.p,self.a,mode_iso,phot)

        sa.create_pops(
            self.p.Rsun,
            self.inp['AMRd0'],
            self.inp['AMRt'],
            self.inp['SFRd0'],
            self.inp['SFRt0'],
            self.p.sigmash,
            imf_ref.number_stars,
            **kwargs
            )

        pop_tabs = {}
        for pop in ['d','t','sh']:
            mode_label = '-'.join(mode_iso.values())
            pop_tabs[pop] = Table.read(
                os.path.join(self.a.T['poptab'],
                ''.join(('SSP_R',str(self.p.Rsun),'_' + pop +'_',mode_label,'.csv')))
                )
            # this is now done with the method create_reference_columns()
            #pop_tabs[pop]['N_ref'] = np.copy(pop_tabs[pop]['N'])

            # add time column for convenience
            pop_tabs[pop]['t'] = tp - pop_tabs[pop]['age'] 

            # remove odd sources
            if phot_cut:
                mask = (pop_tabs[pop]['G_Gaia_DR3'] < 25) &\
                    (pop_tabs[pop]['G_Gaia_DR3'] - pop_tabs[pop]['G_RP_Gaia_DR3'] < 1.65)
                pop_tabs[pop] = pop_tabs[pop][mask]                

        return pop_tabs
    

    def create_reference_columns(self, pop_tabs, colnames_ref):
        r""" 
        Creates copies of columns of the stellar assemblies tables. 
        Copied columns have names with an addition '_ref'. 
        This is needed to ensure that the original columns calculated 
        for the reference IMF, SFR, and IFMR are never modified during the MCMC run. 

        Parameters:
        -----------
        - pop_tabs: dict of pandas.DataFrame objects or Astropy tables
            Stellar assemblies tables for thin/thick disk and halo:
            {'d':table1, 't':table2, 'sh':table3}. 
        - colnames_ref: list of strings
            Names of the columns that should be copied. 

        Returns:
        --------
        - dict of Astropy tables or pandas.DataFrames
            Tables with the added reference columns. 
        """
        pop_tabs_ref = pop_tabs.copy()
        for label in pop_tabs_ref.keys():
            for name in colnames_ref:
                pop_tabs_ref[label][f'{name}_ref'] = pop_tabs_ref[label][name]

        return pop_tabs_ref


    def reset_columns(self,pop_tabs,colnames=['N', 'Mini', 'age', 't_cool']):
        r""" 
        Reset selected columns to their original state using their reference copies
        as created by selt.create_reference_columns(). 

        Parameters:
        -----------
        - pop_tabs: dict of pandas.DataFrame objects or Astropy tables
            Stellar assemblies tables for thin/thick disk and halo:
            {'d':table1, 't':table2, 'sh':table3}. Must include reference columns. 
        - colnames: list of strings
            Names of the columns to reset. By default are:
            ['N', 'Mini', 'age', 't_cool'].
                 
        Returns:
        --------
        - dict of Astropy tables or pandas.DataFrames
            Tables where the specified columns are the same as their reference versions. 
        """
        colnames_copy = colnames.copy()

        # Check that the reference columns exist
        # age_WD column can miss when WD are not modeled
        for col in colnames:
            for label in pop_tabs.keys():
                if f'{col}_ref' not in pop_tabs[label].keys():
                    if col=='t_cool':
                        colnames_copy.remove('t_cool')
                    else:
                        sys.exit(f'Cannot reset columns. No reference column for {col} in [{label}] table!')
        # Reset
        for col in colnames_copy: 
            for label in pop_tabs.keys():
                pop_tabs[label][col] = pop_tabs[label][f'{col}_ref']

        return pop_tabs
    

    def update_pop_tabs(self, opt_params, pop_tabs, **kwargs):
        r""" 
        One of the most important methods. 
        Prescribes recipies for reweighting number densities of the reference tables
        given the new model parameters. Also ages, initial and WD final masses can be updated.
        
        Parameters:
        -----------
        - opt_params: dict of dicts of strings
            Hierarchically structured dictionary with the new parameter values. 
            Output of ParHandler.fill_param_struct() method. 
        - pop_tabs: dict of pandas.DataFrame objects or Astropy tables
            Stellar assemblies tables for thin/thick disk and halo:
            {'d':table1, 't':table2, 'sh':table3}. Must also include reference columns
            for 'N', 'Mini', 'age', 't_cool' (the latter can be skipped if WDs are not modeled). 
        - kwargs: dict
            Optional keyword arguments. Should include:
            - SFR_ref: dict
                Reference SFR. 
            - IMF_new
            - IMF_ref
            - indm
            - ifmr_handler
            - msage_handler

        Returns:
        --------
        
        """
        # In MCMC, no copy of pop_tabs is created to save the memory and execution time. 
        # But for testing, it was useful
        #pop_tabs = copy.deepcopy(pop_tabs) # do not uncomment if you use MCMC!

        # Just in case, reset columns ['N', 'Mini', 'age', 't_cool'] 
        # to their original versions corresponding to the model predictions 
        # for the reference IMR, SFR, IFMR.  
        pop_tabs = self.reset_columns(pop_tabs)

        # Some preparations
        labels = pop_tabs.keys() 
        opt_classes = opt_params.keys()

        indt = copy.deepcopy(kwargs['indt']) 
        SFR_ref = kwargs['SFR_ref']

        # Actions to be done if WD params are updated
        if any(x in opt_classes for x in ['ifmr', 'f_dadb', 'dcool']):
            ind_wd = kwargs['ind_wd'] # WD position indices in the tables
            time_grid = np.round(self.a.t,3)
            min_age = {'d':0,'t':13 - self.p.tt2}
            max_time_idx = {'d':self.a.jd - 1, 't':self.a.jt - 1}
        
        # When SFR is updated
        if 'sfr' in opt_classes:
            SFR_new = kwargs['SFR_new']
        
        # When IMF is updated
        if 'imf' in opt_classes:
            IMF_new = kwargs['IMF_new']
            IMF_ref = kwargs['IMF_ref']
            indm = kwargs['indm']
            imf_weights = np.array([p_new/p_old for p_new,p_old in zip(IMF_new,IMF_ref)])
                                        
        # Iterate over thin/thick disk and halo
        for label in labels:

            if any(x in opt_classes for x in ['ifmr', 'f_dadb', 'dcool']):
                ind_da, ind_db = ind_wd[label]
                all_wd_idx = np.concatenate(ind_wd[label])                 # DA and DB position indices
                age_wd_ref_arr = pop_tabs[label]['t_cool_ref'][all_wd_idx] # WD cooling age
                age_tot_ref_arr = pop_tabs[label]['age'][all_wd_idx]       # WD total age 

            # -------- IFMR: vectorize if present ----------
            # Important: IFMR and dcool corrections are not applied 
            # to the halo WDs because both affect final WD ages 
            # but the halo is modeled as a single age population. 
            # To make this applicable to halo, generation of the stellar assemblies
            # should be modified to include mock halo populations with 0 number densities
            # and some age shift from its mean 13-Gyr age value. Then, when the total 
            # age will be changed, these 'silent' populations could be swithced on. 
            # Anyway, it seems like too much trouble for now and is not worse it - 
            # sample is selected close to the plane, so the halo contribution 
            # will be negligible. 
            if 'ifmr' in opt_classes and label != 'sh':

                # Get WD final masses and metallisities
                Mf_arr = pop_tabs[label]['Mf'][all_wd_idx]
                FeH_arr = pop_tabs[label]['FeH'][all_wd_idx]

                # Update IFMR 
                ifmr_handler = kwargs['ifmr_handler']
                ifmr_handler.update_ifmr(**opt_params['ifmr'])

                # Call ifmr.cummings_r s on the whole array - 
                # update initial masses and write back in one go
                Mini_new, _ = ifmr_handler.cummings_r(Mf_arr)
                pop_tabs[label]['Mini'][all_wd_idx] = Mini_new

                # Update MS ages and total WD ages using a prebuilt interpolator
                # Mini must be in log10 here!! 
                msage_handler = kwargs['msage_handler']
                points = np.column_stack([FeH_arr, np.log10(Mini_new)])
                age_ms = msage_handler.interp_age_ms(points)
                '''
                When Mf are out of new IFMR range, nans in Mini can appear 
                bad = ~np.isfinite(age_ms)

                if bad.any():
                    print('Problematic MS ages:')
                    print('Corresponding Mini_new:', Mini_new[bad])
                    print('Corresponding Mf:',Mf_arr[bad])
                ''';
                age_tot_wd = age_ms + age_wd_ref_arr              # updated total ages
                pop_tabs[label]['age'][all_wd_idx] = age_tot_wd   # write to the table

            elif 'dcool' in opt_classes and label != 'sh':
                # If no IFMR, start from the stored age_tot
                age_tot_wd = age_tot_ref_arr.copy()

            # -------- dcool: apply part of the cooling delay correction ----------

            if 'dcool' in opt_classes and label != 'sh':
                alpha_cool = opt_params['dcool']['alpha_cool']

                age_tot_wd = age_tot_wd + alpha_cool * age_wd_ref_arr

                pop_tabs[label]['age'][all_wd_idx] = age_tot_wd

                pop_tabs[label]['t_cool'][all_wd_idx] = ((1 + alpha_cool) * age_wd_ref_arr)

            # -------- IFMR and dcool (remaining part) correction ----------
            if (('ifmr' in opt_classes) or ('dcool' in opt_classes)) and (label != 'sh'):
                # Create age mask 
                # It ensures that new ages lay within the allowed range 
                # (not larger than the total MW age, not younger than min age of the component)
                bad_mask = (~np.isfinite(age_tot_wd)) | (age_tot_wd > 13.0) | (age_tot_wd < min_age[label])
                good_mask = ~bad_mask

                # Get reference time index column for WD
                indt_ref = indt[label][all_wd_idx]   

                # Then compute new time index array and clip to a valid range
                #t_wd = 13.0 - age_tot_wd
                #indt_new = np.array([np.argmin(np.abs(time - self.a.t)) for time in t_wd]) # Do not use this line! +4s to iteration runtime!
                boundaries = (self.a.t[:-1] + self.a.t[1:]) / 2

                indt_new = np.searchsorted(boundaries, tp - age_tot_wd)

                indt_new = np.clip(indt_new, 0, max_time_idx[label])
                # update indt in place
                indt[label][all_wd_idx] = indt_new

                # Compute SFR weights (only for unmasked rows)
                # This is a correction for the total age bin shift - 
                # Number desities change because populations now belong to other age bins
                if np.any(good_mask):
                    # get reference SFR
                    sfr_ref_arr = SFR_ref[label]                         
                    # gather numerator and denominator vectors
                    num = sfr_ref_arr[indt_new[good_mask]] # new time bins
                    den = sfr_ref_arr[indt_ref[good_mask]] # old time bins
                    den = np.where(den == 0, 1e-30, den)   # avoid div by zero
                    # calculate correction
                    sfr_shift_weights = num / den
                    # apply weights back to the WD number densities 
                    pop_tabs[label]['N'][all_wd_idx[good_mask]] *= sfr_shift_weights
                    
                # zero out invalid rows
                if np.any(bad_mask):
                    pop_tabs[label]['N'][all_wd_idx[bad_mask]] = 0
                    #indt[label][all_wd_idx[bad_mask]] = 0  # formal time index


            # -------- f_dadb: account for change in DA/DB fraction ----------
            # Applies to thin/thick disk and halo
            if 'f_dadb' in opt_classes:

                f_da = opt_params['f_dadb']['f_da']
                
                # Find how the new values changed with respect to the reference values:
                # for DA: 0.8, for DB: 1 - 0.8 = 0.2
                da_weight = f_da/0.8
                db_weight = (1 - f_da)/0.2

                # Check for dependence on Teff
                f_da_teff = kwargs.get('f_da_teff', None)
                teff_factor_da, teff_factor_db = 1, 1

                if f_da_teff:
                    teff_factor_db = fdb_parabola(
                        10**(pop_tabs[label]['logT'][ind_db] - 3)
                    )
                    teff_factor_da = 1 - fdb_parabola(
                        10**(pop_tabs[label]['logT'][ind_da] - 3)
                    )
                
                # Update WD number densities with the new DA/DB fractions
                pop_tabs[label]['N'][ind_da] *= da_weight*teff_factor_da
                pop_tabs[label]['N'][ind_db] *= db_weight*teff_factor_db


            # -------- SFR: apply SFR weights ----------
            # Applies to the thin/thick disk only
            # Halo has a single age of 13 Gyr in any case
            if ('sfr' in opt_classes) and ((label == 'd') or (label == 't' and 't' in opt_params['sfr'].keys())):
                sfr_weights = SFR_new[label]/SFR_ref[label]
                pop_tabs[label]['N'] *= sfr_weights[indt[label]]

            # -------- IMF: apply IMF weights ----------
            # Applies to thin/thick disk and halo
            if 'imf' in opt_classes:
                pop_tabs[label]['N'] *= imf_weights[indm[label]]

        return pop_tabs, indt
    

def extract_model_tables(out,inp):
    """Create a collection of modeled quantites from the JJ-model input dict
    and an output of the local_run() function."""
    output = {'Fi':out['phi'],'AVR':out['avr'],
              'Hd':out['hd'],'Ht':out['ht'],'Hsh':out['hsh']}
    # Will only work when SFR peaks have special kinematics
    if 'Fp0' in inp.keys():
        output['Fp'] = inp['Fp0']
    if 'hdp' in out.keys():
        output['Hdp'] = out['hdp']
    return output
        

def fdb_parabola(Teff):
    # Teff in 10^3 format
    parabola = lambda x, a, b, c: a*x**2 + b*x + c
    parameters = [1.4e-4, -1.15983436e-2, 3.11929944e-1]
    return parabola(Teff,*parameters)


def piecewise_linear_v2(x, x0, b, k1, k2):
    '''2-slope piecewise linear function with breakpoint at x0.'''
    condlist = [x < x0, x >= x0]
    funclist = [lambda x:k1*x + b, lambda x:k2*(x - x0) + b + k1*x0]
    return np.piecewise(x,condlist,funclist)


def piecewise_linear_v3(x, x0, x1, b, k1, k2, k3):
    '''3-slope piecewise linear function with breakpoints at x0 and x1.'''
    condlist = [x < x0, (x >= x0) & (x < x1), x >= x1]
    funclist = [lambda x: k1*x + b, 
                lambda x: k2*(x - x0) + b + k1*x0, 
                lambda x: k3*(x - x1) + b + k1*x0 + k2*(x1 - x0)]
    return np.piecewise(x, condlist, funclist)


def chi_square(y1, y2):
    '''Chi-square normalized to the number of points.'''
    return sum((y1 - y2)**2)/len(y1)


class MSAgeHandler():
    """Class to work with the main sequence ages."""

    def __init__(self,feh=None,**kwargs):
        r""" 
        Initialization. 

        Parameters:
        -----------
        - feh: float
            Metallicity, dex. 
        - param_file: string
            Relative path to a file with parameters for the age interpolator. 
        """
        self.feh_grid = []
        self.params = []

        # Read the parameter file
        if 'param_file' in kwargs:
            param_file = kwargs['param_file']
        else:
            param_file = os.path.join(localpath,'input','ms_lifetimes','tau_ms_params_v1_Mbr1.18.txt')
    
        try:
            with open(param_file,'r') as f:
                for line in f:
                    if line[0] not in ['#','\n']:
                        values = line.split(' ')
                        self.feh_grid.append(float(values[0]))
                        parameter_set = [float(val) for val in values[1:] if val not in ('','\n')]
                        n_parameters = len(parameter_set) # number of parameters can be different
                        if n_parameters == 4:
                            parameter_set[0] = np.log10(parameter_set[0]) # convert Mbr to log
                        if n_parameters == 5:
                            parameter_set[0] = np.log10(parameter_set[0])
                            parameter_set[1] = np.log10(parameter_set[1])
                        self.params.append(parameter_set)
        except:
            pass
        
        # Grid of metallicities for the age interpolator
        self.feh_grid = np.array(self.feh_grid)

        # Type of function to use (2- or 3-slope fit)
        self.pwl = ['pwl2' if len(par)==4 else 'pwl3' for par in self.params]
        self.pwl_funcs = {'pwl2':piecewise_linear_v2,'pwl3':piecewise_linear_v3}

        # Choose one metallicity
        self.set_feh(feh)       

        # Tabulate and create an interpolator 
        if self.params != []:
            _ = self.build_interpolator()


    def set_feh(self,feh):
        r""" 
        Update metallicity value. 

        Parameters:
        -----------
        - feh: float
            Metallicity, dex. 

        Returns:
        --------
        - int
            Position index of the given metallicity in the interpolator's metallicity grid. 
        """
        self.feh = feh
        self.feh_idx = (np.argmin(np.abs(self.feh_grid - self.feh)) if self.feh!=None else None)
        return self.feh_idx
    

    def get_age_ms(self,mass_ini,feh=None):
        r""" 
        Calculate MS lifetime given the initial mass and metallicity. 

        Parameters:
        -----------
        - mass_ini: float or 1d-array
            Initial mass of a star, Msun. Not in log10, linear scale. 
        - feh: float
            Metallicity, dex. Optional, if has already been set. 

        Returns:
        --------
        - float or 1d-array
            Main-sequence lifetime, Gyr. 
        """
        # Check that metallicity is specified
        if feh is not None:
            self.set_feh(feh)

        if self.feh_idx is None:
            raise ValueError(
                'Parameter feh (metallicity [Fe/H]) must be given!'
            )

        # Choose the interpolator corresponding to this metallicity 
        # and type of the function according to the data in the parameter file
        func = self.pwl_funcs[self.pwl[self.feh_idx]]

        # Calculate MS lifetime
        if np.isscalar(mass_ini):
            power = func(np.log10(mass_ini),*self.params[self.feh_idx]) if not np.isnan(mass_ini) else np.nan
        else:
            power = np.array([func(np.log10(mass),*self.params[self.feh_idx]) if not np.isnan(mass) else np.nan for mass in mass_ini])
        tau_ms = 10**power  # in Gyr

        return tau_ms
    

    def compute_age_ms_grid(self,log_mini_grid=np.linspace(np.log10(0.08),np.log10(8.2),100)):
        r""" 
        Compute MS ages for a 2d-grid of initial masses and metallicities. 

        Parameters:
        -----------
        - log_mini_grid: 1d-array of floats
            By default in log space (as the interpolation was performed in log). 
            Min-max masses are 0.08 and 8.2 Msun (WD progenitors). 

        Returns:
        --------
        - 2d-array of floats
            MS lifetimes (ages), Gyr. 
        """
        self.log_mini_grid = log_mini_grid
        feh_grid = self.feh_grid

        # Compute lifetime table
        age_ms_grid = np.zeros((len(feh_grid), len(log_mini_grid)))
        for i, feh in enumerate(feh_grid):
            for j, mini in enumerate(log_mini_grid):
                age_ms_grid[i, j] = self.get_age_ms(10**mini, feh=feh)

        return age_ms_grid
    

    def build_interpolator(self):
        r""" 
        Create interpolator given a pre-computed lifetimes for the grid 
        of initial masses and metallicities. 

        Parameters:
        -----------
        None. 

        Returns:
        --------
        - scipy.interpolate.RegularGridInterpolator object
            Interpolator. 
        """
        # Create MS lifetimes grid
        age_ms_grid = self.compute_age_ms_grid()

        # Create interpolator object
        interp_age_ms = RegularGridInterpolator(
            (self.feh_grid, self.log_mini_grid),
            age_ms_grid,
            bounds_error=False,
            fill_value=None
        )
        self.interp_age_ms = interp_age_ms
        return interp_age_ms
    

    def retrieve_ms_lifetimes(self,FeH_grid,dir_out='MS_lifetime',
                              iso_dir='jjmodel/input/isochrones/Padova/multiband'):
        r""" 
        Estimate MS lifetimes from the JJ-model Padova isochrone grid. 

        Parameters:
        -----------
        - FeH_grid: 1d-array of floats
            Metallicity grid used in the JJ model. 
        - dir_out: string
            Name of the output directory where the results will be saved. 
            By default is 'MS_lifetime'
        - iso_dir: string
            Path to the folder with isochrones. 

        Returns:
        --------
        
        """
        # Create dict for the output data
        age_ms_data = {str(feh):{'mass':[],'age':[]} for feh in FeH_grid}

        # Iterate over metallicities and isochrone files
        for feh in FeH_grid:

            feh_dir = os.path.join(iso_dir,'iso_feh' + str(feh))
            iso_files = os.listdir(feh_dir)

            # For all ages in the metallicity directory
            for iso_file in iso_files:

                # Read isochrone
                iso_data = np.loadtxt(os.path.join(feh_dir,iso_file)).T

                # Get age from the filename
                iso_age = float(iso_file.split('age')[1].split('.txt')[0]) # in Gyr

                # Get turn-off mass (lowest mass with label=2)
                # 0-th column = initial mass
                # 5-th column = evolutionary stage (1 = MS, 2 = SGB)
                turnoff_mass = iso_data[0][np.where(iso_data[5] == 2)[0]][0]

                # Append to the dict
                age_ms_data[str(feh)]['mass'].append(turnoff_mass)
                age_ms_data[str(feh)]['age'].append(iso_age)

            # Convert lists to arrays and format output 
            output = np.array([age_ms_data[str(feh)]['mass'], age_ms_data[str(feh)]['age']])
            ind_u = np.unique(output[0],return_index=True)[1]

            masses = output[0][ind_u]
            ages = [min(output[1][np.where(output[0]==output[0][i])[0]]) for i in ind_u]
            
            output = np.array([masses,ages])
            np.savetxt(os.path.join(dir_out,''.join(('fe'+str(feh)+'.txt'))), 
                       output.T, header='Mini[M_sun]\tAge[Gyr]')
            

    def plot_age_ms(self, mets, dir_data='MS_lifetime', dir_out='MS_lifetime/plots', 
                    cmap='RdBu'):
        '''
        Visualization of the MS-lifetime data for a list of metallicities.
        '''

        fig, ax = plt.subplots(figsize=(8,6))
        cmap = mpl.colormaps[cmap]

        # Create a Normalize instance based on your metallicity values
        norm = mpl.colors.Normalize(vmin=-2.5, vmax=0.5)
        # Map each metallicity value through the colormap using the normalization
        colors = cmap(norm(mets))
        # Take colors at regular intervals spanning the colormap.
        colors = cmap(np.linspace(0, 1, len(mets)))
        
        # Plot data
        for met, c in zip(mets,colors):

            data = np.loadtxt(os.path.join(dir_data,''.join(('fe'+str(met)+'.txt')))).T
            mass = data[0]
            print(np.mean(np.diff(mass)),np.std(np.diff(mass)))
            age = data[1]
            x = np.log10(mass)
            y = np.log10(age)

            ax.loglog(10**x, 10**y, "x", ms=3, color=c)
            ax.set_xlabel(r'$\mathrm{M_{ini}, \ M_{\odot}}$')
            ax.set_ylabel(r'$\mathrm{t_{MS}, \ Gyr}$')

        bottom, top = ax.get_ylim()
        ax.plot([1.18,1.18],[bottom, top], 'k--', lw=0.5, color='grey',label=r'$\mathrm{M_{br}}$')
        ax.set_ylim(bottom,top)

        fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

        # Pass both cmap and norm to ScalarMappable so the colorbar matches your data
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # required for older matplotlib
        cax = fig.colorbar(sm, ax=ax)
        cax.set_label(r'$\mathrm{[Fe/H]}$')

        plt.savefig(os.path.join(dir_out,'MS_lifetimes_allFeH.png'))
        
        return fig, ax
    

    def fit_ms_lifetime(self, met, func='v1', Mbr = None, 
                    dir_data='MS_lifetime', dir_out='MS_lifetime/plots', 
                    plot=False, stat=True, verbose=True):
        '''
        Returns fit function and parameters for the MS-lifetime relation for a given metallicity.
        The fit is done in log-log space.
        Parameter func can be 'v1', 'v2' or 'v3' for 2-slope (free or fixed breakpoint),
        or a 3-slope piecewise linear function, respectively.
        If plot=True, a plot of the data, fit and residuals is shown.
        '''

        # Get the data
        data = np.loadtxt(os.path.join(dir_data,''.join(('fe'+str(met)+'.txt')))).T

        mass = data[0]
        age = data[1]
        x = np.log10(mass)
        y = np.log10(age)
        
        # Fit options
        param = {'v1':{'func': lambda x, b, k1, k2: piecewise_linear_v2(x, np.log10(Mbr), b, k1, k2),
                    'p0': [0.5, -3.5, -2.5],
                    'bounds': ((-10, -5, -5),(10, -1.0, -1.0))
                    },
                'v2':{'func': piecewise_linear_v2, 
                    'p0': [0.04, 0.5, -3.5, -2.5],
                    'bounds': ((0.02, -10, -5, -5),(0.35, 10, -1.0, -1.0))
                    },
                'v3':{'func': piecewise_linear_v3, 
                    'p0': [0.06, 0.12, 1, -3.1, -3.1, -3.1],
                    'bounds': ((0.02, 0.1, -10, -5, -5, -5),(0.12, 0.35, 10, -1.0, -1.0, -1.0))
                    }
                }
        
        # Choose the fit function 
        piecewise_linear = param[func]['func']

        # Perform the fit
        p , _ = optimize.curve_fit(piecewise_linear, x, y, 
                                p0 = param[func]['p0'], 
                                bounds = param[func]['bounds'],
                                #sigma = 1/(x + 1e-12) # weights
                                )
        
        # Output dict for miscellaneous information
        msl = {}

        if stat:
            chi2 = round(chi_square(y, piecewise_linear(x, *p)),5)
            msl['chi2'] = chi2

            if verbose:
                print('Chi^2 = ', chi2)
                if func == 'v1':
                    print('M_br1 = ', round(Mbr,2), 'Msun')
                    print('alpha1 = ', round(p[1],2))
                    print('alpha2 = ', round(p[2],2))
                if func == 'v2':
                    print('M_br1 = ', round(10**p[0],2), 'Msun')
                    print('alpha1 = ', round(p[2],2))
                    print('alpha2 = ', round(p[3],2))
                if func == 'v3':
                    print('M_br1 = ', round(10**p[0],2), 'Msun')
                    print('M_br2 = ', round(10**p[1],2), 'Msun')
                    print('alpha1 = ', round(p[3],2))
                    print('alpha2 = ', round(p[4],2))
                    print('alpha3 = ', round(p[5],2))

        if plot:
            f, ax = plt.subplots(2,1, figsize=(6,8),sharex=True)
            ax[0].loglog(10**x, 10**y, "x", ms=5, c='steelblue',label='Data from isochrones')
            ax[0].loglog(10**x, 10**piecewise_linear(x, *p),c='tomato',label='Fitted function')
            #ax[0].loglog(10**x, 10**piecewise_linear(x, *np.round(p,2)),c='orange')
            kwg = {'ls':'--', 'color':'grey','lw':0.5,'zorder':10}
            if func == 'v1':
                if not Mbr:
                    raise ValueError("For func='v1', Mbr must be provided.")
                ax[0].vlines(Mbr, 0.02, 20,**kwg)
            elif func == 'v2':
                ax[0].vlines(10**p[0], 0.02, 20, **kwg)
            elif func == 'v3':
                ax[0].vlines(10**p[0], 0.02, 20, **kwg)
                ax[0].vlines(10**p[1], 0.02, 20, **kwg)
            ax[0].set_xlim(0.7,10)
            ax[0].set_ylabel('Age [Gyr]')
            ax[0].legend(loc=1)
            
            ax[1].scatter(10**x, 10**(y - piecewise_linear(x, *p)) - 1, s=1)
            if stat:
                chi2 = round(chi_square(y, piecewise_linear(x, *p)),5)
                ax[1].text(0.75, 0.1, r'$\mathrm{\chi^2 \, = \, }$'+ str(chi2), transform=ax[1].transAxes)
            ax[1].set_xticks(np.arange(1,10))
            ax[1].set_xticklabels(np.arange(1,10))
            ax[1].set_ylabel('Age [Gyr]')
            ax[1].set_xlabel('Mass [Msun]')

            if func == 'v1':
                figname = 'MS_lifetime_fit_fe' + str(met) + '_' + func + '_Mbr' + str(Mbr) + '.png'
            else:
                figname = 'MS_lifetime_fit_fe' + str(met) + '_' + func + '.png'

            plt.savefig(os.path.join(dir_out,figname))

            msl['fig'] = (f, ax)
        
        return (piecewise_linear, p), msl
    

    def fit_all_metallicities(self,FeH_grid, func='v1', Mbr=None, dir_data='MS_lifetime', 
                              dir_out='MS_lifetime', plot=True, stat=True):
        '''
        Fit the MS-lifetime data for all metallicities in FeH_grid.
        The fit is done in log-log space.
        Parameter func can be 'v1', 'v2' or 'v3' for 2-slope (free or fixed breakpoint),
        or a 3-slope piecewise linear function, respectively.
        The fit parameters are saved to a text file.
        '''

        msl1 = {}

        if stat:
            chi2_array = []

        if func == 'v1' or func == 'v2':
            labels = ['FeH', 'M_br1', 'b', 'alpha1', 'alpha2']
        elif func == 'v3':
            labels = ['FeH', 'M_br1', 'M_br2', 'b', 'alpha1', 'alpha2', 'alpha3']

        header = ''.join((['{:<10}'.format(l) for l in labels]))
        print(header)

        output = {key:[] for key in labels}

        filename = 'tau_ms_params_' + func + '.txt'
        if func == 'v1':
            filename = 'tau_ms_params_' + func + '_Mbr' + str(Mbr) + '.txt'

        with open(os.path.join(dir_out,filename),"w") as f:
            f.write('# M_br in Msun, transform to log10 before passing to the fit function!\n')
            f.write('# ' + header + '\n')
            for feh in FeH_grid:
                with warnings.catch_warnings(action='ignore'):
                    (_, p), msl2 = self.fit_ms_lifetime(feh,func=func,Mbr=Mbr,
                                                dir_data=dir_data,plot=False,stat=stat,verbose=False);
                    if stat:
                        chi2_array.append(msl2['chi2'])
                if func=='v1':
                    params = [feh, Mbr, *p]
                elif func=='v2':
                    params = [feh, 10**p[0], *p[1:]]
                elif func=='v3':
                    params = [feh, 10**p[0], 10**p[1], *p[2:]]
                
                for key in labels:
                    output[key].append(params[labels.index(key)])
                
                line = ''.join((['{:<10}'.format(str(round(pa,2))) for pa in params]))
                print(line)
                f.write(line + '\n')

            if stat:
                chi2_mean = round(np.mean(chi2_array),5)
                chi2_std = round(np.std(chi2_array),5)
                print('\nMean chi^2 = ', chi2_mean, ' +/- ', chi2_std)
                f.write('\n# Mean chi^2 = ' + str(chi2_mean) + ' +/- ' + str(chi2_std) + '\n')

                msl1['chi2_mean'] = chi2_mean
                msl1['chi2_std'] = chi2_std

        if plot:

            lbs = {'M_br1':{'c':'m','l':r'$\mathrm{M_{br1}}$'},
                'M_br2':{'c':'violet','l':r'$\mathrm{M_{br2}}$'}, 
                'b':{'c':'orange','l':r'$\mathrm{b}$'},
                'alpha1':{'c':'lawngreen','l':r'$\mathrm{\alpha_{1}}$'},
                'alpha2':{'c':'green','l':r'$\mathrm{\alpha_{2}}$'},
                'alpha3':{'c':'blue','l':r'$\mathrm{\alpha_{3}}$'}
                }

            f, ax = plt.subplots(len(params)-1,1,figsize=(8,6))
            ax[0].set_title('Fit parameters of MS lifetime')
            ax[len(params)-2].set_xlabel(r'$\mathrm{[Fe/H]}$')

            if stat:
                ax[0].text(0.05, 0.8, 
                        r'$\mathrm{<\chi^2> \, = \, }$'+ str(chi2_mean) + r'$\, \pm \,$' + str(chi2_std), 
                        transform=ax[0].transAxes
                        )
            
            for i, key in enumerate(labels[1:]):
                    
                ax[i].plot(FeH_grid, output[key], 'o-',c=lbs[key]['c'])
                ax[i].set_ylabel(lbs[key]['l'])
                if i < len(params)-2:
                    ax[i].set_xticklabels([])

            if func != 'v1':
                figname = 'MS_lifetime_fit_params' + '_' + func + '.png'
            else:
                figname = 'MS_lifetime_fit_params' + '_' + func + '_Mbr' + str(Mbr) + '.png'
            plt.savefig(os.path.join(dir_out,figname))

            msl1['fig'] = (f, ax)
            
        return output, msl1

