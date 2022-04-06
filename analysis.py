"""
Created on Mon Feb 20 18:17:18 2017

@author: Skevja
"""

import os
import sys
import inspect
import numpy as np
import warnings
from scipy.special import erf
from scipy.ndimage import gaussian_filter1d
from astropy.table import Table
from scipy.signal import convolve2d 
import matplotlib.pyplot as plt
from fast_histogram import histogram2d
from .control import (CheckIsoInput, inpcheck_radius, inpcheck_age, reduce_kwargs,
                     inpcheck_height,inpcheck_iskwargtype,inpcheck_kwargs,inpcheck_mode_comp,
                     inpcheck_kwargs_compatibility,inpcheck_dz)
from .iof import tab_reader, tab_sorter, TabSaver
from .constants import KM, tp, tr
from .funcs import hgr, RotCurve, RadialPotential, AMR
from .tools import rebin_histogram, reduce_table, rotation_matrix, gauss_weights, convolve2d_gauss
from .geometry import Volume


# =============================================================================
# Selection of special populations using stellar parameters & photometry
# =============================================================================

class GetPopulations():
    """
    Class for selecting different stellar populations using 
    color-magnitude diagram (CMD) and physical stellar parameters. 
    """
    
    def __init__(self,mode_iso,R,p,a):
        """
        p : namedtuple
            Set of the model parameters from the parameter file. 
        a : namedtuple
            Collection of the fixed model parameters, useful 
            quantities and arrays.
        """
        
        this_function = inspect.stack()[0][3]
        ch = CheckIsoInput()
        ch.check_mode_isochrone(mode_iso,this_function)
        self.p, self.a = p, a
        self.mode_iso = mode_iso
        if R!=p.Rsun:
            self.R = inpcheck_radius(R,self.p,this_function)
        else:
            self.R = R
        
        tab_d = Table.read(os.path.join(a.T['poptab'],
                                        ''.join(('SSP_R',str(self.R),'_d_',mode_iso,'.csv'))))
        tab_t = Table.read(os.path.join(a.T['poptab'],
                                        ''.join(('SSP_R',str(self.R),'_t_',mode_iso,'.csv'))))
        tab_sh = Table.read(os.path.join(a.T['poptab'],
                                         ''.join(('SSP_R',str(self.R),'_sh_',mode_iso,'.csv'))))
        self.tables = {'d':tab_d,'t':tab_t,'sh':tab_sh}
        
        
    def custom_population(self,mode_comp,column_list,range_list,**kwargs):
        """
        Allows to select population using custom cuts on the columns
        of the stellar seemblies table. 
        
        Parameters
        ----------
        mode_comp : str
            'd', 't' or 'sh' (thin disk, thick disk or halo).
        column_list : list
            List of column names (str) for which the cuts will be
            applied.
        range_list : list
            List with range of allowed values [min,max] corresponding
            to the columns in column_list.
        **kwargs : dict, optional keyord arguments
            save : boolean
                If True, the output table is saved. 
            tabname : str
                Name of population. If not given, the table is 
                saved with the default name 'custom_population'.
                
        Returns
        -------
        tab : astropy table 
            Selected population.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','tabname'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh'],
                                           'custom population',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function)
        
        tab = self.tables[mode_comp]
        
        n = len(column_list)
        for i in range(n):
            col = column_list[i]
            colmin,colmax = range_list[i]
            tab = tab[np.logical_and.reduce([tab[col]>colmin,tab[col]<colmax])]
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            if 'tabname' in kwargs:
                popname = kwargs['tabname']
            else:
                popname = 'custom_population'
            ts = TabSaver(self.p,self.a,**kwargs)
            ts.poptab_save(tab,mode_comp,self.mode_iso,self.R,popname)
            #tabname = tab_sorter(popname,self.p,self.a.T,R=self.R,mode=mode_comp) 
            #tab.write(tabname,overwrite=True)
            if len(tab['logT'])==0:
                print('\t','{:<3}'.format(mode_comp),': ',popname,' sample is empty.')
        
        return tab 
                                         

    def rc_simple(self,mode_comp,**kwargs):
        """
        Selects Red Clump (RC) population using simple cuts on 
        T_eff, logg and logL: 4250 K  < T_eff < 5250 K && 
        logg < 2.75 && 1.65 < log(L/L_sun) < 1.85. 
          
        Parameters
        ----------
        mode_comp : str
            'd', 't' or 'sh' (thin disk, thick disk or halo).
        **kwargs : dict, optional keyord arguments
            save : boolean
                If True, the output table is saved. 

        Returns
        -------
        rc0 : astropy table
            Table of the stellar assemblies that belong to RC 
            population.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh'],'RC(+HGB) stars',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function)
        
        tab = self.tables[mode_comp]
        if self.mode_iso=='BaSTI':
            rc0 = tab[np.logical_and.reduce([tab['logT']>np.log10(4250),tab['logT']<np.log10(5250),                                 
                                 tab['logL']>1.65,tab['logL']<1.85])]                                 
        else:
            rc0 = tab[np.logical_and.reduce([tab['logT']>np.log10(4250),tab['logT']<np.log10(5250),                                             
                                             tab['logL']>1.65,tab['logL']<1.85,tab['logg']<2.75])]
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            #tabname = tab_sorter('rc+tab',self.p,self.a.T,R=self.R,mode=mode_comp,mode_pop='rc+') 
            #rc0.write(tabname,overwrite=True)
            ts = TabSaver(self.p,self.a,**kwargs)
            ts.poptab_save(rc0,mode_comp,self.mode_iso,self.R,'rc+')
            if len(rc0['logT'])==0:
                print('\t','{:<3}'.format(mode_comp),': RC(+RGB) sample is empty.')
                
        return rc0
    
        
    def rc_clean(self,mode_comp,**kwargs):
        """
        Selects Red Clump (RC) population in the 3D-space 
        {logT,logg,[Fe/H]}. This is a cleaner RC selection, than
        the one given by rc_simple method. 
        
        Parameters
        ----------
        mode_comp : str
            'd', 't' or 'sh' (thin disk, thick disk or halo).
        **kwargs : dict, optional keyord arguments
            fig : boolean
                If True, the selected RC sample is plotted 
                in the 3D-space {logT,logg,[Fe/H]}. 
            close : boolean
                If True, the figure is closed after plotting 
                (active only if fig=True). 
            save : boolean
                If True, the output table (and figure, if plotted)
                is/are saved. 

        Returns
        -------
        rc_sample_tab : astropy table
            Table of the stellar assemblies that belong to RC 
            population.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['fig','close','save'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh'],'RC stars',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function)
        
        if self.mode_iso=='BaSTI':
            message = "Method 'rc_clean' cannot be used with BaSTI isochrones, use 'rc_simple' or "+\
                      "switch to Padova or MISt isochrones (BaSTI isochrones have no logg)."
            sys.exit(message)
            
        rc0 = self.rc_simple(mode_comp)
        axis0 = [0,0,1]
        if mode_comp=='d':
            theta0 = np.radians(25)
            #frac_ignore = 0.05 
            ind_ignore = 1
            dx_large, dx_small = 0.025, 0.005
        if mode_comp=='t':
            theta0 = np.radians(15)
            dx_large, dx_small = 0.1, 0.05
        if mode_comp=='sh':
            theta0 = np.radians(15)
            dx_large, dx_small = 0.3, 0.2
        
        # Switch to the normalized axes. 
        logtmax, logtmin = np.amax(rc0['logT']), np.amin(rc0['logT'])
        loggmax, loggmin = np.amax(rc0['logg']), np.amin(rc0['logg'])
        fehmax, fehmin = np.amax(rc0['FeH']), np.amin(rc0['FeH'])
        
        rc_list = [[i,k,l] for (i,k,l) in zip((rc0['logT']-logtmin)/(logtmax-logtmin),
                                              (rc0['logg']-loggmin)/(loggmax-loggmin),
                                              (rc0['FeH']-fehmin)/(fehmax-fehmin))]
        
        # Each (logT,logg,[Fe/H]) vector is rotated around z-axis, such that RC and HGB 
        # stars are better separated in the new X'Z plane. 
        rc_list_rotated = [np.dot(rotation_matrix(axis0,theta0),vector) for vector in rc_list]
        
        '''
        # Helpful plot, uncomment if something goes wrong, then try to use this method again.
        # Shows the pre-selected RC sample in XZ and X'Z planes - before and after rotation. 
        plt.figure()
        plt.scatter([i[0] for i in rc_list_rotated],
                    [i[2] for i in rc_list_rotated],s=0.5,c='b',label='Before rotation')
        plt.scatter([i[0] for i in rc_list],
                    [i[2] for i in rc_list],s=0.5,c='gray',label='After rotation')
        plt.xlabel("X or X'")
        plt.xlabel("Z")
        plt.legend(loc=1)
        plt.show()
        '''
        
        # Sort the rotated sample by Z-axis ([Fe/H])
        rc_ind, rc_compl_ind = [], []
        
        #rc_list_rotated.sort(key=lambda x: x[2])
        rc_list2 = [i[2] for i in rc_list_rotated]
        
        # Create Z-grid to slice the sorted sample. Step is smaller for z > 0.9,
        # because it's harder to select clean RC there, so finer resolution is needed. 
        x_split = 0.9
        z_grid = np.concatenate((np.linspace(0,x_split,int(x_split/dx_large+1)),
                                 np.linspace(x_split+dx_small,1,int((1-x_split)/dx_small+1))),axis=0)
        
        rc_sample, rc_compl = [], [] 
        for i in range(len(z_grid)):
            try:
                # Select Z-slice
                ind = np.where((np.array(rc_list2) > z_grid[i])&(np.array(rc_list2) <= z_grid[i+1]))[0]
                z_bin = list(np.array(rc_list_rotated)[ind])
                # Sort by X'
                z_bin0 = [k[0] for k in z_bin]
                indx_original = np.argsort(z_bin0)
                z_bin.sort(key=lambda x: x[0])
                z_bin0 = [k[0] for k in z_bin]
                
                # Calculate the derivative of the sorted sequence. As there is a gap between the RC and
                # other ginat stars along the X' axis at each Z, there will be a strong peak in the 
                # derivative marking two subsamples' separation. The position of this peak is used 
                # to define RC sample, the rest of stars are stored in a complementary array.  
                xdif = np.diff(z_bin0)
                
                '''
                indx = np.arange(len(xdif))
                plt.figure()
                plt.plot(indx,xdif,c='k')
                plt.plot(indx[ind_ignore:-ind_ignore],xdif[ind_ignore:-ind_ignore],c='m')
                plt.savefig(''.join(('z_',str(z_grid[i]),'-',str(z_grid[i+1]),'.png')))
                plt.close()
                '''
                
                if mode_comp=='d':
                    #ind_ignore = int(frac_ignore*len(ind))
                    indbr = (np.where(xdif[ind_ignore:-ind_ignore]==\
                                      np.amax(xdif[ind_ignore:-ind_ignore]))[0][0] + ind_ignore + 1)
                else:
                    if mode_comp=='sh' and np.amax(xdif) < 0.2:
                        indbr = len(ind)
                    else:
                        indbr = np.where(xdif==np.amax(xdif))[0][0] + 1 
                       
                rc_sample_rotated = [[k[0],k[1],k[2]] for k in z_bin[indbr:]]
                rc_compl_rotated = [[k[0],k[1],k[2]] for k in z_bin[:indbr]]
                rc_ind.extend(ind[indx_original[indbr:]])
                rc_compl_ind.extend(ind[indx_original[:indbr]])
                
                # Rotate each vector back to return to X-axis. 
                rc_sample_unrotated = [np.dot(rotation_matrix(axis0,-theta0),vector) for vector in rc_sample_rotated]
                rc_compl_unrotated = [np.dot(rotation_matrix(axis0,-theta0),vector) for vector in rc_compl_rotated]
                rc_sample.extend(rc_sample_unrotated)
                rc_compl.extend(rc_compl_unrotated)
            except: pass
        
        rc_sample_tab = Table()
        rc_complm_tab = Table()
        keys = list(rc0.keys())
        
        if rc_sample!=[]:
            '''
            # Un-normalize values and build the output table. 
            rc_sample_logt = np.array([i[0] for i in rc_sample])*(logtmax-logtmin) + logtmin
            rc_sample_logg = np.array([i[1] for i in rc_sample])*(loggmax-loggmin) + loggmin
            rc_sample_feh = np.array([i[2] for i in rc_sample])*(fehmax-fehmin) + fehmin
            
            rc_ind = [np.where((np.abs(rc0['logT']-i)==np.amin(np.abs(rc0['logT']-i)))&
                               (np.abs(rc0['logg']-k)==np.amin(np.abs(rc0['logg']-k)))&
                               (np.abs(rc0['FeH']-l)==np.amin(np.abs(rc0['FeH']-l))))[0][0] for (i,k,l) 
                               in zip(rc_sample_logt,rc_sample_logg,rc_sample_feh)]
            rc_ind = [np.where(np.abs(rc0['FeH']-i)==np.amin(np.abs(rc0['FeH']-i)))[0][0] 
                      for i in rc_sample_feh]
            '''
            for i in range(len(keys)):
                rc_sample_tab[keys[i]]=rc0[keys[i]][np.array(rc_ind)]
                rc_complm_tab[keys[i]]=rc0[keys[i]][np.array(rc_compl_ind)]
        else:
            for i in range(len(keys)):
                rc_sample_tab[keys[i]]=[]
                rc_complm_tab[keys[i]]=[]
        
        if inpcheck_iskwargtype(kwargs,'fig',True,bool,this_function):
            
            ax_orientation = {'d':[2,-118],'t':[2,-105],'sh':[2,-103]}
            
            plt.figure()
            ax = plt.subplot(111, projection='3d')
            ax.view_init(ax_orientation[mode_comp][0],ax_orientation[mode_comp][1])             
            ax.scatter([i[0] for i in rc_sample],
                       [i[1] for i in rc_sample],
                       [i[2] for i in rc_sample],s=0.5,c='r',
                       label='$\mathrm{Selected \ RC \ sample}$')
            ax.scatter([i[0] for i in rc_compl],
                       [i[1] for i in rc_compl],
                       [i[2] for i in rc_compl],s=0.5,c='gray',
                       label='$\mathrm{Complementary \ sample}$')
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            ax.set_zlim(0,1)
            ax.set_xlabel('$\mathrm{logT}$',fontsize=12,labelpad=10)
            ax.set_ylabel('$\mathrm{logg}$',fontsize=12,labelpad=10)
            ax.set_zlabel('$\mathrm{[Fe/H]}$',fontsize=12,labelpad=10)
            lg = plt.legend(prop={'size':11},ncol=2,loc='upper center')
            for handle in lg.legendHandles:
                handle.set_sizes([6.0])
            ax.set_title(''.join(('$\mathrm{R = }$',str(self.R),'$\mathrm{\ kpc}$')),
                         fontsize=12,pad=30)
            figname = os.path.join(self.a.T['popplt'],
                                   ''.join(('Clean_RC_R',str(self.R),'_',mode_comp,'.png')))
            if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
                plt.savefig(figname)
            if inpcheck_iskwargtype(kwargs,'close',True,bool,this_function):
                plt.close()

        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            ts = TabSaver(self.p,self.a,**kwargs)
            ts.poptab_save(rc_sample_tab,mode_comp,self.mode_iso,self.R,'rc')
            ts.poptab_save(rc_complm_tab,mode_comp,self.mode_iso,self.R,'rc_compl')
            if len(rc_sample_tab['logT'])==0:
                print('\t','{:<3}'.format(mode_comp),': RC sample is empty.')

        return rc_sample_tab
            
            
    def a_stars(self,mode_comp,**kwargs):
        """
        Selects A-type stars using cuts on T_eff:
        7500 K < T_eff < 10000 K.
        
        Parameters
        ----------
        mode_comp : str
            'd', 't' or 'sh' (thin disk, thick disk or halo).
        **kwargs : dict, optional keyord arguments
            save : boolean
                If True, the output table is saved. 
        Returns
        -------
        ast : astropy table
            Table of the stellar assemblies that belong to the 
            A-star population.
        """
        
        this_function = inspect.stack()[0][3]
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh'],'A-type stars',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function)
        
        tab = self.tables[mode_comp]
        ast = tab[np.logical_and.reduce([tab['logT'] > np.log10(7500),tab['logT'] < np.log10(10000)])]
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):          
            #tabname = tab_sorter('atab',self.p,self.a.T,R=self.R,mode=mode_comp,mode_pop='a') 
            #ast.write(tabname,overwrite=True)
            ts = TabSaver(self.p,self.a,**kwargs)
            ts.poptab_save(ast,mode_comp,self.mode_iso,self.R,'a')
        if len(ast['logT'])==0:
            print('\t','{:<3}'.format(mode_comp),': A-type stellar sample is empty.')
                
        return ast
    
    
    def f_stars(self,mode_comp,**kwargs):
        """
        Selects F-type stars using cuts on T_eff:
        6000 K < T_eff < 7500 K.
            
        Parameters
        ----------
        mode_comp : str
            'd', 't' or 'sh' (thin disk, thick disk or halo).
        **kwargs : dict, optional keyord arguments
            save : boolean
                If True, the output table is saved. 

        Returns
        -------
        fst : astropy table
            Table of the stellar assemblies that belong to the 
            F-star population.
        """
        
        this_function = inspect.stack()[0][3]
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh'],'F stars',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function)
        
        tab = self.tables[mode_comp]
        fst = tab[np.logical_and.reduce([tab['logT'] > np.log10(6300),tab['logT'] < np.log10(7500)])]
                                         
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):  
            #tabname = tab_sorter('ftab',self.p,self.a.T,R=self.R,mode=mode_comp,mode_pop='f') 
            #fst.write(tabname,overwrite=True)
            ts = TabSaver(self.p,self.a,**kwargs)
            ts.poptab_save(fst,mode_comp,self.mode_iso,self.R,'f')
            if len(fst['logT'])==0:
                print('\t','{:<3}'.format(mode_comp),': F sample is empty.')
                
        return fst

           
    def g_dwarfs(self,mode_comp,**kwargs):
        """
        Selects G-dwarfs using cuts on T_eff and logg:
        5200 K < T_eff < 6000 K && 4.3 < logg < 7. 
            
        Parameters
        ----------
        mode_comp : str
            'd', 't' or 'sh' (thin disk, thick disk or halo).
        **kwargs : dict, optional keyord arguments
            save : boolean
                If True, the RC sample table is saved. 

        Returns
        -------
        gdw : astropy table
            Table of the stellar assemblies that belong to the 
            G-dwarf population.
        """        
        
        this_function = inspect.stack()[0][3]
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh'],'G dwarfs',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function)

        tab = self.tables[mode_comp]
        if self.mode_iso=='BaSTI':
            gdw = tab[np.logical_and.reduce([tab['logT'] > np.log10(5200),
                                             tab['logT'] < np.log10(6000),
                                             tab['logL'] < 0.07,tab['logL'] > -0.55])]
        else:
            gdw = tab[np.logical_and.reduce([tab['logT'] > np.log10(5200),
                                             tab['logT'] < np.log10(6000),
                                             tab['logg'] < 7,tab['logg'] > 4.3])]
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):                
            #tabname = tab_sorter('gdwtab',self.p,self.a.T,R=self.R,mode=mode_comp,mode_pop='gdw') 
            #gdw.write(tabname,overwrite=True)
            ts = TabSaver(self.p,self.a,**kwargs)
            ts.poptab_save(gdw,mode_comp,self.mode_iso,self.R,'gdw')
            if len(gdw['logT'])==0:
                print('\t','{:<3}'.format(mode_comp),': G-dwarf sample is empty.')
                
        return gdw
    
    
    def k_dwarfs(self,mode_comp,**kwargs):
        """
        Selects K-dwarfs using cuts on T_eff and logg:
        3700 K < T_eff < 5200 K && 4.3 < logg < 7. 
        
        Parameters
        ----------
        mode_comp : str
            'd', 't' or 'sh' (thin disk, thick disk or halo).
        **kwargs : dict, optional keyord arguments
            save : boolean
                If True, the RC sample table is saved. 

        Returns
        -------
        kdw : astropy table
            Table of the stellar assemblies that belong to the 
            K-dwarf population.
        """
        
        this_function = inspect.stack()[0][3]
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh'],'K dwarfs',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function)

        tab = self.tables[mode_comp]
        if self.mode_iso=='BaSTI':
            kdw = tab[np.logical_and.reduce([tab['logT'] > np.log10(3700),
                                             tab['logT'] < np.log10(5200),
                                             tab['logL'] < -0.55,tab['logL'] > -1.63])]
        else:
            kdw = tab[np.logical_and.reduce([tab['logT'] > np.log10(3700),
                                             tab['logT'] < np.log10(5200),
                                             tab['logg'] < 7,tab['logg'] > 4.3])]
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):  
            #tabname = tab_sorter('kdwtab',self.p,self.a.T,R=self.R,mode=mode_comp,mode_pop='kdw') 
            #kdw.write(tabname,overwrite=True)
            ts = TabSaver(self.p,self.a,**kwargs)
            ts.poptab_save(kdw,mode_comp,self.mode_iso,self.R,'kdw')
            if len(kdw['logT'])==0:
                print('\t','{:<3}'.format(mode_comp),': K-dwarf sample is empty.')
        return kdw 
    
    
    def m_dwarfs(self,mode_comp,**kwargs):
        """
        Selects M-dwarfs using cuts on T_eff and logg:
        2400 K < T_eff < 3700 K && 4 < logg < 7. 
            
        Parameters
        ----------
        mode_comp : str
            'd', 't' or 'sh' (thin disk, thick disk or halo).
        **kwargs : dict, optional keyord arguments
            save : boolean
                If True, the RC sample table is saved. 

        Returns
        -------
        mdw : astropy table
            Table of the stellar assemblies that belong to the 
            M-dwarf population.
        """
        
        this_function = inspect.stack()[0][3]
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh'],'M dwarfs',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function)
        
        tab = self.tables[mode_comp]
        if self.mode_iso=='BaSTI':
            mdw = tab[np.logical_and.reduce([tab['logT'] > np.log10(2400),
                                             tab['logT'] < np.log10(3700),
                                             tab['logL'] < -0.55,tab['logL'] > -3.5])]
        else:
            mdw = tab[np.logical_and.reduce([tab['logT'] > np.log10(2400),
                                             tab['logT'] < np.log10(3700),
                                             tab['logg'] < 7,tab['logg'] > 4.0])]
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            #tabname = tab_sorter('mdwtab',self.p,self.a.T,R=self.R,mode=mode_comp,mode_pop='mdw') 
            #mdw.write(tabname,overwrite=True)
            ts = TabSaver(self.p,self.a,**kwargs)
            ts.poptab_save(mdw,mode_comp,self.mode_iso,self.R,'mdw')
            if len(mdw['logT'])==0:
                print('\t','{:<3}'.format(mode_comp),': M-dwarf sample is empty.')
                
        return mdw     
    
    def _rr_lyrae_(self,mode_comp,**kwargs):
        """
        Experimental method.
        Selects RR Lyrae stars. HB stars in the instability strip
        with pulsation period < 1 day (Marconi et al. 2015). 
        Dy default, these are the fundamental pulsators (RRab 
        stars), but also the first obertone pulsators can be 
        selected (RRc stars). 
            
        Parameters
        ----------
        mode_comp : str
            'd', 't' or 'sh' (thin disk, thick disk or halo).
        **kwargs : dict, optional keyord arguments
            FO   : boolean
                If True, the first obertone pulsators are selected
                (RRc stars). 
            save : boolean
                If True, the RC sample table is saved. 

        Returns
        -------
        rrl : astropy table
            Table of the stellar assemblies that belong to the 
            RR Lyrae population.
        """
        
        this_function = inspect.stack()[0][3]
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh'],'RR Lyrae stars',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function)
        
        tab = self.tables[mode_comp]
        # Definition of the instability strip (IS) is uses logZ and logP:
        amr = AMR()
        Z = amr.fe2z(tab['FeH'])
        tab['logZ'] = np.log10(Z)
        if inpcheck_iskwargtype(kwargs,'FO',True,bool,this_function):
            tab['logP'] = 11.167 + 0.822*tab['logL'] - 0.56*np.log10(tab['Mf']) -\
                          3.4*tab['logT'] + 0.013*tab['logZ']
        else:
            tab['logP'] = 11.347 + 0.860*tab['logL'] - 0.58*np.log10(tab['Mf']) -\
                          3.43*tab['logT'] + 0.024*tab['logZ']
        
        # Select stars in IS
        rrl = tab[np.logical_and.reduce([tab['logT'] < -0.080*tab['logL'] - 0.012*tab['logZ'] + 3.957,
                                         tab['logT'] > -0.084*tab['logL'] - 0.012*tab['logZ'] + 3.879])]
        # Stars with pulsation periods < 1 day
        rrl = rrl[np.logical_and.reduce([10**rrl['logP'] < 1])]
                                         
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            ts = TabSaver(self.p,self.a,**kwargs)
            ts.poptab_save(rrl,mode_comp,self.mode_iso,self.R,'rrl')
            if len(rrl['logT'])==0:
                print('\t','{:<3}'.format(mode_comp),': RR Lyrae sample is empty.')
                
        return rrl     


    def cepheids_type1(self,mode_comp,**kwargs):
        """
        Selects classical Cepheids. HB stars in the instability 
        strip (IS) with masses in the range 4-20 Msun. IS adopted
        from De Somma et al 2020a (mixing length parameter = 1.5,
        canonical mass-luminosity relation), periods calclated 
        according to De Somma et al.2020b. Dy default, these are 
        the fundamental pulsators, but also the first obertone 
        (FO) pulsators can be selected. 
        
        Parameters
        ----------
        mode_comp : str
            'd', 't' or 'sh' (thin disk, thick disk or halo).
        **kwargs : dict, optional keyord arguments
            FO   : boolean
                If True, the first obertone pulsators are selected.
            save : boolean
                If True, the output table is saved. 

        Returns
        -------
        ceph : astropy table
            Table of the stellar assemblies that belong to the 
            Cepheid type I population.
        """
        
        this_function = inspect.stack()[0][3]
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh'],'Cepheids type I',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function)
        
        tab = self.tables[mode_comp]
        # Definition of the instability strip (IS) is uses logZ and logP:
        amr = AMR()
        Z = amr.fe2z(tab['FeH'])
        tab['logZ'] = np.log10(Z)
        if inpcheck_iskwargtype(kwargs,'FO',True,bool,this_function):
            tab['logP'] = 10.268 - 3.192*tab['logT'] - 0.758*np.log10(tab['Mf']) + 0.919*tab['logL']
        else:
            tab['logP'] = 10.595 - 3.253*tab['logT'] - 0.621*np.log10(tab['Mf']) + 0.804*tab['logL']
        
        # Select stars in IS
        ceph = tab[np.logical_and.reduce([tab['logT'] < 0.104*tab['logL'] - 0.023*tab['logL']**2 + 3.667,
                                          tab['logT'] > 0.030*tab['logL'] - 0.018*tab['logL']**2 + 3.788])]
        # Stars with 4 < mass/Msun < 20
        ceph = ceph[np.logical_and.reduce([ceph['Mf'] > 4,ceph['Mf'] < 20])]
                                         
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            ts = TabSaver(self.p,self.a,**kwargs)
            ts.poptab_save(ceph,mode_comp,self.mode_iso,self.R,'ceph')
            if len(ceph['logT'])==0:
                print('\t','{:<3}'.format(mode_comp),': Cepheid Type I sample is empty.')
                
        return ceph     





# =============================================================================
# Functions not designed for user 
# =============================================================================

def _rhoz_d_(p,a,**kwargs):
    """
    Calculates densities of all thin-disk populations at all R and all z. 
    Can work with stellar assemblies table. 
    """
    if 'zlim' in kwargs:
        zlow,zup = kwargs['zlim']
        indz1,indz2 = int(zlow//p.dz),int(zup//p.dz)
        nz = indz2 - indz1
    else:
        indz1,indz2 = 0,a.n
        nz = a.n
    
    only_local = False
    if 'R' in kwargs:
        if kwargs['R']==p.Rsun:
            only_local = True
        else:
            Rarray = [kwargs['R']]
            Rindex = [int(kwargs['R']//p.dR - p.Rmin//p.dR)]
    else:
        Rarray = a.R
        Rindex = np.arange(a.Rbins)
        
    if only_local:
        Hd0,Phi0,AVRd0 = tab_reader(['Hd0','Phi0','AVR0'],p,a.T)
    else:
        Hd,Phi,AVRd = tab_reader(['Hd','Phi','AVR'],p,a.T)
        if not inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]):
            Hd0,Phi0,AVRd0 = tab_reader(['Hd0','Phi0','AVR0'],p,a.T)        
                                                               
    if ('mode_pop' in kwargs) or ('tab' in kwargs):
        if ('mode_pop' in kwargs):
            if 'mode_iso' not in kwargs:
                mode_iso = 'Padova'
            else:
                mode_iso = kwargs['mode_iso']
                
            if not only_local:
                tabs = [tab_reader(kwargs['mode_pop'],p,a.T,R=radius,mode='d',
                                   mode_iso=mode_iso,tab=True) for radius in Rarray]  
                if not inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]):
                    tab0 = tab_reader(kwargs['mode_pop'],p,a.T,R=p.Rsun,
                                      mode='d',mode_iso=mode_iso,tab=True)
            else:
                tab0 = tab_reader(kwargs['mode_pop'],p,a.T,R=p.Rsun,
                                  mode='d',mode_iso=mode_iso,tab=True)
                
        if ('tab' in kwargs) and ('mode_pop' not in kwargs):
            if only_local:
                tab0 = kwargs['tab']
            else:
                if inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]):
                    if len(Rarray)==1:
                        tabs = [kwargs['tab']] 
                    else:
                        tabs = kwargs['tab']
                else:
                    if len(Rarray)==1:
                        tabs, tab0 = [kwargs['tab'][0]], kwargs['tab'][1]
                    else:
                        tabs, tab0 = kwargs['tab']
        
        if inpcheck_iskwargtype(kwargs,'number',True,bool,inspect.stack()[0][3]):
            column = 'N'
        else:
            column = 'Sigma'
        
        if only_local:
            tab0_rd = reduce_table(tab0,a)
            Sigma0 = tab0_rd[column]
        else:
            tabs_rd = [reduce_table(table,a) for table in tabs]
            Sigma = [table[column] for table in tabs_rd]
            if not inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]):
                tab0_rd = reduce_table(tab0,a)
                Sigma0 = tab0_rd[column]      
    else:
        if not inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]) or only_local:
            SFRd0,gd0 = tab_reader(['SFRd0','gd0'],p,a.T)
            Sigma0 = SFRd0[1]*gd0[1]*tr
        if not only_local:
            SFRd,gd = tab_reader(['SFRd','gd'],p,a.T)
            Sigma = [SFRd[i+1]*gd[i+1]*tr for i in Rindex]

    # Locally at Rsun.
    if p.pkey==1:
        npeak = len(p.sigp)
        if not only_local:
            Hdp = [tab_reader(['Hdp'],p,a.T,R=radius)[0] for radius in Rarray]
            sigp = [table[0] for table in Hdp]
            Fp = [tab_reader(['Fp'],p,a.T,R=radius)[0] for radius in Rarray]
            fpr0 = [1 - np.sum(subarray[1:],axis=0) for subarray in Fp]
        if not inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]) or only_local:
            Fp0, Hdp0 = tab_reader(['Fp0','Hdp0'],p,a.T)
            fp0 = 1 - np.sum(Fp0[1:],axis=0)
            # If there are extra peaks on the thin-disk SFR, that have special kinematics, 
            # the density profile consists of two terms: standard thin-disk part and peaks. 
            if inpcheck_iskwargtype(kwargs,'sigma',True,bool,inspect.stack()[0][3]):
                rho_d0 = fp0*Sigma0
                rho_dp = [Fp0[l+1]*Sigma0 for l in np.arange(npeak)]
            else:
                rho_d0 = fp0*Sigma0/2/Hd0[1]
                rho_dp = [Fp0[l+1]*Sigma0/2/Hdp0[1][l] for l in np.arange(npeak)]
            rho_d0_term = np.array([rho_d0*np.exp(-k/KM**2/AVRd0[1]**2) for k in Phi0[1][indz1:indz2]])
            rho_dp_term = np.array([[rho_dp[l]*np.exp(-k/KM**2/p.sigp[l]**2) for k in Phi0[1][indz1:indz2]]
                           for l in np.arange(npeak)])
            rho_z0 = np.add(rho_d0_term,np.sum(rho_dp_term,axis=0)) 
    else:
        if not inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]) or only_local:
            if inpcheck_iskwargtype(kwargs,'sigma',True,bool,inspect.stack()[0][3]):
                rho_z0 = np.array([Sigma0*np.exp(-k/KM**2/AVRd0[1]**2) for k in Phi0[1][indz1:indz2]])
            else:
                rho_z0 = np.array([Sigma0/2/Hd0[1]*np.exp(-k/KM**2/AVRd0[1]**2) for k in Phi0[1][indz1:indz2]])
    
    if not only_local:
        # All other distances. 
        rho_z = np.zeros((len(Rarray),nz,a.jd))
        for i in range(len(Rarray)):
            if p.pkey==1:
                if inpcheck_iskwargtype(kwargs,'sigma',True,bool,inspect.stack()[0][3]):
                    rho_d0 = fpr0[i]*Sigma[i]
                    rho_dp = [Fp[i][l+1]*Sigma[i] for l in np.arange(npeak)]
                else:
                    rho_d0 = fpr0[i]*Sigma[i]/2/Hd[Rindex[i]+1]
                    rho_dp = [Fp[i][l+1]*Sigma[i]/2/Hdp[i][1][l] for l in np.arange(npeak)]
                rho_d0_term = np.array([rho_d0*np.exp(-k/KM**2/AVRd[Rindex[i]+1]**2) 
                                        for k in Phi[Rindex[i]+1][indz1:indz2]])
                rho_dp_term = np.array([[rho_dp[l]*np.exp(-k/KM**2/sigp[i][l]**2) 
                                         for k in Phi[Rindex[i]+1][indz1:indz2]] 
                                         for l in np.arange(npeak)])
                rho_z[i] = np.add(rho_d0_term,np.sum(rho_dp_term,axis=0)) 
            else:
                if inpcheck_iskwargtype(kwargs,'sigma',True,bool,inspect.stack()[0][3]):
                    rho_z[i] = [Sigma[i]*np.exp(-k/KM**2/AVRd[Rindex[i]+1]**2) 
                                for k in Phi[Rindex[i]+1][indz1:indz2]]
                else:
                    rho_z[i] = [Sigma[i]/2/Hd[Rindex[i]+1]*np.exp(-k/KM**2/AVRd[Rindex[i]+1]**2) 
                                for k in Phi[Rindex[i]+1][indz1:indz2]]
    
    if only_local:
        return rho_z0
    else:
        if not inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]):                   
            return (rho_z, rho_z0)
        else:
            return rho_z


def _rhoz_t_(p,a,**kwargs):
    """
    Calculates densities of all thick-disk populations at all R and all z. 
    Can work with stellar assemblies table. 
    """
    if 'zlim' in kwargs:
        zlow,zup = kwargs['zlim']
        indz1,indz2 = int(zlow//p.dz),int(zup//p.dz)
        nz = indz2 - indz1
    else:
        indz1,indz2 = 0,a.n
        nz = a.n
    
    only_local = False
    if 'R' in kwargs:
        if kwargs['R']==p.Rsun:
            only_local = True
        else:
            Rarray = [kwargs['R']]
            Rindex = [int(kwargs['R']//p.dR - p.Rmin//p.dR)]
    else:
        Rarray = a.R
        Rindex = np.arange(a.Rbins)
    
    if only_local or (not inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3])):
        Ht0, AMRt, Phi0 = tab_reader(['Ht0','AMRt','Phi0'],p,a.T)
    if not only_local:
        Ht, Phi, Sigt = tab_reader(['Ht','Phi','Sigt'],p,a.T)
                                                      
    if ('mode_pop' in kwargs) or ('tab' in kwargs):
        if ('mode_pop' in kwargs):
            if 'mode_iso' not in kwargs:
                mode_iso = 'Padova'
            else:
                mode_iso = kwargs['mode_iso']
            
            if not only_local:
                tabs = [tab_reader(kwargs['mode_pop'],p,a.T,R=radius,mode='t',
                                   mode_iso=mode_iso,tab=True)  for radius in Rarray]
                if not inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]):
                    tab0 = tab_reader(kwargs['mode_pop'],p,a.T,R=p.Rsun,
                                      mode='t',mode_iso=mode_iso,tab=True)
            else:
                tab0 = tab_reader(kwargs['mode_pop'],p,a.T,R=p.Rsun,
                                      mode='t',mode_iso=mode_iso,tab=True)
                
        if ('tab' in kwargs) and ('mode_pop' not in kwargs):
            if only_local:
                tab0 = kwargs['tab']
            else:
                if inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]):
                    if len(Rarray)==1:
                        tabs = [kwargs['tab']] 
                    else:
                        tabs = kwargs['tab']
                else:
                    if len(Rarray)==1:
                        tabs, tab0 = [kwargs['tab'][0]], kwargs['tab'][1]
                    else:
                        tabs, tab0 = kwargs['tab']
        
        if inpcheck_iskwargtype(kwargs,'number',True,bool,inspect.stack()[0][3]):
            column = 'N'
        else:
            column = 'Sigma'
        
        if only_local:
            tab0_rt = reduce_table(tab0,a)
            Sigma0 = tab0_rt[column]
        else:
            tabs_rt = [reduce_table(table,a) for table in tabs]
            Sigma = [table[column] for table in tabs_rt]
            if not inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]):
                tab0_rt = reduce_table(tab0,a)
                Sigma0 = tab0_rt[column]      
    else:
        gt = tab_reader(['gt'],p,a.T)[0]
        if not only_local:
            SFRt = tab_reader(['SFRt'],p,a.T)[0]
            Sigma = [SFRt[i+1]*gt[1]*tr for i in Rindex]
        if not inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]) or only_local:
            SFRt0 = tab_reader(['SFRt0'],p,a.T)[0]
            Sigma0 = SFRt0[1]*gt[1]*tr
            
    if not inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]) or only_local:
        # Locally at Rsun.
        rho_z0 = np.zeros((nz,a.jd))
        if inpcheck_iskwargtype(kwargs,'sigma',True,bool,inspect.stack()[0][3]):
            rho0 = Sigma0
        else:
            rho0 = Sigma0/2/Ht0[1]
        if ('mode_pop' in kwargs) or ('tab' in kwargs):
            rho_z0 = np.array([rho0*np.exp(-k/KM**2/p.sigt**2) for k in Phi0[1][indz1:indz2]])
        else:
            rho_z0[:,:a.jt] = np.array([rho0*np.exp(-k/KM**2/p.sigt**2) for k in Phi0[1][indz1:indz2]])
        
    # All other distances. 
    if not only_local:
        rho_z = np.zeros((len(Rarray),nz,a.jd))
        for i in range(len(Rarray)):
            if inpcheck_iskwargtype(kwargs,'sigma',True,bool,inspect.stack()[0][3]):
                rho0 = Sigma[i]
            else:
                rho0 = Sigma[i]/2/Ht[1][Rindex[i]]
            if ('mode_pop' in kwargs) or ('tab' in kwargs):
                rho_z[i] = [rho0*np.exp(-k/KM**2/Sigt[1][Rindex[i]]**2) 
                            for k in Phi[Rindex[i]+1][indz1:indz2]]
            else:
                rho_z[i,:,:a.jt] = np.array([rho0*np.exp(-k/KM**2/Sigt[1][Rindex[i]]**2)
                                             for k in Phi[Rindex[i]+1][indz1:indz2]])
    
    if only_local:
        return rho_z0
    else:
        if not inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]):
            return (rho_z,rho_z0)
        else:
            return rho_z


def _rhoz_sh_(p,a,**kwargs):
    """
    Calculates densities of stellar halo populations at all R and all z. 
    Can work with stellar assemblies table. 
    """
    if 'zlim' in kwargs:
        zlow,zup = kwargs['zlim']
        indz1,indz2 = int(zlow//p.dz),int(zup//p.dz)
        nz = indz2 - indz1
    else:
        indz1,indz2 = 0,a.n
        nz = a.n
        
    only_local = False
    if 'R' in kwargs:
        if kwargs['R']==p.Rsun:
            only_local = True
        else:
            Rarray = [kwargs['R']]
            Rindex = [int(kwargs['R']//p.dR - p.Rmin//p.dR)]
    else:
        Rarray = a.R
        Rindex = np.arange(a.Rbins)
    
    if not only_local:
        Phi, SR, Hsh = tab_reader(['Phi','SR','Hsh'],p,a.T)
    if not inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]) or only_local:
        Phi0, Hsh0 = tab_reader(['Phi0','Hsh0'],p,a.T) 
                                            
    if ('mode_pop' in kwargs) or ('tab' in kwargs):
        if ('mode_pop' in kwargs):
            if 'mode_iso' not in kwargs:
                mode_iso = 'Padova'
            else:
                mode_iso = kwargs['mode_iso']
            if not only_local:
                tabs = [tab_reader(kwargs['mode_pop'],p,a.T,R=radius,
                                   mode='sh',mode_iso=mode_iso,tab=True) for radius in Rarray]
                if not inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]):
                    tab0 = tab_reader(kwargs['mode_pop'],p,a.T,R=p.Rsun,
                                      mode='sh',mode_iso=mode_iso,tab=True)   
            else:
                tab0 = tab_reader(kwargs['mode_pop'],p,a.T,R=p.Rsun,mode='sh',mode_iso=mode_iso,
                                  tab=True)
            
        if ('tab' in kwargs) and ('mode_pop' not in kwargs):
            if only_local:
                tab0 = kwargs['tab']
            else:
                if inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]):
                    if len(Rarray)==1:
                        tabs = [kwargs['tab']] 
                    else:
                        tabs = kwargs['tab']
                else:
                    if len(Rarray)==1:
                        tabs, tab0 = [kwargs['tab'][0]], kwargs['tab'][1]
                    else:
                        tabs, tab0 = kwargs['tab']
                            
        if inpcheck_iskwargtype(kwargs,'number',True,bool,inspect.stack()[0][3]):
            column = 'N'
        else:
            column = 'Sigma'
        
        if only_local:
            tab0_rsh = reduce_table(tab0,a)
            Sigma0 = tab0_rsh[column]
        else:
            tabs_rsh = [reduce_table(table,a) for table in tabs]
            Sigma = [table[column] for table in tabs_rsh]
            if not inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]):
                tab0_rsh = reduce_table(tab0,a)
                Sigma0 = tab0_rsh[column]   
    else:
        if not only_local:
            Sigma = []
            for i in range(len(Rarray)):
                Sigma_i = np.zeros((a.jd))
                Sigma_i[0] = SR[-1][i]
                Sigma.append(Sigma_i)
        if not inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]) or only_local:
            Sigma0 = np.zeros((a.jd))
            Sigma0[0] = p.sigmash
            
    # Locally at Rsun.
    if not inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]) or only_local:
        rho_sh0 = np.array([Sigma0/2/Hsh0[1]*np.exp(-k/KM**2/p.sigsh**2) 
                          for k in Phi0[1][indz1:indz2]])
    
    # All other distances. 
    if not only_local:
        rho_sh = np.zeros((len(Rarray),nz,a.jd)) 
        for i in range(len(Rarray)):
            if inpcheck_iskwargtype(kwargs,'sigma',True,bool,inspect.stack()[0][3]):
                rho0 = Sigma[i]
            else:
                rho0 = Sigma[i]/2/Hsh[1][Rindex[i]]
            rho_sh[i] = np.array([rho0*np.exp(-k/KM**2/p.sigsh**2)
                                 for k in Phi[Rindex[i]+1][indz1:indz2]])
            
    if only_local:
        return rho_sh0
    else:
        if not inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]):
            return (rho_sh, rho_sh0)
        else:
            return rho_sh


def _rhomet_d_(R,mets,p,a,this_function,**kwargs):
    """
    Calculates thin-disk density of mono-metallicity populations
    given by mets bins, at distance R. 
    """
    nage = len(mets) - 1 
    if R==p.Rsun:
        AMR = tab_reader(['AMRd0'],p,a.T)[0][1]
    else:
        indr = int(_indr_(R,p,a) + 1)
        AMR = tab_reader(['AMRd'],p,a.T)[0][indr]
    indt = _ind_amr2t_(AMR,mets)
    if indt!=[] and len(indt)>1:
        if R==p.Rsun:
            rho_zd = _rhoz_d_(p,a,R=R,**kwargs)
        else:
            if inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]):
                rho_zd = _rhoz_d_(p,a,R=R,**kwargs)[0]
            else:
                rho_zd = _rhoz_d_(p,a,R=R,**kwargs)[0][0]
        rho_z = _rhoz_ages_(rho_zd,indt,this_function,a,between=True,**kwargs)
    else:
        rho_z = np.zeros((nage,a.n))
    return rho_z


def _rhomet_t_(R,mets,p,a,this_function,**kwargs):
    """
    Calculates thick-disk density of mono-metallicity populations
    given by mets bins, at distance R. 
    """
    nage = len(mets)
    AMR = tab_reader(['AMRt'],p,a.T)[0][1]
    indt = _ind_amr2t_(AMR,mets)
    if indt!=[] and len(indt)>1:
        if R==p.Rsun:
            rho_zt = _rhoz_t_(p,a,R=R,**kwargs)
        else:
            if inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]):
                rho_zt = _rhoz_t_(p,a,R=R,**kwargs)[0]
            else:
                rho_zt = _rhoz_t_(p,a,R=R,**kwargs)[0][0]
        rho_z = _rhoz_ages_(rho_zt,indt,this_function,a,between=True,**kwargs)
    else:
        rho_z = np.zeros((a.n,nage-1))
    return rho_z
    

def _rhomet_sh_(R,mets,p,a,this_function,**kwargs):
    """
    Calculates stellar halo density of mono-metallicity populations
    given by mets bins, at distance R. 
    """
    nage = len(mets) - 1
    metmin, metmax = p.FeHsh - 3*p.dFeHsh, p.FeHsh + 3*p.dFeHsh
    if R==p.Rsun:
        rho_zsh = _rhoz_sh_(p,a,R=R,**kwargs)
    else:
        if inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]):
            rho_zsh = _rhoz_sh_(p,a,R=R,**kwargs)[0]
        else:
            rho_zsh = _rhoz_sh_(p,a,R=R,**kwargs)[0][0]
    rho_z = np.zeros((nage,a.n))
    for i in range(nage-1):
        if (mets[i]>=metmin) and (mets[i+1]<=metmax):
            t1 = (mets[i]-p.FeHsh)/np.sqrt(2)/p.dFeHsh
            t2 = (mets[i+1]-p.FeHsh)/np.sqrt(2)/p.dFeHsh
            rho_z[i] = rho_zsh[:,i]*(erf(t2)-erf(t1))/erf(3/np.sqrt(2))  
    return rho_z


def _rhometz_d_(zlim,mets,p,a,this_function,**kwargs):
    """
    Calculates thin-disk density of mono-metallicity populations
    given by mets bins, in horizontal slice zlim. 
    """
    nage = len(mets) - 1 
    rho_r = np.zeros((nage,a.Rbins))
    AMR = tab_reader(['AMRd'],p,a.T)[0]
    
    tables = False
    if 'tab' in kwargs:
        tabs = kwargs['tab']
        del kwargs['tab']
        tables = True
        
    for i in range(a.Rbins):
        indt = _ind_amr2t_(AMR[i+1],mets)
        if indt!=[] and len(indt)>1:
            if inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]):
                if tables:
                    rho_zd = _rhoz_d_(p,a,R=a.R[i],zlim=zlim,**kwargs,tab=tabs[i])[0]
                else:
                    rho_zd = _rhoz_d_(p,a,R=a.R[i],zlim=zlim,**kwargs)[0]
            else:
                if tables:
                    rho_zd = _rhoz_d_(p,a,R=a.R[i],zlim=zlim,**kwargs,tab=tabs[i])[0][0]
                else:
                    rho_zd = _rhoz_d_(p,a,R=a.R[i],zlim=zlim,**kwargs)[0][0]
            rho_r[:,i] = _rhor_ages_(rho_zd,indt,this_function,a,between=True,**kwargs)
    return rho_r


def _rhometz_t_(zlim,mets,p,a,this_function,**kwargs):
    """
    Calculates thick-disk density of mono-metallicity populations
    given by mets bins, in horizontal slice zlim. 
    """
    nage = len(mets) - 1 
    rho_r = np.zeros((nage,a.Rbins))
    AMR = tab_reader(['AMRt'],p,a.T)[0][1]
    indt = _ind_amr2t_(AMR,mets)
    
    tables = False
    if 'tab' in kwargs:
        tabs = kwargs['tab']
        del kwargs['tab']
        tables = True
        
    if indt!=[] and len(indt)>1:
        for i in range(a.Rbins):
            if inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]):
                if tables:
                    rho_zt = _rhoz_t_(p,a,R=a.R[i],zlim=zlim,**kwargs,tab=tabs[i])[0]
                else:
                    rho_zt = _rhoz_t_(p,a,R=a.R[i],zlim=zlim,**kwargs)[0]
            else:
                if tables:
                    rho_zt = _rhoz_t_(p,a,R=a.R[i],zlim=zlim,**kwargs,tab=tabs[i])[0]
                else:
                    rho_zt = _rhoz_t_(p,a,R=a.R[i],zlim=zlim,**kwargs)[0][0]
            rho_r[:,i] = _rhor_ages_(rho_zt,indt,this_function,a,between=True,**kwargs)
    return rho_r
    

def _rhometz_sh_(zlim,mets,p,a,this_function,**kwargs):
    """
    Calculates stellar halo density of mono-metallicity populations
    given by mets bins, in horizontal slice zlim. 
    """
    nage = len(mets) - 1 
    metmin, metmax = p.FeHsh - 3*p.dFeHsh, p.FeHsh + 3*p.dFeHsh
    if inpcheck_iskwargtype(kwargs,'local',False,bool,inspect.stack()[0][3]):
        rho_zsh = _rhoz_sh_(p,a,zlim=zlim,**kwargs)
    else:
        rho_zsh = _rhoz_sh_(p,a,zlim=zlim,**kwargs)[0]
    rho_r = np.zeros((nage,a.Rbins))
    for k in range(a.Rbins):
        for i in range(nage):
            if (mets[i]>=metmin) and (mets[i+1]<=metmax):
                t1 = (mets[i]-p.FeHsh)/np.sqrt(2)/p.dFeHsh
                t2 = (mets[i+1]-p.FeHsh)/np.sqrt(2)/p.dFeHsh
                rho_r[i,k] = np.sum(np.sum(rho_zsh[k]))*(erf(t2)-erf(t1))/erf(3/np.sqrt(2))  
    return rho_r


def _rhoz_ages_(rhoz_input,indt,this_function,a,**kwargs):
    """
    Selects vertical density profiles of rhoz_input that correspond 
    to a set of ages given by the indices indt. 
    """
    if inpcheck_iskwargtype(kwargs,'between',True,bool,this_function):
        nage = len(indt) - 1 
        rho_z = np.zeros((nage,rhoz_input.shape[0]))
        for i in range(nage):
            if (indt[i+1]!=-999) and (indt[i]!=-999):
                indt1, indt2 = np.sort([indt[i+1],indt[i]])
                i1 = np.arange(indt1,indt2,dtype=np.int)
                rho_z[i] = np.sum(rhoz_input.T[i1,:],axis=0)
    else:
        indt_good = np.where(indt!=-999)[0]
        rho_z = rhoz_input.T[indt[indt_good],:]
    return rho_z


def _rhor_ages_(rhor_input,indt,this_function,a,**kwargs):
    """
    Selects radial density profiles of rhoz_input that correspond 
    to a set of ages given by the indices indt. 
    """
    if inpcheck_iskwargtype(kwargs,'between',True,bool,this_function):
        nage = len(indt) - 1 
        rho_r = np.zeros((nage,rhor_input.shape[0]))
        for i in range(nage):
            if (indt[i+1]!=-999) and (indt[i]!=-999):
                indt1, indt2 = np.sort([indt[i+1],indt[i]])
                i1 = np.arange(indt1,indt2,dtype=np.int)
                rho_r[i] = np.sum(rhor_input.T[i1,:],axis=0)
    else:
        indt_good = np.where(indt!=-999)[0]
        rho_r = rhor_input.T[indt[indt_good],:]
    rho_r = np.sum(rho_r,axis=1)
    return rho_r


def _ind_amr2t_(amr_array,mets):
    """
    Returns time-indices corresponding to the given set of 
    metallicities mets using the age-metallicity relation amr_array. 
    """
    metmin, metmax = np.amin(amr_array), np.amax(amr_array)
    indt = [] 
    for i in range(len(mets)):
        if (metmin<=mets[i]) and (mets[i]<=metmax):
            indt.append(np.where(np.abs(amr_array-mets[i])==np.amin(np.abs(amr_array-mets[i])))[0][0])
        else:
            indt.append(-999)
    indt = np.array(indt)
    return indt


def _meanq_d_(indr,indz,indt,tab,Fi,Hd,AVR,p,a,**kwargs):
    """
    Mean number density of thin-disk populations given by indt
    indices at R,z given by indr and indz indices. 
    """
    if p.pkey==1:
        weight1 = kwargs['fp0']*0.5/Hd[indr]*np.exp(-Fi[indr][indz]/KM**2/AVR[indr]**2)
        weights2 = np.array([kwargs['Fp'][l+1]*0.5/kwargs['Hdp'][1][l]*\
                             np.exp(-Fi[indr][indz]/KM**2/kwargs['sigp'][l]**2) 
                             for l in np.arange(kwargs['npeak'])])                    
        Nz_term1 = tab['N']*weight1[indt]
        Nz_term2 = np.array([tab['N']*weights2[l][indt] for l in np.arange(kwargs['npeak'])])
        Nz = np.add(Nz_term1,np.sum(Nz_term2,axis=0))
        return Nz, Nz_term1, Nz_term2
    else:
        weight = 0.5/Hd[indr][indt]*np.exp(-Fi[indr][indz]/KM**2/AVR[indr][indt]**2)
        Nz = tab['N']*weight[indt]
        return Nz
    
    
def _meanq_t_(indr,indz,tab,Fi,Ht,Sigt,p,a,**kwargs):
    """
    Mean number density of thick-disk populations at R,z given 
    by indr and indz indices. 
    """
    weight = 0.5/Ht[1][indr-1]*np.exp(-Fi[indr][indz]/KM**2/Sigt[1][indr-1]**2)
    Nz = tab['N']*weight
    return Nz


def _meanq_sh_(indr,indz,tab,Fi,Hsh,p,a,**kwargs):
    """
    Mean number density of halo populations at R,z given 
    by indr and indz indices. 
    """
    weight = 0.5/Hsh[1][indr-1]*np.exp(-Fi[indr][indz]/KM**2/p.sigsh**2)
    Nz = tab['N']*weight
    return Nz


def _fw_hist_d_(N_d,sig_d,wgrid,dw,p,a,**kwargs):
    """
    Returns thin-disk |W|-velocity distribution given the number
    densities of populations N_d and their velocity dispersions
    sig_d. 
    """    
    sq2 = np.sqrt(2)
    fw_hist = np.zeros((len(wgrid)))
    
    if p.pkey==1:
        N_d0_term, N_dp_term = N_d
        AVR, sigp = sig_d
        npeak = len(sigp)
    else:
        N_d, AVR = N_d, sig_d
    if 'indt' in kwargs:
        AVR = AVR[kwargs['indt']]
        if p.pkey==1:
            N_d0_term = N_d0_term[kwargs['indt']]
            N_dp_term = [subtable[kwargs['indt']] for subtable in N_dp_term]
        else:
            N_d = N_d[kwargs['indt']]
        
    for i in range(len(wgrid)):
        wi = wgrid[i]
        if p.pkey==1:
            term0 = N_d0_term*(erf((wi+dw/2)/sq2/AVR)-erf((wi-dw/2)/sq2/AVR))
            termp = [np.sum(N_dp_term[k])*(erf((wi+dw/2)/sq2/sigp[k])-erf((wi-dw/2)/sq2/sigp[k]))
                            for k in np.arange(npeak)]
            fw_hist[i] = np.sum(term0) + np.sum(termp)
        else:
            fw_hist[i] = np.sum(N_d*(erf((wi+dw/2)/sq2/AVR)-erf((wi-dw/2)/sq2/AVR)))
    return fw_hist
        

def _fw_hist_t_(N_t,sig_t,wgrid,dw,p,a,**kwargs):
    """
    Returns thick-disk |W|-velocity distribution given the number
    densities of populations N_t and their velocity dispersion
    sig_t. 
    """
    sq2 = np.sqrt(2)
    fw_hist = np.zeros((len(wgrid)))
    if 'indt' in kwargs:
        N_t = N_t[kwargs['indt']]
    
    for i in range(len(wgrid)):
        fw_hist[i] = np.sum(N_t*(erf((i+dw/2)/sq2/sig_t)-erf((i-dw/2)/sq2/sig_t)))
    return fw_hist
             
    
def _fw_hist_sh_(N_sh,wgrid,dw,p,a,**kwargs):
    """
    Returns halo |W|-velocity distribution given the number
    densities of populations N_sh.
    """
    sq2 = np.sqrt(2)
    fw_hist = np.zeros((len(wgrid)))
    if 'indt' in kwargs:
        N_sh = N_sh[kwargs['indt']]
        
    for i in range(len(wgrid)):
        fw_hist[i] = np.sum(N_sh*(erf((i+dw/2)/sq2/p.sigsh)-erf((i-dw/2)/sq2/p.sigsh))) 
    return fw_hist
           

def _fw_d_(R,zlim,wgrid,dw,p,a,**kwargs):
    """
    Thin-disk |W|-velocity distribution at R in slice zlim. 
    """
    kwargs_calc = kwargs.copy()
    if R==p.Rsun:
        AVR = tab_reader(['AVR0'],p,a.T)[0][1]
    else:
        indr = int(_indr_(R,p,a) + 1)
        AVR = tab_reader(['AVR'],p,a.T)[0][indr]
    if R==p.Rsun:
        Nz_d = _rhoz_d_(p,a,R=R,zlim=zlim,**kwargs_calc)
    else:
        Nz_d = _rhoz_d_(p,a,R=R,zlim=zlim,**kwargs_calc,local=False)[0]
    N_d = np.sum(Nz_d,axis=0)
    
    if 'mets' in kwargs_calc:
        if R==p.Rsun:
            AMR = tab_reader(['AMRd0'],p,a.T)[0][1]
        else:
            AMR = tab_reader(['AMRd'],p,a.T)[0][indr]
        inddt = _ind_amr2t_(AMR,kwargs_calc['mets'])
        if inddt[0]!=-999 and inddt[1]!=-999:
            kwargs_calc['indt'] = inddt
        else:
            metmin,metmax = np.amin(AMR),np.amax(AMR)
            if ((kwargs_calc['mets'][0]<=metmin and kwargs_calc['mets'][1]<=metmin) or 
                (kwargs_calc['mets'][0]>=metmax and kwargs_calc['mets'][1]>=metmax)):
                kwargs_calc['indt'] = [0,0]
            else:
                i1, i2 = inddt
                if inddt[0]==-999:
                    i1 = 0
                if inddt[1]==-999:
                    i2 = int(a.jd - 1) 
                kwargs_calc['indt'] = np.arange(i1,i2,dtype=np.int)
    if p.pkey==1:
        if R==p.Rsun:
            sigp = p.sigp
            Fp = tab_reader(['Fp0'],p,a.T)[0]                
        else:
            sigp = tab_reader(['Hdp'],p,a.T,R=R)[0][0]
            Fp = tab_reader(['Fp'],p,a.T,R=R)[0]                
        fp = 1 - np.sum(Fp[1:],axis=0)
        N_d0_term = N_d*fp
        N_dp_term = [N_d*subtable for subtable in Fp[1:]]
        fwhist_d = _fw_hist_d_([N_d0_term,N_dp_term],[AVR,sigp],wgrid,dw,p,a,**kwargs_calc)
    else:
        fwhist_d = _fw_hist_d_(N_d,AVR,wgrid,dw,p,a,**kwargs_calc)
    return fwhist_d


def _fw_t_(R,zlim,wgrid,dw,p,a,**kwargs):
    """
    Thick-disk |W|-velocity distribution at R in slice zlim. 
    """
    kwargs_calc = kwargs.copy()
    if 'mets' in kwargs_calc:
        AMR = tab_reader(['AMRt'],p,a.T)[0]
        indtt = _ind_amr2t_(AMR,kwargs['mets'])
        if indtt[0]!=-999 and indtt[1]!=-999:
            kwargs_calc['indt'] = indtt
        else:
            metmin,metmax = np.amin(AMR),np.amax(AMR)
            if ((kwargs_calc['mets'][0]<=metmin and kwargs_calc['mets'][1]<=metmin) or 
                (kwargs_calc['mets'][0]>=metmax and kwargs_calc['mets'][1]>=metmax)):
                kwargs_calc['indt'] = [0,0]
            else:
                i1, i2 = indtt
                if indtt[0]==-999:
                    i1 = 0
                if indtt[1]==-999:
                    i2 = a.jt
                kwargs_calc['indt'] = [i1,i2]
    if R==p.Rsun:
        Nz_t = _rhoz_t_(p,a,R=R,zlim=zlim,**kwargs_calc)
    else:
        Nz_t = _rhoz_t_(p,a,R=R,zlim=zlim,**kwargs_calc,local=False)[0]
    N_t = np.sum(Nz_t,axis=0)
    
    if R==p.Rsun:
        Sigt = p.sigt
    else:
        indr = int(_indr_(R,p,a) + 1)
        Sigt = tab_reader(['Sigt'],p,a.T)[0][1][indr-1]
    fwhist_t = _fw_hist_t_(N_t,Sigt,wgrid,dw,p,a,**kwargs_calc)
    return fwhist_t


def _fw_sh_(R,zlim,wgrid,dw,p,a,**kwargs):
    """
    Halo |W|-velocity distribution at R in slice zlim. 
    """
    kwargs_calc = kwargs.copy()
    if R==p.Rsun:
        Nz_sh = _rhoz_sh_(p,a,R=R,zlim=zlim,**kwargs_calc)
    else:
        Nz_sh = _rhoz_sh_(p,a,R=R,zlim=zlim,**kwargs_calc,local=False)[0]
    N_sh = np.sum(Nz_sh,axis=0)
    if 'mets' in kwargs_calc:
        metmin, metmax = p.FeHsh - 3*p.dFeHsh, p.FeHsh + 3*p.dFeHsh
        if ((kwargs_calc['mets'][0]<=metmin and kwargs_calc['mets'][1]<=metmin) or 
            (kwargs_calc['mets'][0]>=metmax and kwargs_calc['mets'][1]>=metmax)):
            fwhist_sh = np.zeros((len(wgrid)))
        else:
            t1 = np.abs(kwargs_calc['mets'][0]-p.FeHsh)/np.sqrt(2)/p.dFeHsh
            t2 = np.abs(kwargs_calc['mets'][1]-p.FeHsh)/np.sqrt(2)/p.dFeHsh
            if kwargs_calc['mets'][0]<=metmin:
                t1 = 3/np.sqrt(2)
            if kwargs_calc['mets'][1]>=metmax:
                t2 = 3/np.sqrt(2)  
            Nw_sh = N_sh*np.abs(erf(t2)-erf(t1))/erf(3/np.sqrt(2))
            fwhist_sh = _fw_hist_sh_(Nw_sh,wgrid,dw,p,a,**kwargs_calc)         
    else:
        fwhist_sh = _fw_hist_sh_(N_sh,wgrid,dw,p,a,**kwargs_calc)
    return fwhist_sh


def _extend_mag_(mag,tab,bands):
    """
    Concatenates photometric lists.
    """
    mag[0].extend(tab[bands[0]])
    mag[1].extend(tab[bands[1]])
    mag[2].extend(tab[bands[2]])
    return mag


def _indr_(R,p,a):
    if (R >= p.Rmin - p.dR/2) and (R <= p.Rmax + p.dR/2):
        indr = int(np.where(a.R_edges - R < 0)[0][-1])
        return indr
    else:
        raise ValueError('R value is outside of the [Rmin,Rmax] interval.')
    
# =============================================================================
# Description of args and kwargs
# =============================================================================

def analysis_args_description():
    """
    mode_comp : str
        Galactic component, can be 'd', 't', 'sh', 'dt' or 'tot' 
        (thin disk, thick disk, halo, thin+thick disk or total). 
        For some functions not all of the options are available. 
    R : scalar
        Galactocentric distance, kpc.
    zlim : array-like
        Range of heights to be considered, in pc. For some 
        functions, len(zlim) can be > 2, i.e., zlim is a set of 
        horizontal z-slices. 
    mode_geom : str
        Modeled geometry. Applicable for Hess diagram calculation,
        stellar assembly table in volume, volume itself: can be 
        'local_sphere', 'local_cylinder' or 'rphiz_box'. For the
        calculation of disk brightness or colour profile, it is 
        disk orientation with respect to observer: 'face-on' or 
        'edge-on'. 
    ages : array_like
        Set of age-bins to be considered.
    mets : array_like
        Set of metallicity-bins to be considered. 
    quantity : str
        Name of column of the stellar assemblies table, for which 
        the function has to be applied; for velocity dispersion
        use 'sigw'. 
    bands : array-like
        List of names of photometric columns to be used in some 
        function. For calculation of the brightness profile, 
        give single column e.g. ['B'], for colour profile give
        e.g. ['B','V'] for 'B-V'. For Hess diagram three column
        names must be given, e.g. ['B','V','V'] for (B-V) versus 
        M_V. 
    mag_range : array-like
        Min-max magnitudes along the x- and y-axis of the Hess 
        diagram, [[x_min,x_max],[y_min,y_max]]. 
    mag_step : array-like
        Step along the x- and y-axis of the Hess diagram, [dx,dy]. 
    volume : array-like
        Volume in pc^3 in |z|-bins at a given R. The exact values
        in bins depend on the chosen geometry, see mode_geom. 
    p : namedtuple
        Set of the model parameters from the parameter file. 
    a : namedtuple
        Collection of the fixed model parameters, useful 
        quantities and arrays.
    """
    print(analysis_args_description.__doc__)
    

def analysis_kwargs_description():
    """
    save : boolean
        If True, the calculated quantities are saved as tables  
        to the output directory.
    between : boolean
        If true, the output quantity corresponds to the 
        age bins with edges in given ages, not to the individual 
        single ages (i.e. age-bins of width tr).
    mode_pop : str 
        Name of population. Can be one of the pre-defined 
        ('a','f','rc','rc+','gdw','kdw','mdw') or custom, if 
        it was selected and saved as table in advance. 
    mode_iso : str
        Stellar library, 'Padova' or 'MIST'. If not specified, 
        Padova isochrones are used.
    tab : dict or astropy table 
        Stellar assemblies table, alternative to mode_pop.  
    number : boolean
        If true, the calculated quantity is the spatial 
        number density of stars, not matter density. 
        Active only when mode_pop or tab are given. 
    sigma : boolean 
        If True, the result is surface density in Msun/pc^2, 
        otherwise the midplane matter density is calculated, 
        Msun/pc^3. In combination with number=True returns the 
        number surface density, i.e, number of stars/pc^2,
        otherwise the number of stars/pc^3 is returned. In some
        cases the quantity of interest is calculated as a weighted 
        mean with the mass/surface (number) densities as weights. 
    sigma_gauss : scalar 
        Standard deviation of the Gaussian kernel used to smooth
        the distribution. Same units as the calculated quantity.
    metbins : array-like
        Edges of metallicity bins for metallicity distribution. 
    dz : scalar
        Step in |z|, in pc. 
    wmax : scalar
        Maximum value of |W|-velocity, km/s. 
    dw : scalar
        Step in |W|-velocity, km/s. 
    ages : array-like
        Range of ages, in Gyr. 
    mets : array-like
        Range of metallicities. 
    vln : str
        String with info about chosen volume shape. 
    """
    print(analysis_kwargs_description.__doc__)


# =============================================================================
# Functions that calculate quantities of interest
# =============================================================================

def rhoz_monoage(mode_comp,R,ages,p,a,**kwargs):
    """
    Vertical profiles of the mono-age disk subpopulations 
    calculated at Galactocentric distance R.
    
    Parameters
    ----------
    mode_comp ('d', 't', 'sh', 'dt' or 'tot'), R, ages, p, a
    **kwargs : dict, optional keyword arguments
        valid keywords : save, between, mode_pop, tab, number,
        mode_iso
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()

    Returns
    -------
    rhoz : 2d-ndarray
        Array of the shape (len(ages),a.n) with the densities of 
        the selected mono-age subpopulations, in Msun/pc^3 or 
        number/pc^3 if number=True. 
    """
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save','mode_pop','number','tab','between','mode_iso'],this_function)
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','dt','sh','tot'],
                                       'vertical mono-age profiles',this_function)
    if R!=p.Rsun:
        R = inpcheck_radius(R,p,this_function)
    ages = inpcheck_age(ages,this_function)
    inpcheck_kwargs_compatibility(kwargs,this_function)
    
    indt = np.array(np.subtract(tp,ages)//tr,dtype=np.int)
    
    if (mode_comp=='dt' or mode_comp=='tot') and ('tab' in kwargs):
        tabd = kwargs['tab'][0]
        tabt = kwargs['tab'][1]
        if mode_comp=='tot':
            tabsh = kwargs['tab'][2]  
    
    if mode_comp=='d':
        if R==p.Rsun:
            rho_zd = _rhoz_d_(p,a,R=R,local=False,**kwargs)
        else:
            rho_zd = _rhoz_d_(p,a,R=R,local=False,**kwargs)[0]
        rho_z = _rhoz_ages_(rho_zd,indt,this_function,a,**kwargs)
    if mode_comp=='t':
        if R==p.Rsun:
            rho_zt = _rhoz_t_(p,a,R=R,local=False,**kwargs)
        else:
            rho_zt = _rhoz_t_(p,a,R=R,local=False,**kwargs)[0]
        rho_z = _rhoz_ages_(rho_zt,indt,this_function,a,**kwargs)
    if mode_comp=='sh':
        if R==p.Rsun:
            rho_zsh = _rhoz_sh_(p,a,R=R,local=False,**kwargs)
        else:
            rho_zsh = _rhoz_sh_(p,a,R=R,local=False,**kwargs)[0]
        rho_z = _rhoz_ages_(rho_zsh,indt,this_function,a,**kwargs)
        
    if mode_comp=='dt':
        if R==p.Rsun:
            if 'tab' in kwargs:
                del kwargs['tab']
                rho_zd = _rhoz_d_(p,a,R=R,local=False,**kwargs,tab=tabd)
                rho_zt = _rhoz_t_(p,a,R=R,local=False,**kwargs,tab=tabt)
            else:
                rho_zd = _rhoz_d_(p,a,R=R,local=False,**kwargs)
                rho_zt = _rhoz_t_(p,a,R=R,local=False,**kwargs)
        else:
            if 'tab' in kwargs:
                del kwargs['tab']
                rho_zd = _rhoz_d_(p,a,R=R,local=False,**kwargs,tab=tabd)[0]
                rho_zt = _rhoz_t_(p,a,R=R,local=False,**kwargs,tab=tabd)[0]
            else:
                rho_zd = _rhoz_d_(p,a,R=R,local=False,**kwargs)[0]
                rho_zt = _rhoz_t_(p,a,R=R,local=False,**kwargs)[0]
        
        rhod = _rhoz_ages_(rho_zd,indt,this_function,a,**kwargs)
        rhot = _rhoz_ages_(rho_zt,indt,this_function,a,**kwargs)
        rho_z = np.add(rhod,rhot)
        
    if mode_comp=='tot':
        if R==p.Rsun:
            if 'tab' in kwargs:
                del kwargs['tab']
                rho_zd = _rhoz_d_(p,a,R=R,local=False,**kwargs,tab=tabd)
                rho_zt = _rhoz_t_(p,a,R=R,local=False,**kwargs,tab=tabt)
                rho_zsh = _rhoz_sh_(p,a,R=R,local=False,**kwargs,tab=tabsh)
            else:
                rho_zd = _rhoz_d_(p,a,R=R,local=False,**kwargs)
                rho_zt = _rhoz_t_(p,a,R=R,local=False,**kwargs)
                rho_zsh = _rhoz_sh_(p,a,R=R,local=False,**kwargs)
        else:
            if 'tab' in kwargs:
                del kwargs['tab']
                rho_zd = _rhoz_d_(p,a,R=R,local=False,**kwargs,tab=tabd)[0]
                rho_zt = _rhoz_t_(p,a,R=R,local=False,**kwargs,tab=tabt)[0]
                rho_zsh = _rhoz_sh_(p,a,R=R,local=False,**kwargs,tab=tabsh)[0]
            else:
                rho_zd = _rhoz_d_(p,a,R=R,local=False,**kwargs)[0]
                rho_zt = _rhoz_t_(p,a,R=R,local=False,**kwargs)[0]
                rho_zsh = _rhoz_sh_(p,a,R=R,local=False,**kwargs)[0]
                
        rhod = _rhoz_ages_(rho_zd,indt,this_function,a,**kwargs)
        rhot = _rhoz_ages_(rho_zt,indt,this_function,a,**kwargs)
        rhosh = _rhoz_ages_(rho_zsh,indt,this_function,a,**kwargs)
        rho_z = np.add(rhod,np.add(rhot,rhosh))
        
    rho_z[rho_z==0]=np.nan

    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        ts.rhoz_monoage_save(rho_z,mode_comp,R,ages)
         
    return rho_z


def rhoz_monomet(mode_comp,R,mets,p,a,**kwargs):
    """
    Vertical profiles of the mono-age disk subpopulations 
    calculated at Galactocentric distance R.
    
    Parameters
    ----------
    mode_comp ('d', 't', 'sh', 'dt' or 'tot'), R, mets, p, a 
    **kwargs : dict, optional keyword arguments
        valid keywords : save, mode_pop, tab, number, mode_iso
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()
       
    Returns
    -------
    rhoz : 2d-ndarray
        Array of the shape (len(ages),a.n) with the densities of 
        the selected mono-age subpopulations, in Msun/pc^3. Each 
        profile corresponds to z-grid a.z. 
    """
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save','number','mode_pop','tab','mode_iso'],this_function)
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','dt','sh','tot'],
                                       'vertical mono-metallicity profiles',this_function)
    if R!=p.Rsun:   
        R = inpcheck_radius(R,p,this_function)
    inpcheck_kwargs_compatibility(kwargs,this_function)
    
    if (mode_comp=='dt' or mode_comp=='tot') and ('tab' in kwargs):
        tabd = kwargs['tab'][0]
        tabt = kwargs['tab'][1]
        if mode_comp=='tot':
            tabsh = kwargs['tab'][2]  
            
    if mode_comp=='d':
        rho_z = _rhomet_d_(R,mets,p,a,this_function,**kwargs,local=False)
    if mode_comp=='t':
        rho_z = _rhomet_t_(R,mets,p,a,this_function,**kwargs,local=False)
    if mode_comp=='sh':
        rho_z = _rhomet_sh_(R,mets,p,a,this_function,**kwargs,local=False)
    if mode_comp=='dt':
        if 'tab' in kwargs:
            del kwargs['tab']
            rho_zd = _rhomet_d_(R,mets,p,a,this_function,**kwargs,local=False,tab=tabd)
            rho_zt = _rhomet_t_(R,mets,p,a,this_function,**kwargs,local=False,tabd=tabt)
        else:
            rho_zd = _rhomet_d_(R,mets,p,a,this_function,**kwargs,local=False)
            rho_zt = _rhomet_t_(R,mets,p,a,this_function,**kwargs,local=False)
        rho_z = np.add(rho_zd,rho_zt)
    if mode_comp=='tot':
        if 'tab' in kwargs:
            del kwargs['tab']
            rho_zd = _rhomet_d_(R,mets,p,a,this_function,**kwargs,local=False,tab=tabd)
            rho_zt = _rhomet_t_(R,mets,p,a,this_function,**kwargs,local=False,tab=tabt)
            rho_zsh = _rhomet_sh_(R,mets,p,a,this_function,**kwargs,local=False,tab=tabsh)
        else:
            rho_zd = _rhomet_d_(R,mets,p,a,this_function,**kwargs,local=False)
            rho_zt = _rhomet_t_(R,mets,p,a,this_function,**kwargs,local=False)
            rho_zsh = _rhomet_sh_(R,mets,p,a,this_function,**kwargs,local=False)
        rho_z = np.add(rho_zd,np.add(rho_zt,rho_zsh))
    rho_z[rho_z==0]=np.nan
                  
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        ts.rhoz_monomet_save(rho_z,mode_comp,R,mets)
         
    return rho_z


def agez(mode_comp,p,a,**kwargs):
    """
    Age as a function of height z calculated at different radii. 
    
    Parameters
    ----------
    mode_comp ('d', 't', 'sh', 'dt' or 'tot'), p ,a 
    **kwargs : dict, optional keyword arguments
        valid keywords : save, mode_pop, tab, number, mode_iso 
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()

    Returns
    -------
    agezr : 2d-ndarray
        Age as a function of height z at the different 
        Galactocentric distances a.R. Array shape is (a.Rbins,a.n). 
    agezr0 : 1d-ndarray
        Same, but for the Solar radius, array of len a.n. 
    """
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save','tab','mode_pop','number','mode_iso','R'],this_function)
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                       'vertical age profiles',this_function)
    inpcheck_kwargs_compatibility(kwargs,this_function)
    
    if (mode_comp=='dt' or mode_comp=='tot') and 'tab' in kwargs:
        if 'R' in kwargs and kwargs['R']==p.Rsun:
            tabd0, tabt0 = kwargs['tab'][0],kwargs['tab'][1]
            if mode_comp=='tot':
                tabsh0 = kwargs['tab'][2]
        else:
            tabd, tabt = kwargs['tab'][0][0],kwargs['tab'][0][1]
            tabd0, tabt0 = kwargs['tab'][1][0],kwargs['tab'][1][1]
            if mode_comp=='tot':
                tabsh = kwargs['tab'][0][2]
                tabsh0 = kwargs['tab'][1][2]    
    
    if mode_comp=='d':
        if 'R' in kwargs and kwargs['R']==p.Rsun:
            rho_z0 = _rhoz_d_(p,a,**kwargs)
        else:
            rho_z, rho_z0 = _rhoz_d_(p,a,**kwargs)
            age_z = np.array([[np.sum(a.t*rho_z[i][k])/np.sum(rho_z[i][k]) for k in np.arange(a.n)]
                          for i in np.arange(a.Rbins)])
        age_z0 = [np.sum(a.t*rho_z0[i])/np.sum(rho_z0[i]) for i in np.arange(a.n)]
        
    if mode_comp=='t' or mode_comp=='sh':
        if mode_comp=='t': 
            if 'R' in kwargs and kwargs['R']==p.Rsun:
                rho_z0 = _rhoz_t_(p,a,**kwargs)
            else:
                rho_z, rho_z0 = _rhoz_t_(p,a,**kwargs)
        else:
            if 'R' in kwargs and kwargs['R']==p.Rsun:
                rho_z0 = _rhoz_sh_(p,a,**kwargs)
            else:
                rho_z, rho_z0 = _rhoz_sh_(p,a,**kwargs)
        
        age_z0 = np.zeros((a.n))
        age_z0.fill(np.nan)
        if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):
            age_z = np.zeros((a.Rbins,a.n))
            for i in range(a.Rbins):
                age_z[i].fill(np.nan)
        
        for k in range(a.n):    
            if np.sum(rho_z0[k])!=0:
                age_z0[k] = np.sum(a.t*rho_z0[k])/np.sum(rho_z0[k])
            if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):
                for i in range(a.Rbins):
                    if np.sum(rho_z[i][k])!=0:
                        age_z[i,k] = np.sum(a.t*rho_z[i][k])/np.sum(rho_z[i][k])
                      
    if mode_comp=='dt':
        if 'R' in kwargs and kwargs['R']==p.Rsun:
            if 'tab' in kwargs:
                del kwargs['tab']
                rho_zd0 = _rhoz_d_(p,a,**kwargs,tab=tabd0)
                rho_zt0 = _rhoz_t_(p,a,**kwargs,tab=tabt0)
            else:
                rho_zd0 = _rhoz_d_(p,a,**kwargs)
                rho_zt0 = _rhoz_t_(p,a,**kwargs)
        else:
            if 'tab' in kwargs:
                del kwargs['tab']
                rho_zd, rho_zd0 = _rhoz_d_(p,a,**kwargs,tab=[tabd,tabd0])
                rho_zt, rho_zt0 = _rhoz_t_(p,a,**kwargs,tab=[tabt,tabt0])
            else:
                rho_zd, rho_zd0 = _rhoz_d_(p,a,**kwargs)
                rho_zt, rho_zt0 = _rhoz_t_(p,a,**kwargs)
            age_z = np.array([[(np.sum(a.t*rho_zt[i][k]) + np.sum(a.t*rho_zd[i][k]))\
                              /(np.sum(rho_zd[i][k]) + np.sum(rho_zt[i][k])) for k in np.arange(a.n)] 
                              for i in np.arange(a.Rbins)])
        age_z0 = [(np.sum(a.t*rho_zd0[i]) + np.sum(a.t*rho_zt0[i]))\
                    /(np.sum(rho_zd0[i]) + np.sum(rho_zt0[i])) for i in np.arange(a.n)]
    
    if mode_comp=='tot':
        if 'R' in kwargs and kwargs['R']==p.Rsun:
            if 'tab' in kwargs:
                del kwargs['tab']
                rho_zd0 = _rhoz_d_(p,a,**kwargs,tab=tabd0)
                rho_zt0 = _rhoz_t_(p,a,**kwargs,tab=tabt0)
                rho_zsh0 = _rhoz_sh_(p,a,**kwargs,tab=tabsh0)
            else:
                rho_zd0 = _rhoz_d_(p,a,**kwargs)
                rho_zt0 = _rhoz_t_(p,a,**kwargs)
                rho_zsh0 = _rhoz_sh_(p,a,**kwargs)
        else:
            if 'tab' in kwargs:
                del kwargs['tab']
                rho_zd, rho_zd0 = _rhoz_d_(p,a,**kwargs,tab=[tabd,tabd0])
                rho_zt, rho_zt0 = _rhoz_t_(p,a,**kwargs,tab=[tabt,tabt0])
                rho_zsh, rho_zsh0 = _rhoz_sh_(p,a,**kwargs,tab=[tabsh,tabsh0])
            else:
                rho_zd, rho_zd0 = _rhoz_d_(p,a,**kwargs)
                rho_zt, rho_zt0 = _rhoz_t_(p,a,**kwargs)
                rho_zsh, rho_zsh0 = _rhoz_sh_(p,a,**kwargs)
            age_z = np.array([[(np.sum(a.t*rho_zt[i][k]) + np.sum(a.t*rho_zd[i][k]) + np.sum(a.t*rho_zsh[i][k]))\
                          /(np.sum(rho_zd[i][k]) + np.sum(rho_zt[i][k]) + np.sum(rho_zsh[i][k])) 
                          for k in np.arange(a.n)] 
                          for i in np.arange(a.Rbins)])
        age_z0 = [(np.sum(a.t*rho_zd0[i]) + np.sum(a.t*rho_zt0[i]) + np.sum(a.t*rho_zsh0[i]))\
                  /(np.sum(rho_zd0[i]) + np.sum(rho_zt0[i]) + np.sum(rho_zsh0[i])) for i in np.arange(a.n)]

    age_z0 = np.subtract(tp,age_z0)
    if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):
        age_z = np.subtract(tp,age_z)
    
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        if ('R' in kwargs) and (kwargs['R']==p.Rsun):
            ts.agez_save(age_z0,mode_comp)
        else:
            ts.agez_save((age_z,age_z0),mode_comp)
                    
    if ('R' in kwargs) and (kwargs['R']==p.Rsun):
        return age_z0
    else:
        return (age_z,age_z0)
    

def ager(mode_comp,zlim,p,a,**kwargs):
    """
    Age as a function of Galactocentric distance. 
    
    Parameters
    ----------
    mode_comp ('d', 't', 'sh', 'dt' or 'tot'), zlim, p, a 
    **kwargs : dict, optional keyword arguments
        valid keywords : save, mode_pop, tab, number, mode_iso 
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()
        
    Returns
    -------
    ager : ndarray
        Radial age profile, array of length a.Rbins
    """
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save','tab','mode_pop','number','mode_iso'],this_function)
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                       'radial age profile',this_function)
    zlim = inpcheck_height(zlim,p,this_function)
    inpcheck_kwargs_compatibility(kwargs,this_function)
        
    if mode_comp=='d':
        rho_r = _rhoz_d_(p,a,zlim=zlim,local=False,**kwargs)
        rho_r = np.sum(rho_r,axis=1)
        age_r = [np.sum(a.t*rho_r[i])/np.sum(rho_r[i]) for i in np.arange(a.Rbins)]
        
    if mode_comp=='t' or mode_comp=='sh':
        if mode_comp=='t':
            rho_r = _rhoz_t_(p,a,zlim=zlim,local=False,**kwargs)
        else:
            rho_r = _rhoz_sh_(p,a,zlim=zlim,local=False,**kwargs)
        rho_r = np.sum(rho_r,axis=1)
        age_r = np.zeros((a.Rbins))
        age_r.fill(np.nan)
        for i in range(a.Rbins):
            if np.sum(rho_r[i])!=0:
                age_r[i] = np.sum(a.t*rho_r[i])/np.sum(rho_r[i])
                
    if mode_comp=='dt':
        if 'tab' in kwargs:
            tabs = kwargs['tab']
            del kwargs['tab']
            rho_rd = _rhoz_d_(p,a,zlim=zlim,local=False,**kwargs,tab=tabs[0])
            rho_rt = _rhoz_t_(p,a,zlim=zlim,local=False,**kwargs,tab=tabs[1])
        else:
            rho_rd = _rhoz_d_(p,a,zlim=zlim,local=False,**kwargs)
            rho_rt = _rhoz_t_(p,a,zlim=zlim,local=False,**kwargs)
        rho_rd = np.sum(rho_rd,axis=1)
        rho_rt = np.sum(rho_rt,axis=1)
        age_r = [(np.sum(a.t*rho_rd[i]) + np.sum(a.t*rho_rt[i]))\
                 /(np.sum(rho_rd[i]) + np.sum(rho_rt[i])) for i in np.arange(a.Rbins)]
            
    if mode_comp=='tot':
        if 'tab' in kwargs:
            tabs = kwargs['tab']
            del kwargs['tab']
            rho_rd = _rhoz_d_(p,a,zlim=zlim,local=False,**kwargs,tab=tabs[0])
            rho_rt = _rhoz_t_(p,a,zlim=zlim,local=False,**kwargs,tab=tabs[1])
            rho_rsh = _rhoz_sh_(p,a,zlim=zlim,local=False,**kwargs,tab=tabs[2])
        else:
            rho_rd = _rhoz_d_(p,a,zlim=zlim,local=False,**kwargs)
            rho_rt = _rhoz_t_(p,a,zlim=zlim,local=False,**kwargs)
            rho_rsh = _rhoz_sh_(p,a,zlim=zlim,local=False,**kwargs)
        rho_rd = np.sum(rho_rd,axis=1)
        rho_rt = np.sum(rho_rt,axis=1)
        rho_rsh = np.sum(rho_rsh,axis=1)
        age_r = [(np.sum(a.t*rho_rd[i]) + np.sum(a.t*rho_rt[i]) + np.sum(a.t*rho_rsh[i]))\
                 /(np.sum(rho_rd[i]) + np.sum(rho_rt[i]) + np.sum(rho_rsh[i])) 
                 for i in np.arange(a.Rbins)]
            
    age_r = np.subtract(tp,age_r)
    
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        ts.ager_save(age_r,mode_comp,zlim)
                                                                                
    return age_r


def metz(mode_comp,p,a,**kwargs):
    """
    Metallicity as a function of height z. 
    
    Parameters
    ----------
    mode_comp ('d', 't', 'sh', 'dt' or 'tot'), p, a 
    **kwargs : dict, optional keyword arguments
        valid keywords : save, mode_pop, tab, number, mode_iso
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()

    Returns
    -------
    fehzr : 2d-ndarray
        Metallicity as a function of height z at the different 
        Galactocentric distances a.R. Array shape is (a.Rbins,a.n). 
    fehzr0 : ndarray
        Same, but for the Solar radius, array of len a.Rbins. 
    """
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save','mode_pop','tab','number','mode_iso','R'],this_function)
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                       'vertical metallicity profiles',this_function)
    inpcheck_kwargs_compatibility(kwargs,this_function)
        
    if mode_comp=='d':
        if 'R' in kwargs and kwargs['R']==p.Rsun:
            AMRd0 = tab_reader(['AMRd0'],p,a.T)[0]
            rho_z0 = _rhoz_d_(p,a,**kwargs)
        else:
            AMRd,AMRd0 = tab_reader(['AMRd','AMRd0'],p,a.T)
            rho_z, rho_z0 = _rhoz_d_(p,a,**kwargs)
            feh_z = np.array([[np.sum(AMRd[i+1]*rho_z[i][k])/np.sum(rho_z[i][k]) for k in np.arange(a.n)]
                              for i in np.arange(a.Rbins)])
        feh_z0 = [np.sum(AMRd0[1]*rho_z0[i])/np.sum(rho_z0[i]) for i in np.arange(a.n)]     
        
    if mode_comp=='t':
        AMRt = tab_reader(['AMRt'],p,a.T)[0]
        AMRt = np.concatenate((AMRt[1],np.linspace(AMRt[1][-1],AMRt[1][-1],a.jd-a.jt)),axis=0)
        if 'R' in kwargs and kwargs['R']==p.Rsun:
            rho_z0 = _rhoz_t_(p,a,**kwargs)
        else:
            rho_z, rho_z0 = _rhoz_t_(p,a,**kwargs)
            feh_z = np.zeros((a.Rbins,a.n))
            for i in range(a.Rbins):
                feh_z[i].fill(np.nan)
        feh_z0 = np.zeros((a.n))
        feh_z0.fill(np.nan)
        
        for i in range(a.n):
            if np.sum(rho_z0[i])!=0:
                feh_z0[i] = np.sum(AMRt*rho_z0[i])/np.sum(rho_z0[i])
            if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):
                for k in range(a.Rbins):
                    if np.sum(rho_z[k][i])!=0:
                        feh_z[k,i] = np.sum(AMRt*rho_z[k][i])/np.sum(rho_z[k][i])
        
    if mode_comp=='sh':
        if 'R' in kwargs and kwargs['R']==p.Rsun:
            rho_z0 = _rhoz_sh_(p,a,**kwargs)
        else:
            rho_z, rho_z0 = _rhoz_sh_(p,a,**kwargs)
            feh_z = np.zeros((a.Rbins,a.n))
            for i in range(a.Rbins):
                feh_z[i].fill(np.nan)
        feh_z0 = np.zeros((a.n))
        feh_z0.fill(np.nan)
        
        for i in range(a.n):
            if np.sum(rho_z0[i])!=0:
              feh_z0[i] = p.FeHsh
            if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):
                for k in range(a.Rbins):
                    if np.sum(rho_z[k][i])!=0:
                        feh_z[k,i] = p.FeHsh
                          
    if mode_comp=='dt':
        AMRd0,AMRt = tab_reader(['AMRd0','AMRt'],p,a.T)
        AMRt = np.concatenate((AMRt[1],np.linspace(AMRt[1][-1],AMRt[1][-1],a.jd-a.jt)),axis=0)
        if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):
            AMRd = tab_reader(['AMRd'],p,a.T)[0]
            if 'tab' in kwargs:
                tabs = kwargs['tab']
                del kwargs['tab']
                rho_zd, rho_zd0 = _rhoz_d_(p,a,**kwargs,tab=[tabs[0][0],tabs[1][0]])
                rho_zt, rho_zt0 = _rhoz_t_(p,a,**kwargs,tab=[tabs[0][1],tabs[1][1]])
            else:
                rho_zd, rho_zd0 = _rhoz_d_(p,a,**kwargs)
                rho_zt, rho_zt0 = _rhoz_t_(p,a,**kwargs)
        else:
            if 'tab' in kwargs:
                tabs = kwargs['tab']
                del kwargs['tab']
                rho_zd0 = _rhoz_d_(p,a,**kwargs,tab=tabs[0])
                rho_zt0 = _rhoz_t_(p,a,**kwargs,tab=tabs[1])
            else:
                rho_zd0 = _rhoz_d_(p,a,**kwargs)
                rho_zt0 = _rhoz_t_(p,a,**kwargs)
        feh_z0 = [(np.sum(AMRd0[1]*rho_zd0[i]) + np.sum(AMRt*rho_zt0[i]))\
                    /(np.sum(rho_zd0[i]) + np.sum(rho_zt0[i])) for i in np.arange(a.n)]
        if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):
            feh_z = np.array([[(np.sum(AMRt*rho_zt[i][k]) + np.sum(AMRd[i+1]*rho_zd[i][k]))\
                              /(np.sum(rho_zd[i][k]) + np.sum(rho_zt[i][k])) for k in np.arange(a.n)] 
                              for i in np.arange(a.Rbins)])
            
    if mode_comp=='tot':
        AMRd0,AMRt = tab_reader(['AMRd0','AMRt'],p,a.T)
        AMRt = np.concatenate((AMRt[1],np.linspace(AMRt[1][-1],AMRt[1][-1],a.jd-a.jt)),axis=0)
        if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):
            AMRd = tab_reader(['AMRd'],p,a.T)[0]
            if 'tab' in kwargs:
                tabs = kwargs['tab']
                del kwargs['tab']
                rho_zd, rho_zd0 = _rhoz_d_(p,a,**kwargs,tab=[tabs[0][0],tabs[1][0]])
                rho_zt, rho_zt0 = _rhoz_t_(p,a,**kwargs,tab=[tabs[0][1],tabs[1][1]])
                rho_zsh, rho_zsh0 = _rhoz_sh_(p,a,**kwargs,tab=[tabs[0][2],tabs[1][2]])
            else:
                rho_zd, rho_zd0 = _rhoz_d_(p,a,**kwargs)
                rho_zt, rho_zt0 = _rhoz_t_(p,a,**kwargs)
                rho_zsh, rho_zsh0 = _rhoz_sh_(p,a,**kwargs)
        else:
            if 'tab' in kwargs:
                tabs = kwargs['tab']
                del kwargs['tab']
                rho_zd0 = _rhoz_d_(p,a,**kwargs,tab=tabs[0])
                rho_zt0 = _rhoz_t_(p,a,**kwargs,tab=tabs[1])
                rho_zsh0 = _rhoz_sh_(p,a,**kwargs,tab=tabs[2])
            else:
                rho_zd0 = _rhoz_d_(p,a,**kwargs)
                rho_zt0 = _rhoz_t_(p,a,**kwargs)
                rho_zsh0 = _rhoz_sh_(p,a,**kwargs)
        feh_z0 = [(np.sum(AMRd0[1]*rho_zd0[i]) + np.sum(AMRt*rho_zt0[i]) + np.sum(p.FeHsh*rho_zsh0[i]))\
                    /(np.sum(rho_zd0[i]) + np.sum(rho_zt0[i]) + np.sum(rho_zsh0[i])) for i in np.arange(a.n)]
        if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):
            feh_z = np.array([[(np.sum(AMRd[i+1]*rho_zd[i][k]) + np.sum(AMRt*rho_zt[i][k]) +  
                                np.sum(p.FeHsh*rho_zsh[i][k]))\
                              /(np.sum(rho_zd[i][k]) + np.sum(rho_zt[i][k]) + np.sum(rho_zsh[i][k])) 
                              for k in np.arange(a.n)] 
                              for i in np.arange(a.Rbins)])
        
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        if ('R' in kwargs) and (kwargs['R']==p.Rsun):
            ts.metz_save(feh_z0,mode_comp)
        else:
            ts.metz_save((feh_z,feh_z0),mode_comp)
    
    if ('R' in kwargs) and (kwargs['R']==p.Rsun):
        return feh_z0
    else:      
        return (feh_z,feh_z0)
    
                                                                     
def metr(mode_comp,zlim,p,a,**kwargs):
    """
    Metallicity as a function of Galactocentric distance. 
    
    Parameters
    ----------
    mode_comp ('d','t','sh','dt' or'tot'), zlim, p, a 
    **kwargs : dict, optional keyword arguments
        valid keywods : save, mode_pop, tab, number, mode_iso
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()
        
    Returns
    -------
    fehr : ndarray
        Radial metallicity profile, array of len a.Rbins. 
    """
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save','mode_pop','tab','number','mode_iso'],this_function)
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                       'radial metallicity profiles',this_function)
    zlim = inpcheck_height(zlim,p,this_function)
    inpcheck_kwargs_compatibility(kwargs,this_function)
        
    if mode_comp=='d':
        AMRd = tab_reader(['AMRd'],p,a.T)[0]
        rho_r = _rhoz_d_(p,a,zlim=zlim,local=False,**kwargs)
        rho_r = np.sum(rho_r,axis=1)
        feh_r = [np.sum(AMRd[i+1]*rho_r[i])/np.sum(rho_r[i]) for i in np.arange(a.Rbins)]
        
    if mode_comp=='t':
        AMRt = tab_reader(['AMRt'],p,a.T)[0]
        AMRt = np.concatenate((AMRt[1],np.linspace(AMRt[1][-1],AMRt[1][-1],a.jd-a.jt)),axis=0)
        rho_r = _rhoz_t_(p,a,zlim=zlim,local=False,**kwargs)
        rho_r = np.sum(rho_r,axis=1)
        feh_r = np.zeros((a.Rbins))
        feh_r.fill(np.nan)
        for i in range(a.Rbins):
            if np.sum(rho_r[i])!=0:
                feh_r[i] = np.sum(AMRt*rho_r[i])/np.sum(rho_r[i])
        
    if mode_comp=='sh':
        rho_r = _rhoz_sh_(p,a,zlim=zlim,local=False,**kwargs)
        rho_r = np.sum(rho_r,axis=1)
        feh_r = np.zeros((a.Rbins))
        feh_r.fill(np.nan)
        for i in range(a.Rbins):
            if np.sum(rho_r[i])!=0:
                feh_r[i] = p.FeHsh
        
    if mode_comp=='dt':
        AMRd,AMRt = tab_reader(['AMRd','AMRt'],p,a.T)
        AMRt = np.concatenate((AMRt[1],np.linspace(AMRt[1][-1],AMRt[1][-1],a.jd-a.jt)),axis=0)
        if 'tab' in kwargs:
            tabs = kwargs['tab']
            del kwargs['tab']
            rho_rd = _rhoz_d_(p,a,zlim=zlim,local=False,**kwargs,tab=tabs[0])
            rho_rt = _rhoz_t_(p,a,zlim=zlim,local=False,**kwargs,tab=tabs[1])
        else:
            rho_rd = _rhoz_d_(p,a,zlim=zlim,local=False,**kwargs)
            rho_rt = _rhoz_t_(p,a,zlim=zlim,local=False,**kwargs)
        rho_rd = np.sum(rho_rd,axis=1)
        rho_rt = np.sum(rho_rt,axis=1)
        feh_r = [(np.sum(AMRd[i+1]*rho_rd[i]) + np.sum(AMRt*rho_rt[i]))\
                 /(np.sum(rho_rd[i]) + np.sum(rho_rt[i])) for i in np.arange(a.Rbins)]
            
    if mode_comp=='tot':
        AMRd,AMRt = tab_reader(['AMRd','AMRt'],p,a.T)
        AMRt = np.concatenate((AMRt[1],np.linspace(AMRt[1][-1],AMRt[1][-1],a.jd-a.jt)),axis=0)
        if 'tab' in kwargs:
            tabs = kwargs['tab']
            del kwargs['tab']
            rho_rd = _rhoz_d_(p,a,zlim=zlim,local=False,**kwargs,tab=tabs[0])
            rho_rt = _rhoz_t_(p,a,zlim=zlim,local=False,**kwargs,tab=tabs[1])
            rho_rsh = _rhoz_sh_(p,a,zlim=zlim,local=False,**kwargs,tab=tabs[2])
        else:
            rho_rd = _rhoz_d_(p,a,zlim=zlim,local=False,**kwargs)
            rho_rt = _rhoz_t_(p,a,zlim=zlim,local=False,**kwargs)
            rho_rsh = _rhoz_sh_(p,a,zlim=zlim,local=False,**kwargs)
        rho_rd = np.sum(rho_rd,axis=1)
        rho_rt = np.sum(rho_rt,axis=1)
        rho_rsh = np.sum(rho_rsh,axis=1)
        feh_r = [(np.sum(AMRd[i+1]*rho_rd[i]) + np.sum(AMRt*rho_rt[i]) + np.sum(p.FeHsh*rho_rsh[i]))\
                 /(np.sum(rho_rd[i]) + np.sum(rho_rt[i]) + np.sum(rho_rsh[i])) 
                 for i in np.arange(a.Rbins)]
        
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        ts.metr_save(feh_r,mode_comp,zlim)
                                                                                
    return feh_r
                                                

def rhor(p,a,**kwargs):
    """
    Radial density profiles of the Galactic components. 
    
    Parameters
    ----------
    p, a 
    **kwargs : dict, optional keyword arguments
        valid keywords : sigma 
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()
        
    Returns
    -------
    rhor : 2d-ndarray
        Surface (or spatial) matter (or number) density in the 
        Galactic plane as a function of Galactocentric distance R.   
        Array shape is (6,a.Rbins), different columns correspond 
        to the thin disk, molecular gas, atomic gas, thick disk, 
        stellar halo, DM halo. 
    rhor0 : list 
        Same, but at the Solar radius p.Rsun. 
    """
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['sigma'],this_function)
    inpcheck_kwargs_compatibility(kwargs,this_function)
    
    SR,Hdeff,Ht,Hdh,Hsh = tab_reader(['SR','Heffd','Ht','Hdh','Hsh'],p,a.T)
    (hg1,hg10),(hg2,hg20) = hgr(p,a)

    if inpcheck_iskwargtype(kwargs,'sigma',True,bool,this_function):
        rho_r = SR[1:]
        rho_r0 = [p.sigmad,p.sigmag1,p.sigmag2,p.sigmat,p.sigmadh,p.sigmash]
    else:
        rho_rd = [SR[1][k]/2/Hdeff[1][k] for k in a.R_array]  
        rho_g1 = [SR[2][k]/2/hg1[k] for k in a.R_array]  
        rho_g2 = [SR[3][k]/2/hg2[k] for k in a.R_array] 
        rho_rt = [SR[4][k]/2/Ht[1][k] for k in a.R_array]  
        rho_dh = [SR[5][k]/2/Hdh[1][k] for k in a.R_array] 
        rho_sh = [SR[6][k]/2/Hsh[1][k] for k in a.R_array] 
        Hdeff0,Ht0,Hdh0,Hsh0 = tab_reader(['Heffd0','Ht0','Hdh0','Hsh0'],p,a.T)
        rho_r0 = [p.sigmad/2/Hdeff0[1],p.sigmag1/2/hg10,
                  p.sigmag2/2/hg20,p.sigmat/2/Ht0[1],
                  p.sigmadh/2/Hdh0[1],p.sigmash/2/Hsh0[1]]
        rho_r = np.stack((rho_rd,rho_g1,rho_g2,rho_rt,rho_dh,rho_sh),axis=0)
        
    return (rho_r, rho_r0)


def rhor_monoage(mode_comp,zlim,ages,p,a,**kwargs):
    """
    Radial density profiles of the mono-age subpopulations. 
    
    Parameters
    ----------
    mode_comp ('d','t','sh','dt' or 'tot'), zlim, ages, p, a 
    **kwargs : dict, optional keyword arguments
        valid keywords: save, sigma, between, tab, mode_pop, 
        number, mode_iso 
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()
            
    Returns
    -------
    rhor : 2d-ndarray
        Surface (or spatial) matter (or number) density in the 
        Galactic plane as a function of Galactocentric distance R 
        and age. Array shape is (len(ages),a.Rbins), or 
        (len(ages)-1,a.Rbins) if between=True. 
    """
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save','sigma','between','tab','mode_pop',
                            'number','mode_iso'],this_function)
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                       'radial mono-age profiles',this_function)
    ages = inpcheck_age(ages,this_function)
    zlim = inpcheck_height(zlim,p,this_function)
    inpcheck_kwargs_compatibility(kwargs,this_function)
    
    if inpcheck_iskwargtype(kwargs,'between',True,bool,this_function):
        nage = int(len(ages) - 1)
    else:
        nage = len(ages)
                 
    indt = np.array(np.subtract(tp,ages)//tr,dtype=np.int)
    rho_r = np.zeros((nage,a.Rbins))
    
    if (mode_comp=='dt' or mode_comp=='tot') and ('tab' in kwargs):
        tabd = kwargs['tab'][0]
        tabt = kwargs['tab'][1]
        if mode_comp=='tot':
            tabsh = kwargs['tab'][2]  
    
    if mode_comp=='d':
        rho_z = _rhoz_d_(p,a,zlim=zlim,**kwargs,local=False)
        for i in range(a.Rbins):
            rho_r[:,i] = _rhor_ages_(rho_z[i],indt,this_function,a,**kwargs)
            
    if mode_comp=='t':
        rho_z = _rhoz_t_(p,a,zlim=zlim,**kwargs,local=False)
        for i in range(a.Rbins):
            rho_r[:,i] = _rhor_ages_(rho_z[i],indt,this_function,a,**kwargs)
            
    if mode_comp=='sh':
        rho_z = _rhoz_sh_(p,a,zlim=zlim,**kwargs,local=False)
        for i in range(a.Rbins):
            rho_r[:,i] = _rhor_ages_(rho_z[i],indt,this_function,a,**kwargs)
            
    if mode_comp=='dt':
        if 'tab' in kwargs:
            del kwargs['tab']
            rho_zd = _rhoz_d_(p,a,zlim=zlim,**kwargs,local=False,tab=tabd)
            rho_zt = _rhoz_t_(p,a,zlim=zlim,**kwargs,local=False,tab=tabt)
        else:
            rho_zd = _rhoz_d_(p,a,zlim=zlim,**kwargs,local=False)
            rho_zt = _rhoz_t_(p,a,zlim=zlim,**kwargs,local=False)
        for i in range(a.Rbins):
            rho_r[:,i] = _rhor_ages_(rho_zd[i],indt,this_function,a,**kwargs) +\
                       _rhor_ages_(rho_zt[i],indt,this_function,a,**kwargs)
                       
    if mode_comp=='tot':
        if 'tab' in kwargs:
            del kwargs['tab']
            rho_zd = _rhoz_d_(p,a,zlim=zlim,**kwargs,local=False,tab=tabd)
            rho_zt = _rhoz_t_(p,a,zlim=zlim,**kwargs,local=False,tab=tabt)
            rho_zsh = _rhoz_sh_(p,a,zlim=zlim,**kwargs,local=False,tab=tabsh)
        else:
            rho_zd = _rhoz_d_(p,a,zlim=zlim,**kwargs,local=False)
            rho_zt = _rhoz_t_(p,a,zlim=zlim,**kwargs,local=False)
            rho_zsh = _rhoz_sh_(p,a,zlim=zlim,**kwargs,local=False)
        for i in range(a.Rbins):
            rho_r[:,i] = _rhor_ages_(rho_zd[i],indt,this_function,a,**kwargs) +\
                         _rhor_ages_(rho_zt[i],indt,this_function,a,**kwargs) +\
                         _rhor_ages_(rho_zsh[i],indt,this_function,a,**kwargs)
        
    rho_r[rho_r==0]=np.nan
        
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        ts.rhor_monoage_save(rho_r,mode_comp,zlim,ages)
    
    return rho_r


def rhor_monomet(mode_comp,zlim,mets,p,a,**kwargs):
    """
    Radial density profiles of the mono-metallicity subpopulations. 
    
    Parameters
    ----------
    mode_comp ('d','t','dt','sh' or 'tot'), zlim, mets, p, a 
    **kwargs : dict, optional keyword arguments
        valid keywords : save, sigma, mode_pop, tab, number,
        mode_iso
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()

    Returns
    -------
    rhomet : 2d-ndarray
        Surface (or spatial) matter (or number) density in the 
        Galactic plane as a function of Galactocentric distance R 
        and metallicity. Array shape is (len(mets),a.Rbins).         
    """
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save','sigma','mode_pop','tab','number','mode_iso'],this_function)
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','dt','sh','tot'],
                                       'radial mono-metallicity profiles',this_function)
    zlim = inpcheck_height(zlim,p,this_function)
    inpcheck_kwargs_compatibility(kwargs,this_function)
    
    if (mode_comp=='dt' or mode_comp=='tot') and ('tab' in kwargs):
        tabd = kwargs['tab'][0]
        tabt = kwargs['tab'][1]
        if mode_comp=='tot':
            tabsh = kwargs['tab'][2]  
    
    if mode_comp=='d':
        rhomet = _rhometz_d_(zlim,mets,p,a,this_function,**kwargs,local=False)
    if mode_comp=='t':
        rhomet = _rhometz_t_(zlim,mets,p,a,this_function,**kwargs,local=False)
    if mode_comp=='sh':
        rhomet = _rhometz_sh_(zlim,mets,p,a,this_function,**kwargs,local=False)
    if mode_comp=='dt':
        if 'tab' in kwargs:
            del kwargs['tab']
            rhometd = _rhometz_d_(zlim,mets,p,a,this_function,**kwargs,tab=tabd,local=False)
            rhomett = _rhometz_t_(zlim,mets,p,a,this_function,**kwargs,tab=tabt,local=False)
        else:
            rhometd = _rhometz_d_(zlim,mets,p,a,this_function,**kwargs,local=False)
            rhomett = _rhometz_t_(zlim,mets,p,a,this_function,**kwargs,local=False)
        rhomet = np.add(rhometd,rhomett)
    if mode_comp=='tot':
        if 'tab' in kwargs:
            del kwargs['tab']
            rhometd = _rhometz_d_(zlim,mets,p,a,this_function,**kwargs,tab=tabd,local=False)
            rhomett = _rhometz_t_(zlim,mets,p,a,this_function,**kwargs,tab=tabt,local=False)
            rhometsh = _rhometz_sh_(zlim,mets,p,a,this_function,**kwargs,tab=tabsh,local=False)
        else:
            rhometd = _rhometz_d_(zlim,mets,p,a,this_function,**kwargs,local=False)
            rhomett = _rhometz_t_(zlim,mets,p,a,this_function,**kwargs,local=False)
            rhometsh = _rhometz_sh_(zlim,mets,p,a,this_function,**kwargs,local=False)
        rhomet = np.add(rhometd,np.add(rhomett,rhometsh))        
    rhomet[rhomet==0] = np.nan
        
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        ts.rhor_monomet_save(rhomet,mode_comp,zlim,mets)
        
    return rhomet


def agehist(mode_comp,zlim,p,a,**kwargs):
    """
    Calculates age distributions at all Galactocentric radii R. 
    
    Parameters
    ----------
    mode_comp ('d','t','sh','dt' or 'tot'), zlim, p, a 
    **kwargs : dict, optional keyword arguments
        valid keywords : save, mode_pop, tab, number, sigma_gauss,
        mode_iso
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()

    Returns
    -------
    ages : 2d-ndarray
        Age distributions at the different Galactocentric distances, 
        array shape is (a.Rbins,a.jd).
    ages0 : ndarray
        Same, but at the Solar radius, array of len a.jd. 
    """
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save','mode_pop','tab','number','sigma_gauss','mode_iso','R'],
                    this_function)
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                       'age distributions',this_function)
    zlim = inpcheck_height(zlim,p,this_function)
    inpcheck_kwargs_compatibility(kwargs,this_function)
    
    if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):
        ages_ar = np.zeros((a.Rbins,a.jd)) 
    
    if mode_comp=='d':
        if 'R' in kwargs and kwargs['R']==p.Rsun:
            rho_d0 = _rhoz_d_(p,a,zlim=zlim,**kwargs)
        else:
            rho_d, rho_d0 = _rhoz_d_(p,a,zlim=zlim,**kwargs)
        sum_rho0 = np.sum(np.sum(rho_d0))
        ages_ar0 = np.sum(rho_d0,axis=0)/sum_rho0
        if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):
            for i in range(a.Rbins):
                sum_rho = np.sum(np.sum(rho_d[i])) 
                ages_ar[i] = np.sum(rho_d[i],axis=0)/sum_rho
    
    if mode_comp=='t' or mode_comp=='sh':
        if mode_comp=='t':
            if 'R' in kwargs and kwargs['R']==p.Rsun:
                rho0 = _rhoz_t_(p,a,zlim=zlim,**kwargs)
            else:
                rho, rho0 = _rhoz_t_(p,a,zlim=zlim,**kwargs)
        else:
            if 'R' in kwargs and kwargs['R']==p.Rsun:
                rho0 = _rhoz_sh_(p,a,zlim=zlim,**kwargs)
            else:
                rho, rho0 = _rhoz_sh_(p,a,zlim=zlim,**kwargs)
        sum_rho0 = np.sum(np.sum(rho0))
        if sum_rho0!=0:
            ages_ar0 = np.sum(rho0,axis=0)/sum_rho0
        else:
            ages_ar0 = np.zeros((a.jd))
        if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):    
            for i in range(a.Rbins):
                sum_rho = np.sum(np.sum(rho[i]))
                if sum_rho!=0:
                    ages_ar[i] = np.sum(rho[i],axis=0)/sum_rho
                else:
                    ages_ar[i] = np.zeros((a.jd))
                        
    if mode_comp=='dt': 
        if 'R' in kwargs and kwargs['R']==p.Rsun:
            if 'tab' in kwargs:
                tabs = kwargs['tab']
                del kwargs['tab']
                rho_d0 = _rhoz_d_(p,a,zlim=zlim,**kwargs,tab=tabs[0])
                rho_t0 = _rhoz_t_(p,a,zlim=zlim,**kwargs,tab=tabs[1])
            else:
                rho_d0 = _rhoz_d_(p,a,zlim=zlim,**kwargs)
                rho_t0 = _rhoz_t_(p,a,zlim=zlim,**kwargs)
        else:
            if 'tab' in kwargs:
                tabs = kwargs['tab']
                del kwargs['tab']
                rho_d, rho_d0 = _rhoz_d_(p,a,zlim=zlim,**kwargs,tab=[tabs[0][0],tabs[1][0]])
                rho_t, rho_t0 = _rhoz_t_(p,a,zlim=zlim,**kwargs,tab=[tabs[0][1],tabs[1][1]])
            else:
                rho_d, rho_d0 = _rhoz_d_(p,a,zlim=zlim,**kwargs)
                rho_t, rho_t0 = _rhoz_t_(p,a,zlim=zlim,**kwargs)
        sum_rho0 = np.sum(np.sum(rho_d0)) + np.sum(np.sum(rho_t0))
        ages_ar0 = np.add(np.sum(rho_d0,axis=0),np.sum(rho_t0,axis=0))/sum_rho0 
        if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):    
            for i in range(a.Rbins):
                sum_rho = np.sum(np.sum(rho_d[i])) + np.sum(np.sum(rho_t[i]))
                ages_ar[i] = np.add(np.sum(rho_d[i],axis=0),np.sum(rho_t[i],axis=0))/sum_rho
            
    if mode_comp=='tot':   
        if 'R' in kwargs and kwargs['R']==p.Rsun:
            if 'tab' in kwargs:
                tabs = kwargs['tab']
                del kwargs['tab']
                rho_d0 = _rhoz_d_(p,a,zlim=zlim,**kwargs,tab=tabs[0])
                rho_t0 = _rhoz_t_(p,a,zlim=zlim,**kwargs,tab=tabs[1])
                rho_sh0 = _rhoz_sh_(p,a,zlim=zlim,**kwargs,tab=tabs[2])
            else:
                rho_d0 = _rhoz_d_(p,a,zlim=zlim,**kwargs)
                rho_t0 = _rhoz_t_(p,a,zlim=zlim,**kwargs)
                rho_sh0 = _rhoz_sh_(p,a,zlim=zlim,**kwargs)
        else:          
            if 'tab' in kwargs:
                tabs = kwargs['tab']
                del kwargs['tab']
                rho_d, rho_d0 = _rhoz_d_(p,a,zlim=zlim,**kwargs,tab=[tabs[0][0],tabs[1][0]])
                rho_t, rho_t0 = _rhoz_t_(p,a,zlim=zlim,**kwargs,tab=[tabs[0][1],tabs[1][1]])
                rho_sh, rho_sh0 = _rhoz_sh_(p,a,zlim=zlim,**kwargs,tab=[tabs[0][2],tabs[1][2]])
            else:
                rho_d, rho_d0 = _rhoz_d_(p,a,zlim=zlim,**kwargs)
                rho_t, rho_t0 = _rhoz_t_(p,a,zlim=zlim,**kwargs)
                rho_sh, rho_sh0 = _rhoz_sh_(p,a,zlim=zlim,**kwargs)
        sum_rho0 = np.sum(np.sum(rho_d0)) + np.sum(np.sum(rho_t0)) + np.sum(np.sum(rho_sh0))
        ages_ar0 = np.add(np.add(np.sum(rho_d0,axis=0),np.sum(rho_sh0,axis=0)),
                          np.sum(rho_t0,axis=0))/sum_rho0        
        if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):   
            for i in range(a.Rbins):
                sum_rho = np.sum(np.sum(rho_d[i])) + np.sum(np.sum(rho_t[i])) + np.sum(np.sum(rho_sh[i]))
                agesd_ar = np.sum(rho_d[i],axis=0)
                agest_ar = np.sum(rho_t[i],axis=0)
                agessh_ar = np.sum(rho_sh[i],axis=0)
                ages_ar[i] = (agesd_ar + agest_ar + agessh_ar)/sum_rho
    
    if ('sigma_gauss' in kwargs):
        ages_ar0 = gaussian_filter1d(ages_ar0,kwargs['sigma_gauss']//tr)
    if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):   
        for i in range(a.Rbins):
            if ('sigma_gauss' in kwargs):
                ages_ar[i] = gaussian_filter1d(ages_ar[i],kwargs['sigma_gauss']//tr)
                      
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        if 'R' in kwargs and kwargs['R']==p.Rsun:
            ts.agehist_save(ages_ar0,mode_comp,zlim)
        else:
            ts.agehist_save((ages_ar,ages_ar0),mode_comp,zlim)
            
    if 'R' in kwargs and kwargs['R']==p.Rsun:
        return (ages_ar0, sum_rho0)
    else:      
        return (ages_ar, ages_ar0)         
                                              
                                                                           
def methist(mode_comp,zlim,p,a,**kwargs):
    """
    Calculates metallicity distributions at all R. 
    
    Parameters
    ----------
    mode_comp ('d','t','sh','dt' or 'tot'), zlim, p, a 
    **kwargs : dict, optional keyword arguments
        valid keywords : save, tab, mode_pop, number, sigma_gauss, 
        metbins, mode_iso
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()

    Returns
    -------
    metdist : 2d-ndarray
        Metallicity distributions at the different Galactocentric 
        distances, array shape is (a.Rbins,a.jd) or 
        (a.Rbins,len(metbins)-1) if metbins is in kwargs.
    metdist0 : ndarray
        Metallicity distribution at the Solar radius, array 
        of len a.jd or len(metbins)-1.
    """
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save','sigma_gauss','metbins','tab',
                            'mode_pop','number','mode_iso','R'],this_function)
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                       'metallicity distributions',this_function)
    zlim = inpcheck_height(zlim,p,this_function)
    inpcheck_kwargs_compatibility(kwargs,this_function)
    
    if 'metbins' in kwargs:
        metbins = kwargs['metbins']
        mmin, mmax = metbins[0], metbins[-1]
        dmet = round(np.mean(np.diff(metbins)),3)
        metbinsc = np.add(metbins,dmet/2)[:-1]
    else:
        mmin, mmax, dmet = -1.1, 0.8, 0.05
        metbins = np.arange(mmin,mmax+dmet,dmet)
        metbinsc = np.add(metbins,dmet/2)[:-1]
        
    jm = len(metbinsc)
    if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):
        metdist = np.zeros((a.Rbins,jm))
        
    if mode_comp=='d':
        if 'R' in kwargs and kwargs['R']==p.Rsun:
            rho_d0 = _rhoz_d_(p,a,zlim=zlim,**kwargs)
        else:
            rho_d, rho_d0 = _rhoz_d_(p,a,zlim=zlim,**kwargs)
            AMRd = tab_reader(['AMRd'],p,a.T)[0]
        sum_rho0 = np.sum(np.sum(rho_d0))
        AMRd0 = tab_reader(['AMRd0'],p,a.T)[0]
        metdist0 = rebin_histogram(metbins,AMRd0[1],np.sum(rho_d0,axis=0))
        
        if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):
            for i in range(a.Rbins):
                sum_rho = np.sum(np.sum(rho_d[i]))
                sum_age_rho = np.sum(rho_d[i],axis=0)
                metdist[i] = rebin_histogram(metbins,AMRd[i+1],sum_age_rho)
            
    if mode_comp=='t':
        if 'R' in kwargs and kwargs['R']==p.Rsun:
            rho_t0 = _rhoz_t_(p,a,zlim=zlim,**kwargs)
        else:
            rho_t, rho_t0 = _rhoz_t_(p,a,zlim=zlim,**kwargs)
        AMRt = tab_reader(['AMRt'],p,a.T)[0]
        sum_rho0 = np.sum(np.sum(rho_t0))
        if sum_rho0!=0:
            metdist0 = rebin_histogram(metbins,AMRt[1],np.sum(rho_t0,axis=0))
        else:
            metdist0 = np.zeros((jm))
        if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):
            for i in range(a.Rbins):
                sum_rho = np.sum(np.sum(rho_t[i]))
                if sum_rho!=0:
                    metdist[i] = rebin_histogram(metbins,AMRt[1],np.sum(rho_t[i],axis=0))
                else:
                    metdist[i] = np.zeros((jm))
                    
    if mode_comp=='sh':
        if 'R' in kwargs and kwargs['R']==p.Rsun:
            rho_sh0 = _rhoz_sh_(p,a,zlim=zlim,**kwargs)
        else:
            rho_sh, rho_sh0 = _rhoz_sh_(p,a,zlim=zlim,**kwargs)
        AMRsh_spread = np.linspace(p.FeHsh-3*p.dFeHsh,p.FeHsh+3*p.dFeHsh,20*p.n_FeHsh)
        wsh = gauss_weights(AMRsh_spread,p.FeHsh,p.dFeHsh)
        sum_rho0 = np.sum(np.sum(rho_sh0))
        if sum_rho0!=0:
            metdist0 = rebin_histogram(metbins,AMRsh_spread,wsh*sum_rho0)
        else:
            metdist0 = np.zeros((jm))
        if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):
            for i in range(a.Rbins):
                sum_rho = np.sum(np.sum(rho_sh[i]))
                if sum_rho!=0:
                    metdist[i] = rebin_histogram(metbins,AMRsh_spread,wsh*sum_rho)
                else:
                    metdist[i] = np.zeros((jm))
            
    if mode_comp=='dt':  
        if 'R' in kwargs and kwargs['R']==p.Rsun:
            if 'tab' in kwargs:
                tabs = kwargs['tab']
                del kwargs['tab']
                rho_d0 = _rhoz_d_(p,a,zlim=zlim,**kwargs,tab=tabs[0])
                rho_t0 = _rhoz_t_(p,a,zlim=zlim,**kwargs,tab=tabs[1])
            else:
                rho_d0 = _rhoz_d_(p,a,zlim=zlim,**kwargs)
                rho_t0 = _rhoz_t_(p,a,zlim=zlim,**kwargs)
        else:
            if 'tab' in kwargs:
                tabs = kwargs['tab']
                del kwargs['tab']
                rho_d, rho_d0 = _rhoz_d_(p,a,zlim=zlim,**kwargs,tab=[tabs[0][0],tabs[1][0]])
                rho_t, rho_t0 = _rhoz_t_(p,a,zlim=zlim,**kwargs,tab=[tabs[0][1],tabs[1][1]])
            else:
                rho_d, rho_d0 = _rhoz_d_(p,a,zlim=zlim,**kwargs)
                rho_t, rho_t0 = _rhoz_t_(p,a,zlim=zlim,**kwargs)
            AMRd = tab_reader(['AMRd'],p,a.T)[0]
        sum_rho0 = np.sum(np.sum(rho_d0)) + np.sum(np.sum(rho_t0))
        AMRd0, AMRt = tab_reader(['AMRd0','AMRt'],p,a.T)
        metdist0 = rebin_histogram(metbins,np.concatenate((AMRd0[1],AMRt[1]),axis=0),
                                   np.concatenate((np.sum(rho_d0,axis=0),
                                                   np.sum(rho_t0,axis=0)),axis=0))  
        if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):                                                                           
            for i in range(a.Rbins):
                sum_rho = np.sum(np.sum(rho_d[i])) + np.sum(np.sum(rho_t[i]))
                sum_age_rhod = np.sum(rho_d[i],axis=0)
                sum_age_rhot = np.sum(rho_t[i],axis=0)                                      
                metdist[i] = rebin_histogram(metbins,np.concatenate((AMRd[i+1],AMRt[1]),axis=0),
                                             np.concatenate((sum_age_rhod,sum_age_rhot),axis=0))      
    if mode_comp=='tot':   
        if 'R' in kwargs and kwargs['R']==p.Rsun:
            if 'tab' in kwargs:
                tabs = kwargs['tab']
                del kwargs['tab']
                rho_d0 = _rhoz_d_(p,a,zlim=zlim,**kwargs,tab=tabs[0])
                rho_t0 = _rhoz_t_(p,a,zlim=zlim,**kwargs,tab=tabs[1])
                rho_sh0 = _rhoz_sh_(p,a,zlim=zlim,**kwargs,tab=tabs[2])
            else:
                rho_d0 = _rhoz_d_(p,a,zlim=zlim,**kwargs)
                rho_t0 = _rhoz_t_(p,a,zlim=zlim,**kwargs)
                rho_sh0 = _rhoz_sh_(p,a,zlim=zlim,**kwargs)
        else:
            if 'tab' in kwargs:
                tabs = kwargs['tab']
                del kwargs['tab']
                rho_d, rho_d0 = _rhoz_d_(p,a,zlim=zlim,**kwargs,tab=[tabs[0][0],tabs[1][0]])
                rho_t, rho_t0 = _rhoz_t_(p,a,zlim=zlim,**kwargs,tab=[tabs[0][1],tabs[1][1]])
                rho_sh, rho_sh0 = _rhoz_sh_(p,a,zlim=zlim,**kwargs,tab=[tabs[0][2],tabs[1][2]])
            else:
                rho_d, rho_d0 = _rhoz_d_(p,a,zlim=zlim,**kwargs)
                rho_t, rho_t0 = _rhoz_t_(p,a,zlim=zlim,**kwargs)
                rho_sh, rho_sh0 = _rhoz_sh_(p,a,zlim=zlim,**kwargs)
            AMRd = tab_reader(['AMRd'],p,a.T)[0]
        sum_rho0 = np.sum(np.sum(rho_d0)) + np.sum(np.sum(rho_t0)) + np.sum(np.sum(rho_sh0))
        AMRd0, AMRt = tab_reader(['AMRd0','AMRt'],p,a.T)
        AMRsh_spread = np.linspace(p.FeHsh-3*p.dFeHsh,p.FeHsh+3*p.dFeHsh,20*p.n_FeHsh)
        wsh = gauss_weights(AMRsh_spread,p.FeHsh,p.dFeHsh)
        
        amr_met0 = np.concatenate((AMRsh_spread,np.concatenate((AMRd0[1],AMRt[1]),axis=0)),axis=0)
        all_rho0 = np.concatenate((np.sum(np.sum(rho_sh0))*wsh,
                               np.concatenate((np.sum(rho_d0,axis=0),np.sum(rho_t0,axis=0)),axis=0)),
                               axis=0)
        metdist0 = rebin_histogram(metbins,amr_met0,all_rho0)
        
        if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):                                           
            for i in range(a.Rbins):
                sum_rho = np.sum(np.sum(rho_d[i])) + np.sum(np.sum(rho_t[i])) + np.sum(np.sum(rho_sh[i]))
                sum_age_rhod = np.sum(rho_d[i],axis=0)
                sum_age_rhot = np.sum(rho_t[i],axis=0)  
                sum_age_rhosh = np.sum(rho_sh[i],axis=0)[0]*wsh      
                amr_met = np.concatenate((AMRsh_spread,np.concatenate((AMRd[i+1],AMRt[1]),axis=0)),axis=0)
                all_rho = np.concatenate((sum_age_rhosh,
                                          np.concatenate((sum_age_rhod,sum_age_rhot),axis=0)),axis=0)
                metdist[i] = rebin_histogram(metbins,amr_met,all_rho)
    
    sum_rho0 = np.sum(metdist0)*dmet
    if sum_rho0!=0:
        metdist0 = metdist0/sum_rho0
    
    if ('sigma_gauss' in kwargs):
        metdist0 = gaussian_filter1d(metdist0,kwargs['sigma_gauss']//dmet)
        
    if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):   
        for i in range(a.Rbins):
            if np.sum(metdist[i])!=0:
                metdist[i] = metdist[i]/np.sum(metdist[i])/dmet
            if ('sigma_gauss' in kwargs):
                metdist[i] = gaussian_filter1d(metdist[i],kwargs['sigma_gauss']//dmet)
                          
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        if 'R' in kwargs and kwargs['R']==p.Rsun:
            ts.methist_save((metdist0,metbinsc),mode_comp,zlim)
        else:
            ts.methist_save((metdist,metdist0,metbinsc),mode_comp,zlim)
    
    if 'R' in kwargs and kwargs['R']==p.Rsun:
        return (metdist0, sum_rho0)
    else:
        return (metdist, metdist0)


def hr_monoage(mode_comp,ages,p,a,**kwargs):
    """
    Scale heights of the mono-age disk subpopulations as a function 
    of Galactocentric distance R. 
    
    Parameters
    ----------
    mode_comp ('d','dt' or 'tot'), ages, p, a 
    **kwargs : dict, optional keyword arguments
        valid keywords : save, between, mode_pop, tab, number,
        mode_iso
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()
        
    Returns
    -------
    H : 2d-ndarray
        Scale heights in pc, array shape is (a.Rbins,len(ages)) 
        or (a.Rbins,len(ages)-1) if between=True.
    """
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save','mode_pop','tab','between','number','mode_iso'],this_function)
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','dt','tot'],
                                       'mono-age scale heights',this_function)
    ages = inpcheck_age(ages,this_function)
    inpcheck_kwargs_compatibility(kwargs,this_function)
    
    if inpcheck_iskwargtype(kwargs,'between',True,bool,this_function):
        nage = len(ages) - 1 
    else:
        nage = len(ages)
        
    indt = np.array(np.subtract(tp,ages)//tr,dtype=np.int)
    H = np.zeros((a.Rbins,nage))
        
    if mode_comp=='d':
        rhozd = _rhoz_d_(p,a,**kwargs,local=False)
        sigmazd = _rhoz_d_(p,a,sigma=True,**kwargs,local=False)
        for i in range(a.Rbins):
            rhod0 = _rhoz_ages_(rhozd[i],indt,this_function,a,**kwargs)
            sigmad = _rhoz_ages_(sigmazd[i],indt,this_function,a,**kwargs)
            for k in range(nage):
                if rhod0[k,0]!=0:
                    H[i,k] = sigmad[k,0]/2/rhod0[k,0]
                else:
                    H[i,k] = np.nan
            
    if mode_comp=='dt':
        if 'tab' in kwargs:
            tabs = kwargs['tab']
            del kwargs['tab']
            rhozd = _rhoz_d_(p,a,**kwargs,local=False,tab=tabs[0])
            sigmazd = _rhoz_d_(p,a,sigma=True,**kwargs,local=False,tab=tabs[0])
            rhozt = _rhoz_t_(p,a,**kwargs,local=False,tab=tabs[1])
            sigmazt = _rhoz_t_(p,a,sigma=True,**kwargs,local=False,tab=tabs[1])
        else:
            rhozd = _rhoz_d_(p,a,**kwargs,local=False)
            sigmazd = _rhoz_d_(p,a,sigma=True,**kwargs,local=False)
            rhozt = _rhoz_t_(p,a,**kwargs,local=False)
            sigmazt = _rhoz_t_(p,a,sigma=True,**kwargs,local=False)
        for i in range(a.Rbins):
            rhod0 = _rhoz_ages_(rhozd[i],indt,this_function,a,**kwargs)
            sigmad = _rhoz_ages_(sigmazd[i],indt,this_function,a,**kwargs)
            rhot0 = _rhoz_ages_(rhozt[i],indt,this_function,a,**kwargs)
            sigmat = _rhoz_ages_(sigmazt[i],indt,this_function,a,**kwargs)
            for k in range(nage):
                if (rhod0[k,0] + rhot0[k,0])!=0:
                    H[i,k] = (sigmad[k,0] + sigmat[k,0])/2/(rhod0[k,0] + rhot0[k,0])
                else:
                    H[i,k] = np.nan
            
    if mode_comp=='tot':
        if 'tab' in kwargs:
            tabs = kwargs['tab']
            del kwargs['tab']
            rhozd = _rhoz_d_(p,a,**kwargs,local=False,tab=tabs[0])
            sigmazd = _rhoz_d_(p,a,sigma=True,**kwargs,local=False,tab=tabs[0])
            rhozt = _rhoz_t_(p,a,**kwargs,local=False,tab=tabs[1])
            sigmazt = _rhoz_t_(p,a,sigma=True,**kwargs,local=False,tab=tabs[1])
            rhozsh = _rhoz_sh_(p,a,**kwargs,local=False,tab=tabs[2])
            sigmazsh = _rhoz_sh_(p,a,sigma=True,**kwargs,local=False,tab=tabs[2])
        else:
            rhozd = _rhoz_d_(p,a,**kwargs,local=False)
            sigmazd = _rhoz_d_(p,a,sigma=True,**kwargs,local=False)
            rhozt = _rhoz_t_(p,a,**kwargs,local=False)
            sigmazt = _rhoz_t_(p,a,sigma=True,**kwargs,local=False)
            rhozsh = _rhoz_sh_(p,a,**kwargs,local=False)
            sigmazsh = _rhoz_sh_(p,a,sigma=True,**kwargs,local=False)
        for i in range(a.Rbins):
            rhod0 = _rhoz_ages_(rhozd[i],indt,this_function,a,**kwargs)
            sigmad = _rhoz_ages_(sigmazd[i],indt,this_function,a,**kwargs)
            rhot0 = _rhoz_ages_(rhozt[i],indt,this_function,a,**kwargs)
            sigmat = _rhoz_ages_(sigmazt[i],indt,this_function,a,**kwargs)
            rhosh0 = _rhoz_ages_(rhozsh[i],indt,this_function,a,**kwargs)
            sigmash = _rhoz_ages_(sigmazsh[i],indt,this_function,a,**kwargs)
            for k in range(nage):
                if (rhod0[k,0] + rhot0[k,0] + rhosh0[k,0])!=0:
                    H[i,k] = (sigmad[k,0] + sigmat[k,0] + sigmash[k,0])/2\
                            /(rhod0[k,0] + rhot0[k,0] + rhosh0[k,0])
                else:
                    H[i,k] = np.nan
       
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        ts.hr_monoage_save(H,mode_comp,ages)                

    return H


def hr_monomet(mode_comp,mets,p,a,**kwargs):
    """
    Scale heights of the mono-metallicity disk subpopulations as 
    a function of Galactocentric distance R. 
    
    Parameters
    ----------
    mode_comp ('d','dt' or 'tot'), mets, p, a 
    **kwargs : dict, optional keyword arguments
        valid keywords : save, mode_pop, tab, number, mode_iso
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()

    Returns
    -------
    hmet : 2d-ndarray
        Scale heights in pc, array shape is (a.Rbins,len(mets)-1).
    """
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save','mode_pop','tab','number','mode_iso'],this_function)
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','dt','tot'],
                                       'mono-metallicity scale heights',this_function)
    inpcheck_kwargs_compatibility(kwargs,this_function)
    
    nmet = len(mets) - 1 
    H = np.zeros((a.Rbins,nmet))
        
    if mode_comp=='d':
        rhozd = _rhoz_d_(p,a,**kwargs,local=False)
        sigmazd = _rhoz_d_(p,a,sigma=True,**kwargs,local=False)
        AMRd = tab_reader(['AMRd'],p,a.T)[0]
        for i in range(a.Rbins):
            indt = _ind_amr2t_(AMRd[i+1],mets)
            rhod0 = _rhoz_ages_(rhozd[i],indt,this_function,a,between=True,**kwargs)
            sigmad = _rhoz_ages_(sigmazd[i],indt,this_function,a,between=True,**kwargs)
            for k in range(nmet):
                if rhod0[k,0]!=0:
                    H[i,k] = sigmad[k,0]/2/rhod0[k,0]
                else:
                    H[i,k] = np.nan
            
    if mode_comp=='dt':
        if 'tab' in kwargs:
            tabs = kwargs['tab']
            del kwargs['tab']
            rhozd = _rhoz_d_(p,a,**kwargs,local=False,tab=tabs[0])
            sigmazd = _rhoz_d_(p,a,sigma=True,**kwargs,local=False,tab=tabs[0])
            rhozt = _rhoz_t_(p,a,**kwargs,local=False,tab=tabs[1])
            sigmazt = _rhoz_t_(p,a,sigma=True,**kwargs,local=False,tab=tabs[1])
        else:
            rhozd = _rhoz_d_(p,a,**kwargs,local=False)
            sigmazd = _rhoz_d_(p,a,sigma=True,**kwargs,local=False)
            rhozt = _rhoz_t_(p,a,**kwargs,local=False)
            sigmazt = _rhoz_t_(p,a,sigma=True,**kwargs,local=False)
        AMRd, AMRt = tab_reader(['AMRd','AMRt'],p,a.T)
        indtt = _ind_amr2t_(AMRt[1],mets)
        for i in range(a.Rbins):
            inddt = _ind_amr2t_(AMRd[i+1],mets)
            rhod0 = _rhoz_ages_(rhozd[i],inddt,this_function,a,between=True,**kwargs)
            sigmad = _rhoz_ages_(sigmazd[i],inddt,this_function,a,between=True,**kwargs)
            rhot0 = _rhoz_ages_(rhozt[i],indtt,this_function,a,between=True,**kwargs)
            sigmat = _rhoz_ages_(sigmazt[i],indtt,this_function,a,between=True,**kwargs)
            for k in range(nmet):
                if (rhod0[k,0] + rhot0[k,0])!=0:
                    H[i,k] = (sigmad[k,0] + sigmat[k,0])/2/(rhod0[k,0] + rhot0[k,0])
                else:
                    H[i,k] = np.nan
            
    if mode_comp=='tot':
        if 'tab' in kwargs:
            tabs = kwargs['tab']
            del kwargs['tab']
            rhozd = _rhoz_d_(p,a,**kwargs,local=False,tab=tabs[0])
            sigmazd = _rhoz_d_(p,a,sigma=True,**kwargs,local=False,tab=tabs[0])
            rhozt = _rhoz_t_(p,a,**kwargs,local=False,tab=tabs[1])
            sigmazt = _rhoz_t_(p,a,sigma=True,**kwargs,local=False,tab=tabs[1])
            rhozsh = _rhoz_sh_(p,a,**kwargs,local=False,tab=tabs[2])
            sigmazsh = _rhoz_sh_(p,a,sigma=True,**kwargs,local=False,tab=tabs[2])
        else:
            rhozd = _rhoz_d_(p,a,**kwargs,local=False)
            sigmazd = _rhoz_d_(p,a,sigma=True,**kwargs,local=False)
            rhozt = _rhoz_t_(p,a,**kwargs,local=False)
            sigmazt = _rhoz_t_(p,a,sigma=True,**kwargs,local=False)
            rhozsh = _rhoz_sh_(p,a,**kwargs,local=False)
            sigmazsh = _rhoz_sh_(p,a,sigma=True,**kwargs,local=False)
        AMRd, AMRt = tab_reader(['AMRd','AMRt'],p,a.T)
        indtt = _ind_amr2t_(AMRt[1],mets)
        for i in range(a.Rbins):
            inddt = _ind_amr2t_(AMRd[i+1],mets)
            rhod0 = _rhoz_ages_(rhozd[i],inddt,this_function,a,between=True,**kwargs)
            sigmad = _rhoz_ages_(sigmazd[i],inddt,this_function,a,between=True,**kwargs)
            rhot0 = _rhoz_ages_(rhozt[i],indtt,this_function,a,between=True,**kwargs)
            sigmat = _rhoz_ages_(sigmazt[i],indtt,this_function,a,between=True,**kwargs)
            rhosh0 = rhozsh[i]
            sigmash = sigmazsh[i]
            for k in range(nmet):
                if (mets[k]>=p.FeHsh-3*p.dFeHsh) and (mets[k+1]<=p.FeHsh+3*p.dFeHsh):
                    t1 = (mets[k]-p.FeHsh)/np.sqrt(2)/p.dFeHsh
                    t2 = (mets[k+1]-p.FeHsh)/np.sqrt(2)/p.dFeHsh
                    rhosh0_ik = np.sum(rhosh0[:,0])*(erf(t2)-erf(t1))/erf(3/np.sqrt(2))
                    sigmash_ik = np.sum(sigmash[:,0])*(erf(t2)-erf(t1))/erf(3/np.sqrt(2))
                else:
                    rhosh0_ik, sigmash_ik = 0,0
                if (rhod0[k,0] + rhot0[k,0] + rhosh0_ik)!=0:
                    H[i,k] = (sigmad[k,0] + sigmat[k,0] + sigmash_ik)/2\
                            /(rhod0[k,0] + rhot0[k,0] + rhosh0_ik)
                else:
                    H[i,k] = np.nan
       
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        ts.hr_monomet_save(H,mode_comp,mets)                 
    
    return H 


def sigwz(mode_comp,p,a,**kwargs):
    """
    W-velocity dispersion as a function of height z. 
    
    Parameters
    ----------
    mode_comp ('d','dt' or 'tot'), p, a 
    **kwargs : dict, optional keyword arguments
        valid keywords : save, mode_pop, tab, number, mode_iso
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()

    Returns
    -------
    sigwzr : 2d-ndarray
        Vertical profile of the W-velocity dispersion (km/s) for
        the different Galactocentric distances R, array shape is 
        (a.Rbins, a.n). 
    sigwzr0 : ndarray
        Same, but for the Solar radius p.Rsun, array of len a.n. 
    """
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save','mode_pop','tab','number','mode_iso','R'],this_function)
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','dt','tot'],
                                       'W-velocity dispersion',this_function)
    inpcheck_kwargs_compatibility(kwargs,this_function)
    
    AVRd0 = tab_reader(['AVR0'],p,a.T)[0]
    if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):
        AVRd = tab_reader(['AVR'],p,a.T)[0]
        if mode_comp=='dt' or mode_comp=='tot':
            Sigt = tab_reader(['Sigt'],p,a.T)[0]      
        
        rho_sigw = np.zeros((a.Rbins,a.n))
        rho_sum = np.zeros((a.Rbins,a.n))   
    
    if p.pkey==1:
        npeak = len(p.sigp)
        Fp0, Hdp0 = tab_reader(['Fp0','Hdp0'],p,a.T)
        fp0 = 1 - np.sum(Fp0[1:],axis=0)
        if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):
            Hdp = [tab_reader(['Hdp'],p,a.T,R=radius)[0] for radius in a.R]
            sigp = [table[0] for table in Hdp]
            Fp = [tab_reader(['Fp'],p,a.T,R=radius)[0] for radius in a.R]
            fpr0 = [1 - np.sum(subarray[1:],axis=0) for subarray in Fp]
        
    if 'R' in kwargs and kwargs['R']==p.Rsun:
        if 'tab' in kwargs and (mode_comp=='dt' or mode_comp=='tot'):
            tabs = kwargs['tab']
            del kwargs['tab']
            rho_z0 = _rhoz_d_(p,a,**kwargs,tab=tabs[0]) 
        else:
            rho_z0 = _rhoz_d_(p,a,**kwargs)   
    else:
        if 'tab' in kwargs and (mode_comp=='dt' or mode_comp=='tot'):
            tabs = kwargs['tab']
            del kwargs['tab']
            rho_z, rho_z0 = _rhoz_d_(p,a,**kwargs,tab=[tabs[0][0],tabs[1][0]])   
        else:
            rho_z, rho_z0 = _rhoz_d_(p,a,**kwargs)   
            
    rho_sum0 = np.sum(rho_z0,axis=-1)
    if p.pkey==1:
        rho_sigw0 = [(np.sum(AVRd0[1]*fp0*rho_z0[k]) + np.sum([p.sigp[l]*Fp0[l+1]*rho_z0[k]
                      for l in np.arange(npeak)])) for k in np.arange(a.n)]
    else:
        rho_sigw0 = [np.sum(AVRd0[1]*rho_z0[k]) for k in np.arange(a.n)] 
    if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):    
        for i in range(a.Rbins):
            rho_sum[i] = np.sum(rho_z[i],axis=-1)
            if p.pkey==1:
                rho_sigw[i] = [(np.sum(AVRd[i+1]*fpr0[i]*rho_z[i][k])
                             +np.sum([sigp[i][l]*Fp[i][l+1]*rho_z[i][k] for l in np.arange(npeak)]))
                             for k in np.arange(a.n)]
            else:
                rho_sigw[i] = [np.sum(AVRd[i+1]*rho_z[i][k]) for k in np.arange(a.n)] 
        
    if mode_comp=='dt' or mode_comp=='tot':
        if 'R' in kwargs and kwargs['R']==p.Rsun:
            if 'tab' in kwargs:
                rho_zt0 = _rhoz_t_(p,a,**kwargs,tab=tabs[1])
            else:
                rho_zt0 = _rhoz_t_(p,a,**kwargs)
        else:
            if 'tab' in kwargs:
                rho_zt, rho_zt0 = _rhoz_t_(p,a,**kwargs,tab=[tabs[0][1],tabs[1][1]])
            else:
                rho_zt, rho_zt0 = _rhoz_t_(p,a,**kwargs)
        rho_sigw0 = [rho_sigw0[k] + np.sum(p.sigt*rho_zt0[k]) for k in np.arange(a.n)] 
        rho_sum0 = rho_sum0 + np.sum(rho_zt0,axis=-1)
        if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):
            for i in range(a.Rbins):
                rho_sum[i] = np.add(rho_sum[i],np.sum(rho_zt[i],axis=-1))
                rho_sigw[i] = [rho_sigw[i][k] + np.sum(Sigt[1][i]*rho_zt[i][k]) for k in np.arange(a.n)]
                                 
    if mode_comp=='tot':
        if 'R' in kwargs and kwargs['R']==p.Rsun:
            if 'tab' in kwargs:
                rho_zsh0 = _rhoz_sh_(p,a,**kwargs,tab=tabs[2])
            else:
                rho_zsh0 = _rhoz_sh_(p,a,**kwargs)
        else:
            if 'tab' in kwargs:
                rho_zsh, rho_zsh0 = _rhoz_sh_(p,a,**kwargs,tab=[tabs[0][2],tabs[1][2]])
            else:
                rho_zsh, rho_zsh0 = _rhoz_sh_(p,a,**kwargs)
        rho_sigw0 = [rho_sigw0[k] + np.sum(p.sigsh*rho_zsh0[k]) for k in np.arange(a.n)] 
        rho_sum0 = rho_sum0 + np.sum(rho_zsh0,axis=-1)
        if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):
            for i in range(a.Rbins):
                rho_sum[i] = np.add(rho_sum[i],np.sum(rho_zsh[i],axis=-1))
                rho_sigw[i] = [rho_sigw[i][k] + np.sum(p.sigsh*rho_zsh[i][k]) for k in np.arange(a.n)]
            
    sigw_z0 = rho_sigw0/rho_sum0
    if ('R' not in kwargs) or ('R' in kwargs and kwargs['R']!=p.Rsun):
        sigw_z = rho_sigw/rho_sum
                
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        if 'R' in kwargs and kwargs['R']==p.Rsun:
            ts.sigwz_save(sigw_z0,mode_comp)   
        else:
            ts.sigwz_save((sigw_z,sigw_z0),mode_comp)   
    
    if 'R' in kwargs and kwargs['R']==p.Rsun:
        return sigw_z0
    else:
        return (sigw_z, sigw_z0)


def sigwr(mode_comp,zlim,p,a,**kwargs):
    """
    W-velocity dispersion as a function of Galactocentric 
    distance R. 
    
    Parameters
    ----------
    mode_comp ('d','dt' or 'tot'), zlim, p, a 
    **kwargs : dict, optional keyword arguments
        valid keywords : save, mode_pop, tab, number, mode_iso
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()

    Returns
    -------
    sigwr : ndarray
        Radial profile of the W-velocity dispersion, in km/s.
        Array of len a.Rbins. 
    """    
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save','mode_pop','number','tab','mode_iso'],this_function)
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','dt','tot'],
                                       'W-velocity dispersion',this_function)
    zlim = inpcheck_height(zlim,p,this_function)
    inpcheck_kwargs_compatibility(kwargs,this_function)
    AVRd = tab_reader(['AVR'],p,a.T)[0]
    if mode_comp=='dt' or mode_comp=='tot':
        Sigt = tab_reader(['Sigt'],p,a.T)[0]
        
    if p.pkey==1:
        Hdp = [tab_reader(['Hdp'],p,a.T,R=radius)[0] for radius in a.R]
        sigp = [table[0] for table in Hdp]
        Fp = [tab_reader(['Fp'],p,a.T,R=radius)[0] for radius in a.R]
        fpr0 = [1 - np.sum(subarray[1:],axis=0) for subarray in Fp]
        npeak = len(p.sigp)
    
    nz = int(len(zlim) - 1)
    indz = np.array([int(i//p.dz) for i in zlim])
    sigw_r = np.zeros((a.Rbins,nz))
    
    table = False
    if 'tab' in kwargs:
        tabs = kwargs['tab']
        del kwargs['tab']
        table = True
        rho_rdz = _rhoz_d_(p,a,**kwargs,local=False,tab=tabs[0])
    else:
        rho_rdz = _rhoz_d_(p,a,**kwargs,local=False)
        
    if mode_comp=='dt' or mode_comp=='tot':
        if table:
            rho_rtz = _rhoz_t_(p,a,**kwargs,local=False,tab=tabs[1])
        else:
            rho_rtz = _rhoz_t_(p,a,**kwargs,local=False)
    if mode_comp=='tot':
        if table:
            rho_rshz = _rhoz_sh_(p,a,**kwargs,local=False,tab=tabs[2])
        else:
            rho_rshz = _rhoz_sh_(p,a,**kwargs,local=False)
        
    for i in range(a.Rbins):
        rho_rd = [np.sum(rho_rdz[i][indz[k]:indz[k+1]],axis=0) for k in range(nz)]
        if p.pkey==1:
            sigw_rho = [(np.sum(AVRd[i+1]*fpr0[i]*rho_rd[k]) + 
                        np.sum([sigp[i][l]*Fp[i][l+1]*rho_rd[k] for l in np.arange(npeak)]))
                        for k in range(nz)]
        else:
            sigw_rho = [np.sum(AVRd[i+1]*rho_rd[k]) for k in range(nz)]
        rho_sum = [np.sum(rho_rd[k]) for k in range(nz)]
        
        if mode_comp=='dt' or mode_comp=='tot':
            rho_rt = [np.sum(rho_rtz[i][indz[k]:indz[k+1]],axis=0) for k in range(nz)]
            sigw_rho = np.add(sigw_rho,[np.sum(Sigt[1][i]*rho_rt[k]) for k in range(nz)])
            rho_sum = np.add(rho_sum,[np.sum(rho_rt[k]) for k in range(nz)])
            
        if mode_comp=='tot':
            rho_rsh = [np.sum(rho_rshz[i][indz[k]:indz[k+1]],axis=0) for k in range(nz)]
            sigw_rho = np.add(sigw_rho,[np.sum(p.sigsh*rho_rsh[k]) for k in range(nz)])
            rho_sum = np.add(rho_sum,[np.sum(rho_rsh[k]) for k in range(nz)])

        sigw_r[i] = np.divide(sigw_rho,rho_sum)            
    
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        ts.sigwr_save(sigw_r,mode_comp,zlim)   
                                                                                
    return sigw_r
                                                                                
                                                
def sigwr_monoage(mode_comp,zlim,ages,p,a,**kwargs):
    """
    W-velocity dispersion as a function of Galactocentric 
    distance R for the mono-age subpoplations. 
    
    Parameters
    ----------
    mode_comp ('d','dt','tot'), zlim, ages, p, a 
    **kwargs : dict, optional keyword arguments
        valid keywords : save, between, mode_pop, tab, number,
        mode_iso
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()

    Returns
    -------
    sigwr : 2d-ndarray
        Radial profile of the W-velocity dispersion (km/s) of mono-
        age subpopulations. Array shape is (a.Rbins,len(ages)) or
        (a.Rbins,len(ages)-1) if between=True. 
    """

    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save','mode_pop','tab','number','between','mode_iso'],this_function)
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','dt','tot'],
                                       'W-velocity dispersion',this_function)
    ages = inpcheck_age(ages,this_function)
    zlim = inpcheck_height(zlim,p,this_function)
    inpcheck_kwargs_compatibility(kwargs,this_function)

    AVRd = tab_reader(['AVR'],p,a.T)[0]   
    if mode_comp=='dt' or mode_comp=='tot':
       Sigt = tab_reader(['Sigt'],p,a.T)[0] 
            
    if p.pkey==1:
        Hdp = [tab_reader(['Hdp'],p,a.T,R=radius)[0] for radius in a.R]
        sigp = [table[0] for table in Hdp]
        Fp = [tab_reader(['Fp'],p,a.T,R=radius)[0] for radius in a.R]
        fpr0 = [1 - np.sum(subarray[1:],axis=0) for subarray in Fp]
        npeak = len(p.sigp)
    
    indt = np.array(np.subtract(tp,ages)//tr,dtype=np.int)
    if inpcheck_iskwargtype(kwargs,'between',True,bool,this_function):
        nage = int(len(ages) - 1)
    else:
        nage = len(ages)
    
    sigw_r = np.zeros((a.Rbins,nage))
    
    kwargs_calc = kwargs.copy()   
    if 'tab' in kwargs:
        del kwargs_calc['tab']
    
    for i in range(a.Rbins):
        if 'tab' in kwargs:
            if mode_comp!='d':
                rhord = _rhoz_d_(p,a,zlim=zlim,R=a.R[i],**kwargs_calc,
                                 local=False,tab=kwargs['tab'][0][i])[0]
            else:
                rhord = _rhoz_d_(p,a,zlim=zlim,R=a.R[i],**kwargs_calc,
                                 local=False,tab=kwargs['tab'][i])[0]
        else:
            rhord = _rhoz_d_(p,a,zlim=zlim,R=a.R[i],**kwargs_calc,local=False)[0]
        
        if inpcheck_iskwargtype(kwargs_calc,'between',True,bool,this_function):
            kwargs_calc2 = kwargs_calc.copy()
            del kwargs_calc2['between']
            sigw_rho, rho_sum = np.zeros((nage)), np.zeros((nage)) 
            for k in range(nage):
                indt_list = np.arange(indt[k+1],indt[k],dtype=np.int)
                rho_rd = _rhoz_ages_(rhord,indt_list,this_function,a,**kwargs_calc2)
                rho_rd = np.sum(rho_rd,axis=1)
                if p.pkey==1:
                    sigw_rho[k] = (np.sum(AVRd[i+1][indt_list]*fpr0[i][indt_list]*rho_rd) + 
                                   np.sum(np.sum([sigp[i][l]*Fp[i][l+1][indt_list]*rho_rd 
                                                  for l in np.arange(npeak)],axis=0)))
                else:
                    sigw_rho[k] = np.sum(AVRd[i+1][indt_list]*rho_rd)
                rho_sum[k] = np.sum(rho_rd)
        else:
            rho_rd = _rhoz_ages_(rhord,indt,this_function,a,**kwargs)
            rho_rd = np.sum(rho_rd,axis=1)
            if p.pkey==1:
                sigw_rho = (AVRd[i+1][indt]*fpr0[i][indt]*rho_rd + 
                            np.sum([sigp[i][l]*Fp[i][l+1][indt]*rho_rd for l in np.arange(npeak)],axis=0))
            else:
                sigw_rho = AVRd[i+1][indt]*rho_rd
            rho_sum = rho_rd
        
        if mode_comp=='dt' or mode_comp=='tot':
            if 'tab' in kwargs:
                rhort = _rhoz_t_(p,a,zlim=zlim,R=a.R[i],**kwargs_calc,
                                 local=False,tab=kwargs['tab'][1][i])[0]
            else:
                rhort = _rhoz_t_(p,a,zlim=zlim,R=a.R[i],**kwargs_calc,local=False)[0]
            rho_rt = _rhoz_ages_(rhort,indt,this_function,a,**kwargs)
            rho_rt = np.sum(rho_rt,axis=1)
            sigw_rho = sigw_rho + Sigt[1][i]*rho_rt
            rho_sum = rho_sum + rho_rt
            
        if mode_comp=='tot':
            if 'tab' in kwargs:
                rhorsh = _rhoz_sh_(p,a,zlim=zlim,R=a.R[i],**kwargs_calc,
                                   local=False,tab=kwargs['tab'][2][i])[0]
            else:
                rhorsh = _rhoz_sh_(p,a,zlim=zlim,R=a.R[i],**kwargs_calc,local=False)[0]
            rho_rsh = _rhoz_ages_(rhorsh,indt,this_function,a,**kwargs)
            rho_rsh = np.sum(rho_rsh,axis=1)
            sigw_rho = sigw_rho + p.sigsh*rho_rsh
            rho_sum = rho_sum + rho_rsh      
        
        for k in range(nage):
            if rho_sum[k]!=0:
                sigw_r[i,k] = sigw_rho[k]/rho_sum[k]
            else:
                sigw_r[i,k] = np.nan
                              
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        ts.sigwr_monoage_save(sigw_r,mode_comp,zlim,ages)   
            
    return sigw_r
            

def sigwr_monomet(mode_comp,zlim,mets,p,a,**kwargs):
    """
    W-velocity dispersion as a function of Galactocentric 
    distance R for the mono-metallicity subpoplations.

    Parameters
    ----------
    mode_comp ('d','dt' or 'tot'), zlim, mets, p, a 
    **kwargs : dict, optional keyword arguments
        valid keywords : save, mode_pop, tab, number, mode_iso
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()

    Returns
    -------
    sigwmet : 2d-ndarray
        Radial profile of the W-velocity dispersion (km/s) for
        the mono-metallicity bins. Array shape is (a.Rbins,
        len(mets)-1).          
    """
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save','mode_pop','tab','number','mode_iso'],this_function)
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','dt','tot'],
                                       'W-velocity dispersion',this_function)
    zlim = inpcheck_height(zlim,p,this_function)
    inpcheck_kwargs_compatibility(kwargs,this_function)
    
    AMRd,AVRd = tab_reader(['AMRd','AVR'],p,a.T)
    if mode_comp=='dt' or mode_comp=='tot':
        AMRt,Sigt = tab_reader(['AMRt','Sigt'],p,a.T)
    
    if p.pkey==1:
        Hdp = [tab_reader(['Hdp'],p,a.T,R=radius)[0] for radius in a.R]
        sigp = [table[0] for table in Hdp]
        Fp = [tab_reader(['Fp'],p,a.T,R=radius)[0] for radius in a.R]
        fpr0 = [1 - np.sum(subarray[1:],axis=0) for subarray in Fp]
        npeak = len(p.sigp)
        
    nage = int(len(mets)-1)
    sigw_r = np.zeros((nage,a.Rbins))
        
    table = False
    if 'tab' in kwargs:
        table = True
        tabs = kwargs['tab']
        del kwargs['tab']
    
    for i in range(a.Rbins):
        if table:
            if mode_comp!='d':
                rhord = _rhoz_d_(p,a,zlim=zlim,R=a.R[i],**kwargs,local=False,tab=tabs[0][i])[0]   
            else:
                rhord = _rhoz_d_(p,a,zlim=zlim,R=a.R[i],**kwargs,local=False,tab=tabs[i])[0] 
        else:
            rhord = _rhoz_d_(p,a,zlim=zlim,R=a.R[i],**kwargs,local=False)[0]    
        inddt = _ind_amr2t_(AMRd[i+1],mets)
        sigw_rho, rho_sum = np.zeros((nage)), np.zeros((nage)) 
        for k in range(nage):
            ind1, ind2 = np.sort([inddt[k+1],inddt[k]])
            if ind1!=-999 and ind2!=-999:
                indt_list = np.arange(ind1,ind2,dtype=np.int)
                rho_rd = _rhoz_ages_(rhord,indt_list,this_function,a,**kwargs)
                rho_rd = np.sum(rho_rd,axis=1)
                if p.pkey==1:
                    sigw_rho[k] = (np.sum(AVRd[i+1][indt_list]*fpr0[i][indt_list]*rho_rd) + 
                                   np.sum(np.sum([sigp[i][l]*Fp[i][l+1][indt_list]*rho_rd 
                                                  for l in np.arange(npeak)],axis=0)))
                else:
                    sigw_rho[k] = np.sum(AVRd[i+1][indt_list]*rho_rd)
                rho_sum[k] = np.sum(rho_rd)
            
        if mode_comp=='dt' or mode_comp=='tot':
            if table:
                rhort = _rhoz_t_(p,a,zlim=zlim,R=a.R[i],**kwargs,local=False,tab=tabs[1][i])[0]
            else:
                rhort = _rhoz_t_(p,a,zlim=zlim,R=a.R[i],**kwargs,local=False)[0]
            indtt = _ind_amr2t_(AMRt[1],mets)
            for k in range(nage):
                ind1, ind2 = np.sort([indtt[k+1],indtt[k]])
                if ind1!=-999 and ind2!=-999:
                    indt_list = np.arange(ind1,ind2,dtype=np.int)
                    rho_rt = _rhoz_ages_(rhort,indt_list,this_function,a,**kwargs)
                    rho_rt = np.sum(rho_rt,axis=1)
                    sigw_rho[k] = sigw_rho[k] + np.sum(Sigt[1][i]*rho_rt)
                    rho_sum[k] = rho_sum[k] + np.sum(rho_rt)
            
        if mode_comp=='tot':
            if table:
                rhorsh = _rhoz_sh_(p,a,zlim=zlim,R=a.R[i],**kwargs,local=False,tab=tabs[2][i])[0]
            else:
                rhorsh = _rhoz_sh_(p,a,zlim=zlim,R=a.R[i],**kwargs,local=False)[0]
            for k in range(nage):
                if (mets[k]>=p.FeHsh-3*p.dFeHsh) and (mets[k+1]<=p.FeHsh+3*p.dFeHsh):
                    t1 = (mets[k]-p.FeHsh)/np.sqrt(2)/p.dFeHsh
                    t2 = (mets[k+1]-p.FeHsh)/np.sqrt(2)/p.dFeHsh
                    rho_rsh = rhorsh*(erf(t2)-erf(t1))/erf(3/np.sqrt(2))  
                    sigw_rho[k] = sigw_rho[k] + np.sum(p.sigsh*rho_rsh)
                    rho_sum[k] = rho_sum[k] + np.sum(rho_rsh)      

        for k in range(nage):
            if rho_sum[k]!=0:
                sigw_r[k,i] = sigw_rho[k]/rho_sum[k]
            else:
                sigw_r[k,i] = np.nan                  

    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        ts.sigwr_monomet_save(sigw_r,mode_comp,zlim,mets)   
                    
    return sigw_r


def mean_quantity(mode_comp,R,zlim,quantity,p,a,**kwargs):
    """
    Calculates mean value of some quantity as a function of 
    height z, weighted by the spatial number densities of the 
    stellar assemblies. 
    
    Parameters
    ----------
    mode_comp ('d','t','sh','dt','tot'), R, zlim, quantity, p, a
    **kwargs : dict, optional keyword arguments
        valid keywords : save, mode_pop, tab, mode_iso, ages, mets 
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()
    
    Returns
    -------
    Q_mean : ndarray
        Vertical profile of the given quantity in the selected 
        horizontal slice zlim. 
        Array of len (zlim[1]-zlim[0])//p.dz. 
    """    
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save','mode_pop','tab','mode_iso','ages','mets'],this_function)
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],quantity,this_function)
    zlim = inpcheck_height(zlim,p,this_function)
    R = inpcheck_radius(R,p,this_function)   
    inpcheck_kwargs_compatibility(kwargs,this_function)
    
    indr = int(_indr_(R,p,a) + 1)
    indz1, indz2 = int(zlim[0]//p.dz), int(zlim[1]//p.dz)
    Q_mean = np.zeros((int(indz2-indz1)))

    if 'ages' in kwargs: 
        column = 'age'
        bins = kwargs['ages']
    if 'mets' in kwargs:
        column = 'FeH'
        bins = kwargs['mets']
    
    Fi = tab_reader(['Phi'],p,a.T)[0]
    
    if mode_comp=='d' or mode_comp=='dt' or mode_comp=='tot':
        Hd,AVR = tab_reader(['Hd','AVR'],p,a.T)
        if p.pkey==1:
            Hdp = tab_reader(['Hdp'],p,a.T,R=R)[0] 
            sigp = Hdp[0]
            Fp = tab_reader(['Fp'],p,a.T,R=R)[0] 
            fp0 = 1 - np.sum(Fp[1:],axis=0) 
            npeak = len(p.sigp)
        if 'mode_pop' not in kwargs and 'tab' not in kwargs:
            tabd = tab_reader('ssp',p,a.T,R=R,mode='d',tab=True,**kwargs)
        else:
            if 'mode_pop' in kwargs:
                tabd = tab_reader(kwargs['mode_pop'],p,a.T,R=R,mode='d',tab=True,**kwargs)
            else:
                if mode_comp=='d':
                    tabd = kwargs['tab']
                if mode_comp=='dt' or mode_comp=='tot':
                    tabd = kwargs['tab'][0]
        if ('ages' in kwargs) or ('mets' in kwargs):
            ind_sub = np.where((tabd[column]>bins[0])&(tabd[column]<bins[1]))[0]
            tabd = tabd[ind_sub]
        inddt = np.array(np.subtract(tp,tabd['age'])//tr,dtype=np.int)
        disk_ages = tp - a.t
        
    if mode_comp=='t' or mode_comp=='dt' or mode_comp=='tot':
        Ht,Sigt = tab_reader(['Ht','Sigt'],p,a.T)
        if 'mode_pop' not in kwargs and 'tab' not in kwargs:
            tabt = tab_reader('ssp',p,a.T,R=R,mode='t',tab=True,**kwargs)
        else:
            if 'mode_pop' in kwargs:
                tabt = tab_reader(kwargs['mode_pop'],p,a.T,R=R,mode='t',tab=True,**kwargs)
            else:
                if mode_comp=='t':
                    tabt = kwargs['tab']
                if mode_comp=='dt' or mode_comp=='tot':
                    tabt = kwargs['tab'][1]
        if ('ages' in kwargs) or ('mets' in kwargs):
            ind_sub = np.where((tabt[column]>bins[0])&(tabt[column]<bins[1]))[0]
            tabt = tabt[ind_sub]
            
    if mode_comp=='sh' or mode_comp=='tot':
        Hsh = tab_reader(['Hsh'],p,a.T)[0]
        if 'mode_pop' not in kwargs and 'tab' not in kwargs:
            tabsh = tab_reader('ssp',p,a.T,R=R,mode='sh',tab=True,**kwargs)
        else:
            if 'mode_pop' in kwargs:
                tabsh = tab_reader(kwargs['mode_pop'],p,a.T,R=R,mode='sh',tab=True,**kwargs)
            else:
                if mode_comp=='sh':
                    tabsh = kwargs['tab']
                if mode_comp=='tot':
                    tabsh = kwargs['tab'][2]
        if ('ages' in kwargs) or ('mets' in kwargs):
            ind_sub = np.where((tabsh[column]>bins[0])&(tabsh[column]<bins[1]))[0]
            tabsh = tabsh[ind_sub]
            
    for i in range(indz1,indz2):
        weighted_nzsum, nzsum = 0, 0  
        if mode_comp=='d' or mode_comp=='dt' or mode_comp=='tot':
            if p.pkey==1:
                Nz_d, Nz_term1_d, Nz_term2_d = _meanq_d_(indr,i,inddt,tabd,Fi,Hd,AVR,p,a,fp0=fp0,
                                                         Fp=Fp,Hdp=Hdp,sigp=sigp,npeak=npeak)
            else:
                Nz_d = _meanq_d_(indr,i,inddt,tabd,Fi,Hd,AVR,p,a)
            nzsum += np.sum(Nz_d)
        if mode_comp=='t' or mode_comp=='dt' or mode_comp=='tot':
            Nz_t = _meanq_t_(indr,i,tabt,Fi,Ht,Sigt,p,a)
            nzsum += np.sum(Nz_t)
        if mode_comp=='sh' or mode_comp=='tot':
            Nz_sh = _meanq_sh_(indr,i,tabsh,Fi,Hsh,p,a)
            nzsum += np.sum(Nz_sh)
        
        if nzsum!=0:
            if quantity=='sigw':
                if mode_comp=='d' or mode_comp=='dt' or mode_comp=='tot':
                    if p.pkey==1:
                        weighted_nzsum += np.sum(Nz_term1_d*AVR[indr][inddt])
                        for k in range(npeak):
                            weighted_nzsum += np.sum(Nz_term2_d[k]*sigp[k]) 
                    else:
                        weighted_nzsum += np.sum(Nz_d*AVR[indr][inddt])
                if mode_comp=='t' or mode_comp=='dt' or mode_comp=='tot':
                    weighted_nzsum += np.sum(Nz_t*Sigt[1][indr-1])
                if mode_comp=='sh' or mode_comp=='tot':
                    weighted_nzsum += np.sum(Nz_sh*p.sigsh)
            else:
                if mode_comp=='d' or mode_comp=='dt' or mode_comp=='tot':
                    weighted_nzsum += np.sum(Nz_d*tabd[quantity])
                if mode_comp=='t' or mode_comp=='dt' or mode_comp=='tot':
                    weighted_nzsum += np.sum(Nz_t*tabt[quantity])
                if mode_comp=='sh' or mode_comp=='tot':
                    weighted_nzsum += np.sum(Nz_sh*tabsh[quantity])
            Q_mean[i] =  weighted_nzsum/nzsum
        else:
            Q_mean[i] = np.nan
           
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        ts.mean_quantity_save(Q_mean,mode_comp,R,zlim,quantity)   
                
    return Q_mean


def pops_in_volume(mode_comp,R,volume,p,a,**kwargs):
    """
    Calculates the number of stars in a volume.  
    
    Parameters
    ----------
    mode_comp : str
        model component(s) - 'd','t','sh','dt' or 'tot'. 
    R : scalar
        Galactocentric distance, kpc. 
    volume : scalar or array-like
        If scalar, then this volume is the same for all z-slices,
        e.g. cylinder can be modeled in this way. If value is an 
        array (must match the model grid a.z or be consistent with 
        the assumed zlim and zmax in kwargs, see below), then each 
        volumes of z-slices can be different. This is suitable 
        for the modeling of a sphere or a cone. Units are pc^3. 
        Note that if the modeled volume is located at both positive
        and negative z, this must be included here, as the function
        works with the absolute z-values. 
    p, a : standard collections of parameters and useful arrays
    **kwargs : dict, optional keyword arguments
        valid keywords: save, mode_pop, tab, mode_iso, vln, zlim 
        zlim : array-like
            Range of heights [zmim,zmax] to be processed. Note 
            that zmax-zmin > p.dz, the slice cannot be thinner
            than the model z-resolution. If this parameter is 
            given, and volume is an array, be sure that 
            len(volume) equals to (zmax-zlim)//p.dz. If no zlim
            is given, then all z from 0 to p.zmax are considered.
        
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()
    
    Returns
    -------
    tab : astropy table
        Column 'Nz' contains the number of stars (stellar 
        assembly populations) located in the volume specified by
        the parameters R (where in the disk), zlim (which z range),
        and volume (what is the volume of a z-slice = allows to
        model different shapes). 
    """
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save','mode_pop','tab','mode_iso','vln','zlim'],this_function)                            
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                       'stellar assemblies number density',this_function)
    if R!=p.Rsun:
        R = inpcheck_radius(R,p,this_function)  
    inpcheck_kwargs_compatibility(kwargs,this_function)
    
    popname = 'ssp'
    if 'tab' in kwargs:
        tab = kwargs['tab']
        if mode_comp=='d': 
            tabd = tab
        if mode_comp=='t': 
            tabt = tab
        if mode_comp=='sh':
            tabsh = tab
        if mode_comp=='dt':
            tabd, tabt = tab 
        if mode_comp=='tot':
            tabd, tabt, tabsh = tab
    else:    
        if 'mode_pop' in kwargs:
            popname = kwargs['mode_pop']        
        if mode_comp=='d' or mode_comp=='dt' or mode_comp=='tot':
            tabd = tab_reader(popname,p,a.T,R=R,mode='d',tab=True,**kwargs)
        if mode_comp=='t' or mode_comp=='dt' or mode_comp=='tot':
            tabt = tab_reader(popname,p,a.T,R=R,mode='t',tab=True,**kwargs)
        if mode_comp=='sh' or mode_comp=='tot':
            tabsh = tab_reader(popname,p,a.T,R=R,mode='sh',tab=True,**kwargs)
    
    if 'zlim' in kwargs:
        indz1, indz2 = int(abs(kwargs['zlim'][0])//p.dz), int(abs(kwargs['zlim'][1]//p.dz))
        indz1, indz2 = np.sort([indz1,indz2])
    else:
        indz1, indz2 = 0, int(len(a.n_array))
        
    if R==p.Rsun:
        Fi = tab_reader(['Phi0'],p,a.T)[0]
        indr = 1 
    else:
        Fi = tab_reader(['Phi'],p,a.T)[0]
        indr = int(_indr_(R,p,a) + 1)
        
    if mode_comp=='d' or mode_comp=='dt' or mode_comp=='tot':
        if R==p.Rsun:
            Hd, AVR = tab_reader(['Hd0','AVR0'],p,a.T)
            if p.pkey==1:
                Hdp, Fp = tab_reader(['Hdp0','Fp0'],p,a.T)
        else:
            Hd, AVR = tab_reader(['Hd','AVR'],p,a.T)
            if p.pkey==1:
                Hdp, Fp = tab_reader(['Hdp','Fp'],p,a.T,R=R)
        
        if p.pkey==1:
            sigp = Hdp[0]
            fpr0 = 1 - np.sum(Fp[1:],axis=0) 
            npeak = len(p.sigp)
        
            wd0 = np.array([fpr0[i]/2/Hd[indr][i]*\
                            np.sum(np.exp(-Fi[indr][indz1:indz2]/KM**2/AVR[indr][i]**2)*volume) 
                            for i in a.jd_array])
            wdp = np.array([np.sum([Fp[k+1][i]/2/Hdp[1][k]*\
                            np.sum(np.exp(-Fi[indr][indz1:indz2]/KM**2/sigp[k]**2)*volume) 
                            for k in np.arange(npeak)]) for i in a.jd_array])
        else:
            wd = np.array([0.5/Hd[indr][i]*\
                           np.sum(np.exp(-Fi[indr][indz1:indz2]/KM**2/AVR[indr][i]**2)*volume)
                           for i in a.jd_array])
            
    if mode_comp=='t' or mode_comp=='dt' or mode_comp=='tot':
        if R==p.Rsun:
            Ht = tab_reader(['Ht0'],p,a.T)[0]
            wt = 0.5/Ht[1]*np.sum(np.exp(-Fi[1][indz1:indz2]/KM**2/p.sigt**2)*volume)
        else:
            Ht, Sigt = tab_reader(['Ht','Sigt'],p,a.T)
            wt = 0.5/Ht[1][indr-1]*np.sum(np.exp(-Fi[indr][indz1:indz2]/KM**2/Sigt[1][indr-1]**2)*volume) 
    if mode_comp=='sh' or mode_comp=='tot':
        if R==p.Rsun:
            Hsh = tab_reader(['Hsh0'],p,a.T)[0]
            wsh = 0.5/Hsh[1]*np.sum(np.exp(-Fi[indr][indz1:indz2]/KM**2/p.sigsh**2)*volume) 
        else:
            Hsh = tab_reader(['Hsh'],p,a.T)[0]
            wsh = 0.5/Hsh[1][indr-1]*np.sum(np.exp(-Fi[indr][indz1:indz2]/KM**2/p.sigsh**2)*volume) 
    
    if mode_comp=='d' or mode_comp=='dt' or mode_comp=='tot':
        indt_tab = np.array(np.subtract(tp,tabd['age'])//tr,dtype=np.int)
        if p.pkey==1:
            sum_Nzd = np.add(tabd['N']*wd0[indt_tab],tabd['N']*wdp[indt_tab])
        else:
            sum_Nzd = tabd['N']*wd[indt_tab]
    if mode_comp=='t' or mode_comp=='dt' or mode_comp=='tot':
        sum_Nzt = tabt['N']*wt
    if mode_comp=='sh' or mode_comp=='tot':
        sum_Nzsh = tabsh['N']*wsh
    
    if mode_comp=='d':
        sum_Nz = np.array(sum_Nzd)
        tabd['Nz'] = sum_Nz
        tab = tabd
    if mode_comp=='t':
        sum_Nz = np.array(sum_Nzt)
        tabt['Nz'] = sum_Nz
        tab = tabt
    if mode_comp=='sh':
        sum_Nz = np.array(sum_Nzsh)
        tabsh['Nz'] = sum_Nz
        tab = tabsh
    if mode_comp=='dt':
        sum_Nz = (np.array(sum_Nzd), np.array(sum_Nzt))
        tabd['Nz'] = sum_Nzd
        tabt['Nz'] = sum_Nzt
        tab = (tabd, tabt)
    if mode_comp=='tot':
        sum_Nz = (np.array(sum_Nzd), np.array(sum_Nzt), np.array(sum_Nzsh))
        tabd['Nz'] = sum_Nzd
        tabt['Nz'] = sum_Nzt
        tabsh['Nz'] = sum_Nzsh
        tab = (tabd, tabt, tabsh)
        
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        ts.pops_in_volume_save(tab,mode_comp,R,popname)
    
    return tab


def disk_brightness(mode_comp,mode_geom,bands,p,a,**kwargs):
    """
    MW as an external galaxy. Calculates the surface brightness 
    or colour profile of the MW if it is viewed edge-on or face-on 
    (individual model components and stellar populations can be
    selected). 
    
    Parameters
    ----------
    mode_comp ('d','t','sh','dt','tot'), mode_geom, bands, p, a
    bands : string or list
        If bands is a string, it corresponds to the band of the 
        surface brightness profile. If it is a list, then it gives
        the two band names to be used for the colour profilie - e.g. 
        ['U','V'] for U-V.
    mode_geom : string
        Can be 'face-on' or 'edge-on', defines orientation. 
    **kwargs : dict, optional keyword arguments
        valid keywords: save, zlim, mode_iso 
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()
    
    Returns
    -------
    profile : ndarray
        Disk surface brightness profile (mag/arcsec^2) or colour 
        profile (mag). Use together with a.R array. 
    """
    
    this_function = inspect.stack()[0][3]
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                           'disk brightness/colour profile',this_function)

    inpcheck_kwargs(kwargs,['zlim','save','mode_iso'],this_function)
    inpcheck_kwargs_compatibility(kwargs,this_function)
    zlim = [0,p.zmax]
    if 'zlim' in kwargs :
        zlim = kwargs['zlim']
        if mode_geom=='face-on':
            print(this_function + ": Unnecessary input. Keyword 'zlim' "+\
                  "doesn't work for a 'face-on' disk.")   
    mode_iso = 'Padova'
    if 'mode_iso' in kwargs:
        mode_iso = kwargs['mode_iso']
    Magsun = {'U':5.61,'B':5.44,'V':4.81,'I':4.1,'K':3.27,
              'GBP_EDR3':5.0,'GRP_EDR3':4.18,'G_EDR3':4.67}
    Lsun_bol = 1            # In Solar units 
    Msun_bol = 4.74         # In mag 
    Lsun = {key:Lsun_bol*10**(0.4*(Msun_bol - Magsun[key])) for key in Magsun}
    # look here: http://mips.as.arizona.edu/~cnaw/sun.html
    
    if mode_comp=='d' or mode_comp=='dt' or mode_comp=='tot':
        if 'tab' in kwargs:
            if mode_comp!='d':
                tabd = kwargs['tab'][0]
            else:
                tabd = kwargs['tab']
        else:
            tabd = [Table.read(os.path.join(a.T['poptab'],
                          ''.join(('SSP_R',str(i),'_d_',mode_iso,'.csv')))) for i in a.R]
        if type(bands)==str:
            goodd = [np.where((table[bands]>-15))[0] for table in tabd]
        else:
            goodd = [np.where((table[bands[0]]>-15)&(table[bands[1]]>-15))[0] for table in tabd]
    if mode_comp=='t' or mode_comp=='dt':
        if 'tab' in kwargs:
            if mode_comp!='t':
                tabt = kwargs['tab'][1]
            else:
                tabt = kwargs['tab']
        else:
            tabt = [Table.read(os.path.join(a.T['poptab'],
                          ''.join(('SSP_R',str(i),'_t_',mode_iso,'.csv')))) for i in a.R]
        if type(bands)==str:
            goodt = [np.where((table[bands]>-15))[0] for table in tabt]
        else:
            goodt = [np.where((table[bands[0]]>-15)&(table[bands[1]]>-15))[0] for table in tabt]
    if mode_comp=='sh' or mode_comp=='tot':
        if 'tab' in kwargs:
            if mode_comp!='sh':
                tabt = kwargs['tab'][2]
            else:
                tabt = kwargs['tab']
        else:
            tabsh = [Table.read(os.path.join(a.T['poptab'],
                          ''.join(('SSP_R',str(i),'_sh_',mode_iso,'.csv'))))  for i in a.R]
        if type(bands)==str:
            goodsh = [np.where((table[bands]>-15))[0] for table in tabsh]
        else:
            goodsh = [np.where((table[bands[0]]>-15)&(table[bands[1]]>-15))[0] for table in tabsh]
    
    if mode_geom=='edge-on':
        V = Volume(p,a)
        volume = V.none()
        volume = V.zcut(volume,zlim)
        sum_Nzd, sum_Nzt, sum_Nzsh = [], [], [] 
        for i in range(a.Rbins):
            if mode_comp=='d' or mode_comp=='dt' or mode_comp=='tot':
                sum_Nzd.append(pops_in_volume('d',a.R[i],volume,p,a,tab=tabd[i])['Nz'])
            if mode_comp=='t' or mode_comp=='dt':
                sum_Nzt.append(pops_in_volume('t',a.R[i],volume,p,a,tab=tabt[i])['Nz'])
            if mode_comp=='sh' or mode_comp=='tot':
                sum_Nzsh.append(pops_in_volume('sh',a.R[i],volume,p,a,tab=tabsh[i])['Nz'])
                
    profile = np.zeros((a.Rbins))
    if type(bands)==str:
        bands_str = bands
    else:
        bands_str, bands_str1 = bands
    for i in range(a.Rbins):
        Nz_tot, tab_tot, tab_tot1 = [], [], []
        # FACE-ON
        #--------------------------------------------------------------------------
        if mode_geom=='face-on':
            if mode_comp=='d' or mode_comp=='dt' or mode_comp=='tot':        
                Nz_tot.extend(tabd[i]['N'][goodd[i]])
                tab_tot.extend(tabd[i][bands_str][goodd[i]])
                if type(bands)!=str:
                    tab_tot1.extend(tabd[i][bands_str1][goodd[i]])
            if mode_comp=='t' or mode_comp=='dt':     
                Nz_tot.extend(tabt[i]['N'][goodt[i]])
                tab_tot.extend(tabt[i][bands_str][goodt[i]])
                if type(bands)!=str:
                    tab_tot1.extend(tabt[i][bands_str1][goodt[i]])
            if mode_comp=='sh' or mode_comp=='tot':
                Nz_tot.extend(tabsh[i]['N'][goodsh[i]])
                tab_tot.extend(tabsh[i][bands_str][goodsh[i]])  
                if type(bands)!=str:
                    tab_tot1.extend(tabsh[i][bands_str1][goodsh[i]])  
        # EDGE-ON
        #--------------------------------------------------------------------------
        else: 
            for k in range(a.Rbins-i):
                if mode_comp=='d' or mode_comp=='dt' or mode_comp=='tot':
                    Nz_tot.extend(sum_Nzd[i+k][goodd[i+k]])
                    tab_tot.extend(tabd[i+k][bands_str][goodd[i+k]]) 
                    if type(bands)!=str:
                        tab_tot1.extend(tabd[i+k][bands_str1][goodd[i+k]]) 
                if mode_comp=='t' or mode_comp=='dt':
                    Nz_tot.extend(sum_Nzt[i+k][goodt[i+k]])
                    tab_tot.extend(tabt[i+k][bands_str][goodt[i+k]]) 
                    if type(bands)!=str:
                        tab_tot1.extend(tabt[i+k][bands_str1][goodt[i+k]]) 
                if mode_comp=='sh' or mode_comp=='tot':
                    Nz_tot.extend(sum_Nzsh[i+k][goodsh[i+k]])
                    tab_tot.extend(tabsh[i+k][bands_str][goodsh[i+k]]) 
                    if type(bands)!=str:
                        tab_tot1.extend(tabsh[i+k][bands_str1][goodsh[i+k]]) 
        Nz_tot = np.array(Nz_tot)
        tab_tot = np.array(tab_tot)
        if type(bands)==str:
            # Brightness profile
            profile[i] = Magsun[bands] + 21.572 - 2.5*\
                         np.log10(np.sum(Nz_tot*Lsun[bands]*10**(0.4*(Magsun[bands]-tab_tot))))  
        else:
            # Colour profile
            tab_tot1 = np.array(tab_tot1)
            profile[i] = np.sum(Nz_tot*(tab_tot-tab_tot1))/np.sum(Nz_tot)
                                                                                       
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        ts.disk_brightness_save(profile,mode_comp,mode_geom,bands)  
        
    return profile
  
    
def rz_map(mode_comp,p,a,**kwargs):
    """
    Matter (or number) volume density map in Rz coordinates. 
    
    Parameters
    ----------
    mode_comp ('d','t','sh','dt' or 'tot'), p, a 
    **kwargs : dict, optional keyword arguments
        valid keywords : save, mode_pop, tab, number, ages, mets, 
        dz, mode_iso
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()

    Returns
    -------
    rz_grid : 2d-ndarray
        Map of matter or stellar density in Galactic coordinates.
    """
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save','mets','ages','number','dz','mode_pop',
                            'tab','mode_iso'],this_function)
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                       'RZ-density map',this_function)
    inpcheck_kwargs_compatibility(kwargs,this_function)
    
    dz = 100 # pc
    if 'dz' in kwargs:
        dz = kwargs['dz']        
        
    z_edges = np.arange(0,p.zmax,dz)
    zbins = len(z_edges)
    rz_grid = np.zeros((a.Rbins,zbins-1))
    z_coef = dz//p.dz

    for i in range(a.Rbins):
        if mode_comp=='d' or mode_comp=='t' or mode_comp=='sh':
            if 'ages' in kwargs or 'mets' in kwargs:
                if 'ages' in kwargs:
                    kwargs_calc = reduce_kwargs(kwargs,['mode_pop','number','mode_iso','tab'])
                    if 'tab' in kwargs:
                        kwargs_calc['tab'] = kwargs['tab'][i]
                    rhoz = rhoz_monoage(mode_comp,a.R[i],kwargs['ages'],p,a,between=True,**kwargs_calc)[0]
                else:
                    kwargs_calc = reduce_kwargs(kwargs,['mode_pop','number','mode_iso','tab'])
                    if 'tab' in kwargs:
                        kwargs_calc['tab'] = kwargs['tab'][i]
                    rhoz = rhoz_monomet(mode_comp,a.R[i],kwargs['mets'],p,a,**kwargs_calc)[0]
            else:
                kwargs_calc = reduce_kwargs(kwargs,['mode_pop','number','mode_iso','tab'])
                if 'tab' in kwargs:
                        kwargs_calc['tab'] = kwargs['tab'][i]
                rhoz = rhoz_monoage(mode_comp,a.R[i],[0,tp],p,a,between=True,**kwargs_calc)[0]
        else:
            if mode_comp=='dt':
                if 'ages' in kwargs or 'mets' in kwargs:
                    if 'ages' in kwargs:
                        kwargs_calc = reduce_kwargs(kwargs,['mode_pop','number','mode_iso','tab'])
                        kwargs_calcd, kwargs_calct = kwargs_calc.copy(), kwargs_calc.copy()
                        if 'tab' in kwargs:
                            kwargs_calcd['tab'] = kwargs['tab'][0][i]
                            kwargs_calct['tab'] = kwargs['tab'][1][i]
                        rhozd = rhoz_monoage('d',a.R[i],kwargs['ages'],p,a,between=True,**kwargs_calcd)[0]
                        rhozt = rhoz_monoage('t',a.R[i],kwargs['ages'],p,a,between=True,**kwargs_calct)[0]
                    else:
                        kwargs_calc = reduce_kwargs(kwargs,['mode_pop','number','mode_iso','tab'])
                        kwargs_calcd, kwargs_calct = kwargs_calc.copy(), kwargs_calc.copy()
                        if 'tab' in kwargs:
                            kwargs_calcd['tab'] = kwargs['tab'][0][i]
                            kwargs_calct['tab'] = kwargs['tab'][1][i]
                        rhozd = rhoz_monomet('d',a.R[i],kwargs['mets'],p,a,**kwargs_calcd)[0]
                        rhozt = rhoz_monomet('t',a.R[i],kwargs['mets'],p,a,**kwargs_calct)[0]
                else:
                    kwargs_calc = reduce_kwargs(kwargs,['mode_pop','number','mode_iso','tab'])
                    kwargs_calcd, kwargs_calct = kwargs_calc.copy(), kwargs_calc.copy()
                    if 'tab' in kwargs:
                            kwargs_calcd['tab'] = kwargs['tab'][0][i]
                            kwargs_calct['tab'] = kwargs['tab'][1][i]
                    rhozd = rhoz_monoage('d',a.R[i],[0,tp],p,a,between=True,**kwargs_calcd)[0]
                    rhozt = rhoz_monoage('t',a.R[i],[0,tp],p,a,between=True,**kwargs_calct)[0]
                rhozd[rhozd*0!=0] = 0 
                rhozt[rhozt*0!=0] = 0 
                rhoz = np.add(rhozd,rhozt)
                
            if mode_comp=='tot':
                if 'ages' in kwargs or 'mets' in kwargs:
                    if 'ages' in kwargs:
                        kwargs_calc = reduce_kwargs(kwargs,['mode_pop','number','mode_iso','tab'])
                        kwargs_calcd, kwargs_calct, kwargs_calcsh =\
                                kwargs_calc.copy(), kwargs_calc.copy(), kwargs_calc.copy()
                        if 'tab' in kwargs:
                            kwargs_calcd['tab'] = kwargs['tab'][0][i]
                            kwargs_calct['tab'] = kwargs['tab'][1][i]
                            kwargs_calcsh['tab'] = kwargs['tab'][2][i]
                        rhozd = rhoz_monoage('d',a.R[i],kwargs['ages'],p,a,between=True,**kwargs_calcd)[0]
                        rhozt = rhoz_monoage('t',a.R[i],kwargs['ages'],p,a,between=True,**kwargs_calct)[0]
                        rhozsh = rhoz_monoage('sh',a.R[i],kwargs['ages'],p,a,between=True,**kwargs_calcsh)[0]
                    else:
                        kwargs_calc = reduce_kwargs(kwargs,['mode_pop','number','mode_iso','tab'])
                        kwargs_calcd, kwargs_calct, kwargs_calcsh =\
                                kwargs_calc.copy(), kwargs_calc.copy(), kwargs_calc.copy()
                        if 'tab' in kwargs:
                            kwargs_calcd['tab'] = kwargs['tab'][0][i]
                            kwargs_calct['tab'] = kwargs['tab'][1][i]
                            kwargs_calcsh['tab'] = kwargs['tab'][2][i]
                        rhozd = rhoz_monomet('d',a.R[i],kwargs['mets'],p,a,**kwargs_calcd)[0]
                        rhozt = rhoz_monomet('t',a.R[i],kwargs['mets'],p,a,**kwargs_calct)[0]
                        rhozsh = rhoz_monomet('sh',a.R[i],kwargs['mets'],p,a,**kwargs_calcsh)[0]
                else:
                    kwargs_calc = reduce_kwargs(kwargs,['mode_pop','number','mode_iso','tab'])
                    kwargs_calcd, kwargs_calct, kwargs_calcsh =\
                            kwargs_calc.copy(), kwargs_calc.copy(), kwargs_calc.copy()
                    if 'tab' in kwargs:
                        kwargs_calcd['tab'] = kwargs['tab'][0][i]
                        kwargs_calct['tab'] = kwargs['tab'][1][i]
                        kwargs_calcsh['tab'] = kwargs['tab'][2][i]
                    rhozd = rhoz_monoage('d',a.R[i],[0,tp],p,a,between=True,**kwargs_calcd)[0]
                    rhozt = rhoz_monoage('t',a.R[i],[0,tp],p,a,between=True,**kwargs_calct)[0]
                    rhozsh = rhoz_monoage('sh',a.R[i],[0,tp],p,a,between=True,**kwargs_calcsh)[0]
                rhozd[rhozd*0!=0] = 0 
                rhozt[rhozt*0!=0] = 0
                rhozsh[rhozsh*0!=0] = 0
                rhoz = np.add(rhozd,np.add(rhozt,rhozsh))
                
        rz_grid[i] = [np.mean(rhoz[int(k*z_coef):int((k+1)*z_coef)]) for k in np.arange(zbins-1)]
        rhoz[rhoz==0] = np.nan
        
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        ts.rz_map_save(rz_grid,mode_comp)  
    
    return rz_grid
    

def rz_map_quantity(mode_comp,quantity,p,a,**kwargs):
    """
    Calculates mean value of some quantity in Rz-bins (use 'sigw'
    or names of stellar assemblies tables' columns). 
    
    Parameters
    ----------
    mode_comp ('d','t','sh','dt' or 'tot'), quantity, p, a
    **kwargs : dict, optional keyword arguments
        valid keywords : save, mode_pop, tab, dz, ages, mets, 
        mode_iso
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()

    Returns
    -------
    rz_grid : 2d-ndarray
        Map of chosen quantity in Galactic coordinates.
    """
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save','dz','mode_pop','tab','mode_iso',
                            'ages','mets'],this_function)
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                       'RZ-density map',this_function)
    if 'ages' in kwargs:
        kwargs['ages'] = inpcheck_age(kwargs['ages'],this_function)
    if 'dz' in kwargs:
        dz = inpcheck_dz(kwargs['dz'],p,this_function)
    else:
        dz = 100 # pc
    inpcheck_kwargs_compatibility(kwargs,this_function)
     
    z_edges = np.arange(0,p.zmax,dz)
    zbins = len(z_edges)
    rz_grid = np.zeros((a.Rbins,zbins-1))
    z_coef = dz//p.dz
                
    kwargs_calc = reduce_kwargs(kwargs,['mode_iso','mode_pop','ages','mets'])
    for i in range(a.Rbins):
        if 'tab' in kwargs:
            if mode_comp=='dt' or mode_comp=='tot':
                if mode_comp=='dt':
                    kwargs_calc['tab'] = [kwargs['tab'][0][i],kwargs['tab'][1][i]]
                else:
                    kwargs_calc['tab'] = [kwargs['tab'][0][i],kwargs['tab'][1][i],kwargs['tab'][2][i]]
            else:
                kwargs_calc['tab'] = kwargs['tab'][i]
        Q_r = mean_quantity(mode_comp,a.R[i],[0,p.zmax],quantity,p,a,**kwargs_calc)
        rz_grid[i] = [np.mean(Q_r[int(k*z_coef):int((k+1)*z_coef)]) for k in np.arange(zbins-1)]
        
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        ts.rz_map_quantity_save(rz_grid,mode_comp,quantity)  
        
    return rz_grid

 
def fw_hist(mode_comp,R,zlim,p,a,**kwargs):
    """
    |W|-velocity distribution function. 
    
    Parameters
    ----------
    mode_comp ('d','t','sh','dt' or 'tot'), R, zlim, p, a 
    **kwargs : dict, optional keyword arguments
        valid keywords : save, mode_pop, tab, number, mode_iso, 
        dw, wmax, ages, mets        
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()
        
    Returns
    -------
    fwhist : ndarray
        Normalized on area W-velocity distribution (histogram), km/s.
    wgrid : ndarray    
        Corresponding |W|-bin centers, km/s.  
    """
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save','dw','wmax','mode_pop','tab','number','mode_iso',
                            'ages','mets'],this_function)
    ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                       'f(|W|) distribution',this_function)
    inpcheck_kwargs_compatibility(kwargs,this_function)
    
    wmax = 60 
    if 'wmax' in kwargs:
        wmax = kwargs['wmax']        
    if 'dw' in kwargs:
        dw = kwargs['dw']
        del kwargs['dw']
    else:
        dw = 2 
    wgrid = np.arange(dw/2,wmax+dw/2,dw)
            
    if 'ages' in kwargs:
        i1 = (tp - kwargs['ages'][0])//tr
        i2 = (tp - kwargs['ages'][1])//tr
        indt = np.arange(i2,i1,dtype=np.int)
        kwargs['indt'] = indt
    
    if mode_comp=='d':
        fwhist = _fw_d_(R,zlim,wgrid,dw,p,a,**kwargs)
    if mode_comp=='t':
        fwhist = _fw_t_(R,zlim,wgrid,dw,p,a,**kwargs)
    if mode_comp=='sh':
        fwhist = _fw_sh_(R,zlim,wgrid,dw,p,a,**kwargs)
    if mode_comp=='dt':
        if 'tab' in kwargs:
            tabs = kwargs['tab']
            del kwargs['tab']
            fwhist_d = _fw_d_(R,zlim,wgrid,dw,p,a,**kwargs,tab=tabs[0])
            fwhist_t = _fw_t_(R,zlim,wgrid,dw,p,a,**kwargs,tab=tabs[1])
        else:
            fwhist_d = _fw_d_(R,zlim,wgrid,dw,p,a,**kwargs)
            fwhist_t = _fw_t_(R,zlim,wgrid,dw,p,a,**kwargs)
        fwhist = np.add(fwhist_d,fwhist_t)
    if mode_comp=='tot':
        if 'tab' in kwargs:
            tabs = kwargs['tab']
            del kwargs['tab']
            fwhist_d = _fw_d_(R,zlim,wgrid,dw,p,a,**kwargs,tab=tabs[0])
            fwhist_t = _fw_t_(R,zlim,wgrid,dw,p,a,**kwargs,tab=tabs[1])
            fwhist_sh = _fw_sh_(R,zlim,wgrid,dw,p,a,**kwargs,tab=tabs[2])
        else:
            fwhist_d = _fw_d_(R,zlim,wgrid,dw,p,a,**kwargs)
            fwhist_t = _fw_t_(R,zlim,wgrid,dw,p,a,**kwargs)
            fwhist_sh = _fw_sh_(R,zlim,wgrid,dw,p,a,**kwargs)
        fwhist = np.add(np.add(fwhist_d,fwhist_t),fwhist_sh)
    
    if np.sum(fwhist)!=0:
        fwhist = fwhist/np.sum(fwhist*dw)
        
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,**kwargs)
        ts.fw_save((fwhist,wgrid),mode_comp,R,zlim)  
      
    return (fwhist, wgrid)


def hess_simple(mode_comp,mode_geom,bands,mag_range,mag_step,p,a,**kwargs):
    """
    
    Parameters
    ----------
    mode_comp : TYPE
        DESCRIPTION.
    mode_geom : TYPE
        DESCRIPTION.
    bands : TYPE
        DESCRIPTION.
    mag_range : TYPE
        DESCRIPTION.
    mag_step : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    """
        
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['zlim','save','mode_pop','r_minmax','R_minmax','dphi',
                            'smooth','mode_iso'],this_function)
    inpcheck_kwargs_compatibility(kwargs,this_function)
    zlim = [0,p.zmax]
    if 'zlim' in kwargs:
        zlim = kwargs['zlim']
    mode_iso = 'Padova'
    if 'mode_iso' in kwargs:
        mode_iso = kwargs['mode_iso']
    
    V = Volume(p,a)
    if mode_geom=='local_sphere':
        if p.run_mode==0:
            volume, vln = V.local_sphere(kwargs['r_minmax'][0],kwargs['r_minmax'][1])
        else:
            volume, indrs, vln = V.local_sphere(kwargs['r_minmax'][0],kwargs['r_minmax'][1])
            
    if mode_geom=='local_cylinder':
        if p.run_mode==0:
            volume, vln = V.local_cylinder(kwargs['r_minmax'][0],kwargs['r_minmax'][1],
                                           zlim[0],zlim[1])
        else:
            volume, indrs, vln = V.local_cylinder(kwargs['r_minmax'][0],kwargs['r_minmax'][1],
                                                  zlim[0],zlim[1])
    if mode_geom=='rphiz_box':
        volume, indrs, vln = V.rphiz_box(kwargs['R_minmax'][0],kwargs['R_minmax'][1],kwargs['dphi'],
                                    zlim[0],zlim[1])
    
    if p.run_mode==0:
        if mode_comp=='d' or mode_comp=='dt' or mode_comp=='tot':
            tabd = Table.read(os.path.join(a.T['poptab'],
                              ''.join(('SSP_R',str(p.Rsun),'_d_',mode_iso,'.csv'))))
        if mode_comp=='t' or mode_comp=='dt' or mode_comp=='tot':
            tabt = Table.read(os.path.join(a.T['poptab'],
                             ''.join(('SSP_R',str(p.Rsun),'_t_',mode_iso,'.csv'))))
        if mode_comp=='sh' or mode_comp=='tot':
            tabsh = Table.read(os.path.join(a.T['poptab'],
                              ''.join(('SSP_R',str(p.Rsun),'_sh_',mode_iso,'.csv'))))
    else:
        if mode_comp=='d' or mode_comp=='dt' or mode_comp=='tot':
            tabd = [Table.read(os.path.join(a.T['poptab'],
                      ''.join(('SSP_R',str(i),'_d_',mode_iso,'.csv')))) for i in a.R[indrs]]
        if mode_comp=='t' or mode_comp=='dt' or mode_comp=='tot':
            tabt = [Table.read(os.path.join(a.T['poptab'],
                      ''.join(('SSP_R',str(i),'_t_',mode_iso,'.csv')))) for i in a.R[indrs]]
        if mode_comp=='sh' or mode_comp=='tot':
            tabsh = [Table.read(os.path.join(a.T['poptab'],
                      ''.join(('SSP_R',str(i),'_sh_',mode_iso,'.csv'))))  for i in a.R[indrs]]        

    sum_Nz, mag = [], [[], [], []]
    kwargs_calc = reduce_kwargs(kwargs,['mode_pop','mode_iso'])
    
    if p.run_mode==0:
        if mode_comp=='d' or mode_comp=='dt' or mode_comp=='tot':
            sum_Nz.extend(pops_in_volume('d',p.Rsun,volume,p,a,tab=tabd,**kwargs_calc)['Nz'])
            mag = _extend_mag_(mag,tabd,bands)
        if mode_comp=='t' or mode_comp=='dt' or mode_comp=='tot':
            sum_Nz.extend(pops_in_volume('t',p.Rsun,volume,p,a,tab=tabt,**kwargs_calc)['Nz'])
            mag = _extend_mag_(mag,tabt,bands)
        if mode_comp=='sh' or mode_comp=='tot':
            sum_Nz.extend(pops_in_volume('sh',p.Rsun,volume,p,a,tab=tabsh,**kwargs_calc)['Nz'])
            mag = _extend_mag_(mag,tabsh,bands)
    else:
        for i in range(len(indrs)):
            if mode_comp=='d' or mode_comp=='dt' or mode_comp=='tot':
                sum_Nz.extend(pops_in_volume('d',a.R[indrs[i]],volume[i],p,a,
                                             tab=tabd[i],**kwargs_calc)['Nz'])
                mag = _extend_mag_(mag,tabd[i],bands)
            if mode_comp=='t' or mode_comp=='dt' or mode_comp=='tot':
                sum_Nz.extend(pops_in_volume('t',a.R[indrs[i]],volume[i],p,a,
                                             tab=tabt[i],**kwargs_calc)['Nz'])
                mag = _extend_mag_(mag,tabt[i],bands)
            if mode_comp=='sh' or mode_comp=='tot':
                sum_Nz.extend(pops_in_volume('sh',a.R[indrs[i]],volume[i],p,a,
                                             tab=tabsh[i],**kwargs_calc)['Nz'])
                mag = _extend_mag_(mag,tabsh[i],bands)
    sum_Nz = np.array(sum_Nz)
    
    dx, dy = mag_step
    xlen = int(round(abs((mag_range[0][0]-mag_range[0][1])/mag_step[0]),0)) 
    ylen = int(round(abs((mag_range[1][0]-mag_range[1][1])/mag_step[1]),0))
    
    hess = np.zeros((xlen,ylen))
    hess = histogram2d(np.subtract(mag[1],mag[2]),mag[0],
                                weights=sum_Nz,bins=[xlen,ylen],range=mag_range)               
    if 'smooth' in kwargs:
        hess = convolve2d_gauss(hess,kwargs['smooth'],mag_range)
        
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a,vln=vln,**kwargs)
        ts.hess_save(hess,mode_comp,mode_geom,bands,mag_range,mag_step) 
    
    return hess.T


def fi_iso(ah,p,a,**kwargs):
    """
    Calculates the normalized vertical gravitational potential as 
    a function of R.
    
    Parameters
    ----------
    **kwargs : dict, optional keyword arguments
        valid keywords : save, save_format, close
    For details call help(Plotting) and from analysis
    analysis_args_description() and analysis_kwargs_description() 
    """
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save'],this_function)
    
    Phi = tab_reader(['Phi'],p,a.T)[0]
    
    sigma_r, sigma_r0 = rhor(p,a,sigma=True)
    rho_r, rho_r0 = rhor(p,a)
    FiR = np.zeros((6,a.Rbins))
    
    fi_r = RadialPotential(p.Rsun,a.R)
    
    ind_hole_g1 = np.where(np.abs(a.R-p.Rg10)==np.amin(np.abs(a.R-p.Rg10)))[0][0]
    ind_hole_g2 = np.where(np.abs(a.R-p.Rg20)==np.amin(np.abs(a.R-p.Rg20)))[0][0]
    
    FiR[0] = fi_r.exp_disk(sigma_r0[0]*np.exp(p.Rsun/p.Rd),p.Rd)
    FiR[1] = fi_r.exp_disk(sigma_r0[1]*np.exp(p.Rsun/p.Rt),p.Rt)
    FiR[2] = fi_r.exp_disk(sigma_r0[2]*np.exp(p.Rsun/p.Rg1),p.Rg1)
    FiR[3] = fi_r.exp_disk(sigma_r0[3]*np.exp(p.Rsun/p.Rg2),p.Rg2)
    # No gas in the center
    FiR[2][:ind_hole_g1] = 0 
    FiR[3][:ind_hole_g2] = 0 
    FiR[4] = fi_r.cored_iso_sphere(rho_r0[4],ah)
    FiR[5] = fi_r.pow_law(rho_r0[5],-p.a_in)
    FiR_tot = np.sum(FiR,axis=0)
    
    Phi[1:] = [Phi[i+1] + FiR_tot[i] for i in np.arange(a.Rbins)]
    
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a)
        ts.fi_iso_save(Phi) 
    
    return Phi


def rot_curve(ah,p,a,**kwargs):
    """
    Calculatess the circular velocity Vc as a function of R.
    
    Parameters
    ----------
    **kwargs : dict, optional keyword arguments
        valid keywords : save, R  
    For more details call (first from analysis import ...)
    analysis_args_description() and analysis_kwargs_description()

    """
    
    this_function = inspect.stack()[0][3]
    inpcheck_kwargs(kwargs,['save', 'R'],this_function)

    Hdh0,Hsh0 = tab_reader(['Hdh0','Hsh0'],p,a.T)
    rhodh0 = p.sigmadh/2/Hdh0[1]
    rhosh0 = p.sigmash/2/Hsh0[1]
        
    R = np.linspace(0.01,p.Rmax,1000)
    if 'R' in kwargs: 
        R = np.array(kwargs['R'])
    
    RC = RotCurve(p.Rsun,R)
    vcd = RC.vc_disk(p.sigmad,p.Rd,0)
    vct = RC.vc_disk(p.sigmat,p.Rt,0)
    vcg1 = RC.vc_disk(p.sigmag1,p.Rg1,p.Rg10)
    vcg2 = RC.vc_disk(p.sigmag2,p.Rg2,p.Rg20)
    vcb = RC.vc_bulge(p.Mb)
    vcdh = RC.vc_halo_cored_iso_sphere(rhodh0,ah)
    #vcsh = RC.vc_halo_power_law(rhosh0,-p.a_in)
    
    vc = RC.vc_tot([vcb,vcd,vct,vcg1,vcg2,vcdh])
    vc0 = RC.vc0(vc)
    
    if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
        ts = TabSaver(p,a)
        ts.rot_curve_save(np.stack((R,vc,vcb,vcd,vct,vcg1,vcg2,vcdh),axis=-1)) 

    return {'r':R,'d':vcd,'t':vct,'g1':vcg1,'g2':vcg2,'b':vcb,'dh':vcdh,
            'tot':vc,'vc0':vc0}
    




