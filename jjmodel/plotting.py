"""
Created on Mon Apr 25 19:10:51 2016

@author: Skevja
"""

import os
import sys
import inspect
import warnings
import numpy as np 
from scipy.ndimage import zoom
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from .iof import tab_sorter, tab_reader, TabSaver
from .constants import KM, SIGMA_E, tp
from .tools import Timer
from .control import (inpcheck_radius,inpcheck_age,inpcheck_dz,inpcheck_height,inpcheck_iskwargtype,
                     inpcheck_kwargs,inpcheck_mode_comp,reduce_kwargs,inpcheck_kwargs_compatibility)
from .analysis import (rhoz_monoage,rhoz_monomet,metz,metr,rhor_monoage,rhor_monomet,agehist,
                       methist,hr_monoage,hr_monomet,sigwz,sigwr,sigwr_monoage,sigwr_monomet,rz_map, 
                       rz_map_quantity,agez,ager,rhor,fw_hist,disk_brightness,hess_simple,fi_iso, 
                       rot_curve)                       



class Plotting():
    """
    Class for plotting. 
    All methods of this class return figure and axes (f, ax).        
    """
    
    def __init__(self,p,a,inp):
        """Class instance is initialized by three basic parameters. 
        
        :param p: Set of model parameters from the parameter file. 
        :type p: namedtuple
        :param a: Collection of the fixed model parameters, useful quantities, and arrays.
        :type a: namedtuple
        :param inp: Collection of the input functions including SFR, AVR, AMR, and IMF.
        :type inp: dict  
        """
        
        self.p, self.a, self.inp = p, a, inp
        self.name = ['d','g1','g2','t','dh','sh']
        self.labs = {'d':r'$\mathrm{Thin \ disk}$', 't':r'$\mathrm{Thick \ disk}$',
                     'g1':r'$\mathrm{Molecular \ gas}$', 'g2':r'$\mathrm{Atomic \ gas}$',
                     'dh':r'$\mathrm{DM \ halo}$', 'sh':r'$\mathrm{Stellar \ halo}$',
                     'r0':'$\mathrm{R_\odot=}$' + str(self.p.Rsun) + '$\mathrm{\ kpc}$',
                     'dt':r'$\mathrm{Total \ disk}$', 'tot':r'$\mathrm{Total \ disk + halo}$',
                     'b':r'$\mathrm{Bulge}$'}
        self.pops = {'rc':'$\mathrm{RC \ stars}$', 'rc+':'$\mathrm{RC+HGB \ stars}$',
                     'a':'$\mathrm{A \ stars}$', 'f':'$\mathrm{F \ stars}$',
                     'gdw':'$\mathrm{G \ dwarfs}$', 'kdw':'$\mathrm{K \ dwarfs}$',                     
                     'mdw':'$\mathrm{M \ dwarfs}$','ceph':'$\mathrm{Cepheids \ Type \ I}$'}
        self.popcols = {'rc':'r', 'rc+':'tomato',
                        'a':'cyan', 'f':'steelblue',
                        'gdw':'orange', 'kdw':'magenta','ceph':'indigo'                   
                        }
        self.cols = {'d':'dodgerblue','g1':'cyan','g2':'lawngreen','t':'orange',
                    'dh':'darkmagenta','sh':'r','dt':'violet','tot':'k','b':'y'}
        self.axts = {'r':'$\mathrm{R, \ kpc}$', 't':'$\mathrm{t, \ Gyr}$',                          
                     'tau':r'$\mathrm{\tau, \ Gyr}$', 'num':r'$\mathrm{N, \ pc^{-2}}$',
                     'sigma':r'$\mathrm{\Sigma, \ M_\odot \ pc^{-2}}$',
                     'z':'$\mathrm{|z|, \ pc}$', 'zkpc':'$\mathrm{|z|, \ kpc}$',
                     'h':'$\mathrm{h, \ pc}$', 'nsfr':'$\mathrm{SFR/<SFR>}$',
                     'rho':r'$\mathrm{\rho, \ M_\odot \ pc^{-3}}$',
                     'sigw':r'$\mathrm{\sigma_W, \ km \ s^{-1}}$', 'fe':'$\mathrm{[Fe/H]}$',
                     'kz':'$\mathrm{K_{|z|}, \ km^2 \ s^{-2} \ kpc^{-1}}$',
                     'n':'$\mathrm{N, \ pc^{-3}}$'}
        self.xt = {'ticks':np.arange(tp+1),
                   'labels':['0','','2','','4','','6','','8','','10','','12',''] }
        self.lw = {'main':2,'secondary':1}
        self.fnt = {'main':14,'secondary':11}
        self.cmaps = {'r':mpl.cm.jet_r,'z':mpl.cm.jet,'tau':mpl.cm.gnuplot,'feh':mpl.cm.gnuplot2_r,
                      'dens':mpl.cm.magma,'viridis':mpl.cm.viridis}
        self.cmapnames = {'r':'jet_r','z':'jet','tau':'gnuplot','feh':'gnuplot2_r',
                          'dens':'magma','viridis':'viridis'}
        self.cbar_settings = {'FeH':[0.2,[-2.5,1],self.axts['fe']],
                              'sigw':[10,[0,100],self.axts['sigw']],
                              'logg':[1,[-10,10],'$\mathrm{logg}$'],
                              'logL':[1,[-10,10],'$\mathrm{logL}$'],
                              'logT':[1,[0,5],'$\mathrm{logT}$'],
                              'Mf':[1,[0,100],'$\mathrm{M_f, \ M_\odot}$'],
                              'Mini':[1,[0,100],'$\mathrm{M_{ini}, \ M_\odot}$'],                   
                              'U':[2,[-10,20],'$\mathrm{U, \ mag}$'],
                              'B':[2,[-10,20],'$\mathrm{B, \ mag}$'],
                              'V':[2,[-10,20],'$\mathrm{V, \ mag}$'],
                              'R':[2,[-10,20],'$\mathrm{R, \ mag}$'],
                              'I':[2,[-10,20],'$\mathrm{I, \ mag}$'],
                              'K':[2,[-10,20],'$\mathrm{K, \ mag}$'],
                              'H':[2,[-10,20],'$\mathrm{H, \ mag}$'],
                              'J':[2,[-10,20],'$\mathrm{J, \ mag}$'],
                              'G_DR2':[2,[-10,20],'$\mathrm{G\_DR2, \ mag}$'],
                              'GBPbr_DR2':[2,[-10,20],'$\mathrm{GBPbr\_DR2, \ mag}$'],
                              'GBPft_DR2':[2,[-10,20],'$\mathrm{GBPft\_DR2, \ mag}$'],
                              'GRP_DR2':[2,[-10,20],'$\mathrm{GRP\_DR2, \ mag}$'],
                              'G_EDR3':[2,[-10,20],'$\mathrm{G\_EDR3, \ mag}$'],
                              'GBP_EDR3':[2,[-10,20],'$\mathrm{GBP\_EDR3, \ mag}$'],
                              'GRP_EDR3':[2,[-10,20],'$\mathrm{GRP\_EDR3, \ mag}$']}                             
        label_size = 12
        mpl.rcParams['xtick.labelsize'] = label_size 
        mpl.rcParams['ytick.labelsize'] = label_size 
    
    
    def _figclose_(self,this_function,**kwargs):
        """
        Closes figure. 
        """
        if inpcheck_iskwargtype(kwargs,'close',True,bool,this_function):
            plt.close()
            

    def _figformat_(self,**kwargs):
        """
        Returns figure format (defualt png or other if specified in kwargs).
        """
        if 'save_format' in kwargs:
            fmt = kwargs['save_format']
        else:
            fmt = 'png'
        return fmt
    

    def _figsave_(self,figname,this_function,**kwargs):
        """
        Saves figure with the name figname.
        """
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            plt.savefig(figname)
            
        
    def _figcolorbar_(self,default,nbins_default,**kwargs):
        """ 
        Creates colorbar (continious or binned).
        """
        if 'cbar' in kwargs:
            if 'cbar_bins' in kwargs and kwargs['cbar_bins']==True:
                colorbar = mpl.cm.get_cmap(kwargs['cbar'],int(nbins_default))
            else:
                colorbar = kwargs['cbar']
        else:
            if 'cbar_bins' in kwargs and kwargs['cbar_bins']==True:
                colorbar = mpl.cm.get_cmap(self.cmapnames[default],int(nbins_default))
            else: 
                colorbar = self.cmaps[default]
        return colorbar 
    
    
    def _cbarminmax_(self,cmin,cmax,dc,**kwargs):
        """ 
        Calculates correct min and max values for the colorbar 
        (depend on whether the colorbar is continious or binned).
        """
        if 'cbar_bins' in kwargs and kwargs['cbar_bins']==True:
            cmin, cmax = cmin - dc/2, cmax + dc/2  
        return (cmin, cmax)
    
  
    def _dynamic_ylim_log_(self,array,yminmax_default):
        """ 
        Finds the range to be displayed on the plot axis for the data in array (log scale). 
        Array has multiple columns. 
        """
        nx, ny = array.shape
        ymind, ymaxd = yminmax_default
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",category=RuntimeWarning)
                ar_mins = [np.nanmin(array[i]) for i in np.arange(nx)]
                ar_maxs = [np.nanmax(array[i]) for i in np.arange(nx)]
                ar_mins_clean, ar_maxs_clean = [], [] 
                for i in range(nx):
                    if ar_mins[i]!=0 and ar_mins[i]*0==0:
                        ar_mins_clean.append(ar_mins[i])
                    if ar_maxs[i]!=0 and ar_maxs[i]*0==0:
                        ar_maxs_clean.append(ar_maxs[i])
            minmin = np.amin(ar_mins_clean)
            maxmax = np.amax(ar_maxs_clean)
            ymin = 10**np.int(np.log10(minmin)//1)
            ymax = 10**np.int(np.log10(maxmax)//1 + 1)
            if ymin < ymind: 
                ymin = ymind
        except: 
            ymin, ymax = ymind, ymaxd
        return (ymin,ymax)
    
    
    def _dynamic_ylim_lin_(self,array,yminmax_default,factor):
        """ 
        Finds the range to be displayed on the plot axis for the data in array (linear scale). 
        Array has multiple columns. 
        """
        nx, ny = array.shape
        ymind, ymaxd = yminmax_default
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",category=RuntimeWarning)
                ar_mins = [np.nanmin(array[i]) for i in np.arange(nx)]
                ar_maxs = [np.nanmax(array[i]) for i in np.arange(nx)]
                ar_mins_clean, ar_maxs_clean = [], [] 
                for i in range(nx):
                    if ar_mins[i]*0==0:
                        ar_mins_clean.append(ar_mins[i])
                    if ar_maxs[i]*0==0:
                        ar_maxs_clean.append(ar_maxs[i])
            minmin = np.amin(ar_mins_clean)
            maxmax = np.amax(ar_maxs_clean)
            ymin = factor*(minmin//factor)
            ymax = (factor*(maxmax//factor) + factor)
            if ymin < ymind: 
                ymin = ymind
            if ymax > ymaxd:
                ymax = ymaxd
        except: 
            ymin, ymax = ymind, ymaxd
        return (ymin,ymax)
    
    def _dynamic_ylim_1d_lin_(self,array,yminmax_default,factor):
        """ 
        Finds the range to be displayed on the plot axis for the data in array (linear scale). 
        Array has one column. 
        """
        ymind, ymaxd = yminmax_default
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",category=RuntimeWarning)
                mins = np.nanmin(array) 
                maxs = np.nanmax(array) 
            ymin = factor*(mins//factor) 
            ymax = (factor*(maxs//factor) + factor)
            if ymin < ymind: 
                ymin = ymind
            if ymax > ymaxd:
                ymax = ymaxd
        except: 
            ymin, ymax = ymind, ymaxd
        return (ymin,ymax)
    
    
    def rhor_plt(self,**kwargs):
        """
        Radial density profiles of the Galactic components 
        (thin and thick disk, molecular and atomic gas, DM and stellar halo). 
        
        :param sigma: Optional. If True, surface density profiles (:math:`\mathrm{M_\odot \ pc^{-2}}`) will be plotted. 
            By default, the plot displayes midplane densities (:math:`\mathrm{M_\odot \ pc^{-3}}`). 
        :type sigma: boolean
        :param save: Optional. If True, the figure will be saved (call ``a.T['inpplt']`` 
            to display the folder name).
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean
        
        :return: Figure and axes. 
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','sigma'],this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function)
        kwargs_calc = reduce_kwargs(kwargs,['sigma'])
        format_ = self._figformat_(**kwargs)
        
        rho_r, rho_r0 = rhor(self.p,self.a,**kwargs_calc)

        round_eval = "%.1e"
        if inpcheck_iskwargtype(kwargs,'sigma',True,bool,this_function):
            round_eval = "%.2e"
            ymin,ymax = self._dynamic_ylim_log_(rho_r,[0.1,350])
            ylabel = self.axts['sigma']
            figname = os.path.join(self.a.T['inpplt'],''.join(('SigmaR.',format_)))
        else:
            ymin,ymax = self._dynamic_ylim_log_(rho_r,[1e-5,1e2])
            ylabel = self.axts['rho']
            figname = os.path.join(self.a.T['inpplt'],''.join(('RhoR.',format_)))
        
        f,ax = plt.subplots(figsize=(9,7))
        for i in range(6):
            ax.semilogy(self.a.R,rho_r[i],label=self.labs[self.name[i]],
                         color=self.cols[self.name[i]],lw=self.lw['main'])      
            ax.semilogy(self.p.Rsun,rho_r0[i],
                         color=self.cols[self.name[i]],ls='none',marker='o',markersize=8)
            ax.text(self.p.Rsun+0.1,1.01*rho_r0[i],str(eval(round_eval % (rho_r0[i]))),
                    fontsize=9,c='grey')
        ax.semilogy([self.p.Rsun,self.p.Rsun],[ymin,ymax],ls='--',lw=self.lw['secondary'],
                     c='darkgrey',label=self.labs['r0'])
        ax.set_xlabel(self.axts['r'],fontsize=self.fnt['main'])
        ax.set_ylabel(ylabel,fontsize=self.fnt['main'])
        plt.legend(prop={'size':self.fnt['secondary']},loc=1,ncol=2)
        ax.set_xlim(self.p.Rmin,self.p.Rmax)
        ax.set_ylim(ymin,ymax)
                
        self._figsave_(figname,this_function,**kwargs)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax) 
        
    
    def nsfr_plt(self,mode_comp,**kwargs):
        """
        Normalized star formation rate (NSFR) as a function of Galactocentric distance. 
                
        :param mode_comp: Galactic component. Can be ``'d'`` (thin disk), ``'t'`` (thick disk), or 
            ``'dt'`` (thin + thick disk).
        :type mode_comp: str
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :type cbar_bins: boolean
        :param save: Optional. If True, the figure will be saved (call ``a.T['inpplt']`` 
            to display the folder name).
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean
        
        :return: Figure and axes.
        """
        
        this_function = inspect.stack()[0][3]
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','dt'],'SFR',this_function)
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins'],this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
        if (mode_comp=='t' and (('cbar' in kwargs) or 
                                ('cbar_bins' in kwargs and kwargs['cbar_bins']==True))):
            print(this_function + ": Unnecessary input. Keywords 'cbar' and 'cbar_bins'"+\
                                  " don't work with mode_comp='t'.")
                  
        f, ax = plt.subplots(figsize=(10,7))
        ax.set_xticks(self.xt['ticks'])
        ax.set_xticklabels(self.xt['labels'])
        ax.set_xlabel(self.axts['t'],fontsize=self.fnt['main'],labelpad=15)
        ax.set_ylabel(self.axts['nsfr'],fontsize=self.fnt['main'],labelpad=15)
        axx = ax.twiny()
        axx.set_xticks(self.xt['ticks'])
        axx.set_xticklabels(self.xt['labels'][::-1])
        axx.set_xlabel(self.axts['tau'],fontsize=self.fnt['main'],labelpad=15)
        ax.text(0.82,0.85,ln,transform=ax.transAxes,fontsize=self.fnt['main'])
        f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.86)
        if mode_comp!='t':
            colorbar = self._figcolorbar_('r',self.a.Rbins,**kwargs)
            cmin, cmax = self._cbarminmax_(self.p.Rmin,self.p.Rmax,self.p.dR,**kwargs)
            if mode_comp=='d':
                NSFR, NSFR0 = tab_reader(['NSFRd','NSFRd0'],self.p,self.a.T)
            else:
                NSFR, NSFR0 = tab_reader(['NSFRtot','NSFRtot0'],self.p,self.a.T)
            ymax = round(np.amax(NSFR[1]),1) + 0.2
            line_segments = LineCollection([list(zip(NSFR[0], NSFR[i+1])) for i in self.a.R_array],
                                           linewidths=self.lw['main'],cmap = colorbar,
                                           norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)                                                                       
                                           )                                      
            line_segments.set_array(self.a.R)
            im = ax.add_collection(line_segments)
            ax.plot(NSFR0[0],NSFR0[1],ls='--',color='k',lw=self.lw['secondary'],label=self.labs['r0'])
            ax.legend(loc=1,prop={'size':self.fnt['secondary']})
            pos = ax.get_position()
            cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
            cbar = f.colorbar(im,cax=cax,orientation='vertical')
            cbar.set_ticks(np.arange(self.a.R[0],self.a.R[-1]+1))
            cbar.set_label(self.axts['r'],fontsize=self.fnt['main'])
        else:
            NSFR = tab_reader(['NSFRt'],self.p,self.a.T)[0]
            ymax = round(np.amax(NSFR[1]),2) + 0.01
            ax.plot(NSFR[0],NSFR[1],lw=self.lw['main'],c=self.cols['t'])
        ax.set_xlim((0,tp))
        ax.set_ylim((0,ymax))
        
        format_ = self._figformat_(**kwargs)
        figname = os.path.join(self.a.T['inpplt'],''.join(('NSFR_',mode_comp,'.',str(format_))))
        self._figsave_(figname,this_function,**kwargs)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)        
    

    def nsfr_rsun_plt(self,mode_comp,**kwargs):
        """
        Normalized star formation rate (NSFR) for the Solar neighbourhood, ``p.Rsun``.
        
        :param mode_comp: Galactic component. Can be ``'d'`` (thin disk), ``'t'`` (thick disk), or 
            ``'dt'`` (thin + thick disk).
        :type mode_comp: str
        :param save: Optional. If True, the figure will be saved (call ``a.T['inpplt']`` 
            to display the folder name).
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean
        
        :return: Figure and axes.
        """
        
        this_function = inspect.stack()[0][3]
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','dt'],'SFR',this_function)
        inpcheck_kwargs(kwargs,['save','save_format','close'],this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
                  
        f, ax = plt.subplots(figsize=(10,7))
        ax.set_xticks(self.xt['ticks'])
        ax.set_xticklabels(self.xt['labels'])
        ax.set_xlabel(self.axts['t'],fontsize=self.fnt['main'],labelpad=15)
        ax.set_ylabel(self.axts['nsfr'],fontsize=self.fnt['main'],labelpad=15)
        axx = ax.twiny()
        axx.set_xticks(self.xt['ticks'])
        axx.set_xticklabels(self.xt['labels'][::-1])
        axx.set_xlabel(self.axts['tau'],fontsize=self.fnt['main'],labelpad=15)
        f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.91)
        if mode_comp!='t':
            if mode_comp=='d':
                NSFRd0 = tab_reader(['NSFRd0'],self.p,self.a.T)[0]
                ymax = round(np.amax(NSFRd0[1]),1) + 0.2
                ax.plot(NSFRd0[0],NSFRd0[1],color=self.cols['d'],lw=self.lw['main'])
            else:
                NSFRtot0,SFRt0,SFRd0 = tab_reader(['NSFRtot0','SFRt0','SFRd0'],self.p,self.a.T)
                SFRdt_mean = np.mean(np.concatenate((SFRt0[1] + SFRd0[1][:self.a.jt],
                                                     SFRd0[1][self.a.jt:]),axis=0))
                NSFRd0 = SFRd0[1]/SFRdt_mean
                NSFRt0 = SFRt0[1]/SFRdt_mean
                ymax = round(np.amax(NSFRtot0[1]),1) + 0.2
                ax.plot(SFRd0[0],NSFRd0,color=self.cols['d'],lw=self.lw['main'],
                        label=self.labs['d'])
                ax.plot(SFRt0[0],NSFRt0,color=self.cols['t'],lw=self.lw['main'],
                        label=self.labs['t'])
                ax.plot(NSFRtot0[0],NSFRtot0[1],color=self.cols['dt'],lw=self.lw['main'],
                        label=self.labs['dt'],ls='--')
                ax.legend(loc=1,prop={'size':self.fnt['secondary']})
        else:
            NSFR = tab_reader(['NSFRt'],self.p,self.a.T)[0]
            ymax = round(np.amax(NSFR[1]),2) + 0.01
            ax.plot(NSFR[0],NSFR[1],lw=self.lw['main'],c=self.cols['t'])
        ax.set_xlim((0,tp))
        ax.set_ylim((0,ymax))
        
        format_ = self._figformat_(**kwargs)
        figname = os.path.join(self.a.T['inpplt'],''.join(('NSFR_R',str(self.p.Rsun),'_',
                                                           mode_comp,'.',str(format_))))
        self._figsave_(figname,this_function,**kwargs)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)        

    
    def amrr_plt(self,mode_comp,**kwargs):
        """
        Age-metallicity relation (AMR) of the disk as a function of Galactocentric distance.
        
        :param mode_comp: Galactic component. Can be ``'d'`` (thin disk), ``'t'`` (thick disk), or 
            ``'dt'`` (thin + thick disk).
        :type mode_comp: str
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :type cbar_bins: boolean
        :param save: Optional. If True, the figure will be saved (call ``a.T['inpplt']`` 
            to display the folder name).
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean
        
        :return: Figure and axes.
        """
        
        this_function = inspect.stack()[0][3]
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','dt'],'AMR',this_function,merged=False)
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins'],this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
        if (mode_comp=='t' and (('cbar' in kwargs) or 
                                ('cbar_bins' in kwargs and kwargs['cbar_bins']==True))):
            print(this_function + ": Unnecessary input. Keywords 'cbar' and 'cbar_bins'"+\
                                  " don't work with mode_comp='t'.")
        
        f, ax = plt.subplots(figsize=(10,7))
        ax.set_xlim((0,tp))
        ax.set_xticks(self.xt['ticks'])
        ax.set_xticklabels(self.xt['labels'])
        ax.set_xlabel(self.axts['t'],fontsize=self.fnt['main'],labelpad=15)
        ax.set_ylabel(self.axts['fe'],fontsize=self.fnt['main'],labelpad=15)
        axx = ax.twiny()
        axx.set_xticks(self.xt['ticks'])
        axx.set_xticklabels(self.xt['labels'][::-1])
        axx.set_xlabel(self.axts['tau'],fontsize=self.fnt['main'],labelpad=15)
        f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.86)
        if mode_comp!='t':
            colorbar = self._figcolorbar_('r',self.a.Rbins,**kwargs)
            cmin, cmax = self._cbarminmax_(self.p.Rmin,self.p.Rmax,self.p.dR,**kwargs)
            ax.text(0.76,0.18,ln,transform=ax.transAxes,fontsize=self.fnt['main'])
            if mode_comp=='d':
                AMR, AMR0 = tab_reader(['AMRd','AMRd0'],self.p,self.a.T)
            else:
                AMR, AMR0 = tab_reader(['AMRd','AMRd0'],self.p,self.a.T)
                AMRt = tab_reader(['AMRt'],self.p,self.a.T)[0]
            line_segments = LineCollection([list(zip(AMR[0], AMR[i+1])) for i in self.a.R_array],
                                           linewidths=self.lw['main'],cmap = colorbar,
                                           norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)                                                                       
                                           )                                      
            line_segments.set_array(self.a.R)
            im = ax.add_collection(line_segments)
            ax.plot(AMR0[0],AMR0[1],ls='--',color='k',lw=self.lw['secondary'],label=self.labs['r0'])
            pos = ax.get_position()
            cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
            cbar = f.colorbar(im,cax=cax,orientation='vertical')
            cbar.set_ticks(np.arange(self.a.R[0],self.a.R[-1]+1))
            cbar.set_label(self.axts['r'],fontsize=self.fnt['main'])
            if mode_comp=='dt':
                ax.plot(AMRt[0][:self.a.jt],AMRt[1][:self.a.jt],lw=self.lw['main'],
                        c='k',label=self.labs['t'])
        else:
            AMR = tab_reader(['AMRt'],self.p,self.a.T)[0]
            ax.plot(AMR[0][:self.a.jt],AMR[1][:self.a.jt],lw=self.lw['main'],
                    c='k',label=self.labs['t'])
        ymin, ymax = self._dynamic_ylim_lin_(AMR[1:],[-1.5,1.0],0.2)
        ax.set_ylim((ymin,ymax))
        ax.legend(loc=4,prop={'size':self.fnt['secondary']})
        
        format_ = self._figformat_(**kwargs)
        figname = os.path.join(self.a.T['inpplt'],''.join(('AMR_',mode_comp,'.',str(format_))))
        self._figsave_(figname,this_function,**kwargs)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)     
    

    def amr_rsun_plt(self,mode_comp,**kwargs):
        """
        Age-metallicity relation (AMR) for the Solar neighbourhood, ``p.Rsun``.
        
        :param mode_comp: Galactic component. Can be ``'d'`` (thin disk), ``'t'`` (thick disk), or 
            ``'dt'`` (thin + thick disk).
        :type mode_comp: str
        :param save: Optional. If True, the figure will be saved (call ``a.T['inpplt']`` 
            to display the folder name).
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean
        
        :return: Figure and axes.       
        """
        
        this_function = inspect.stack()[0][3]
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','dt'],'AMR',this_function,merged=False)
        inpcheck_kwargs(kwargs,['save','save_format','close'],this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
        f, ax = plt.subplots(figsize=(10,7))
        ax.set_xlim((0,tp))
        ax.set_xticks(self.xt['ticks'])
        ax.set_xticklabels(self.xt['labels'])
        ax.set_xlabel(self.axts['t'],fontsize=self.fnt['main'],labelpad=15)
        ax.set_ylabel(self.axts['fe'],fontsize=self.fnt['main'],labelpad=15)
        axx = ax.twiny()
        axx.set_xticks(self.xt['ticks'])
        axx.set_xticklabels(self.xt['labels'][::-1])
        axx.set_xlabel(self.axts['tau'],fontsize=self.fnt['main'],labelpad=15)
        f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.91)
        if mode_comp!='t':            
            AMR = tab_reader(['AMRd0'],self.p,self.a.T)[0]
            ax.plot(AMR[0],AMR[1],color=self.cols['d'],lw=self.lw['main'],label=self.labs['d'])
            ymin, ymax = self._dynamic_ylim_lin_(AMR[1].reshape(len(AMR[1]),1),[-1.5,1.0],0.2)
            if mode_comp=='dt':
                AMRt0 = tab_reader(['AMRt'],self.p,self.a.T)[0]
                ax.plot(AMRt0[0],AMRt0[1],color=self.cols['t'],
                        lw=self.lw['main'],label=self.labs['t'])
                ax.legend(loc=4,prop={'size':self.fnt['secondary']})
                AMRt0_long = np.concatenate((AMRt0[1],np.linspace(AMRt0[1][-1],
                                             AMRt0[1][-1],self.a.jd-self.a.jt)),axis=0)
                ymin, ymax = self._dynamic_ylim_lin_(np.stack((AMR[1],AMRt0_long),axis=-1),
                                                     [-1.5,1.0],0.2)
        else:
            AMR = tab_reader(['AMRt'],self.p,self.a.T)[0]
            ax.plot(AMR[0],AMR[1],lw=self.lw['main'],
                    c=self.cols['t'],label=self.labs['t'])
            ymin, ymax = self._dynamic_ylim_lin_(AMR[1].reshape(len(AMR[1]),1),[-1.5,1.0],0.2)
        ax.set_ylim((ymin,ymax))
        
        format_ = self._figformat_(**kwargs)
        figname = os.path.join(self.a.T['inpplt'],''.join(('AMR_R',str(self.p.Rsun),'_',
                                                           mode_comp,'.',str(format_))))
        self._figsave_(figname,this_function,**kwargs)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)     


    def gr_plt(self,mode_comp,**kwargs):
        """
        Mass loss as a function of time and Galactocentric distance.
        
        :param mode_comp: Galactic component. Can be ``'d'`` (thin disk), ``'t'`` (thick disk), or 
            ``'dt'`` (thin + thick disk).
        :type mode_comp: str
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :type cbar_bins: boolean
        :param save: Optional. If True, the figure will be saved (call ``a.T['inpplt']`` 
            to display the folder name).
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean
        
        :return: Figure and axes. 
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','dt'],
                                           'mass loss',this_function,merged=False)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
        if (mode_comp=='t' and (('cbar' in kwargs) or 
                                ('cbar_bins' in kwargs and kwargs['cbar_bins']==True))):
            print(this_function + ": Unnecessary input. Keywords 'cbar' and 'cbar_bins'"+\
                                  " don't work with mode_comp='t'.")
                
        f, ax = plt.subplots(figsize=(10,7))
        ax.set_xlim((0,tp))
        ax.set_ylim((0.6,1))
        ax.set_xticks(self.xt['ticks'])
        ax.set_xticklabels(self.xt['labels'])
        ax.set_xlabel(self.axts['t'],fontsize=self.fnt['main'],labelpad=15)
        ax.set_ylabel('$\mathrm{Mass \ fraction \ in \ stars \ and \ remnants}$',
                      fontsize=self.fnt['main'],labelpad=15)
        axx = ax.twiny()
        axx.set_xticks(self.xt['ticks'])
        axx.set_xticklabels(self.xt['labels'][::-1])
        axx.set_xlabel(self.axts['tau'],fontsize=self.fnt['main'],labelpad=15)
        f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.86)
        if mode_comp!='t':
            colorbar = self._figcolorbar_('r',self.a.Rbins,**kwargs)
            cmin, cmax = self._cbarminmax_(self.p.Rmin,self.p.Rmax,self.p.dR,**kwargs)
            ax.text(0.4,0.9,ln,transform=ax.transAxes,fontsize=self.fnt['main'])
            if mode_comp=='d':
                g, g0 = tab_reader(['gd','gd0'],self.p,self.a.T)
            else:
                g, g0, gt = tab_reader(['gd','gd0','gt'],self.p,self.a.T)
            line_segments = LineCollection([list(zip(g[0], g[i+1])) for i in self.a.R_array],
                                           linewidths=self.lw['main'],cmap = colorbar,
                                           norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)                                                                       
                                           )                                      
            line_segments.set_array(self.a.R)
            im = ax.add_collection(line_segments)
            ax.plot(g0[0],g0[1],ls='--',color='k',lw=self.lw['secondary'],label=self.labs['r0'])
            pos = ax.get_position()
            cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
            cbar = f.colorbar(im,cax=cax,orientation='vertical')
            cbar.set_ticks(np.arange(self.a.R[0],self.a.R[-1]+1))
            cbar.set_label(self.axts['r'],fontsize=self.fnt['main'])
            if mode_comp!='d':
                ax.plot(gt[0],gt[1],lw=self.lw['main'],c='k',label=self.labs['t'])
        else:
            gt = tab_reader(['gt'],self.p,self.a.T)[0]
            ax.plot(gt[0],gt[1],lw=self.lw['main'],c='k',label=self.labs['t'])
        ax.legend(loc=2,prop={'size':self.fnt['secondary']})
        
        format_ = self._figformat_(**kwargs)
        figname = os.path.join(self.a.T['inpplt'],''.join(('g_',mode_comp,'.',str(format_))))
        self._figsave_(figname,this_function,**kwargs)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)         


    def g_rsun_plt(self,mode_comp,**kwargs):
        """
        Plots mass loss function for the Solar neighbourhood, ``p.Rsun``.
        
        :param mode_comp: Galactic component. Can be ``'d'`` (thin disk), ``'t'`` (thick disk), or 
            ``'dt'`` (thin + thick disk).
        :type mode_comp: str
        :param save: Optional. If True, the figure will be saved (call ``a.T['inpplt']`` 
            to display the folder name).
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean
        
        :return: Figure and axes.  
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','dt'],
                                           'mass loss',this_function,merged=False)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
                        
        f, ax = plt.subplots(figsize=(10,7))
        ax.set_xlim((0,tp))
        ax.set_ylim((0.6,1))
        ax.set_xticks(self.xt['ticks'])
        ax.set_xticklabels(self.xt['labels'])
        ax.set_xlabel(self.axts['t'],fontsize=self.fnt['main'],labelpad=15)
        ax.set_ylabel('$\mathrm{Mass \ fraction \ in \ stars \ and \ remnants}$',
                      fontsize=self.fnt['main'],labelpad=15)
        axx = ax.twiny()
        axx.set_xticks(self.xt['ticks'])
        axx.set_xticklabels(self.xt['labels'][::-1])
        axx.set_xlabel(self.axts['tau'],fontsize=self.fnt['main'],labelpad=15)
        f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.91)
        if mode_comp!='t':
            gd0 = tab_reader(['gd0'],self.p,self.a.T)[0]
            ax.plot(gd0[0],gd0[1],color=self.cols['d'],lw=self.lw['main'],label=self.labs['d'])
            if mode_comp=='dt':
                gt = tab_reader(['gt'],self.p,self.a.T)[0]
                ax.plot(gt[0],gt[1],color=self.cols['t'],lw=self.lw['main'],label=self.labs['t'])
                ax.legend(loc=2,prop={'size':self.fnt['secondary']})
        else:
            gt = tab_reader(['gt'],self.p,self.a.T)[0]
            ax.plot(gt[0],gt[1],color=self.cols['t'],lw=self.lw['main'],label=self.labs['t'])
        
        format_ = self._figformat_(**kwargs)
        figname = os.path.join(self.a.T['inpplt'],''.join(('g_R',str(self.p.Rsun),'_',
                                                           mode_comp,'.',str(format_))))
        self._figsave_(figname,this_function,**kwargs)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)         


    def rhoz_plt(self,R,**kwargs):
        """
        Vertical density profiles of the Galactic components (thin and thick disk, 
        molecular and atomic gas, DM and stellar halo) at some radius.
        
        :param R: Galactocentric distance, kpc.
        :type R: scalar
        :param normalized: Optional. If True, the profiles are normalized at each height  
            on the total density at this height. 
        :type normalized: boolean
        :param cumulative: Optional. If True, the normalized cumulative mass profiles are plotted. 
            At each height z, profiles are normalized on the total mass up to this z. 
        :type cumulative: boolean
        :param save: Optional. If True, the figure will be saved (call ``a.T['densplt']`` 
            to display the folder name).
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean
        
        :return: Figure and axes.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','normalized','cumulative'],this_function)
        if R!=self.p.Rsun:
            R = inpcheck_radius(R,self.p,this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
        rhoz = tab_reader(['rhoz'],self.p,self.a.T,R=R)[0]
        rhoztot = np.sum(np.sum(rhoz[1:],axis=0))
        legendloc = 'best'
        
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_ylim(0,0.8)
        if inpcheck_iskwargtype(kwargs,'cumulative',True,bool,this_function):
            ylabelname = r'$\mathrm{\sum_{0}^{|z|} M_i \ / \ \sum_{0}^{|z|} M_{tot}}$'
            rhoztot = np.cumsum(np.sum(rhoz[1:],axis=0))
            for i in range(6):
                ax.plot(rhoz[0],np.cumsum(rhoz[i+1])/rhoztot,label=self.labs[self.name[i]],
                        lw=self.lw['main'],color=self.cols[self.name[i]])
            ax.plot([0,self.p.zmax],[1,1],lw=0.5,ls='--',c='k')
            ax.set_ylim(0,1.05)
        else:
            if inpcheck_iskwargtype(kwargs,'normalized',True,bool,this_function):
                ylabelname = r'$\mathrm{\rho_i(|z|) \ / \ \rho_{tot}(|z|)}$'
                rhoztot = np.sum(rhoz[1:],axis=0)
                for i in range(6):
                    ax.plot(rhoz[0],rhoz[i+1]/rhoztot,label=self.labs[self.name[i]],
                            lw=self.lw['main'],color=self.cols[self.name[i]])
                ax.plot([0,self.p.zmax],[1,1],lw=0.5,ls='--',c='k')
                ax.set_ylim(0,1.05)
            else:
                ylabelname = r'$\mathrm{\rho_i(|z|), \ M_\odot \ pc^{-3}}$'
                for i in range(6):
                    ax.semilogy(rhoz[0],rhoz[i+1],label=self.labs[self.name[i]],
                                lw=self.lw['main'],color=self.cols[self.name[i]])
                ax.set_ylim(1e-7,1.05)
                ylabelname,legendloc = self.axts['rho'],1
        ax.set_ylabel(ylabelname,fontsize=self.fnt['main'],labelpad=15)
        ax.set_xlabel(self.axts['z'],fontsize=self.fnt['main'],labelpad=15)
        ax.set_title('$\mathrm{R=}$'+str(float(R))+'$\mathrm{\ kpc}$',fontsize=self.fnt['main'],pad=15)
        ax.set_xlim(0,self.p.zmax)
        plt.legend(prop={'size':self.fnt['secondary']},loc=legendloc,ncol=2)
        f.subplots_adjust(left=0.14,top=0.86,bottom=0.15,right=0.86)
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            kwargs_calc = reduce_kwargs(kwargs,['normalized','cumulative'])
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,save_format=format_,fig=True,**kwargs_calc)
            ts.rhoz_save(rhoz,R)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax) 
    
 
    def rhoz_monoage_plt(self,mode_comp,R,ages,**kwargs):
        """
        Vertical density profiles of the mono-age subpopulations
        (plotting for :func:`jjmodel.analysis.rhoz_monoage`).
        
        :param mode_comp: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
        :type mode_comp: str
        :param R: Galactocentric distance, kpc.
        :type R: scalar
        :param ages: Set of age bins, Gyr. 
        :type ages: array-like
        :param between: Optional. If True, the output quantity corresponds to the age intervals 
            specified by parameter **ages**. Otherwise the individual single mono-age subpopulations 
            are returned (i.e., age-bins of width ``tr``, model age resolution).
        :type between: boolean
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'dt'``, **tab** must be organized as a list of tables for the thin and thick disk: 
            [*table_d,table_t*]. If **mode_comp** = ``'tot'``, **tab** must be [*table_d,table_t,table_sh*]. 
        :type tab: astropy table or list[astropy table] 
        :param number: Optional. If True, calculated quantity is the spatial number density of stars in :math:`\mathrm{number \ pc^{-3}}`,
             not matter density in :math:`\mathrm{M_\odot \ pc^{-3}}`. Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str 
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :type cbar_bins: boolean
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.rhoz_monoage_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.  
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins',
                                'mode_pop','number','tab','between','mode_iso'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','dt','sh','tot'],
                                           'vertical mono-age profiles',this_function)
        if R!=self.p.Rsun:
            R = inpcheck_radius(R,self.p,this_function)
        ages = inpcheck_age(ages,this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        nage = len(ages)
        if inpcheck_iskwargtype(kwargs,'between',True,bool,this_function):
            nage = nage - 1        
        kwargs_calc = reduce_kwargs(kwargs,['save','mode_pop','number','tab','between','mode_iso'])
        rhoz = rhoz_monoage(mode_comp,R,ages,self.p,self.a,**kwargs_calc)
        
        colorbar = self._figcolorbar_('tau',len(ages),**kwargs)
        cmin, cmax = self._cbarminmax_(0,tp,np.mean(np.diff(ages)),**kwargs)

        ylabel = self.axts['rho']
        if ('mode_pop' in kwargs) or ('tab' in kwargs):
            if inpcheck_iskwargtype(kwargs,'number',True,bool,this_function):
                ylabel = self.axts['n']
            if 'mode_pop' in kwargs:
                ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
            if 'tab' in kwargs:
                ln += '$\mathrm{\ (custom \ population)}$'
        ln += '$\mathrm{\ at \ R=}$'+str(R)+'$\mathrm{\ kpc}$'
        
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_xlim((0,self.p.zmax))
        ymin, ymax = self._dynamic_ylim_log_(rhoz,[1e-12,1e-2])
        ax.set_ylim(ymin,ymax)                     
        ax.set_yscale('log')
        ax.set_xlabel(self.axts['z'],fontsize=self.fnt['main'])
        ax.set_ylabel(ylabel,fontsize=self.fnt['main'])
        ax.set_title(ln,fontsize=self.fnt['main'],pad=15)
        line_segments = LineCollection([list(zip(self.a.z,rhoz[i])) for i in np.arange(nage)],
                                       linewidths=self.lw['main'],cmap = colorbar,                        
                                       norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)
                                       )
        line_segments.set_array(np.array(ages))
        im = ax.add_collection(line_segments)
        f.subplots_adjust(left=0.14,top=0.86,bottom=0.15,right=0.86)
        pos = ax.get_position()
        cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
        cbar = f.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_ticks(self.xt['ticks'])
        cbar.set_ticklabels(self.xt['labels'])
        cbar.set_label(self.axts['tau'],fontsize=self.fnt['main'])
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs_calc)
            ts.rhoz_monoage_save(rhoz,mode_comp,R,ages)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)
    
    
    def rhoz_monomet_plt(self,mode_comp,R,mets,**kwargs):
        """
        Vertical density profiles of the mono-metallicity subpopulations
        (plotting for jjmodel.analysis.rhoz_monomet).
        
        :param mode_comp: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
        :type mode_comp: str
        :param R: Galactocentric distance, kpc.
        :type R: scalar
        :param mets: Set of metallicity bins, dex. 
        :type mets: array-like
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'dt'``, **tab** must be organized as a list of tables for the thin and thick disk: 
            [*table_d,table_t*]. If **mode_comp** = ``'tot'``, **tab** must be [*table_d,table_t,table_sh*]. 
        :type tab: astropy table or list[astropy table] 
        :param number: Optional. If True, calculated quantity is the spatial number density of stars in :math:`\mathrm{number \ pc^{-3}}`,
             not matter density in :math:`\mathrm{M_\odot \ pc^{-3}}`. Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str 
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :type cbar_bins: boolean
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.rhoz_monomet_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.    
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins',
                                'mode_pop','number','tab','mode_iso'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','dt','sh','tot'],
                                           'vertical mono-metallicity profiles',this_function)
        if R!=self.p.Rsun:
            R = inpcheck_radius(R,self.p,this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        nage = len(mets) - 1
        
        kwargs_calc = reduce_kwargs(kwargs,['save','mode_pop','number','tab','mode_iso'])
        rhoz = rhoz_monomet(mode_comp,R,mets,self.p,self.a,**kwargs_calc)
        
        colorbar = self._figcolorbar_('feh',len(mets),**kwargs)
        cmin, cmax = self._cbarminmax_(mets[0],mets[-1],np.mean(np.diff(mets)),**kwargs)

        ylabel = self.axts['rho']
        if ('mode_pop' in kwargs) or ('tab' in kwargs):
            if inpcheck_iskwargtype(kwargs,'number',True,bool,this_function):
                ylabel = self.axts['n']
            if 'mode_pop' in kwargs:
                ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
            if 'tab' in kwargs:
                ln += '$\mathrm{\ (custom \ population)}$'
        ln += '$\mathrm{\ at \ R=}$'+str(R)+'$\mathrm{\ kpc}$'
                               
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_xlim((0,self.p.zmax))
        ymin, ymax = self._dynamic_ylim_log_(rhoz,[1e-12,1e-2])
        ax.set_ylim(ymin,ymax)                     
        ax.set_yscale('log')
        ax.set_xlabel(self.axts['z'],fontsize=self.fnt['main'])
        ax.set_ylabel(ylabel,fontsize=self.fnt['main'])
        ax.set_title(ln,fontsize=self.fnt['main'],pad=15)                     
        line_segments = LineCollection([list(zip(self.a.z,rhoz[i])) for i in np.arange(nage)],
                                       linewidths=self.lw['main'],cmap = colorbar,                        
                                       norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)
                                       )
        line_segments.set_array(np.array(mets))
        im = ax.add_collection(line_segments)
        f.subplots_adjust(left=0.14,top=0.86,bottom=0.15,right=0.86)
        pos = ax.get_position()
        cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
        cbar = f.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_label(self.axts['fe'],fontsize=self.fnt['main'])
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs_calc)
            ts.rhoz_monomet_save(rhoz,mode_comp,R,mets)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)
    
    
    def rz_map_plt(self,mode_comp,**kwargs):
        """
        Density distribution in R and z Galactic coordinates. Can be matter or number density. 
        Plotting for :func:`jjmodel.analysis.rz_map`. 
        
        :param mode_comp: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total).
        :type mode_comp: str
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'tot'``, **tab** must be [[*table_d,table_t,table_sh*]_ *rmin*,...,[*table_d,table_t,table_sh*]_ *rmax*], 
            where *rmin* and *rmax* are ``p.Rmin`` and ``p.Rmax``. 
        :type tab: list[astropy table] or list[list[astropy table]
        :param number: Optional. If True, calculated quantity is weighted by the spatial number density of stars (:math:`\mathrm{number \ pc^{-3}}`),
             not matter density (:math:`\mathrm{M_\odot \ pc^{-3}}`). Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str 
        :param ages: Optional. Set of age bins, Gyr. 
        :type ages: array-like
        :param mets: Optional. Set of metallicity bins.
        :type mets: array-like
        :param dz: Vertical resolution, pc. 
        :type dz: scalar
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.rz_map_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.  
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','mode_pop',
                                'tab','number','dz','ages','mets','mode_iso'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','dt','sh','tot'],
                                           'RZ-density map',this_function)
        if 'dz' in kwargs:
            kwargs['dz'] = inpcheck_dz(kwargs['dz'],self.p,this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function)
        colorbar = self._figcolorbar_('dens',10,**kwargs)
        
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'
        if 'ages' in kwargs:
            ln += r'$\mathrm{\ of \ \tau=}$' + '${}$'.format(str(kwargs['ages'])) +\
                  r'$\mathrm{ \ Gyr}$'                          
        if 'mets' in kwargs:
            ln += '$\mathrm{\ of \ [Fe/H]=[}$' + str(kwargs['mets'][0]) + ',' +\
                  str(kwargs['mets'][1]) + '$\mathrm{]}$'
        
        cbarlabel = r'$\mathrm{log_{10} \ \rho \ [M_\odot \ pc^{-3}]}$'
        if inpcheck_iskwargtype(kwargs,'number',True,bool,this_function):
            cbarlabel = r'$\mathrm{log_{10} \ N \ [pc^{-3}]}$'
                   
        kwargs_calc = reduce_kwargs(kwargs,['save','mets','ages','number','dz',
                                            'mode_pop','tab','mode_iso'])
        rz_grid = rz_map(mode_comp,self.p,self.a,**kwargs_calc)
            
        ycmin, ycmax = self._dynamic_ylim_log_(rz_grid,[1e-8,1e1])
        ycmin, ycmax = np.log10(ycmin), np.log10(ycmax)
        ycbar = np.arange(ycmin,ycmax + 0.5,0.5)
        ycbar_lb = []
        for i in range(len(ycbar)):
            if ycbar[i]%1==0:
                ycbar_lb.append('${}$'.format(str(ycbar[i])))
            else:
                ycbar_lb.append('')
                
        rz_grid_flip = np.flip(rz_grid.T,axis=0)
        rz_grid_flip[rz_grid_flip==0] = np.nan
        log10_rho = np.log10(rz_grid_flip)
        
        f, ax = plt.subplots(figsize=(14,3.5))
        im = ax.imshow(log10_rho,cmap=colorbar, 
                   extent=[self.p.Rmin-self.p.dR/2,self.p.Rmax+self.p.dR/2,0,self.p.zmax/1e3],
                   norm = mpl.colors.Normalize(vmin=ycmin,vmax=ycmax))
        f.subplots_adjust(left=0.08,top=0.86,bottom=0.15,right=0.86)
        pos = ax.get_position()
        cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
        ax.set_xlabel(self.axts['r'],fontsize=self.fnt['main'],labelpad=10)
        ax.set_ylabel('$\mathrm{|z|, \ kpc}$',fontsize=self.fnt['main'],labelpad=10)
        ax.set_title(ln,fontsize=self.fnt['main'],pad=10)
        cbar = plt.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_ticks(ycbar)
        cbar.set_ticklabels(ycbar_lb) 
        cbar.set_label(cbarlabel,fontsize=self.fnt['main'],labelpad=15)
            
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs_calc)
            ts.rz_map_save(rz_grid,mode_comp)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)
        
    
    def rz_map_quantity_plt(self,mode_comp,quantity,**kwargs):
        """
        Distribution of some quantity Q in Rz-plane. Q can be W-velocity dispersion or 
        stellar physical parameter from isochrones. Plotting for :func:`jjmodel.analysis.rz_map_quantity`. 
        
        :param mode_comp: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total).
        :type mode_comp: str
        :param quantity: :param quantity: Name of the column in a stellar assemblies table to which 
            the function has to be applied; for velocity dispersion use ``'sigw'``. 
        :type quantity: str
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'tot'``, **tab** must be [[*table_d,table_t,table_sh*]_ *rmin*,...,[*table_d,table_t,table_sh*]_ *rmax*], 
            where *rmin* and *rmax* are ``p.Rmin`` and ``p.Rmax``. 
        :type tab: list[astropy table] or list[list[astropy table]
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str     
        :param ages: Optional. Set of age bins, Gyr. 
        :type ages: array-like
        :param mets: Optional. Set of metallicity bins.
        :type mets: array-like
        :param dz: Vertical resolution, pc. 
        :type dz: scalar
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.rz_map_quantity_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.  
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','ages','mets',
                                'dz','mode_pop','tab','mode_iso'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                           'RZ-quantity map',this_function)
        if 'dz' in kwargs:
            kwargs['dz'] = inpcheck_dz(kwargs['dz'],self.p,this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)        
        colorbar = self._figcolorbar_('dens',10,**kwargs)
        
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'
        if 'ages' in kwargs:
            ln += r'$\mathrm{\ of \ \tau=}$' + '${}$'.format(str(kwargs['ages'])) +\
                  r'$\mathrm{ \ Gyr}$'                          
        if 'mets' in kwargs:
            ln += '$\mathrm{\ of \ [Fe/H]=[}$' + str(kwargs['mets'][0]) + ',' +\
                  str(kwargs['mets'][1]) + '$\mathrm{]}$'
        
        kwargs_calc = reduce_kwargs(kwargs,['save','dz','mode_pop','tab','ages','mets','mode_iso'])
        rz_grid = rz_map_quantity(mode_comp,quantity,self.p,self.a,**kwargs_calc)
        mean_q = np.flip(rz_grid.T,axis=0)
    
        if quantity=='age':
            ycbar, ycbar_ticklabels = self.xt['ticks'], self.xt['labels']
            ycbar_label = self.axts['tau']
            ycmin, ycmax = 0, tp 
        else:
            try:
                ycbar_label = self.cbar_settings[quantity][2]
                dy1 = self.cbar_settings[quantity][0]
                interval = np.nanmax(rz_grid) - np.nanmin(rz_grid)
                interval_decimal_points = len(str(eval("%.0e" % interval)).split('.')[1])
                while dy1 > interval or interval/dy1 < 5:
                    dy1 = dy1/2
                ss = "%."+str(int(interval_decimal_points))+"e"
                dy1 = eval(ss % dy1)
                dy = round(dy1,interval_decimal_points)
                if dy==0:
                    dy = round(dy1,interval_decimal_points+1)
                while interval/dy > 6:
                    dy = dy*2
            except:
                dy = (np.nanmax(rz_grid) - np.nanmin(rz_grid))/4
                interval_decimal_points = 2 
            ycmin, ycmax = self._dynamic_ylim_1d_lin_(rz_grid,self.cbar_settings[quantity][1],dy)
            ybins = np.arange(ycmin,ycmax+dy,dy,dtype=np.float)   
            ybins = [round(i,interval_decimal_points+3) for i in ybins]
            test = True
            for k in range(len(ybins)):
                pv = int(str(ybins[k]).split('.')[1])
                if pv!=0:
                    test = False
            if test==True:
                ybins = [int(i) for i in ybins]
            ycbar, ycbar_ticklabels = ybins, ['${}$'.format(str(i)) for i in ybins] 
                
        f, ax = plt.subplots(figsize=(14,3.5))
        im = ax.imshow(mean_q,cmap=colorbar, 
                       extent=[self.p.Rmin-self.p.dR/2,self.p.Rmax+self.p.dR/2,0,self.p.zmax/1e3],
                       norm = mpl.colors.Normalize(vmin=ycmin,vmax=ycmax))
        f.subplots_adjust(left=0.08,top=0.86,bottom=0.15,right=0.86)
        pos = ax.get_position()
        cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
        ax.set_xlabel(self.axts['r'],fontsize=self.fnt['main'],labelpad=10)
        ax.set_ylabel('$\mathrm{|z|, \ kpc}$',fontsize=self.fnt['main'],labelpad=10)
        ax.set_title(ln,fontsize=self.fnt['main'],pad=10)
        cbar = plt.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_ticks(ycbar)
        cbar.set_ticklabels(ycbar_ticklabels) 
        cbar.set_label(ycbar_label,fontsize=self.fnt['main'],labelpad=15)
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs_calc)
            ts.rz_map_quantity_save(rz_grid,mode_comp,quantity)
        self._figclose_(this_function,**kwargs)


    def agez_plt(self,mode_comp,**kwargs):
        """
        Mean age as a function of height z and Galactocentric distance R.
        Plotting for :func:`jjmodel.analysis.agez`. 
        
        :param mode_comp: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total).
        :type mode_comp: str
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'tot'``, **tab** must be [[*table_d,table_t,table_sh*]_ *rmin*,...,[*table_d,table_t,table_sh*]_ *rmax*], 
            where *rmin* and *rmax* are ``p.Rmin`` and ``p.Rmax``. 
        :type tab: list[astropy table] or list[list[astropy table]] 
        :param number: Optional. If True, calculated quantity is weighted by the spatial number density of stars (:math:`\mathrm{number \ pc^{-3}}`),
             not by matter density in (:math:`\mathrm{M_\odot \ pc^{-3}}`). Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str 
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :type cbar_bins: boolean
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.agez_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.  
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins',
                                'tab','mode_pop','number','mode_iso'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','dt','tot'],
                                           'vertical age profiles',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
        colorbar = self._figcolorbar_('r',self.a.Rbins,**kwargs)
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'
        cmin, cmax = self._cbarminmax_(self.p.Rmin,self.p.Rmax,self.p.dR,**kwargs)
            
        kwargs_calc = reduce_kwargs(kwargs,['save','tab','mode_pop','number','mode_iso'])
        agezr, agezr0 = agez(mode_comp,self.p,self.a,**kwargs_calc)
            
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_xlim((0,self.p.zmax))
        ymin, ymax = 0, tp+0.1
        ax.set_ylim(ymin,ymax)                     
        ax.set_xlabel(self.axts['z'],fontsize=self.fnt['main'])
        ax.set_ylabel(self.axts['tau'],fontsize=self.fnt['main'])
        ax.set_title(ln,fontsize=self.fnt['main'],pad=10)
        
        f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.86)
        line_segments = LineCollection([list(zip(self.a.z,agezr[i])) for i in np.arange(self.a.Rbins)],
                                       linewidths=self.lw['main'],cmap = colorbar,                        
                                       norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)                                                                   
                                       )
        line_segments.set_array(self.a.R)
        im = ax.add_collection(line_segments)
        if mode_comp=='dt' or mode_comp=='tot':
            agezt, agezt0 = agez('t',self.p,self.a)
            ax.plot(self.a.z,agezt0,ls='--',color='orange',lw=2,label='$\mathrm{Thick \ disk}$')
        if mode_comp=='tot':
            agezsh, agezsh0 = agez('sh',self.p,self.a)
            ax.plot(self.a.z,agezsh0,ls='--',color='magenta',lw=2,label='$\mathrm{Halo}$')
        ax.plot(self.a.z,agezr0,ls='--',color='k',lw=self.lw['secondary'],label=self.labs['r0'])
        plt.legend(loc=4,prop={'size':self.fnt['secondary']})
        pos = ax.get_position()
        cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
        cbar = f.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_ticks(np.arange(self.a.R[0],self.a.R[-1]+1))
        cbar.set_label(self.axts['r'],fontsize=self.fnt['main'])
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs_calc)
            ts.agez_save((agezr,agezr0),mode_comp)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax) 
    

    def agez_rsun_plt(self,mode_comp,**kwargs):
        """
        Mean age as a function of height z for the Solar neighbourhood, ``p.Rsun``.
        Plotting for :func:`jjmodel.analysis.agez`. 
        
        :param mode_comp: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total).
        :type mode_comp: str
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table, parameter alternative to mode_pop. 
            For **mode_pop** = ``tot``, *tab* is constructed as [*table_d,table_t,table_sh*] with tables for ``p.Rsun``. 
        :type tab: astropy table or list[astropy table] 
        :param number: Optional. If True, calculated quantity is weighted by the spatial number density of stars (:math:`\mathrm{number \ pc^{-3}}`),
             not by matter density in (:math:`\mathrm{M_\odot \ pc^{-3}}`). Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str 
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.agez_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean
        
        :return: Figure and axes. 
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close',
                                'tab','mode_pop','number','mode_iso'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','dt','tot'],
                                           'vertical age profiles',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'
            
        kwargs_calc = reduce_kwargs(kwargs,['save','tab','mode_pop','number','mode_iso'])
        
        if 'tab' in kwargs and (mode_comp=='dt' or mode_comp=='tot'):
            del kwargs_calc['tab']
            agezr0 = agez('d',self.p,self.a,**kwargs_calc,R=self.p.Rsun,tab=kwargs['tab'][0])
        else:
            agezr0 = agez('d',self.p,self.a,**kwargs_calc,R=self.p.Rsun)
            
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_xlim((0,self.p.zmax))
        ymin, ymax = 0, tp+0.1
        ax.set_ylim(ymin,ymax)                     
        ax.set_xlabel(self.axts['z'],fontsize=self.fnt['main'])
        ax.set_ylabel(self.axts['tau'],fontsize=self.fnt['main'])
        ax.set_title(ln,fontsize=self.fnt['main'],pad=10)
        
        f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.91)
        ax.plot(self.a.z,agezr0,color=self.cols['d'],lw=self.lw['main'],label=self.labs['d'])
        if mode_comp=='dt':
            if 'tab' in kwargs:
                agezt0 = agez('t',self.p,self.a,R=self.p.Rsun,**kwargs_calc,tab=kwargs['tab'][1])
                agezdt0 = agez('dt',self.p,self.a,R=self.p.Rsun,**kwargs_calc,
                               tab=[kwargs['tab'][0],kwargs['tab'][1]])
            else:
                agezt0 = agez('t',self.p,self.a,R=self.p.Rsun,**kwargs_calc)
                agezdt0 = agez('dt',self.p,self.a,R=self.p.Rsun,**kwargs_calc)
            ax.plot(self.a.z,agezt0,color=self.cols['t'],lw=self.lw['main'],label=self.labs['t'])
            ax.plot(self.a.z,agezdt0,color=self.cols['dt'],lw=self.lw['main'],label=self.labs['dt'])
        if mode_comp=='tot':
            if 'tab' in kwargs:
                agezt0 = agez('t',self.p,self.a,R=self.p.Rsun,**kwargs_calc,tab=kwargs['tab'][1])
                agezsh0 = agez('sh',self.p,self.a,R=self.p.Rsun,**kwargs_calc,tab=kwargs['tab'][2])
                ageztot0 = agez('tot',self.p,self.a,R=self.p.Rsun,**kwargs_calc,
                                tab=[kwargs['tab'][0],kwargs['tab'][1],kwargs['tab'][2]])
            else:
                agezt0 = agez('t',self.p,self.a,R=self.p.Rsun,**kwargs_calc)
                agezsh0 = agez('sh',self.p,self.a,R=self.p.Rsun,**kwargs_calc)
                ageztot0 = agez('tot',self.p,self.a,R=self.p.Rsun,**kwargs_calc)
            ax.plot(self.a.z,agezt0,color=self.cols['t'],lw=self.lw['main'],label=self.labs['t'])
            ax.plot(self.a.z,agezsh0,color=self.cols['sh'],lw=self.lw['main'],label=self.labs['sh'])
            ax.plot(self.a.z,ageztot0,color=self.cols['tot'],lw=self.lw['main'],label=self.labs['tot'])
        if mode_comp=='dt' or mode_comp=='tot':
            plt.legend(loc=4,prop={'size':self.fnt['secondary']})
                
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs_calc,R=self.p.Rsun)
            ts.agez_save(agezr0,mode_comp)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax) 


    def ager_plt(self,mode_comp,zlim,**kwargs):
        """
        Radial age profiles. Plotting for :func:`jjmodel.analysis.ager`. 
        
        :param mode_comp: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
        :type mode_comp: str
        :param zlim: Range of heights [*zmin,zmax*], pc. 
        :type zlim: array-like 
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'tot'``, **tab** must be [[*table_d,table_t,table_sh*]_ *rmin*,...,[*table_d,table_t,table_sh*]_ *rmax*], 
            where *rmin* and *rmax* are ``p.Rmin`` and ``p.Rmax``. 
        :type tab: list[astropy table] or list[list[astropy table]] 
        :param number: Optional. If True, calculated quantity is weighted by the spatial number density of stars (:math:`\mathrm{number \ pc^{-3}}`),
             not by matter density (:math:`\mathrm{M_\odot \ pc^{-3}}`). Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str 
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :type cbar_bins: boolean
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.ager_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.  
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins',
                                'mode_pop','tab','number','mode_iso'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','dt','tot'],
                                           'radial age profiles',this_function)
        zlim = inpcheck_height(zlim,self.p,this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'        
        
        nz = int(len(zlim) - 1)
        colorbar = self._figcolorbar_('z',nz,**kwargs)
        cmin, cmax = 0, self.p.zmax/1e3
        
        age_r = np.zeros((nz,self.a.Rbins))
        kwargs_calc = reduce_kwargs(kwargs,['save','tab','mode_pop','number','mode_iso'])
        for i in range(nz):
            age_r[i] = ager(mode_comp,[zlim[i],zlim[i+1]],self.p,self.a,**kwargs_calc)        
    
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_xlim((self.p.Rmin,self.p.Rmax))
        ymin, ymax = 0, tp + 0.1
        ax.set_ylim((ymin,ymax))           
        ax.set_xlabel(self.axts['r'],fontsize=self.fnt['main'])
        ax.set_ylabel(self.axts['tau'],fontsize=self.fnt['main'])
        ax.set_title(ln,fontsize=self.fnt['main'],pad=10)
    
        f.subplots_adjust(left=0.14,top=0.86,bottom=0.15,right=0.86)
        line_segments = LineCollection([list(zip(self.a.R,age_r[i])) for i in np.arange(nz)],
                                       linewidths=self.lw['main'],cmap = colorbar,                        
                                       norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)                                                                   
                                       )
        dz = np.mean(np.diff(zlim))
        line_segments.set_array(np.add(zlim,np.mean(np.diff(zlim))/2)[:-1]/1e3)
        im = ax.add_collection(line_segments)
        if mode_comp=='dt' or mode_comp=='tot':
            agert = ager('t',[zlim[0],zlim[-1]],self.p,self.a)
            ax.plot(self.a.R,agert,ls='--',color='orange',lw=2,label='$\mathrm{Thick \ disk}$')
        if mode_comp=='tot':
            agersh = ager('sh',[zlim[0],zlim[-1]],self.p,self.a)
            ax.plot(self.a.R,agersh,ls='--',color='magenta',lw=2,label='$\mathrm{Halo}$')
        ax.plot([self.p.Rsun,self.p.Rsun],[ymin,ymax],ls='--',lw=self.lw['secondary'],
                     c='darkgrey',label=self.labs['r0'])
        plt.legend(prop={'size':self.fnt['secondary']},loc=3,ncol=1)
        pos = ax.get_position()
        cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
        cbar = f.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_ticks(np.arange(zlim[0]+dz/2,zlim[-1]+dz/2,dz)/1e3)
        cbar.set_label(self.axts['zkpc'],fontsize=self.fnt['main'])
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs_calc)
            ts.ager_save(age_r,mode_comp,zlim)
        self._figclose_(this_function,**kwargs) 
        
        return (f, ax) 


    def metz_plt(self,mode_comp,**kwargs):
        """ 
        Mean metallicity as a function of height z and Galactocentric distance R.
        Plotting for :func:`jjmodel.analysis.metz`. 
        
        :param mode_comp: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
        :type mode_comp: str
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'tot'``, **tab** must be [[*table_d,table_t,table_sh*]_ *rmin*,...,[*table_d,table_t,table_sh*]_ *rmax*], 
            where *rmin* and *rmax* are ``p.Rmin`` and ``p.Rmax``. 
        :type tab: list[astropy table] or list[list[astropy table]] 
        :param number: Optional. If True, calculated quantity is weighted by the spatial number density of stars (:math:`\mathrm{number \ pc^{-3}}`),
             not by matter density (:math:`\mathrm{M_\odot \ pc^{-3}}`). Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str 
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :type cbar_bins: boolean
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.metz_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.  
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins',
                                'mode_pop','tab','number','mode_iso'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','dt','tot'],
                                           'metallicity profiles',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
        colorbar = self._figcolorbar_('r',self.a.Rbins,**kwargs)
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'
        cmin, cmax = self._cbarminmax_(self.p.Rmin,self.p.Rmax,self.p.dR,**kwargs)
               
        kwargs_calc = reduce_kwargs(kwargs,['save','tab','mode_pop','number','mode_iso'])
        fehzr, fehzr0 = metz(mode_comp,self.p,self.a,**kwargs_calc)     
            
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_xlim((0,self.p.zmax))
        ymin, ymax = self._dynamic_ylim_1d_lin_(fehzr,[-0.6,1],0.2)

        ax.set_ylim(ymin,ymax)                     
        ax.set_xlabel(self.axts['z'],fontsize=self.fnt['main'])
        ax.set_ylabel(self.axts['fe'],fontsize=self.fnt['main'])
        ax.set_title(ln,fontsize=self.fnt['main'],pad=10)
        
        f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.86)
        line_segments = LineCollection([list(zip(self.a.z,fehzr[i])) for i in np.arange(self.a.Rbins)],
                                       linewidths=self.lw['main'],cmap = colorbar,                        
                                       norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)                                                                   
                                       )
        line_segments.set_array(self.a.R)
        im = ax.add_collection(line_segments)
        if mode_comp=='dt' or mode_comp=='tot':
            fehzt, fehzt0 = metz('t',self.p,self.a)
            ax.plot(self.a.z,fehzt0,ls='--',color='orange',lw=2,label='$\mathrm{Thick \ disk}$')
        ax.plot(self.a.z,fehzr0,ls='--',color='k',lw=self.lw['secondary'],label=self.labs['r0'])
        plt.legend(loc=1,prop={'size':self.fnt['secondary']})
        pos = ax.get_position()
        cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
        cbar = f.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_ticks(np.arange(self.a.R[0],self.a.R[-1]+1))
        cbar.set_label(self.axts['r'],fontsize=self.fnt['main'])
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs_calc)
            ts.metz_save((fehzr,fehzr0),mode_comp)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax) 
    
    
    def metz_rsun_plt(self,mode_comp,**kwargs):
        """        
        Mean metallicity as a function of height z for the Solar neighbourhood, p.Rsun.
        Plotting for jjmodel.analysis.metz. 
        
        :param mode_comp:Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
        :type mode_comp: str
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            For **mode_comp** = ``tot``,  
            *tab* is constructed as [*table_d,table_t,table_sh*] with tables for ``p.Rsun``. 
        :type tab: astropy table or list[astropy table] 
        :param number: Optional. If True, calculated quantity is weighted by the spatial number density of stars (:math:`\mathrm{number \ pc^{-3}}`),
             not by matter density (:math:`\mathrm{M_\odot \ pc^{-3}}`). Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str 
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.metz_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.  
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close',
                                'mode_pop','tab','number','mode_iso'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','dt','tot'],
                                           'metallicity profiles',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'
               
        kwargs_calc = reduce_kwargs(kwargs,['save','tab','mode_pop','number','mode_iso'])
        if ('tab' in kwargs) and (mode_comp=='dt' or mode_comp=='tot'):
            del kwargs_calc['tab']
            fehzr0 = metz('d',self.p,self.a,**kwargs_calc,R=self.p.Rsun,tab=kwargs['tab'][0])
        else:
            fehzr0 = metz('d',self.p,self.a,**kwargs_calc,R=self.p.Rsun)     
            
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_xlim((0,self.p.zmax))
        ax.set_xlabel(self.axts['z'],fontsize=self.fnt['main'])
        ax.set_ylabel(self.axts['fe'],fontsize=self.fnt['main'])
        ax.set_title(ln,fontsize=self.fnt['main'],pad=10)
        f.subplots_adjust(left=0.13,top=0.86,bottom=0.15,right=0.91)
        
        ax.plot(self.a.z,fehzr0,color=self.cols['d'],lw=self.lw['main'],label=self.labs['d'])
        ymin, ymax = self._dynamic_ylim_1d_lin_(fehzr0,[-0.6,1],0.2)
                                                    
        if mode_comp=='dt':
            if 'tab' in kwargs:
                fehzt0 = metz('t',self.p,self.a,R=self.p.Rsun,**kwargs_calc,tab=kwargs['tab'][1])
                fehzdt0 = metz('dt',self.p,self.a,R=self.p.Rsun,**kwargs_calc,tab=kwargs['tab'])
            else:
                fehzt0 = metz('t',self.p,self.a,R=self.p.Rsun,**kwargs_calc)
                fehzdt0 = metz('dt',self.p,self.a,R=self.p.Rsun,**kwargs_calc)
            ax.plot(self.a.z,fehzt0,color=self.cols['t'],lw=self.lw['main'],label=self.labs['t'])
            ax.plot(self.a.z,fehzdt0,color=self.cols['dt'],lw=self.lw['main'],label=self.labs['dt'])
            ymin, ymax = self._dynamic_ylim_1d_lin_(np.stack((fehzr0,fehzt0,fehzdt0),axis=-1),
                                                    [-0.6,1],0.2)
        if mode_comp=='tot':
            if 'tab' in kwargs:
                fehzt0 = metz('t',self.p,self.a,R=self.p.Rsun,**kwargs_calc,tab=kwargs['tab'][1])
                fehzsh0 = metz('sh',self.p,self.a,R=self.p.Rsun,**kwargs_calc,tab=kwargs['tab'][2])
                fehztot0 = metz('tot',self.p,self.a,R=self.p.Rsun,**kwargs_calc,tab=kwargs['tab'])
            else:
                fehzt0 = metz('t',self.p,self.a,R=self.p.Rsun,**kwargs_calc)
                fehzsh0 = metz('sh',self.p,self.a,R=self.p.Rsun,**kwargs_calc)
                fehztot0 = metz('tot',self.p,self.a,R=self.p.Rsun,**kwargs_calc)
            ax.plot(self.a.z,fehzt0,color=self.cols['t'],lw=self.lw['main'],label=self.labs['t'])
            ax.plot(self.a.z,fehzsh0,color=self.cols['sh'],lw=self.lw['main'],label=self.labs['sh'])
            ax.plot(self.a.z,fehztot0,color=self.cols['tot'],lw=self.lw['main'],label=self.labs['tot'])
            ymin, ymax = self._dynamic_ylim_1d_lin_(np.stack((fehzr0,fehzt0,fehzsh0,fehztot0),axis=-1),
                                                    [-0.6,1],0.2)
        ax.set_ylim(ymin,ymax)      
        if mode_comp=='dt' or mode_comp=='tot':
            plt.legend(loc=1,prop={'size':self.fnt['secondary']})
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs_calc,R=self.p.Rsun)
            ts.metz_save(fehzr0,mode_comp)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax) 

    
    def metr_plt(self,mode_comp,zlim,**kwargs):
        """        
        Radial metallicity profiles. Plotting for :func:`jjmodel.analysis.metr`. 
        
        :param mode_comp: Galactic component. Can be ``'d'`` (thin disk), ``'dt'`` (thin + thick disk), 
            or ``'tot'`` (total: thin + thick disk + stellar halo). 
        :type mode_comp: str
        :param zlim: Range of heights [*zmin,zmax*], pc. 
        :type zlim: array-like 
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'tot'``, **tab** must be [[*table_d,table_t,table_sh*]_ *rmin*,...,[*table_d,table_t,table_sh*]_ *rmax*], 
            where *rmin* and *rmax* are ``p.Rmin`` and ``p.Rmax``. 
        :type tab: list[astropy table] or list[list[astropy table]] 
        :param number: Optional. If True, calculated quantity is weighted by the spatial number density of stars (:math:`\mathrm{number \ pc^{-3}}`),
             not by matter density (:math:`\mathrm{M_\odot \ pc^{-3}}`). Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str 
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :type cbar_bins: boolean
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.metr_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.  
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins',
                                'mode_pop','tab','number','mode_iso'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','dt','tot'],
                                           'metallicity profiles',this_function)
        zlim = inpcheck_height(zlim,self.p,this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'
        cmin, cmax = 0, self.p.zmax/1e3
        
        nz = int(len(zlim) - 1)
        colorbar = self._figcolorbar_('z',nz,**kwargs)
        fehr = np.zeros((nz,self.a.Rbins))
              
        kwargs_calc = reduce_kwargs(kwargs,['save','tab','mode_pop','number','mode_iso'])
        for i in range(nz):
            fehr[i] = metr(mode_comp,[zlim[i],zlim[i+1]],self.p,self.a,**kwargs_calc)        
    
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_xlim((self.p.Rmin,self.p.Rmax))
        ax.set_xlabel(self.axts['r'],fontsize=self.fnt['main'])
        ax.set_ylabel(self.axts['fe'],fontsize=self.fnt['main'])
        ax.set_title(ln,fontsize=self.fnt['main'],pad=10)
        
        f.subplots_adjust(left=0.14,top=0.86,bottom=0.15,right=0.86)
        line_segments = LineCollection([list(zip(self.a.R,fehr[i])) for i in np.arange(nz)],
                                       linewidths=self.lw['main'],cmap = colorbar,                        
                                       norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)                                                                  
                                       )
        dz = np.mean(np.diff(zlim))
        line_segments.set_array(np.add(zlim,dz/2)[:-1]/1e3)
        im = ax.add_collection(line_segments)
        if mode_comp=='dt' or mode_comp=='tot':
            if 'tab' in kwargs:
                del kwargs_calc['tab']
                fehrt = metr('t',[zlim[0],zlim[-1]],self.p,self.a,**kwargs_calc,
                             tab=kwargs['tab'][1])
            else:
                fehrt = metr('t',[zlim[0],zlim[-1]],self.p,self.a,**kwargs_calc)
            ax.plot(self.a.R,fehrt,ls='--',color='orange',lw=2,label='$\mathrm{Thick \ disk}$')
        ymin, ymax = self._dynamic_ylim_lin_(fehr,[-1.4,1],0.2)
        ax.set_ylim((ymin,ymax))      
        ax.plot([self.p.Rsun,self.p.Rsun],[ymin,ymax],ls='--',lw=self.lw['secondary'],
                     c='darkgrey',label=self.labs['r0'])
        plt.legend(prop={'size':self.fnt['secondary']},loc=1,ncol=1)
        pos = ax.get_position()
        cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
        cbar = f.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_ticks(np.arange(zlim[0]+dz/2,zlim[-1]+dz/2,dz)/1e3)
        cbar.set_label(self.axts['zkpc'],fontsize=self.fnt['main'])
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs_calc)
            ts.metr_save(fehr,mode_comp,zlim)
        self._figclose_(this_function,**kwargs) 
        
        return (f, ax) 

  
    def rhor_monoage_plt(self,mode_comp,zlim,ages,**kwargs):
        """
        Radial densiy profiles of the mono-age subpopulations. 
        Plotting for :func:`jjmodel.analysis.rhor_monoage`. 
        
        :param mode_comp: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
        :type mode_comp: str
        :param zlim: Range of heights [*zmin,zmax*], pc. 
        :type zlim: array-like 
        :param ages: Set of age bins, Gyr. 
        :type ages: array-like
        :param sigma: Optional. If True, the result is surface density in :math:`\mathrm{M_\odot \ pc^{-2}}`, 
            otherwise the midplane mass density in :math:`\mathrm{M_\odot \ pc^{-3}}` is calculated. 
            In combination with **number** = True, returns the *number* surface density in :math:`\mathrm{number \ pc^{-2}}`.
        :type sigma: boolean
        :param between: Optional. If True, the output quantity corresponds to the age intervals 
            specified by parameter **ages**. Otherwise the individual single mono-age subpopulations 
            are returned (i.e., age-bins of width ``tr``, model age resolution).
        :type between: boolean
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'tot'``, **tab** must be [[*table_d,table_t,table_sh*]_ *rmin*,...,[*table_d,table_t,table_sh*]_ *rmax*], 
            where *rmin* and *rmax* are ``p.Rmin`` and ``p.Rmax``. 
        :type tab: list[astropy table] or list[list[astropy table]] 
        :param number: Optional. If True, calculated quantity is the spatial number density of stars in :math:`\mathrm{number \ pc^{-3}}`,
             not matter density in :math:`\mathrm{M_\odot \ pc^{-3}}`. Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str 
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :type cbar_bins: boolean
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.rhor_monoage_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.  
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','mode_pop',
                                'cbar_bins','sigma','between','number','tab','mode_iso'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                           'radial mono-age profiles',this_function)
        ages = inpcheck_age(ages,this_function)
        zlim = inpcheck_height(zlim,self.p,this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'
        ln += '$\mathrm{\ at \ |z|=[}$' + str(zlim[0]/1e3) + ',' + str(zlim[1]/1e3) + '$\mathrm{] \ kpc}$'
        
        nage = len(ages)
        if inpcheck_iskwargtype(kwargs,'between',True,bool,this_function):
            nage = len(ages) - 1 
            
        colorbar = self._figcolorbar_('tau',len(ages),**kwargs)
        cmin, cmax = self._cbarminmax_(0,tp,np.mean(np.diff(ages)),**kwargs)
                                       
        kwargs_calc = reduce_kwargs(kwargs,['save','sigma','between','tab','mode_pop','number','mode_iso'])
        rhor = rhor_monoage(mode_comp,zlim,ages,self.p,self.a,**kwargs_calc)
        
        ylabel = self.axts['rho']
        if inpcheck_iskwargtype(kwargs,'sigma',True,bool,this_function):
            ylabel = self.axts['sigma']
        if ('mode_pop' in kwargs) or ('tab' in kwargs):
            if inpcheck_iskwargtype(kwargs,'sigma',True,bool,this_function):
                ylabel = self.axts['sigma']
                if inpcheck_iskwargtype(kwargs,'number',True,bool,this_function):
                    ylabel = '$\mathrm{N, \ pc^{-2}}$'                   
            else:
                if inpcheck_iskwargtype(kwargs,'number',True,bool,this_function):
                    ylabel = '$\mathrm{N, \ pc^{-3}}$'
       
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_xlim((self.p.Rmin,self.p.Rmax))        
        ymin, ymax = self._dynamic_ylim_log_(rhor,[1e-12,1.05])
        ax.set_ylim(ymin,ymax)                     
        ax.set_yscale('log')
        ax.set_xlabel(self.axts['r'],fontsize=self.fnt['main'])
        ax.set_ylabel(ylabel,fontsize=self.fnt['main'])
        ax.set_title(ln,fontsize=self.fnt['main'],pad=10)
        line_segments = LineCollection([list(zip(self.a.R,rhor[i])) for i in np.arange(nage)],
                                       linewidths=self.lw['main'],cmap = colorbar,                        
                                       norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)
                                       )
        line_segments.set_array(np.array(ages))
        im = ax.add_collection(line_segments)
        ax.plot([self.p.Rsun,self.p.Rsun],[ymin,ymax],ls='--',lw=self.lw['secondary'],
                c='darkgrey',label=self.labs['r0'])
        plt.legend(prop={'size':self.fnt['secondary']},loc=1,ncol=1)
        f.subplots_adjust(left=0.14,top=0.86,bottom=0.15,right=0.86)
        pos = ax.get_position()
        cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
        cbar = f.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_ticks(self.xt['ticks'])
        cbar.set_ticklabels(self.xt['labels'])
        cbar.set_label(self.axts['tau'],fontsize=self.fnt['main'])
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs_calc)
            ts.rhor_monoage_save(rhor,mode_comp,zlim,ages)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax) 


    def rhor_monomet_plt(self,mode_comp,zlim,mets,**kwargs):
        """
        Radial densiy profiles of the mono-metallicity subpopulations. 
        Plotting for :func:`jjmodel.analysis.rhor_monomet`. 
        
        :param mode_comp: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
        :type mode_comp: str
        :param zlim: Range of heights [*zmin,zmax*], pc. 
        :type zlim: array-like 
        :param mets: Set of metallicity bins, dex. 
        :type mets: array-like
        :param sigma: Optional. If True, the result is surface density in :math:`\mathrm{M_\odot \ pc^{-2}}`, 
            otherwise the midplane mass density in :math:`\mathrm{M_\odot \ pc^{-3}}` is calculated. 
            In combination with **number** = True, returns the *number* surface density in :math:`\mathrm{number \ pc^{-2}}`.
        :type sigma: boolean
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'tot'``, **tab** must be [[*table_d,table_t,table_sh*]_ *rmin*,...,[*table_d,table_t,table_sh*]_ *rmax*], 
            where *rmin* and *rmax* are ``p.Rmin`` and ``p.Rmax``. 
        :type tab: list[astropy table] or list[list[astropy table]] 
        :param number: Optional. If True, calculated quantity is the spatial number density of stars in :math:`\mathrm{number \ pc^{-3}}`,
             not matter density in :math:`\mathrm{M_\odot \ pc^{-3}}`. Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str 
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :type cbar_bins: boolean
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.rhor_monomet_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.  
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins',
                                'sigma','mode_pop','tab','number','mode_iso'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                           'radial mono-metallicity profiles',this_function)   
        zlim = inpcheck_height(zlim,self.p,this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'
        ln += '$\mathrm{\ at \ |z|=[}$' + str(zlim[0]/1e3) + ',' + str(zlim[1]/1e3) + '$\mathrm{] \ kpc}$'

        nmet = len(mets) - 1     
        colorbar = self._figcolorbar_('feh',len(mets),**kwargs)
        cmin, cmax = self._cbarminmax_(round(np.amin(mets),2),round(np.amax(mets),2),
                                       np.mean(np.diff(mets)),**kwargs)
                    
        kwargs_calc = reduce_kwargs(kwargs,['save','sigma','mode_pop','tab','number','mode_iso'])
        rhomet = rhor_monomet(mode_comp,zlim,mets,self.p,self.a,**kwargs_calc)
        
        ylabel = self.axts['rho']
        if inpcheck_iskwargtype(kwargs,'sigma',True,bool,this_function):
            ylabel = self.axts['sigma']
        if ('mode_pop' in kwargs) or ('tab' in kwargs):
            if inpcheck_iskwargtype(kwargs,'sigma',True,bool,this_function):
                ylabel = self.axts['sigma']
                if inpcheck_iskwargtype(kwargs,'number',True,bool,this_function):
                    ylabel = '$\mathrm{N, \ pc^{-2}}$'                   
            else:
                if inpcheck_iskwargtype(kwargs,'number',True,bool,this_function):
                    ylabel = '$\mathrm{N, \ pc^{-3}}$'
        
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_xlim((self.p.Rmin,self.p.Rmax))
        ymin, ymax = self._dynamic_ylim_log_(rhomet,[1e-12,1.05])
        ax.set_ylim(ymin,ymax)                     
        ax.set_yscale('log')
        ax.set_xlabel(self.axts['r'],fontsize=self.fnt['main'])
        ax.set_ylabel(ylabel,fontsize=self.fnt['main'])
        ax.set_title(ln,fontsize=self.fnt['main'])
        line_segments = LineCollection([list(zip(self.a.R,rhomet[i])) for i in np.arange(nmet)],
                                       linewidths=self.lw['main'],cmap = colorbar,                        
                                       norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)                                                                   
                                       )
        line_segments.set_array(np.array(mets))
        im = ax.add_collection(line_segments)
        ax.plot([self.p.Rsun,self.p.Rsun],[ymin,ymax],ls='--',lw=self.lw['secondary'],
                c='darkgrey',label=self.labs['r0'])
        plt.legend(prop={'size':self.fnt['secondary']},loc=1,ncol=1)
        f.subplots_adjust(left=0.14,top=0.86,bottom=0.15,right=0.86)
        pos = ax.get_position()
        cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
        cbar = f.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_label(self.axts['fe'],fontsize=self.fnt['main'])
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs_calc)
            ts.rhor_monomet_save(rhomet,mode_comp,zlim,mets)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax) 


    def agehist_plt(self,mode_comp,zlim,**kwargs):
        """
        Age distribution (normalized on area) as a function of height and Galactocentric distance. 
        Plotting for :func:`jjmodel.analysis.agehist`. 
        
        :param mode_comp: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
        :type mode_comp: str
        :param zlim: Range of heights [*zmin,zmax*], pc. 
        :type zlim: array-like 
        :param cumulative: Optional. If True, the normalized cumulative age distributions will be plotted. 
        :type cumulative: boolean
        :param sigma_gauss: Optional. Standard deviation of the Gaussian kernel used to smooth
            the age distributions, Gyr.
        :type sigma_gauss: scalar
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'tot'``, **tab** must be [[*table_d,table_t,table_sh*]_ *rmin*,...,[*table_d,table_t,table_sh*]_ *rmax*], 
            where *rmin* and *rmax* are ``p.Rmin`` and ``p.Rmax``. 
        :type tab: list[astropy table] or list[list[astropy table]] 
        :param number: Optional. If True, calculated quantity is weighted the spatial number density of stars (:math:`\mathrm{number \ pc^{-3}}`),
             not by matter density (:math:`\mathrm{M_\odot \ pc^{-3}}`). Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set.  
        :type mode_iso: str 
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :type cbar_bins: boolean
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.agehist_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.  
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins','cumulative',
                                'mode_pop','tab','sigma_gauss','number','mode_iso'],this_function)
        zlim = inpcheck_height(zlim,self.p,this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                           'age distributions',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'
        ln += '\n'+'$\mathrm{at \ |z|=[}$' + str(zlim[0]/1e3) +\
              ',' + str(zlim[1]/1e3) + '$\mathrm{] \ kpc}$'

        colorbar = self._figcolorbar_('r',self.a.Rbins,**kwargs)
        cmin, cmax = self._cbarminmax_(self.p.Rmin,self.p.Rmax,self.p.dR,**kwargs)
        
        kwargs_calc = reduce_kwargs(kwargs,['save','mode_pop','tab','sigma_gauss','number','mode_iso'])
        ages, ages0 = agehist(mode_comp,zlim,self.p,self.a,**kwargs_calc)
        
        if inpcheck_iskwargtype(kwargs,'cumulative',True,bool,this_function):
            ages = [np.cumsum(i) for i in ages]
            ages = [i/i[-1] for i in ages]
            ages0 = np.cumsum(ages0)
            ages0 = ages0/ages0[-1]
        
        f, ax = plt.subplots(figsize=(10,7))
        ax.set_xlim((0,tp))
        if inpcheck_iskwargtype(kwargs,'cumulative',True,bool,this_function):
            ax.set_ylim((0,1.01))
            ax.plot([0,tp],[1,1],lw=0.5,ls='--',c='grey')
        else:
            ymax = round(np.amax(ages),3) + (2 - round(1000*np.amax(ages),0)%2)*0.001
            ax.set_ylim((0,ymax))
        ax.set_xticks(self.xt['ticks'])
        ax.set_xticklabels(self.xt['labels'])
        ax.set_xlabel(self.axts['t'],fontsize=self.fnt['main'],labelpad=15)
        ax.set_ylabel(r'$\mathrm{f \, (\tau)}$',fontsize=self.fnt['main'],labelpad=15)
        axx = ax.twiny()
        axx.set_xticks(self.xt['ticks'])
        axx.set_xticklabels(self.xt['labels'][::-1])
        axx.set_xlabel(self.axts['tau'],fontsize=self.fnt['main'],labelpad=15)
        if ((('mode_pop' in kwargs) and 
            (kwargs['mode_pop']=='gdw' or kwargs['mode_pop']=='kdw' or kwargs['mode_pop']=='rc') and 
            inpcheck_iskwargtype(kwargs,'cumulative',True,bool,this_function)) or 'tab' in kwargs):
            ax.text(0.55,0.05,ln,transform=ax.transAxes,fontsize=self.fnt['secondary'])
        else:
            ax.text(0.25,0.9,ln,transform=ax.transAxes,fontsize=self.fnt['secondary'])
        line_segments = LineCollection([list(zip(self.a.t,ages[i])) for i in self.a.R_array],
                                       linewidths=self.lw['main'],cmap = colorbar,
                                       norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)                                                                   
                                       )                                      
        line_segments.set_array(self.a.R)
        im = ax.add_collection(line_segments)
        ax.plot(self.a.t,ages0,ls='--',color='k',lw=self.lw['secondary'],label=self.labs['r0'])
        ax.legend(loc='upper left',prop={'size':self.fnt['secondary']})
        f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.86)
        pos = ax.get_position()
        cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
        cbar = f.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_ticks(np.arange(self.a.R[0],self.a.R[-1]+1))
        cbar.set_label(self.axts['r'],fontsize=self.fnt['main'])
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            if 'save_format' in kwargs: 
                del kwargs['save_format']
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs)
            ts.agehist_save((ages,ages0),mode_comp,zlim)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)


    def agehist_rsun_plt(self,mode_comp,zlim,**kwargs):
        """
        Age distribution (normalized on area) as a function of height for the Solar neighbourhood, ``p.Rsun``.
        Plotting for :func:`jjmodel.analysis.agehist`. 
        
        :param mode_comp: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
        :type mode_comp: str
        :param zlim: Range of heights [*zmin,zmax*], pc. Or set of z-bin edges, pc. 
        :type zlim: array-like 
        :param cumulative: Optional. If True, the normalized cumulative age distributions will be plotted. 
        :type cumulative: boolean
        :param sigma_gauss: Optional. Standard deviation of the Gaussian kernel used to smooth
            the age distributions, Gyr.
        :type sigma_gauss: scalar
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'tot'``, **tab** must be [*table_d,table_t,table_sh*] with tables for ``p.Rsun``.  
        :type tab: astropy table or list[astropy table] 
        :param number: Optional. If True, calculated quantity is weighted the spatial number density of stars (:math:`\mathrm{number \ pc^{-3}}`),
             not by matter density (:math:`\mathrm{M_\odot \ pc^{-3}}`). Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set.  
        :type mode_iso: str 
        :param cbar: Optional. Matplotlib colormap name. Needed only when **zlim** is not a single 
            slice, but a set of z-bins. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.agehist_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins','cumulative',
                                'mode_pop','tab','sigma_gauss','number','mode_iso'],this_function)
        zlim = inpcheck_height(zlim,self.p,this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                           'age distributions',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'
        ln += '\n'+'$\mathrm{at \ |z|=[}$' + str(zlim[0]/1e3) + ',' +\
              str(zlim[-1]/1e3) + '$\mathrm{] \ kpc}$'
        
        if len(zlim)>2:
            nz = int(len(zlim) - 1)
            colorbar = self._figcolorbar_('z',nz,**kwargs)
            cmin, cmax = 0, self.p.zmax/1e3
        
        kwargs_calc = reduce_kwargs(kwargs,['save','mode_pop','tab','sigma_gauss','number','mode_iso'])
        if len(zlim)>2: 
            ages0 = np.zeros((nz,self.a.jd))
            for i in range(nz):
                ages0[i] = agehist(mode_comp,[zlim[i],zlim[i+1]],self.p,self.a,**kwargs_calc,
                                   R=self.p.Rsun)[0]
        else:
            ages0, sum_rho0 = agehist(mode_comp,zlim,self.p,self.a,**kwargs_calc,R=self.p.Rsun)
            if mode_comp=='dt' or mode_comp=='tot':
                if 'tab' in kwargs:
                    del kwargs_calc['tab']
                    agesd0, sum_rhod0 = agehist('d',zlim,self.p,self.a,**kwargs_calc,
                                                R=self.p.Rsun,tab=kwargs['tab'][0])
                    agest0, sum_rhot0 = agehist('t',zlim,self.p,self.a,**kwargs_calc,
                                                R=self.p.Rsun,tab=kwargs['tab'][1])
                else:
                    agesd0, sum_rhod0 = agehist('d',zlim,self.p,self.a,**kwargs_calc,R=self.p.Rsun)                                                
                    agest0, sum_rhot0 = agehist('t',zlim,self.p,self.a,**kwargs_calc,R=self.p.Rsun)
                                                
                if mode_comp=='tot':
                    if 'tab' in kwargs:
                        agessh0, sum_rhosh0 = agehist('sh',zlim,self.p,self.a,**kwargs_calc,
                                                      R=self.p.Rsun,tab=kwargs['tab'][2])
                    else:
                        agessh0, sum_rhosh0 = agehist('sh',zlim,self.p,self.a,**kwargs_calc,
                                                      R=self.p.Rsun)
        if len(zlim)>2: 
            if inpcheck_iskwargtype(kwargs,'cumulative',True,bool,this_function):
                ages0 = [np.cumsum(i) for i in ages0]
                ages0 = [i/i[-1] for i in ages0]
        else:
            if inpcheck_iskwargtype(kwargs,'cumulative',True,bool,this_function):
                ages0 = np.cumsum(ages0*sum_rho0)
                agemax = ages0[-1]
                if mode_comp=='dt' or mode_comp=='tot':
                    agesd0 = np.cumsum(agesd0*sum_rhod0)
                    agest0 = np.cumsum(agest0*sum_rhot0)
                    if mode_comp=='tot':
                        agessh0 = np.cumsum(agessh0*sum_rhosh0)
            else:
                ages0 = ages0*sum_rho0
                agemax = sum_rho0
                if mode_comp=='dt' or mode_comp=='tot':
                    agesd0 = agesd0*sum_rhod0
                    agest0 = agest0*sum_rhot0
                    if mode_comp=='tot':
                        agessh0 = agessh0*sum_rhosh0
            ages0 = ages0/agemax
            if mode_comp=='dt' or mode_comp=='tot':
                agesd0 = agesd0/agemax
                agest0 = agest0/agemax  
                if mode_comp=='tot':
                    agessh0 = agessh0/agemax               
                
        f, ax = plt.subplots(figsize=(10,7))
        ax.set_xlim((0,tp))
        if inpcheck_iskwargtype(kwargs,'cumulative',True,bool,this_function):
            ax.set_ylim((0,1.01))
            ax.plot([0,tp],[1,1],lw=0.5,ls='--',c='grey')
        else:
            ymax = round(np.amax(ages0),3) + (2 - round(1000*np.amax(ages0),0)%2)*0.001
            ax.set_ylim((0,ymax))
        ax.set_xticks(self.xt['ticks'])
        ax.set_xticklabels(self.xt['labels'])
        ax.set_xlabel(self.axts['t'],fontsize=self.fnt['main'],labelpad=15)
        ax.set_ylabel(r'$\mathrm{f \, (\tau)}$',fontsize=self.fnt['main'],labelpad=15)
        axx = ax.twiny()
        axx.set_xticks(self.xt['ticks'])
        axx.set_xticklabels(self.xt['labels'][::-1])
        axx.set_xlabel(self.axts['tau'],fontsize=self.fnt['main'],labelpad=15)
        # if (('mode_pop' in kwargs) and 
        #     (kwargs['mode_pop']=='gdw' or kwargs['mode_pop']=='kdw' or kwargs['mode_pop']=='rc') and 
        if inpcheck_iskwargtype(kwargs,'cumulative',True,bool,this_function):
            ax.text(0.55,0.05,ln,transform=ax.transAxes,fontsize=self.fnt['secondary'])
        else:
            ax.text(0.3,0.9,ln,transform=ax.transAxes,fontsize=self.fnt['secondary'])
        
        if len(zlim)>2: 
            f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.86)
            line_segments = LineCollection([list(zip(self.a.t,ages0[i])) for i in np.arange(nz)],
                                           linewidths=self.lw['main'],cmap = colorbar,
                                           norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)                                                                   
                                           )   
            dz = np.mean(np.diff(zlim))
            line_segments.set_array(np.add(zlim,dz/2)[:-1]/1e3)                                   
            im = ax.add_collection(line_segments)
            pos = ax.get_position()
            cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
            cbar = f.colorbar(im,cax=cax,orientation='vertical')
            cbar.set_ticks(np.arange(zlim[0]+dz/2,zlim[-1]+dz/2,dz)/1e3)
            cbar.set_label(self.axts['zkpc'],fontsize=self.fnt['main'])
        else:
            f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.91)
            if mode_comp=='dt' or mode_comp=='tot':
                ax.plot(self.a.t,agesd0,color=self.cols['d'],lw=self.lw['main'],label=self.labs['d'])
                ax.plot(self.a.t,agest0,color=self.cols['t'],lw=self.lw['main'],label=self.labs['t'])
                if mode_comp=='tot':
                    ax.plot(self.a.t,agessh0,color=self.cols['sh'],lw=self.lw['main'],label=self.labs['sh'])  
                ax.plot(self.a.t,ages0,ls='--',color=self.cols[mode_comp],lw=self.lw['main'],
                        label=self.labs[mode_comp])
            else:
                ax.plot(self.a.t,ages0,color=self.cols[mode_comp],lw=self.lw['main'],
                        label=self.labs[mode_comp])
            if mode_comp=='dt' or mode_comp=='tot':
                if inpcheck_iskwargtype(kwargs,'cumulative',True,bool,this_function):
                    ax.legend(loc='upper left',prop={'size':self.fnt['secondary']})
                else:
                    ax.legend(loc='upper right',prop={'size':self.fnt['secondary']})
                        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            if 'save_format' in kwargs:
                del kwargs['save_format']
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs,R=self.p.Rsun)
            ts.agehist_save(ages0,mode_comp,zlim)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)


    def methist_plt(self,mode_comp,zlim,**kwargs):
        """     
        Metallicity distribution (normalized on area) as a function of height and Galactocentric distance. 
        Plotting for :func:`jjmodel.analysis.methist`. 
        
        :param mode_comp: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
        :type mode_comp: str
        :param zlim: Range of heights [*zmin,zmax*], pc. 
        :type zlim: array-like 
        :param metbins: Optional. Set of metallicity bins. If not given, the grid is -1.1,...,0.8 dex 
            with 0.05 dex step. 
        :type metbins: array-like
        :param cumulative: Optional. If True, the normalized cumulative metallicity distributions will be plotted. 
        :type cumulative: boolean
        :param sigma_gauss: Optional. Standard deviation of the Gaussian kernel used to smooth
            the metallicity distributions, dex.
        :type sigma_gauss: scalar
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'tot'``, **tab** must be [[*table_d,table_t,table_sh*]_ *rmin*,...,[*table_d,table_t,table_sh*]_ *rmax*], 
            where *rmin* and *rmax* are ``p.Rmin`` and ``p.Rmax``. 
        :type tab: list[astropy table] or list[list[astropy table]] 
        :param number: Optional. If True, calculated quantity is weighted by the spatial number density of stars (:math:`\mathrm{number \ pc^{-3}}`),
             not by matter density (:math:`\mathrm{M_\odot \ pc^{-3}}`). Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set.  
        :type mode_iso: str 
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.methist_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean
        
        :return: Figure and axes.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins','cumulative',
                                'sigma_gauss','metbins','mode_pop','tab','number','mode_iso'],
                        this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                           'metallicity distributions',this_function)
        zlim = inpcheck_height(zlim,self.p,this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'
        ln += '$\mathrm{\ at \ |z|=[}$' + str(zlim[0]/1e3) + ',' + str(zlim[-1]/1e3) + '$\mathrm{] \ kpc}$'

        colorbar = self._figcolorbar_('r',self.a.Rbins,**kwargs)
        cmin, cmax = self._cbarminmax_(self.p.Rmin,self.p.Rmax,self.p.dR,**kwargs) 

        mmin, mmax, dmet = -1.1, 0.8, 0.05
        metbins = np.arange(mmin,mmax+dmet,dmet)
        if 'metbins' in kwargs:
            metbins = kwargs['metbins']
            mmin, mmax, dmet = metbins[0], metbins[-1], np.mean(np.diff(metbins))           
        metbinsc = np.add(metbins,dmet/2)[:-1]
        
        kwargs_calc = reduce_kwargs(kwargs,['save','mode_pop','tab','sigma_gauss',
                                            'number','metbins','mode_iso'])
        metdist, metdist0 = methist(mode_comp,zlim,self.p,self.a,**kwargs_calc)

        if inpcheck_iskwargtype(kwargs,'cumulative',True,bool,this_function):
            metdist = [np.cumsum(i) for i in metdist]
            metdist = [i/i[-1] for i in metdist]
            metdist0 = np.cumsum(metdist0)
            metdist0 = metdist0/metdist0[-1]
            
        f, ax = plt.subplots(figsize=(10,7))
        ax.set_xlim((mmin,mmax))
        if inpcheck_iskwargtype(kwargs,'cumulative',True,bool,this_function):
            ax.set_ylim((0,1.01))
            ax.plot([0,tp],[1,1],lw=0.5,ls='--',c='grey')
        else:
            ymax = 0.1*(np.nanmax(metdist)//0.1) + 0.2
            ax.set_ylim((0,ymax))
        ax.set_xlabel(self.axts['fe'],fontsize=self.fnt['main'],labelpad=15)
        ax.set_ylabel(r'$\mathrm{f \, ([Fe/H])}$',fontsize=self.fnt['main'],labelpad=15)
        ax.set_title(ln,fontsize=self.fnt['main'],pad=10)
        f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.86)
        line_segments = LineCollection([list(zip(metbinsc,metdist[i])) for i in self.a.R_array],
                                       linewidths=self.lw['main'],cmap = colorbar,
                                       norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)                                                                   
                                       )                                      
        line_segments.set_array(self.a.R)
        im = ax.add_collection(line_segments)
        ax.plot(metbinsc,metdist0,ls='--',color='k',lw=self.lw['secondary'],label=self.labs['r0'])
        ax.legend(loc=2,prop={'size':self.fnt['secondary']})
        pos = ax.get_position()
        cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
        cbar = f.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_ticks(np.arange(self.a.R[0],self.a.R[-1]+1))
        cbar.set_label(self.axts['r'],fontsize=self.fnt['main'])
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs)
            ts.methist_save((metdist,metdist0,metbinsc),mode_comp,zlim)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)


    def methist_rsun_plt(self,mode_comp,zlim,**kwargs):
        """
        Metallicity distribution (normalized on area) as a function of height 
        for the Solar neighbourhood, p. Rsun. Plotting for jjmodel.analysis.methist. 
        
        :param mode_comp: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
        :type mode_comp: str
        :param zlim: Range of heights [*zmin,zmax*], pc. Or set of z-bin edges, pc. 
        :type zlim: array-like 
        :param metbins: Optional. Set of metallicity bins. If not given, the grid is -1.1,...,0.8 dex 
            with 0.05 dex step. 
        :type metbins: array-like
        :param cumulative: Optional. If True, the normalized cumulative age distributions will be plotted. 
        :type cumulative: boolean
        :param sigma_gauss: Optional. Standard deviation of the Gaussian kernel used to smooth
            the metallicity distributions, dex.
        :type sigma_gauss: scalar
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'tot'``, **tab** must be [*table_d,table_t,table_sh*] with tables for ``p.Rsun``.  
        :type tab: astropy table or list[astropy table] 
        :param number: Optional. If True, calculated quantity is weighted by the spatial number density of stars (:math:`\mathrm{number \ pc^{-3}}`),
             not by matter density (:math:`\mathrm{M_\odot \ pc^{-3}}`). Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set.  
        :type mode_iso: str 
        :param cbar: Optional. Matplotlib colormap name. Needed only when **zlim** is not a single 
            slice, but a set of z-bins. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.methist_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins','cumulative',
                                'sigma_gauss','metbins','mode_pop','tab','number','mode_iso'],
                        this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                           'metallicity distributions',this_function)
        zlim = inpcheck_height(zlim,self.p,this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'
        ln += '$\mathrm{\ at \ |z|=[}$' + str(zlim[0]/1e3) + ',' + str(zlim[-1]/1e3) + '$\mathrm{] \ kpc}$'

        if len(zlim)>2:
            nz = int(len(zlim) - 1)
            colorbar = self._figcolorbar_('z',nz,**kwargs)
            cmin, cmax = self._cbarminmax_(0,self.p.zmax/1e3,np.mean(np.diff(zlim)),**kwargs)

        mmin, mmax, dmet = -1.1, 0.8, 0.05
        metbins = np.arange(mmin,mmax+dmet,dmet)
        if 'metbins' in kwargs:
            metbins = kwargs['metbins']
            mmin, mmax, dmet = metbins[0], metbins[-1], np.mean(np.diff(metbins))           
        metbinsc = np.add(metbins,dmet/2)[:-1]
        
        kwargs_calc = reduce_kwargs(kwargs,['save','mode_pop','tab','sigma_gauss',
                                            'number','metbins','mode_iso'])
        if len(zlim)>2: 
            metdist0 = np.zeros((nz,len(metbinsc)))
            for i in range(nz):
                metdist0[i] = methist(mode_comp,[zlim[i],zlim[i+1]],self.p,self.a,**kwargs_calc,
                                      R=self.p.Rsun)[0]
        else:
            metdist0, sum_rho0 = methist(mode_comp,zlim,self.p,self.a,**kwargs_calc,R=self.p.Rsun)
            if mode_comp=='dt' or mode_comp=='tot':
                if 'tab' in kwargs:
                    del kwargs_calc['tab']
                    metdistd0, sum_rhod0 = methist('d',zlim,self.p,self.a,**kwargs_calc,
                                                   R=self.p.Rsun,tab=kwargs['tab'][0])
                    metdistt0, sum_rhot0 = methist('t',zlim,self.p,self.a,**kwargs_calc,
                                                   R=self.p.Rsun,tab=kwargs['tab'][1])
                else:
                    metdistd0, sum_rhod0 = methist('d',zlim,self.p,self.a,**kwargs_calc,R=self.p.Rsun)
                    metdistt0, sum_rhot0 = methist('t',zlim,self.p,self.a,**kwargs_calc,R=self.p.Rsun)
                if mode_comp=='tot':
                    if 'tab' in kwargs:
                        metdistsh0, sum_rhosh0 = methist('sh',zlim,self.p,self.a,**kwargs_calc,
                                                         R=self.p.Rsun,tab=kwargs['tab'][2])
                    else:
                        metdistsh0, sum_rhosh0 = methist('sh',zlim,self.p,self.a,**kwargs_calc,
                                                         R=self.p.Rsun)        
        if len(zlim)>2: 
            if inpcheck_iskwargtype(kwargs,'cumulative',True,bool,this_function):
                metdist0 = [np.cumsum(i) for i in metdist0]
                metdist0 = [i/i[-1] for i in metdist0]
        else:
            if inpcheck_iskwargtype(kwargs,'cumulative',True,bool,this_function):
                metdist0 = np.cumsum(metdist0*sum_rho0)
                metmax = metdist0[-1]
                if mode_comp=='dt' or mode_comp=='tot':
                    metdistd0 = np.cumsum(metdistd0*sum_rhod0)
                    metdistt0 = np.cumsum(metdistt0*sum_rhot0)
                    if mode_comp=='tot':
                        metdistsh0 = np.cumsum(metdistsh0*sum_rhosh0)
            else:
                metdist0 = metdist0*sum_rho0
                metmax = sum_rho0
                if mode_comp=='dt' or mode_comp=='tot':
                    metdistd0 = metdistd0*sum_rhod0
                    metdistt0 = metdistt0*sum_rhot0
                    if mode_comp=='tot':
                        metdistsh0 = metdistsh0*sum_rhosh0
            metdist0 = metdist0/metmax
            if mode_comp=='dt' or mode_comp=='tot':
                metdistd0 = metdistd0/metmax
                metdistt0 = metdistt0/metmax  
                if mode_comp=='tot':
                    metdistsh0 = metdistsh0/metmax          
        
        f, ax = plt.subplots(figsize=(10,7))
        ax.set_xlim((mmin,mmax))
        if inpcheck_iskwargtype(kwargs,'cumulative',True,bool,this_function):
            ax.set_ylim((0,1.01))
            ax.plot([mmin,mmax],[1,1],lw=0.5,ls='--',c='grey')
        else:
            ymax = 0.1*(np.amax(metdist0)//0.1) + 0.2
            ax.set_ylim((0,ymax))
        ax.set_xlabel(self.axts['fe'],fontsize=self.fnt['main'],labelpad=15)
        ax.set_ylabel(r'$\mathrm{f \, ([Fe/H])}$',fontsize=self.fnt['main'],labelpad=15)
        ax.set_title(ln,fontsize=self.fnt['main'],pad=10)
        if len(zlim)>2: 
            f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.86)
            line_segments = LineCollection([list(zip(metbinsc,metdist0[i])) for i in np.arange(nz)],
                                           linewidths=self.lw['main'],cmap = colorbar,
                                           norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)                                                                   
                                           )                                      
            dz = np.mean(np.diff(zlim))
            line_segments.set_array(np.add(zlim,dz/2)[:-1]/1e3) 
            im = ax.add_collection(line_segments)
            pos = ax.get_position()
            cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
            cbar = f.colorbar(im,cax=cax,orientation='vertical')
            cbar.set_ticks(np.arange(zlim[0]/1e3,(zlim[-1]+dz)/1e3,dz/1e3))
            cbar.set_label(self.axts['zkpc'],fontsize=self.fnt['main'])
        else:
            f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.91)
            if mode_comp=='dt' or mode_comp=='tot':
                ax.plot(metbinsc,metdistd0,color=self.cols['d'],lw=self.lw['main'],label=self.labs['d'])
                ax.plot(metbinsc,metdistt0,color=self.cols['t'],lw=self.lw['main'],label=self.labs['t'])
                if mode_comp=='tot':
                    ax.plot(metbinsc,metdistsh0,color=self.cols['sh'],lw=self.lw['main'],
                            label=self.labs['sh'])
                ax.plot(metbinsc,metdist0,ls='--',color=self.cols[mode_comp],lw=self.lw['main'],
                        label=self.labs[mode_comp])
            else:
                ax.plot(metbinsc,metdist0,color=self.cols[mode_comp],lw=self.lw['main'],
                        label=self.labs[mode_comp])
            if mode_comp=='dt' or mode_comp=='tot':
                if inpcheck_iskwargtype(kwargs,'cumulative',True,bool,this_function):
                    ax.legend(loc='upper left',prop={'size':self.fnt['secondary']})
                else:
                    ax.legend(loc='upper right',prop={'size':self.fnt['secondary']})
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            if 'save_format' in kwargs:
                del kwargs['save_format']
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs,R=self.p.Rsun)
            ts.methist_save((metdist0,metbinsc),mode_comp,zlim)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)


    def h_plt(self,**kwargs):
        """
        MW thin-disk's scale height as a function of time (and age).
        
        :param cbar: Optional. Matplotlib colomap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :param save: Optional. If True, the figure will be saved. 
            The output directory is ``a.T['heightplt']``. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean
        
        :return: Figure and axes. 
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins'],this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
        colorbar = self._figcolorbar_('r',self.a.Rbins,**kwargs)
        cmin, cmax = self._cbarminmax_(self.p.Rmin,self.p.Rmax,self.p.dR,**kwargs)
        
        Hd, Hd0 = tab_reader(['Hd','Hd0'],self.p,self.a.T)
        if self.p.pkey==1:
            Hdp = [tab_reader(['Hdp'],self.p,self.a.T,R=radius)[0] for radius in self.a.R]
            Hdp = [table[1] for table in Hdp]
            npeak = len(self.p.sigp)
        
        f, ax = plt.subplots(figsize=(10,7))
        ax.set_xlim((0,tp))
        hmax = np.amax(Hd[1:,])
        if self.p.pkey==1:
            hmax = np.amax([np.amax(Hd[1:,]),np.amax(Hdp)])
        ymax = 100*round(hmax/100,0) + 100*(2 - round(np.amax(Hd[1:,])/100,0)%2)
        ax.set_ylim((0,ymax))
        ax.set_xticks(self.xt['ticks'])
        ax.set_xticklabels(self.xt['labels'])
        ax.set_xlabel(self.axts['t'],fontsize=self.fnt['main'],labelpad=15)
        ax.set_ylabel(self.axts['h'],fontsize=self.fnt['main'],labelpad=15)
        axx = ax.twiny()
        axx.set_xticks(self.xt['ticks'])
        axx.set_xticklabels(self.xt['labels'][::-1])
        axx.set_xlabel(self.axts['tau'],fontsize=self.fnt['main'],labelpad=15)
        ax.text(0.075,0.1,self.labs['d'],transform=ax.transAxes,fontsize=self.fnt['secondary'])
        line_segments = LineCollection([list(zip(Hd[0], Hd[i+1])) for i in self.a.R_array],
                                       linewidths=self.lw['main'],cmap = colorbar,
                                       norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)                                                                   
                                       )                                      
        line_segments.set_array(self.a.R)
        im = ax.add_collection(line_segments)
        ax.plot(Hd0[0],Hd0[1],ls='--',lw=self.lw['secondary'],color='k',label=self.labs['r0'])
        if self.p.pkey==1:
            for i in range(npeak):
                ax.scatter(np.linspace(self.p.tpk[i],self.p.tpk[i],self.a.Rbins),
                           [table[i] for table in Hdp],c=self.a.R,cmap=colorbar,marker='x',s=20,
                           zorder=30)
        ax.legend(loc=3,prop={'size':self.fnt['secondary']})
        f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.86)
        pos = ax.get_position()
        cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
        cbar = f.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_ticks(np.arange(self.a.R[0],self.a.R[-1]+1))
        cbar.set_label(self.axts['r'],fontsize=self.fnt['main'])
        
        format_ = self._figformat_(**kwargs)
        figname = os.path.join(self.a.T['heightplt'],''.join(('Htau_d.',str(format_))))
        self._figsave_(figname,this_function,**kwargs)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)         


    def h_rsun_plt(self,**kwargs):
        """
        MW thin-disk's scale height as a function of time (and age)
        for the Solar neighbourhood, at ``p.Rsun``.
        
        :param save: Optional. If True, the figure will be saved. 
            The output directory is ``a.T['heightplt']``. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean
        
        :return: Figure and axes.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close'],this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
                
        Hd0 = tab_reader(['Hd0'],self.p,self.a.T)[0]
        if self.p.pkey==1:
            Hdp = tab_reader(['Hdp0'],self.p,self.a.T)[0][1]
            npeak = len(self.p.sigp)
        
        f, ax = plt.subplots(figsize=(10,7))
        ax.set_xlim((0,tp))
        hmax = np.amax(Hd0[1])
        if self.p.pkey==1:
            hmax = np.amax([np.amax(Hd0[1]),np.amax(Hdp)])
        ymax = 100*round(hmax/100,0) + 100*(2 - round(np.amax(Hd0[1])/100,0)%2)
        ax.set_ylim((0,ymax))
        ax.set_xticks(self.xt['ticks'])
        ax.set_xticklabels(self.xt['labels'])
        ax.set_xlabel(self.axts['t'],fontsize=self.fnt['main'],labelpad=15)
        ax.set_ylabel(self.axts['h'],fontsize=self.fnt['main'],labelpad=15)
        axx = ax.twiny()
        axx.set_xticks(self.xt['ticks'])
        axx.set_xticklabels(self.xt['labels'][::-1])
        axx.set_xlabel(self.axts['tau'],fontsize=self.fnt['main'],labelpad=15)
        ax.plot(Hd0[0],Hd0[1],lw=self.lw['main'],color=self.cols['d'],label=self.labs['d'])
        if self.p.pkey==1:
            for i in range(npeak):
                if i==0:
                    ax.scatter(self.p.tpk[i],Hdp[i],c='k',marker='x',s=40,
                               label='$\mathrm{Special \ subpopulations}$')
                else:
                    ax.scatter(self.p.tpk[i],Hdp[i],c='k',marker='x',s=40)                           
            ax.legend(loc=3,prop={'size':self.fnt['secondary']})
        f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.91)
                
        format_ = self._figformat_(**kwargs)
        figname = os.path.join(self.a.T['heightplt'],
                               ''.join(('Htau_d_R',str(self.p.Rsun),'.',str(format_))))
        self._figsave_(figname,this_function,**kwargs)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)         


    def hr_monoage_plt(self,mode_comp,ages,**kwargs):
        """
        Scale heights of the disk's mono-age subpopulations as a function of Galactocentric distance.
        Plotting for :func:`jjmodel.analysis.hr_monoage`. 
        
        :param mode_comp: Galactic component, can be ``'d'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thin+thick disk, or total). 
        :type mode_comp: str
        :param ages: Set of age bins, Gyr. 
        :type ages: array-like
        :param between: Optional. If True, the output quantity corresponds to the age intervals 
            specified by parameter **ages**. Otherwise the individual single mono-age subpopulations 
            are returned (i.e., age-bins of width ``tr``, model age resolution).
        :type between: boolean
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'tot'``, **tab** must be [[*table_d,table_t,table_sh*]_ *rmin*,...,[*table_d,table_t,table_sh*]_ *rmax*], 
            where *rmin* and *rmax* are ``p.Rmin`` and ``p.Rmax``
        :type tab: list[astropy table] or list[list[astropy table]] 
        :param number: Optional. If True, calculated quantity is weighted by the spatial number density of stars (:math:`\mathrm{number \ pc^{-3}}`),
             not matter density (:math:`\mathrm{M_\odot \ pc^{-3}}`). Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str 
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.hr_monoage_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins','mode_pop','tab',
                                'between','number','mode_iso'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','dt','tot'],'scale heights',this_function)
        ages = inpcheck_age(ages,this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'

        colorbar = self._figcolorbar_('tau',self.a.Rbins,**kwargs)
        cmin, cmax = self._cbarminmax_(ages[0],ages[-1],np.mean(np.diff(ages)),**kwargs)
        
        Heffd, Ht, Hsh = tab_reader(['Heffd','Ht','Hsh'],self.p,self.a.T)
        kwargs_calc = reduce_kwargs(kwargs,['save','mode_pop','tab','between','number','mode_iso'])
        H = hr_monoage(mode_comp,ages,self.p,self.a,**kwargs_calc)
            
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_xlim((self.p.Rmin,self.p.Rmax))
        ymin, ymax = self._dynamic_ylim_lin_(H,[0,self.p.zmax],50)
        if mode_comp=='d' and np.amax(Heffd[1]) > ymax:
            ymax = 1.05*np.amax(Heffd[1])
        if mode_comp=='dt' and np.amax(Ht[1]) > ymax:
            ymax = 1.05*np.amax(Ht[1])
        if mode_comp=='tot' and np.amax(Hsh[1]) > ymax:
            ymax = 1.05*np.amax(Hsh[1])
        ax.set_ylim(0,ymax)                     
        ax.set_xlabel(self.axts['r'],fontsize=self.fnt['main'])
        ax.set_ylabel(self.axts['h'],fontsize=self.fnt['main'])
        ax.set_title(ln,fontsize=self.fnt['main'])
        line_segments = LineCollection([list(zip(self.a.R,H[:,i])) for i in np.arange(H.shape[1])],
                                       linewidths=self.lw['main'],cmap = colorbar,                        
                                       norm = mpl.colors.Normalize(vmin=0,vmax=tp)
                                       )
        line_segments.set_array(np.array(ages))
        im = ax.add_collection(line_segments)
        ax.plot(self.a.R,Heffd[1],ls='--',lw=self.lw['main'],c=self.cols['d'],
                label='$\mathrm{Thin}$'+'-'+'$\mathrm{disk \ half}$'+'-'+'$\mathrm{thickness}$')
        if mode_comp=='dt' or mode_comp=='tot':
            ax.plot(self.a.R,Ht[1],lw=self.lw['main'],ls='--',
                    c=self.cols['t'],label='$\mathrm{Thick}$'+'-'+'$\mathrm{disk \ scale \ height}$')
        if mode_comp=='tot':
            ax.plot(self.a.R,Hsh[1],lw=self.lw['main'],ls='--',c=self.cols['sh'],
                    label='$\mathrm{Halo \ scale \ height}$')
        f.subplots_adjust(left=0.14,top=0.86,bottom=0.15,right=0.86)
        ax.legend(loc=4,ncol=2,prop={'size':self.fnt['secondary']})
        pos = ax.get_position()
        cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
        cbar = f.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_ticks(self.xt['ticks'])
        cbar.set_ticklabels(self.xt['labels'])
        cbar.set_label(self.axts['tau'],fontsize=self.fnt['main'])
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs_calc)
            ts.hr_monoage_save(H,mode_comp,ages)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)
    
    
    def hr_monomet_plt(self,mode_comp,mets,**kwargs):
        """
        Scale heights of the disk's mono-metallicity sub-populations as a function of Galactocentric distance.
        Plotting for :func:`jjmodel.analysis.hr_monomet`. 
        
        :param mode_comp: Galactic component, can be ``'d'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thin+thick disk, or total). 
        :type mode_comp: str
        :param mets: Set of metallicity bins. 
        :type mets: array-like
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'tot'``, **tab** must be [[*table_d,table_t,table_sh*]_ *rmin*,...,[*table_d,table_t,table_sh*]_ *rmax*], 
            where *rmin* and *rmax* are ``p.Rmin`` and ``p.Rmax``
        :type tab: list[astropy table] or list[list[astropy table]] 
        :param number: Optional. If True, calculated quantity is weighted by the spatial number density of stars (:math:`\mathrm{number \ pc^{-3}}`),
             not matter density (:math:`\mathrm{M_\odot \ pc^{-3}}`). Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str 
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.hr_monomet_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins',
                                'mode_pop','tab','number','mode_iso'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','dt','tot'],
                                           'mono-metallicity scale heights',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'
            
        colorbar = self._figcolorbar_('feh',self.a.Rbins,**kwargs)
        cmin, cmax = self._cbarminmax_(mets[0],mets[-1],np.mean(np.diff(mets)),**kwargs)
        
        kwargs_calc = reduce_kwargs(kwargs,['save','mode_pop','tab','number','mode_iso'])
        H = hr_monomet(mode_comp,mets,self.p,self.a,**kwargs_calc)
        
        Heffd, Ht, Hsh = tab_reader(['Heffd','Ht','Hsh'],self.p,self.a.T)
        
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_xlim((self.p.Rmin,self.p.Rmax))
        ymin, ymax = self._dynamic_ylim_lin_(H,[0,self.p.zmax],50)
        if mode_comp=='d' and np.amax(Heffd[1]) > ymax:
            ymax = 1.05*np.amax(Heffd[1])
        if mode_comp=='dt' and np.amax(Ht[1]) > ymax:
            ymax = 1.05*np.amax(Ht[1])
        if mode_comp=='tot' and np.amax(Hsh[1]) > ymax:
            ymax = 1.05*np.amax(Hsh[1])
        ax.set_ylim((0,ymax))                     
        ax.set_xlabel(self.axts['r'],fontsize=self.fnt['main'])
        ax.set_ylabel(self.axts['h'],fontsize=self.fnt['main'])
        ax.set_title(ln,fontsize=self.fnt['main'])
        line_segments = LineCollection([list(zip(self.a.R,H[:,i])) for i in np.arange(len(mets)-1)],
                                       linewidths=self.lw['main'],cmap = colorbar,                        
                                       norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)                                                                   
                                       )
        line_segments.set_array(np.array(mets))
        im = ax.add_collection(line_segments)
        ax.plot(self.a.R,Heffd[1],ls='--',lw=self.lw['main'],c=self.cols['d'],
                label='$\mathrm{Thin}$'+'-'+'$\mathrm{disk \ half}$'+'-'+'$\mathrm{thickness}$')
        if mode_comp=='dt' or mode_comp=='tot':
            ax.plot(self.a.R,Ht[1],lw=self.lw['main'],ls='--',
                    c=self.cols['t'],label='$\mathrm{Thick}$'+'-'+'$\mathrm{disk \ scale \ height}$')
        if mode_comp=='tot':
            ax.plot(self.a.R,Hsh[1],lw=self.lw['main'],ls='--',
                    c=self.cols['sh'],label='$\mathrm{Halo \ scale \ height}$')
        f.subplots_adjust(left=0.14,top=0.86,bottom=0.15,right=0.86)
        ax.legend(loc=3,ncol=2,prop={'size':self.fnt['secondary']})
        pos = ax.get_position()
        cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
        cbar = f.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_label(self.axts['fe'],fontsize=self.fnt['main'])
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs_calc)
            ts.hr_monomet_save(H,mode_comp,mets)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax) 


    def hr_gas_plt(self,**kwargs):
        """
        Scale heights of molecular and atomic gas as functions of Galactocentric distance.
        
        :param save: Optional. If True, the figure will be saved. 
            The output directory is ``a.T['inpplt']``. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close'],this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
        H2, HI, Hg1, Hg2 = tab_reader(['H2','HI','Hg1','Hg2'],self.p,self.a.T)
        
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_xlim((self.p.Rmin,self.p.Rmax))
        indr = np.where(abs(HI[0]-self.p.Rmax)==np.amin(abs(HI[0]-self.p.Rmax)))[0][0]
        ymax = 50*round(np.amax(HI[1][:indr]/2+HI[2][:indr]/2)/50,0) + 50 
        ax.set_ylim(0,ymax)                     
        ax.set_xlabel(self.axts['r'],fontsize=self.fnt['main'])
        ax.set_ylabel(self.axts['h'],fontsize=self.fnt['main'])
        ax.errorbar(H2[0],H2[1]/2,yerr=H2[2]/2,ls='none',marker='o',markersize=5,
                    color='b',label='$\mathrm{H_2 \ from \ Nakanishi \ and \ Sofue \ (2016)}$')
        ax.errorbar(HI[0],HI[1]/2,yerr=HI[2]/2,ls='none',marker='o',markersize=5,
                    color='darkgreen',label='$\mathrm{HI \ from \ Nakanishi \ and \ Sofue \ (2016)}$')
        ax.plot(Hg1[0],Hg1[1],lw=self.lw['main'],color=self.cols['g1'],label=self.labs['g1'])
        ax.plot(Hg2[0],Hg2[1],lw=self.lw['main'],color=self.cols['g2'],label=self.labs['g2'])
        f.subplots_adjust(left=0.14,top=0.86,bottom=0.15,right=0.86)
        ax.legend(loc=2,prop={'size':self.fnt['secondary']})
        
        format_ = self._figformat_(**kwargs)
        figname = os.path.join(self.a.T['inpplt'],''.join(('HR_g.',format_)))
        self._figsave_(figname,this_function,**kwargs)
        self._figclose_(this_function,**kwargs) 
        
        return (f, ax) 
    
    
    def avr_plt(self,**kwargs):
        """
        Plots age - W-velocity dispersion relation (AVR).
        
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :param save: Optional. If true, the figure will be saved. 
            The output directory is ``a.T['kinplt']``. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins'],this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
        colorbar = self._figcolorbar_('r',self.a.Rbins,**kwargs)
        
        AVRd, AVRd0 = tab_reader(['AVR','AVR0'],self.p,self.a.T)
        cmin, cmax = self._cbarminmax_(self.p.Rmin,self.p.Rmax,self.p.dR,**kwargs)
        
        if self.p.pkey==1:
            Hdp = [tab_reader(['Hdp'],self.p,self.a.T,R=radius)[0] for radius in self.a.R]
            sigp = [table[0] for table in Hdp]
            npeak = len(self.p.sigp)
            
        f, ax = plt.subplots(figsize=(10,7))
        ax.set_xlim(0,tp)
        ymin, ymax = self._dynamic_ylim_lin_(AVRd[1:,],[0,100],10)
        if self.p.pkey==1 and ymax < np.amax(sigp):
            ymax = 10*round(np.amax(sigp)/10,0) + 10
        ax.set_ylim(0,ymax)
        ax.set_xticks(self.xt['ticks'])
        ax.set_xticklabels(self.xt['labels'])
        ax.set_xlabel(self.axts['t'],fontsize=self.fnt['main'],labelpad=15)
        ax.set_ylabel(self.axts['sigw'],fontsize=self.fnt['main'],labelpad=15)
        axx = ax.twiny()
        axx.set_xticks(self.xt['ticks'])
        axx.set_xticklabels(self.xt['labels'][::-1])
        axx.set_xlabel(self.axts['tau'],fontsize=self.fnt['main'],labelpad=15)
        line_segments = LineCollection([list(zip(AVRd[0], AVRd[i+1])) for i in self.a.R_array],
                                       linewidths=self.lw['main'],cmap = colorbar,
                                       norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)                                                                  
                                       )                                      
        line_segments.set_array(self.a.R)
        im = ax.add_collection(line_segments)
        ax.plot(AVRd0[0],AVRd0[1],ls='--',lw=self.lw['secondary'],color='k',label=self.labs['r0'])
        if self.p.pkey==1:
            for i in range(npeak):
                ax.scatter(np.linspace(self.p.tpk[i],self.p.tpk[i],self.a.Rbins),
                            [table[i] for table in sigp],c=self.a.R,cmap=colorbar,marker='x',s=20,
                            zorder=30)
        ax.legend(loc=1,prop={'size':self.fnt['secondary']})
        f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.86)
        pos = ax.get_position()
        cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
        cbar = f.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_ticks(np.arange(self.a.R[0],self.a.R[-1]+1))
        cbar.set_label(self.axts['r'],fontsize=self.fnt['main'])
        
        format_ = self._figformat_(**kwargs)
        figname = os.path.join(self.a.T['kinplt'],''.join(('AVR.',str(format_))))
        self._figsave_(figname,this_function,**kwargs)
        self._figclose_(this_function,**kwargs) 
        
        return (f, ax)         


    def avr_rsun_plt(self,**kwargs):
        """
        Plots age - W-velocity dispersion relation (AVR) 
        for the Solar neighbourhood, at ``p.Rsun``.
        
        :param save: Optional. If True, the figure will be saved. 
            The output directory is ``a.T['kinplt']``. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close'],this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
                
        AVRd0 = tab_reader(['AVR0'],self.p,self.a.T)[0]
            
        f, ax = plt.subplots(figsize=(10,7))
        ax.set_xlim(0,tp)
        ymin, ymax = self._dynamic_ylim_1d_lin_(AVRd0[1],[0,100],10)
        if self.p.pkey==1 and ymax < np.amax(self.p.sigp):
            ymax = 10*round(np.amax(self.p.sigp)/10,0) + 10
        ax.set_ylim(0,ymax)
        ax.set_xticks(self.xt['ticks'])
        ax.set_xticklabels(self.xt['labels'])
        ax.set_xlabel(self.axts['t'],fontsize=self.fnt['main'],labelpad=15)
        ax.set_ylabel(self.axts['sigw'],fontsize=self.fnt['main'],labelpad=15)
        axx = ax.twiny()
        axx.set_xticks(self.xt['ticks'])
        axx.set_xticklabels(self.xt['labels'][::-1])
        axx.set_xlabel(self.axts['tau'],fontsize=self.fnt['main'],labelpad=15)
        ax.plot(AVRd0[0],AVRd0[1],lw=self.lw['main'],color=self.cols['d'],label=self.labs['d'])
        if self.p.pkey==1:
            for i in range(len(self.p.sigp)):
                if i==0:
                    ax.scatter(self.p.tpk[i],self.p.sigp[i],c='k',marker='x',s=40,
                               label='$\mathrm{Special \ subpopulations}$')
                else:
                    ax.scatter(self.p.tpk[i],self.p.sigp[i],c='k',marker='x',s=40)        
            ax.legend(loc=3,prop={'size':self.fnt['secondary']})
        f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.91)
        
        format_ = self._figformat_(**kwargs)
        figname = os.path.join(self.a.T['kinplt'],''.join(('AVR_R',str(self.p.Rsun),
                                                           '.',str(format_))))
        self._figsave_(figname,this_function,**kwargs)
        self._figclose_(this_function,**kwargs) 
        
        return (f, ax)         

    
    def sigwr_thick_plt(self,**kwargs):
        """
        Thick-disk W-velocity dispersion as a function of Galactocentric distance.
        
        :param save: Optional. If True, the figure will be saved. 
            The output directory is ``a.T['kinplt']``. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close'],this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
                
        Sigt = tab_reader(['Sigt'],self.p,self.a.T)[0]
        
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_xlim((self.p.Rmin,self.p.Rmax))
        ax.set_ylim((0,10*round(np.amax(Sigt[1])/10,0)+10))                 
        ax.set_xlabel(self.axts['r'],fontsize=self.fnt['main'])
        ax.set_ylabel(self.axts['sigw'],fontsize=self.fnt['main'])
        ax.set_title(self.labs['t'],fontsize=self.fnt['main'])
        ax.plot(Sigt[0],Sigt[1],lw=self.lw['main'],color=self.cols['t'],label=self.labs['t'])
        f.subplots_adjust(left=0.14,top=0.86,bottom=0.15,right=0.86)
        
        format_ = self._figformat_(**kwargs)
        figname = os.path.join(self.a.T['kinplt'],''.join(('SigWR_t.',format_)))
        self._figsave_(figname,this_function,**kwargs)
        self._figclose_(this_function,**kwargs) 

        return (f, ax) 
    
    
    def sigwz_plt(self,mode_comp,**kwargs):
        """
        W-velocity dispersion as a function of height z and Galactocentric distance. 
        Plotting for :func:`jjmodel.analysis.sigwz`. 
        
        :param mode_comp: Galactic component, can be ``'d'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thin+thick disk, or total). 
        :type mode_comp: str
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'tot'``, **tab** must be [[*table_d,table_t,table_sh*]_ *rmin*,...,[*table_d,table_t,table_sh*]_ *rmax*], 
            where *rmin* and *rmax* are ``p.Rmin`` and ``p.Rmax``. 
        :type tab: list[astropy table] or list[list[astropy table]] 
        :param number: Optional. If True, calculated quantity is weighted by the spatial number density of stars (:math:`\mathrm{number \ pc^{-3}}`),
             not by matter density (:math:`\mathrm{M_\odot \ pc^{-3}}`). Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str 
        :param cbar: Optional. Matplotlib colomap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.sigwz_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes. 
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins',
                                'mode_pop','tab','number','mode_iso'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','dt','tot'],
                                           'W-velocity dispersion',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'
        
        colorbar = self._figcolorbar_('r',self.a.Rbins,**kwargs)
        cmin, cmax = self._cbarminmax_(self.p.Rmin,self.p.Rmax,self.p.dR,**kwargs)

        kwargs_calc = reduce_kwargs(kwargs,['save','mode_pop','tab','number','mode_iso'])
        sigwzr, sigwzr0 = sigwz(mode_comp,self.p,self.a,**kwargs_calc)
        
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_xlim((0,self.p.zmax))
        ymin, ymax = self._dynamic_ylim_lin_(sigwzr,[0,100],10)
        ax.set_ylim(0,ymax)                     
        ax.set_xlabel(self.axts['z'],fontsize=self.fnt['main'])
        ax.set_ylabel(self.axts['sigw'],fontsize=self.fnt['main'])
        ax.set_title(ln,fontsize=self.fnt['main'],pad=10)
        f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.86)
        line_segments = LineCollection([list(zip(self.a.z,sigwzr[i])) for i in self.a.R_array],
                                       linewidths=self.lw['main'],cmap = colorbar,                        
                                       norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)                                                                  
                                       )
        line_segments.set_array(self.a.R)
        im = ax.add_collection(line_segments)
        ax.plot(self.a.z,sigwzr0,ls='--',color='k',
                lw=self.lw['secondary'],label=self.labs['r0'])
        plt.legend(loc=4,prop={'size':self.fnt['secondary']})
        pos = ax.get_position()
        cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
        cbar = f.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_ticks(np.arange(self.a.R[0],self.a.R[-1]+1))
        cbar.set_label(self.axts['r'],fontsize=self.fnt['main'])
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs_calc)
            ts.sigwz_save((sigwzr,sigwzr0),mode_comp)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax) 


    def sigwz_rsun_plt(self,mode_comp,**kwargs):
        """
        W-velocity dispersion as a function of height z for the Solar neighbourhood, at ``p.Rsun``. 
        Plotting for :func:`jjmodel.analysis.sigwz`. 
        
        :param mode_comp: Galactic component, can be ``'d'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thin+thick disk, or total). 
        :type mode_comp: str
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'tot'``, **tab** must be [*table_d,table_t,table_sh*] with tables for ``p.Rsun``. 
        :type tab: astropy table or list[astropy table] 
        :param number: Optional. If True, calculated quantity is weighted by the spatial number density of stars (:math:`\mathrm{number \ pc^{-3}}`),
             not by matter density (:math:`\mathrm{M_\odot \ pc^{-3}}`). Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str 
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.sigwz_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes. 
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close',
                                'mode_pop','tab','number','mode_iso'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','dt','tot'],
                                           'W-velocity dispersion',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'
        
        kwargs_calc = reduce_kwargs(kwargs,['save','mode_pop','tab','number','mode_iso'])
        sigwzr0 = sigwz(mode_comp,self.p,self.a,**kwargs_calc,R=self.p.Rsun)
        
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_xlim((0,self.p.zmax))
        ymin, ymax = self._dynamic_ylim_1d_lin_(sigwzr0,[0,100],10)
        ax.set_ylim(0,ymax)                     
        ax.set_xlabel(self.axts['z'],fontsize=self.fnt['main'])
        ax.set_ylabel(self.axts['sigw'],fontsize=self.fnt['main'])
        ax.set_title(ln,fontsize=self.fnt['main'],pad=10)
        f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.91)
        ax.plot(self.a.z,sigwzr0,color=self.cols[mode_comp],lw=self.lw['main'])
                        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs_calc,R=self.p.Rsun)
            ts.sigwz_save(sigwzr0,mode_comp)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax) 


    def sigwr_plt(self,mode_comp,zlim,**kwargs):
        """
        W-velocity dispersion as a function of Galactocentric distance.
        Plotting for :func:`jjmodel.analysis.sigwr`. 
        
        :param mode_comp: Galactic component, can be ``'d'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thin+thick disk, or total). 
        :type mode_comp: str
        :param zlim: Range of heights [*zmin,zmax*], pc. Or set of z-bin edges, pc. 
        :type zlim: array-like 
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'tot'``, **tab** must be [[*table_d,table_t,table_sh*]_ *rmin*,...,[*table_d,table_t,table_sh*]_ *rmax*], 
            where *rmin* and *rmax* are ``p.Rmin`` and ``p.Rmax``
        :type tab: list[astropy table] or list[list[astropy table]] 
        :param number: Optional. If True, calculated quantity is weighted by the spatial number density of stars (:math:`\mathrm{number \ pc^{-3}}`),
             not by matter density (:math:`\mathrm{M_\odot \ pc^{-3}}`). Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str 
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.sigwr_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes. 
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins',
                                'mode_pop','number','tab','mode_iso'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','dt','tot'],
                                           'W-velocity dispersion',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'
            
        nz = len(zlim)                        
        kwargs_calc = reduce_kwargs(kwargs,['save','mode_pop','number','tab','mode_iso'])
        if nz > 2:
            colorbar = self._figcolorbar_('z',nz,**kwargs)
            cmin, cmax = self._cbarminmax_(zlim[0]/1e3,zlim[-1]/1e3,np.mean(np.diff(zlim))/1e3,**kwargs)
            '''
            sigw_r = np.zeros((nz-1,self.a.Rbins))
            for i in range(nz-1):
                sigw_r[i] = sigwr(mode_comp,[zlim[i],zlim[i+1]],self.p,self.a,**kwargs_calc)
            ymin, ymax = self._dynamic_ylim_log_(sigw_r,[1,120])
            '''
        sigw_r = sigwr(mode_comp,zlim,self.p,self.a,**kwargs_calc).T
        ymin, ymax = self._dynamic_ylim_1d_lin_(sigw_r,[5,120],10)
                             
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_xlim((self.p.Rmin,self.p.Rmax))            
        if mode_comp=='dt' or mode_comp=='tot':
            Sigt = tab_reader(['Sigt'],self.p,self.a.T)[0]
            if ymax < np.amax(Sigt[1]):
                ymax = 5*np.amax(Sigt[1])//5 + 5
        if mode_comp=='tot':
            if ymax <= np.amax(self.p.sigsh):
                ymax = 5*(self.p.sigsh//5 + 2)
        ax.set_ylim((ymin,ymax))
        ax.set_yscale('log')           
        ax.set_xlabel(self.axts['r'],fontsize=self.fnt['main'])
        ax.set_ylabel(self.axts['sigw'],fontsize=self.fnt['main'])
        f.subplots_adjust(left=0.14,top=0.86,bottom=0.15,right=0.86)
                 
        if nz > 2:
            ax.set_title(ln,fontsize=self.fnt['main'],pad=10)
            line_segments = LineCollection([list(zip(self.a.R,sigw_r[i])) for i in np.arange(nz-1)],
                                           linewidths=self.lw['main'],cmap = colorbar,                        
                                           norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)                                                                       
                                           )
            line_segments.set_array(np.add(zlim[:-1]/1e3,np.diff(zlim)/2e3))
            im = ax.add_collection(line_segments)
            pos = ax.get_position()
            cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
            cbar = f.colorbar(im,cax=cax,orientation='vertical')
            cbar.set_ticks(np.arange(0,(self.p.zmax+0.2)/1e3,0.2))
            cbar.set_label(self.axts['zkpc'],fontsize=self.fnt['main'])
        else: 
            ax.set_title(ln+'$\mathrm{ \ at \ |z|=[}$'+str(round(zlim[0],0)/1e3)+','+\
                         str(round(zlim[1],0)/1e3)+'$\mathrm{] \ kpc}$',fontsize=self.fnt['main'],pad=10)                                
            ax.plot(self.a.R,sigw_r,lw=self.lw['main'],color=self.cols[mode_comp])  
        if mode_comp=='dt' or mode_comp=='tot':
            ax.plot(self.a.R,Sigt[1],ls='--',lw=self.lw['main'],c=self.cols['t'],label=self.labs['t'])
        if mode_comp=='tot':
            ax.plot(self.a.R,[self.p.sigsh for i in self.a.R],ls='--',lw=self.lw['main'],
                    c=self.cols['sh'],label=self.labs['sh'])
        ax.plot([self.p.Rsun,self.p.Rsun],[ymin,ymax],ls='--',lw=self.lw['secondary'],
                     c='darkgrey',label=self.labs['r0'])      
        ax.legend(loc=3,prop={'size':self.fnt['secondary']})
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs_calc)
            ts.sigwr_save(sigw_r,mode_comp,zlim)
        self._figclose_(this_function,**kwargs) 
        
        return (f, ax) 
    

    def sigwr_monoage_plt(self,mode_comp,zlim,ages,**kwargs):
        """
        W-velocity dispersion of mono-age sub-populations as a function of Galactocentric distance.
        Plotting for :func:`jjmodel.analysis.sigwr_monoage`. 
        
        :param mode_comp:  Galactic component, can be ``'d'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thin+thick disk, or total). 
        :type mode_comp: str
        :param zlim: Range of heights [*zmin,zmax*], pc. Or set of z-bin edges, pc. 
        :type zlim: array-like 
        :param ages: Set of age bins, Gyr. 
        :type ages: array-like
        :param between: Optional. If True, the output quantity corresponds to the age intervals 
            specified by parameter **ages**. Otherwise the individual single mono-age subpopulations 
            are returned (i.e., age-bins of width ``tr``, model age resolution).
        :type between: boolean
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'tot'``, **tab** must be [[*table_d,table_t,table_sh*]_ *rmin*,...,[*table_d,table_t,table_sh*]_ *rmax*], 
            where *rmin* and *rmax* are ``p.Rmin`` and ``p.Rmax``
        :type tab: list[astropy table] or list[list[astropy table]] 
        :param number: Optional. If True, calculated quantity is weighted by the spatial number density of stars (:math:`\mathrm{number \ pc^{-3}}`),
             not matter density (:math:`\mathrm{M_\odot \ pc^{-3}}`). Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str 
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.sigwr_monoage_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes. 
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins',
                                'mode_pop','tab','number','between','mode_iso'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','dt','tot'],
                                           'W-velocity dispersion',this_function)
        ages = inpcheck_age(ages,this_function)
        zlim = inpcheck_height(zlim,self.p,this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'
        ln += '$\mathrm{ \ at \ |z|=[}$'+str(round(zlim[0],0)/1e3)+','+\
                         str(round(zlim[1],0)/1e3)+'$\mathrm{] \ kpc}$'
        nage = len(ages)
        if inpcheck_iskwargtype(kwargs,'between',True,bool,this_function):
            nage = int(len(ages) - 1)
                   
        colorbar = self._figcolorbar_('tau',len(ages),**kwargs)
        cmin, cmax = self._cbarminmax_(0,tp,np.mean(np.diff(ages)),**kwargs)
        
        kwargs_calc = reduce_kwargs(kwargs,['save','mode_pop','tab','number','between','mode_iso'])
        sigw_r = sigwr_monoage(mode_comp,zlim,ages,self.p,self.a,**kwargs_calc)
        
        ymin, ymax = self._dynamic_ylim_lin_(sigw_r,[1,200],10)
        if mode_comp=='dt' or mode_comp=='tot':
            Sigt = tab_reader(['Sigt'],self.p,self.a.T)[0]
            if ymin >= np.amin(Sigt[1]):
                ymin = 10*(np.amin(Sigt[1])//10 - 1) 
            if ymax <= np.amax(Sigt[1]):
                ymax = 10*(np.amax(Sigt[1])//10 + 2)
        if mode_comp=='tot':
            if ymax <= np.amax(self.p.sigsh):
                ymax = 10*(self.p.sigsh//10 + 2)
            
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_xlim((self.p.Rmin,self.p.Rmax))
        ax.set_ylim(ymin,ymax)                     
        ax.set_yscale('log')
        ax.set_xlabel(self.axts['r'],fontsize=self.fnt['main'])
        ax.set_ylabel(self.axts['sigw'],fontsize=self.fnt['main'])
        ax.set_title(ln,fontsize=self.fnt['main'])
        line_segments = LineCollection([list(zip(self.a.R,sigw_r[:,i])) for i in np.arange(nage)],
                                       linewidths=self.lw['main'],cmap = colorbar,                        
                                       norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)
                                       )
        line_segments.set_array(np.array(ages))
        im = ax.add_collection(line_segments)
        if mode_comp=='dt' or mode_comp=='tot':
            ax.plot(self.a.R,Sigt[1],ls='--',lw=self.lw['main'],c=self.cols['t'],label=self.labs['t'])
        if mode_comp=='tot':
            ax.plot(self.a.R,[self.p.sigsh for i in self.a.R],ls='--',lw=self.lw['main'],
                    c=self.cols['sh'],label=self.labs['sh'])
        ax.plot([self.p.Rsun,self.p.Rsun],[ymin,ymax],ls='--',lw=self.lw['secondary'],
                c='darkgrey',label=self.labs['r0'])   
        f.subplots_adjust(left=0.14,top=0.86,bottom=0.15,right=0.86)
        plt.legend(loc=3,ncol=1,prop={'size':self.fnt['secondary']})
        pos = ax.get_position()
        cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
        cbar = f.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_ticks(self.xt['ticks'])
        cbar.set_ticklabels(self.xt['labels'])
        cbar.set_label(self.axts['tau'],fontsize=self.fnt['main'])
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs_calc)
            ts.sigwr_monoage_save(sigw_r,mode_comp,zlim,ages)
        self._figclose_(this_function,**kwargs)

        return (f, ax)    


    def sigwr_monomet_plt(self,mode_comp,zlim,mets,**kwargs):
        """
        W-velocity dispersion of mono-metallicity subpopulations as a function of Galactocentric distance.
        Plotting for :func:`jjmodel.analysis.sigwr_monomet`. 

        :param mode_comp: Galactic component, can be ``'d'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thin+thick disk, or total). 
        :type mode_comp: str
        :param zlim: Range of heights [*zmin,zmax*], pc. Or set of z-bin edges, pc. 
        :type zlim: array-like 
        :param mets: Set of metallicity bins. 
        :type mets: array-like
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'tot'``, **tab** must be [[*table_d,table_t,table_sh*]_ *rmin*,...,[*table_d,table_t,table_sh*]_ *rmax*], 
            where *rmin* and *rmax* are ``p.Rmin`` and ``p.Rmax``
        :type tab: list[astropy table] or list[list[astropy table]] 
        :param number: Optional. If True, calculated quantity is weighted by the spatial number density of stars (:math:`\mathrm{number \ pc^{-3}}`),
             not matter density (:math:`\mathrm{M_\odot \ pc^{-3}}`). Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str 
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :param save: Optional. If True, the figure will be saved. The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.sigwr_monomet_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes. 
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins',
                                'mode_pop','tab','number','mode_iso'],
                        this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','dt','tot'],
                                           'W-velocity dispersion',this_function)
        zlim = inpcheck_height(zlim,self.p,this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'
        ln += '$\mathrm{ \ at \ |z|=[}$'+str(round(zlim[0],0)/1e3)+','+\
                         str(round(zlim[1],0)/1e3)+'$\mathrm{] \ kpc}$'

        nmet = int(len(mets)-1)
        colorbar = self._figcolorbar_('feh',len(mets),**kwargs)
        cmin, cmax = self._cbarminmax_(mets[0],mets[-1],np.mean(np.diff(mets)),**kwargs)
        
        kwargs_calc = reduce_kwargs(kwargs,['save','mode_pop','tab','number','mode_iso'])
        sigwmet = sigwr_monomet(mode_comp,zlim,mets,self.p,self.a,**kwargs_calc)
        
        ymin, ymax = self._dynamic_ylim_lin_(sigwmet,[1,200],10)
        if mode_comp=='dt' or mode_comp=='tot':
            Sigt = tab_reader(['Sigt'],self.p,self.a.T)[0]
            if ymin >= np.amin(Sigt[1]):
                ymin = 10*(np.amin(Sigt[1])//10 - 1) 
            if ymax <= np.amax(Sigt[1]):
                ymax = 10*(np.amin(Sigt[1])//10 + 2)
        if mode_comp=='tot':
            if ymax <= np.amax(self.p.sigsh):
                ymax = 10*(self.p.sigsh//10 + 2)  
                
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_xlim((self.p.Rmin,self.p.Rmax))
        ax.set_ylim((ymin,ymax))                
        ax.set_yscale('log')
        ax.set_xlabel(self.axts['r'],fontsize=self.fnt['main'])
        ax.set_ylabel(self.axts['sigw'],fontsize=self.fnt['main'])
        ax.set_title(ln,fontsize=self.fnt['main'])
        line_segments = LineCollection([list(zip(self.a.R,sigwmet[i])) for i in np.arange(nmet)],
                                       linewidths=self.lw['main'],cmap = colorbar,                        
                                       norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)                                                                   
                                       )
        line_segments.set_array(np.array(mets))
        im = ax.add_collection(line_segments)
        if mode_comp=='dt' or mode_comp=='tot':
            ax.plot(self.a.R,Sigt[1],lw=self.lw['main'],color=self.cols['t'],ls='--',label=self.labs['t'])
        if mode_comp=='tot':
            ax.plot(self.a.R,[self.p.sigsh for i in self.a.R],lw=self.lw['main'],
                    color=self.cols['sh'],ls='--',label=self.labs['sh'])
        ax.plot([self.p.Rsun,self.p.Rsun],[ymin,ymax],ls='--',lw=self.lw['secondary'],
                c='darkgrey',label=self.labs['r0'])   
        f.subplots_adjust(left=0.14,top=0.86,bottom=0.15,right=0.86)
        plt.legend(loc=3,ncol=1,prop={'size':self.fnt['secondary']})
        pos = ax.get_position()
        cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
        cbar = f.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_label(self.axts['fe'],fontsize=self.fnt['main'])
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs_calc)
            ts.sigwr_monomet_save(sigwmet,mode_comp,zlim,mets)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)  


    def sigwr_gas_plt(self,**kwargs):
        """
        W-velocity dispersion of the atomic and molecular gas as a function of Galactocentric distance.
        
        :param save: Optional. If True, the figure will be saved. 
            The output directory is ``a.T['kinplt']``. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close'],this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
        Sigg1, Sigg2 = tab_reader(['Sigg1','Sigg2'],self.p,self.a.T)
        
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_xlim((self.p.Rmin,self.p.Rmax))
        ymin, ymax = self._dynamic_ylim_1d_lin_(Sigg2[1],[0,30],10)
        ax.set_ylim(0,ymax)                 
        ax.set_xlabel(self.axts['r'],fontsize=self.fnt['main'])
        ax.set_ylabel(self.axts['sigw'],fontsize=self.fnt['main'])
        ax.set_yticks(np.arange(0,ymax+5,5))
        ax.plot(Sigg1[0],Sigg1[1],lw=self.lw['main'],color=self.cols['g1'],label=self.labs['g1'])
        ax.plot(Sigg2[0],Sigg2[1],lw=self.lw['main'],color=self.cols['g2'],label=self.labs['g2'])
        ax.plot([self.p.Rsun,self.p.Rsun],[0,ymax],ls='--',lw=self.lw['secondary'],
                c='darkgrey',label=self.labs['r0'])
        f.subplots_adjust(left=0.14,top=0.86,bottom=0.15,right=0.86)
        plt.legend(loc=1,prop={'size':self.fnt['secondary']})
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            figname = os.path.join(self.a.T['kinplt'],''.join(('SigWR_g.',format_)))
            self._figsave_(figname,this_function,**kwargs)
        self._figclose_(this_function,**kwargs) 

        return (f, ax) 
    
    
    def fi_plt(self,**kwargs):
        """
        Vertical gravitational potential as a function of Galactocentric distance. 
        Potential is normalized to ``SIGMA_E^2``. 

        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :param save: Optional. If True, the figure will be saved. 
            The output directory is ``a.T['fiplt']``. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins'],this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
        colorbar = self._figcolorbar_('r',self.a.Rbins,**kwargs)
        cmin, cmax = self._cbarminmax_(self.p.Rmin,self.p.Rmax,self.p.dR,**kwargs)
        
        Phi, Phi0 = tab_reader(['Phi','Phi0'],self.p,self.a.T)
        
        f, ax = plt.subplots(figsize=(10,7))
        ax.set_xlim((0,self.p.zmax))
        ax.set_ylim((0,round(np.amax(Phi[1]/SIGMA_E**2/KM**2),1)+0.2))
        ax.set_xlabel(self.axts['z'],fontsize=self.fnt['main'],labelpad=15)
        ax.set_ylabel(r'$\mathrm{\phi/\sigma_e^2}$',fontsize=self.fnt['main'],labelpad=15)
        line_segments = LineCollection([list(zip(Phi[0],Phi[i+1]/SIGMA_E**2/KM**2)) 
                                        for i in self.a.R_array],                                                                   
                                       linewidths=self.lw['main'],cmap = colorbar,
                                       norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)                                                                   
                                       )                                      
        line_segments.set_array(self.a.R)
        im = ax.add_collection(line_segments)
        ax.plot(Phi0[0],Phi0[1]/SIGMA_E**2/KM**2,ls='--',lw=self.lw['secondary'],
                color='k',label=self.labs['r0'])
        ax.legend(loc=2,prop={'size':self.fnt['secondary']})
        f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.86)
        pos = ax.get_position()
        cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
        cbar = f.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_ticks(np.arange(self.a.R[0],self.a.R[-1]+1))
        cbar.set_label(self.axts['r'],fontsize=self.fnt['main'])
        
        format_ = self._figformat_(**kwargs)
        figname = os.path.join(self.a.T['fiplt'],''.join(('Fiz.',str(format_))))
        self._figsave_(figname,this_function,**kwargs)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)     


    def fi_rsun_plt(self,**kwargs):
        """
        The normalized vertical gravitational potential  
        for the Solar neighbourhood, at ``p.Rsun``.

        :param save: Optional. If True, the figure will be saved. 
            The output directory is ``a.T['fiplt']``. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes. 
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close'],this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
        Phi0 = tab_reader(['Phi0'],self.p,self.a.T)[0]
        
        f, ax = plt.subplots(figsize=(10,7))
        ax.set_xlim((0,self.p.zmax))
        ax.set_ylim((0,round(np.amax(Phi0[1]/SIGMA_E**2/KM**2),1)+0.2))
        ax.set_xlabel(self.axts['z'],fontsize=self.fnt['main'],labelpad=15)
        ax.set_ylabel(r'$\mathrm{\phi/\sigma_e^2}$',fontsize=self.fnt['main'],labelpad=15)
        ax.plot(Phi0[0],Phi0[1]/SIGMA_E**2/KM**2,lw=self.lw['main'],color=self.cols['tot'])
                
        f.subplots_adjust(left=0.11,top=0.86,bottom=0.15,right=0.91)
        
        format_ = self._figformat_(**kwargs)
        figname = os.path.join(self.a.T['fiplt'],''.join(('Fiz_R',str(self.p.Rsun),
                                                          '.',str(format_))))
        self._figsave_(figname,this_function,**kwargs)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)     


    def fi_iso_plt(self,**kwargs):
        """
        2d map (R-z plane) of the normalized gravitational potential with isolines. 
        
        :param save: Optional. If True, the figure will be saved. 
            The output directory is ``a.T['fiplt']``. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close'],this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
        Phi = fi_iso(self.inp['ah'],self.p,self.a)
        PHI = np.vstack((np.flip(Phi[1:],axis=0),Phi[1:]))
        PHI = zoom(PHI,3)/1e9
        phimin, phimax = np.amin(PHI), 0.98*np.amax(PHI)
        
        f, ax = plt.subplots(figsize=(12,7))
        ax.imshow(PHI,extent=[self.p.Rmin,self.p.Rmax,-self.p.zmax/1e3,self.p.zmax/1e3],
                  cmap='gist_earth',norm = mpl.colors.Normalize(vmin=phimin,vmax=phimax))                                                           
        ax.contour(PHI,10,extent=[self.p.Rmin,self.p.Rmax,-self.p.zmax/1e3,self.p.zmax/1e3],
                  cmap='Reds_r',norm = mpl.colors.Normalize(vmin=phimin,vmax=phimax),linewidths=2)
        ax.set_xlabel(self.axts['r'],fontsize=self.fnt['main'],labelpad=15)
        ax.set_ylabel('$\mathrm{z, \ kpc}$',fontsize=self.fnt['main'],labelpad=15)
        ax.plot([self.p.Rmin,self.p.Rmax],[0,0],c='lightgrey',ls='--')
        f.subplots_adjust(left=0.11,bottom=0.15)
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_)
            ts.fi_iso_save(Phi)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)     


    def kz_plt(self,R,**kwargs):
        """
        Vertical gravitational force of the different model components calculated 
        for a fixed Galactocentric distance.
        
        :param R: Galactocentric distance, kpc. 
        :type R: scalar 
        :param save: Optional. If True, the figure will be saved. 
            The output directory is ``a.T['fiplt']``. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes. 
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close'],this_function)
        if R!=self.p.Rsun:
            R = inpcheck_radius(R,self.p,this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
        Kz = tab_reader(['Kz'],self.p,self.a.T,R=R)[0]

        f, ax = plt.subplots(figsize=(9,7))
        for i in range(6):
            ax.plot(Kz[0][:-1],Kz[i+1][:-1],label=self.labs[self.name[i]],
                    lw=self.lw['main'],color=self.cols[self.name[i]])
        ax.set_xlabel(self.axts['z'],fontsize=self.fnt['main'],labelpad=15)
        ax.set_ylabel(self.axts['kz'],fontsize=self.fnt['main'],labelpad=15)
        ax.set_title('$\mathrm{R=}$'+str(R)+'$\mathrm{\ kpc}$',fontsize=self.fnt['main'],pad=15)
        plt.legend(prop={'size':self.fnt['secondary']},loc=2,ncol=2)
        ax.set_xlim(0,self.p.zmax)
        ax.set_ylim(0,1.05*np.amax(Kz[1:]))
        f.subplots_adjust(left=0.14,top=0.86,bottom=0.15,right=0.86)
        
        format_ = self._figformat_(**kwargs)
        figname = os.path.join(self.a.T['fiplt'],''.join(('Kz_R',str(R),'.',format_)))
        self._figsave_(figname,this_function,**kwargs)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)     

    
    def rot_curve_plt(self,**kwargs):
        """
        Rotation curve as follows from the assumed MW mass model.
        
        :param R: Optional. Galactocentric distance grid, kpc. 
        :type R: array-like 
        :param save: Optional. If True, the figure will be saved. 
            The output directory is ``a.T['kinplt']``. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean

        :return: Figure and axes.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','R'],this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function)
        kwargs_calc = reduce_kwargs(kwargs,['save','R'])
        
        rc = rot_curve(self.inp['ah'],self.p,self.a,**kwargs_calc)
        components = ['b','d','t','g1','g2','dh']
        
        f, ax = plt.subplots(figsize=(10,7))
        ax.plot(rc['r'],rc['tot'],c='k',lw=self.lw['main'],label='$\mathrm{Total}$')
        for i in range(len(components)):
            k = components[i]
            ax.plot(rc['r'],rc[k],c=self.cols[k],label=self.labs[k],lw=self.lw['main'])
        ax.plot([self.p.Rsun,self.p.Rsun],[0,350],ls='--',c='gray',lw=self.lw['secondary'],
                label=self.labs['r0'])   
        ax.set_xlim(0,rc['r'][-1])
        ax.set_ylim(0,350)
        ax.set_xlabel(self.axts['r'],fontsize=self.fnt['main'],labelpad=15)
        ax.set_ylabel('$\mathrm{V_c,\ km \ s^{-1}}$',fontsize=self.fnt['main'],labelpad=15)
        plt.legend(loc=1,ncol=3)          

        format_ = self._figformat_(**kwargs)
        figname = os.path.join(self.a.T['kinplt'],''.join(('Vc_R.',format_)))
        self._figsave_(figname,this_function,**kwargs)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax) 


    def fw_hist_plt(self,mode_comp,R,zlim,**kwargs):
        """
        W-velocity distribution function. Plotting for :func:`jjmodel.analysis.fw_hist`. 
        
        :param mode_comp: Galactic component. Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total).
        :type mode_comp: str
        :param R: Galactocentric distance, kpc. Can be a single distance bin or a set of bins. 
        :type R: scalar or array-like 
        :param zlim: Range of heights, pc. Can be a single slice [*zmin,zmax*] or a set of z-bin edges. 
        :type zlim: array-like 
        :param ages: Optional. Set of age bins, Gyr. 
        :type ages: array-like
        :param mets: Optional. Set of metallicity bins. 
        :type mets: array-like
        :param wmax: Maximum value of W-velocity, :math:`\mathrm{km \ s^{-1}}`. 
        :type wmax: scalar
        :param dw: Step in W-velocity, :math:`\mathrm{km \ s^{-1}}`. 
        :type dw: scalar
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table, parameter alternative to **mode_pop**. 
            For a single radius **R** and **mode_comp** = ``'tot'``, 
            **tab** must be organized as a list of tables corresponding to this **R**: [*table_d*,*table_t*,table_sh*]. 
            If **R** is an array, then **tab** is [[*table_d,table_t,table_sh*]_ *rmin*,...,[*table_d,table_t,table_sh*]_ *rmax*], 
            where *rmin* and *rmax* are the minimum and maximum values from **R**. If **R** is not specified, *rmin* and *rmax* 
            correspond to ``p.Rmin`` and ``p.Rmax``. 
        :type tab: astropy table or list[astropy table], or list[list[astropy table]]
        :param number: Optional. If True, calculated quantity is weighted by the spatial number density of stars (:math:`\mathrm{number \ pc^{-3}}`),
             not matter density (:math:`\mathrm{M_\odot \ pc^{-3}}`). Active only when **mode_pop** or **tab** is given. 
        :type number: boolean
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str     
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :param save: Optional. If True, the figure will be saved. 
            The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.fw_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean
        
        :return: Figure and axes.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','wmax','dw','mode_pop','cbar',
                                'cbar_bins','tab','number','ages','mets','mode_iso'],this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                           'W-velocity dispersion',this_function)
        if self.p.run_mode==1:
            try:
                R = inpcheck_radius(R,self.p,this_function)
            except:
                R = [inpcheck_radius(i,self.p,this_function) for i in R]       
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
        kwargs_calc = reduce_kwargs(kwargs,['save','dw','wmax','zlim','mode_iso',
                                            'mode_pop','tab','number','ages','mets'])
        inpcheck_kwargs_compatibility(kwargs,this_function)   
        wmax = 60              
        if 'wmax' in kwargs:
            wmax = kwargs['wmax']           
        
        if 'mode_pop' in kwargs_calc and type(kwargs_calc['mode_pop'])!=list:
            ln += '$\mathrm{ \ (}$' + self.pops[kwargs_calc['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs_calc:
            ln += '$\mathrm{ \ (custom \ population)}$'
        if 'ages' in kwargs_calc and len(kwargs_calc['ages'])==2:
            ln += r'$\mathrm{\ of \ \tau = [}$'+str(kwargs_calc['ages'][0]) + ',' +\
                    str(kwargs_calc['ages'][1]) + '$\mathrm{] \ Gyr}$'
        if 'mets' in kwargs_calc and len(kwargs_calc['mets'])==2:
            ln += r'$\mathrm{\ of \ [Fe/H] = [}$' + str(kwargs_calc['mets'][0]) + ',' +\
                str(kwargs_calc['mets'][1]) + r'$\mathrm{]}$'
        if (('cbar' in kwargs or ('cbar_bins' in kwargs and kwargs['cbar_bins']==True)) and
            (('ages' not in kwargs or len(kwargs['ages'])==2) and type(R)!=list and 
             ('mets' not in kwargs or len(kwargs['mets'])==2) and len(zlim)==2)):
            print(this_function + ": Unnecessary input. Keywords 'cbar' and 'cbar_bins' "+\
                  "don't work with this combination of parameters.")
        if type(R)!=list:
            ln += r'$\mathrm{\ at \ R = }$'+str(R) + '$\mathrm{\ kpc}$'
            if len(zlim)==2:
                ln += r'$\mathrm{\ and \ |z| = [}$'+str(zlim[0]/1e3)+','+str(zlim[1]/1e3)+'$\mathrm{]\ kpc}$' 
        else:
            if len(zlim)==2:
                ln += r'$\mathrm{\ at \ |z| = [}$'+str(zlim[0]/1e3)+','+str(zlim[1]/1e3)+'$\mathrm{]\ kpc}$' 
        
        f, ax = plt.subplots(figsize=(9,7))
        ax.set_xlim(0,wmax)
        ax.set_xlabel('$\mathrm{|W|, \ km \ s^{-1}}$',fontsize=self.fnt['main'],labelpad=15)
        ax.set_ylabel('$\mathrm{f(|W|)}$',fontsize=self.fnt['main'],labelpad=15)
        ax.set_title(ln,fontsize=self.fnt['secondary'],pad=15)        
            
        if ('mode_pop' in kwargs_calc and type(kwargs_calc['mode_pop'])==list):
            if type(R)==list:
                raise KeyError(this_function+": Multiple populations and multiple R-bins are not allowed.")
            if len(zlim)>2:
                raise KeyError(this_function+": Multiple populations and multiple z-bins are not allowed.")
            if 'ages' in kwargs_calc and len(kwargs_calc['ages'])>2:
                raise KeyError(this_function+": Multiple populations and multiple age-bins are not allowed.")
            if 'mets' in kwargs_calc and len(kwargs_calc['mets'])>2:
                raise KeyError(this_function+": Multiple populations and multiple [Fe/H]-bins are not allowed.")
            
            if 'mode_pop' in kwargs_calc:
                npop = len(kwargs_calc['mode_pop'])
                xname = 'mode_pop'
                xcols = [self.popcols[i] for i in kwargs_calc['mode_pop']]
                xlabels = [self.pops[i] for i in kwargs_calc['mode_pop']]
            fw_list = [] 
            for i in range(npop):
                kwargs_calc2 = kwargs_calc.copy()
                del kwargs_calc2[xname]
                kwargs_calc2[xname] = kwargs_calc[xname][i]
                fw, wgrid = fw_hist(mode_comp,R,zlim,self.p,self.a,**kwargs_calc2) 
                fw_list.append(fw)
                ax.plot(wgrid,fw,lw=self.lw['main'],color=xcols[i],label=xlabels[i])                                                
            ax.set_ylim(0,round(np.nanmax(fw_list),2)+0.01)
            plt.legend(prop={'size':self.fnt['secondary']},loc=1,ncol=1)
            
        if (('ages' in kwargs_calc and len(kwargs_calc['ages'])>2) or 
            ('mets' in kwargs_calc and len(kwargs_calc['mets'])>2)):
            if type(R)==list:
                raise KeyError(this_function+": Multiple age- or [Fe/H]-bins and multiple R-bins are not allowed.")
            if len(zlim)>2:
                raise KeyError(this_function+": Multiple age- or [Fe/H]-bins and multiple z-bins are not allowed.")
            if 'mode_pop' in kwargs_calc and type(kwargs_calc['mode_pop'])==list:
                raise KeyError(this_function+": Multiple age- or [Fe/H]-bins and multiple populations are not allowed.")

            if 'ages' in kwargs_calc:
                npop = len(kwargs_calc['ages']) - 1
                xcols = self._figcolorbar_('tau',npop,**kwargs)
                xname = 'ages'
                xbar = 'tau'
                xmin, xmax = self._cbarminmax_(0,tp,np.mean(np.diff(kwargs_calc['ages'])),**kwargs)
            else:
                npop = len(kwargs_calc['mets']) - 1
                xcols = self._figcolorbar_('feh',npop,**kwargs)
                xname = 'mets'
                xbar = 'fe'
                xmin, xmax = self._cbarminmax_(kwargs_calc['mets'][0],kwargs_calc['mets'][-1],
                                               np.mean(np.diff(kwargs_calc['mets'])),**kwargs)  
                '''
                cbar_ticks = np.arange(kwargs_calc['mets'][0],kwargs_calc['mets'][-1]+0.1,0.1)
                if 'cbar_bins' in kwargs and kwargs['cbar_bins']==True:
                    dmet = np.mean(np.diff(kwargs_calc['mets']))
                    #cbar_ticks = np.add(kwargs_calc['mets'],dmet/2)[:-1]
                '''
            fw_list = [] 
            for i in range(npop):
                kwargs_calc2 = kwargs_calc.copy()
                del kwargs_calc2[xname]
                kwargs_calc2[xname] = [kwargs_calc[xname][i],kwargs_calc[xname][i+1]]
                fw, wgrid = fw_hist(mode_comp,R,zlim,self.p,self.a,**kwargs_calc2)
                fw_list.append(fw)
            line_segments = LineCollection([list(zip(wgrid,i)) for i in fw_list],   
                                       linewidths=self.lw['main'],cmap = xcols,
                                       norm = mpl.colors.Normalize(vmin=xmin,vmax=xmax)                                                                   
                                       )                                      
            line_segments.set_array(kwargs_calc[xname])
            im = ax.add_collection(line_segments)
            f.subplots_adjust(left=0.14,top=0.86,bottom=0.15,right=0.86)
            pos = ax.get_position()
            cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
            cbar = f.colorbar(im,cax=cax,orientation='vertical')
            if 'ages' in kwargs_calc:
                cbar.set_ticks(self.xt['ticks'])
                cbar.set_ticklabels(self.xt['labels'])
            #else:
            #    cbar.set_ticks(cbar_ticks)
            cbar.set_label(self.axts[xbar],fontsize=self.fnt['main'])   
            ax.set_ylim(0,round(np.nanmax(fw_list),2)+0.01)
            
        if len(zlim)>2 or type(R)==list:
            if 'ages' in kwargs_calc and len(kwargs_calc['ages'])>2:
                raise KeyError(this_function+": Multiple R- or z-bins and multiple age-bins are not allowed.")
            if 'mets' in kwargs_calc and len(kwargs_calc['mets'])>2:
                raise KeyError(this_function+": Multiple R- or z-bins and multiple [Fe/H]-bins are not allowed.")
            if 'mode_pop' in kwargs_calc and type(kwargs_calc['mode_pop'])==list:
                raise KeyError(this_function+": Multiple R- or z-bins and multiple populations are not allowed.")
            
            try:
                npop = len(R)
                xcols = self._figcolorbar_('r',npop,**kwargs)
                R_array = R
                zlim_array = [zlim for i in np.arange(npop)]
                xmin, xmax = self._cbarminmax_(self.p.Rmin,self.p.Rmax,self.p.dR,**kwargs)
                xarray = self.a.R
                xname = 'r'
                cbar_ticks = np.arange(self.p.Rmin,self.p.Rmax+1)
                if 'cbar_bins' in kwargs and kwargs['cbar_bins']==True:
                    cbar_ticks = np.add(self.a.R,self.p.dR/2)[:-1]
            except:
                npop = len(zlim) - 1
                xcols = self._figcolorbar_('z',npop,**kwargs)
                zlim_array = [[zlim[i],zlim[i+1]] for i in np.arange(npop)]
                R_array = [R for i in np.arange(npop)]
                xmin, xmax = 0, self.p.zmax/1e3
                xarray = np.array(zlim)/1e3
                xname = 'zkpc'
                dz = np.mean(np.diff(zlim))
                cbar_ticks = np.arange(zlim[0],zlim[-1]+dz,dz)/1e3
                if 'cbar_bins' in kwargs and kwargs['cbar_bins']==True:
                    cbar_ticks = np.arange(zlim[0]+dz/2,zlim[-1]+dz/2,dz)/1e3
                
            fw_list = [] 
            for i in range(npop):
                fw, wgrid = fw_hist(mode_comp,R_array[i],zlim_array[i],self.p,self.a,**kwargs_calc)
                fw_list.append(fw)
            line_segments = LineCollection([list(zip(wgrid,i)) for i in fw_list],   
                                       linewidths=self.lw['main'],cmap = xcols,
                                       norm = mpl.colors.Normalize(vmin=xmin,vmax=xmax)                                                                                                                                      
                                       )                                      
            line_segments.set_array(xarray)
            f.subplots_adjust(left=0.14,top=0.86,bottom=0.15,right=0.86)
            im = ax.add_collection(line_segments)
            pos = ax.get_position()
            cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
            cbar = f.colorbar(im,cax=cax,orientation='vertical')
            cbar.set_label(self.axts[xname],fontsize=self.fnt['main'])
            cbar.set_ticks(cbar_ticks)
            ax.set_ylim(0,round(np.nanmax(fw_list),2)+0.01)
            
                       
        if (('mode_pop' not in kwargs or type(kwargs['mode_pop'])!=list) and
            ('ages' not in kwargs or len(kwargs['ages'])==2) and 
            ('mets' not in kwargs or len(kwargs['mets'])==2) and
            (len(zlim)==2) and type(R)!=list):
            
            fw, wgrid = fw_hist(mode_comp,R,zlim,self.p,self.a,**kwargs_calc) 
            ax.plot(wgrid,fw,lw=self.lw['main'],color='g')
            ax.set_ylim(0,1.05*(np.amax(fw)+0.001))
            f.subplots_adjust(left=0.14,top=0.86,bottom=0.15,right=0.86)
            
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs_calc)
            ts.fw_save((fw,wgrid),mode_comp,R,zlim)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)     
    
    
    def disk_brightness_plt(self,mode_comp,mode_geom,bands,**kwargs):
        """
        Surface brightness or colour profile of the MW viewed edge-on or face-on. 
        Plotting for :func:`jjmodel.analysis.disk_brightness`. 
    
        :param mode_comp: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
        :type mode_comp: str
        :param mode_geom: Modeled geometry. Disk orientation with respect to the observer: ``'face-on'`` or ``'edge-on'``. 
        :type mode_geom: str
        :param bands: If a string, this parameter corresponds to the band for the surface brightness profile. 
            If it is a list, then **bands** gives the names of two bands to be used for the color profile - e.g. 
            ``['U','V']`` for *U-V*.
        :type bands: str or list
        :param zlim: Optional. Range of heights [*zmim,zmax*] to be considered, pc. If not given, 
            all heights up to ``p.zmax`` are taken into account.
        :type zlim: array-like
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param tab: Optional. Stellar assemblies table(s), parameter alternative to **mode_pop**. 
            If **mode_comp** = ``'tot'``, **tab** must be [[*table_d,table_t,table_sh*]_ *rmin*,...,[*table_d,table_t,table_sh*]_ *rmax*], 
            where *rmin* and *rmax* are ``p.Rmin`` and ``p.Rmax``
        :type tab: list[astropy table] or list[list[astropy table]] 
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str 
        :param save: Optional. If True, the figure will be saved. 
            The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.disk_brightness_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean
        
        :return: Figure and axes.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','zlim','mode_pop','tab','mode_iso'],
                        this_function)
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                           'disk brightness/colour profile',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        if mode_geom=='face-on':
            ln += '$\mathrm{\ face-on}$'
        else:
            ln += '$\mathrm{\ edge-on}$'
        if 'mode_pop' in kwargs:
            ln += '$\mathrm{\ (}$' + self.pops[kwargs['mode_pop']] + '$\mathrm{)}$'
        if 'tab' in kwargs:
            ln += '$\mathrm{\ (custom \ population)}$'
        if mode_geom=='edge-on':
            zlim = [0,self.p.zmax]
            if 'zlim' in kwargs:
                zlim = kwargs['zlim']
            ln += '$\mathrm{ \ for \ |z|=[}$'+str(round(zlim[0],0)/1e3)+','+\
                             str(round(zlim[1],0)/1e3)+'$\mathrm{] \ kpc}$'
                         
        kwargs_calc = reduce_kwargs(kwargs,['save','zlim','mode_iso'])                
        profile = disk_brightness(mode_comp,mode_geom,bands,self.p,self.a,**kwargs_calc)

        f, ax = plt.subplots(figsize=(9,7))
        ax.plot(self.a.R,profile,lw=self.lw['main'],marker='o',markersize=4,color='g')
        ax.set_xlabel(self.axts['r'],fontsize=self.fnt['main'],labelpad=15)
        if type(bands)==str:
            bands_string = bands.split('_')
            if len(bands_string)==2:
                bands_string = bands_string[0]+'\_'+bands_string[1]
            ax.set_ylabel('$\mathrm{\mu_{'+bands_string+'}, \ mag \ arcsec^{-2}}$',
                          fontsize=self.fnt['main'],labelpad=15)
        else:
            bands_string1 = bands[0].split('_')
            bands_string2 = bands[1].split('_')
            bands_string1 = bands_string1[0]+'\_'+bands_string1[1]
            bands_string2 = bands_string2[0]+'\_'+bands_string2[1]
            ax.set_ylabel('$\mathrm{'+bands_string1+'-'+bands_string2+'}$',
                          fontsize=self.fnt['main'],labelpad=15)
        #plt.legend(prop={'size':self.fnt['secondary']},loc=2,ncol=2)
        ax.set_xlim(self.p.Rmin,self.p.Rmax)
        ymin, ymax = round((np.amin(profile)-0.5),0), round((np.amax(profile)+0.5),0)
        if ymin==ymax:
            ymin = ymin - 0.5 
            ymax = ymin + 1 
        if type(bands)==str:
            ax.set_ylim(ymax,ymin)
        else:
            ax.set_ylim(ymin,ymax)
        f.subplots_adjust(left=0.14,top=0.86,bottom=0.15,right=0.86)
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            if 'save_format' in kwargs:
                del kwargs['save_format']
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs)
            ts.disk_brightness_save(profile,mode_comp,mode_geom,bands)  
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)     
    
    
    def hess_simple_plt(self,mode_comp,mode_geom,bands,mag_range,mag_step,**kwargs):
        """
        Hess diagram for the simple volumes. PLotting for :func:`jjmodel.analysis.hess_simple`. 
        
        :param mode_comp: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total).
        :type mode_comp: str
        :param mode_geom: Modeled geometry. Can be ``'local_sphere'``, ``'local_cylinder'``, or ``'rphiz_box'``. 
        :type mode_geom: str
        :param bands: List of names of photometric columns. 
            Three column names must be given, e.g. ``['B','V','V']`` for (*B-V*) versus :math:`M_\mathrm{V}`. 
        :type bands: list[str]
        :param mag_range: Minimal and maximal magnitude along the x- and y-axis of the Hess 
            diagram, [[*x_min,x_max*],[*y_min,y_max*]]. 
        :type mag_range: list[list[scalar]]
        :param mag_step: Step along the x- and y-axis of the Hess diagram, [*dx,dy*]. 
        :type mag_step: array-like
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). 
        :type mode_pop: str
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str
        :param r_minmax: Optional. Minimal and maximal radius of the spherical shell if **mode_geom** is ``'sphere'`` 
            or minimal and maximal radius of the cylindrical shell if **mode_geom** is ``'cylinder'``, pc.   
        :type r_minmax: array-like
        :param R_minmax: Optional. Minimal and maximal Galactocentric distance if **mode_geom** is ``'rphiz_box'``, kpc. 
        :type R_minmax: array-like
        :param dphi: Optional. Minimal and maximal Galactic angle :math:`\\phi` if **mode_geom** is 'rphiz_box', deg.
        :type dphi: array-like
        :param smooth: Optional. Width of the smoothing window in x and y, mag. 
        :type smooth: array-like
        :param zlim: Optional. Range of heights [*zmim,zmax*] to be considered, pc.
        :type zlim: array-like
        :param cbar: Optional. Matplotlib colormap name. 
        :type cbar: str
        :param cbar_bins: Optional. If True, the colorbar will be discrete. 
            By default, it is continious.  
        :param save: Optional. If True, the figure will be saved. 
            The output directory and figure name 
            are prescribed by :meth:`jjmodel.iof.TabSaver.hess_save`. 
        :type save: boolean
        :param save_format: Optional. Format of the figure. 
        :type save_format: str 
        :param close: Optional. If True, the figure window is closed in the end.
        :type close: boolean
        
        :return: Figure and axes.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['save','save_format','close','cbar','cbar_bins','mode_iso',
                                'zlim','mode_pop','r_minmax','R_minmax','smooth','dphi'],this_function)
        kwargs_calc = reduce_kwargs(kwargs,['zlim','save','mode_pop','r_minmax','R_minmax','smooth',
                                            'dphi','mode_iso'])
        ln, mode_comp = inpcheck_mode_comp(mode_comp,['d','t','sh','dt','tot'],
                                           'Hess diagram',this_function)
        inpcheck_kwargs_compatibility(kwargs,this_function,plt_only=True)
        
        hess = hess_simple(mode_comp,mode_geom,bands,mag_range,mag_step,self.p,self.a,**kwargs_calc)
        colorbar = self._figcolorbar_('z',10,**kwargs)
        
        Ntot = int(round(np.sum(np.sum(hess)),0))
        hess[hess==0]=np.nan
        log10_hess = np.log10(hess)
        xmin, xmax = mag_range[0]
        ymin, ymax = mag_range[1]
        hmin, hmax = 0,np.nanmax(log10_hess)
                        
        f, ax = plt.subplots(figsize=(10,8))        
        im = ax.imshow(log10_hess,interpolation='none',extent=[xmin,xmax,ymax,ymin],
                       cmap=colorbar,vmin=hmin,vmax=hmax)
        ax.set_aspect((xmax-xmin)/(ymax-ymin))
        
        b = [] 
        for i in range(len(bands)):
            split = bands[i].split('_')
            bsplit = ''
            for k in range(len(split)):
                if k!=0:
                    bsplit += '\_' + split[k]
                else:
                    bsplit += split[k]
            b.append(bsplit)
        
        ax.set_xlabel('$\mathrm{'+b[1]+'-'+b[2]+'}$',fontsize=self.fnt['main'])
        ax.set_ylabel('$\mathrm{'+b[0]+'}$',fontsize=self.fnt['main'])
        ax.text(0.2,0.05,'$\mathrm{N='+str(Ntot)+'}$',fontsize=self.fnt['secondary'],transform=ax.transAxes)
        f.subplots_adjust(bottom=0.18,left=0.08,right=0.94,top=0.93)
        pos = ax.get_position()
        cax = f.add_axes([pos.x0+pos.width+0.02,pos.y0,0.025,pos.height])
        cbar = f.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_label(r'$\mathrm{log_{10} \ N \ ['+str(mag_step[0])+
                       r'\times'+str(mag_step[1])+'] \ mag^{-2}}$',fontsize=self.fnt['main'],labelpad=10)
        
        if inpcheck_iskwargtype(kwargs,'save',True,bool,this_function):
            format_ = self._figformat_(**kwargs)
            ts = TabSaver(self.p,self.a,fig=True,save_format=format_,**kwargs_calc)
            ts.hess_save(hess,mode_comp,mode_geom,bands,mag_range,mag_step)
        self._figclose_(this_function,**kwargs)
        
        return (f, ax)     
        
            

class PlotBlocks():
    """
    Class to plot and save many figures automatically. 
    """
    
    def __init__(self,p,a,inp):
        """
        Class is initialized with tuples p and a and dictionary ``inp``. 
        
        :param p: Set of model parameters from the parameter file. 
        :type p: namedtuple
        :param a: Collection of the fixed model parameters, useful quantities, and arrays.
        :type a: namedtuple
        :param inp: Collection of the input functions including SFR, AVR, AMR, and IMF.
        :type inp: dict   
        """
        self.P = Plotting(p,a,inp)
        self.a, self.p, self.inp = a, p, inp
        self.modes_full = ['d','t','sh','dt','tot']
        self.modes_disk = ['d','t','dt']
        self.modes_dtot = ['d','dt','tot']
        self.kwargs = {'close':True,'save':True}
        self.pops = ['a','f','rc','gdw','kdw']
        
        
    def model_input(self,**kwgs):
        """
        Plots and saves all model input: radial density profiles 
        of the Galactic components, SFR, mass loss, gas scale heights. 
        
        :param print_time: Optional. If True, calculation time is printed. 
        :type print_time: boolean 
        
        :return: None. 
        """
        
        this_function = inspect.stack()[0][3]
        if not inpcheck_iskwargtype(kwgs,'print_time',False,bool,this_function):
            timer = Timer()
            t_start = timer.start()
    
        # For the Solar neighbourhood
        if self.p.run_mode==0:
            self.P.amr_rsun_plt('dt',**self.kwargs)
            self.P.g_rsun_plt('dt',**self.kwargs)
            [self.P.nsfr_rsun_plt(k,**self.kwargs) for k in self.modes_disk]
    
        # For the whole disk
        else:
            self.P.amrr_plt('dt',**self.kwargs)
            self.P.gr_plt('dt',**self.kwargs)
            [self.P.nsfr_plt(k,**self.kwargs) for k in self.modes_disk]
            self.P.rhor_plt(**self.kwargs)
            self.P.hr_gas_plt(**self.kwargs)
        
        if not inpcheck_iskwargtype(kwgs,'print_time',False,bool,this_function):
            print('\nPlotBlocks.model_input : ',timer.stop(t_start))

        
    def densities(self,zlim,ages,mets,**kwgs):
        """
        Plots and saves predicted vertical and radial profiles  
        of the Galactic components and of the disk's mono-age and mono-metallicity subpopulations. 
        
        :param zlim: Single z-bin [*zmin,zmax*], where the quantity will be calculated, pc. 
        :type zlim: array-like
        :param ages: Set of age bins, Gyr. 
        :type ages: array-like
        :param mets: Set of meallicity bins. 
        :type mets: array-like
        :param print_time: Optional. If True, calculation time is printed. 
        :type print_time: boolean 
        
        :return: None. 
        """
        
        this_function = inspect.stack()[0][3]
        if not inpcheck_iskwargtype(kwgs,'print_time',False,bool,this_function):
            timer = Timer()
            t_start = timer.start()
        
        # In any case
        self.P.rhoz_plt(self.p.Rsun,**self.kwargs)    
        self.P.rhoz_plt(self.p.Rsun,normalized=True,**self.kwargs)
        self.P.rhoz_plt(self.p.Rsun,cumulative=True,**self.kwargs)
        for k in range(len(self.modes_disk)):
            self.P.rhoz_monoage_plt(self.modes_disk[k],self.p.Rsun,ages,between=True,**self.kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",category=UserWarning)
                self.P.rhoz_monomet_plt(self.modes_disk[k],self.p.Rsun,mets,**self.kwargs)      
                
        # For the whole disk
        if self.p.run_mode!=0:
            for i in range(self.a.Rbins):
                self.P.rhoz_plt(self.a.R[i],**self.kwargs)    
                self.P.rhoz_plt(self.a.R[i],normalized=True,**self.kwargs)
                self.P.rhoz_plt(self.a.R[i],cumulative=True,**self.kwargs)
                for k in range(len(self.modes_disk)):
                    self.P.rhoz_monoage_plt(self.modes_disk[k],self.a.R[i],ages,between=True,**self.kwargs)
                    self.P.rhor_monoage_plt(self.modes_disk[k],zlim,ages,between=True,**self.kwargs)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore",category=UserWarning)
                        self.P.rhoz_monomet_plt(self.modes_disk[k],self.a.R[i],mets,**self.kwargs)
                        self.P.rhor_monomet_plt(self.modes_disk[k],zlim,mets,**self.kwargs)
        
        if not inpcheck_iskwargtype(kwgs,'print_time',False,bool,this_function):
            print('\nPlotBlocks.densities : ',timer.stop(t_start))
        
        
    def ages(self,zlim_set,age_sm,**kwgs):
        """
        Plots and saves disk's age distributions at different heights and Galactocentric distances. 
        
        :param zlim_set: Edges of z-bins where the quantity will be calculated. 
        :type zlim_set: array-like
        :param age_sm: Standard deviation of the Gaussian kernel, which is used 
            to smooth age distributions, Gyr. 
        :type age_sm: scalar   
        :param print_time: Optional. If True, calculation time is printed. 
        :type print_time: boolean 
        
        :return: None. 
        """
        
        this_function = inspect.stack()[0][3]
        if not inpcheck_iskwargtype(kwgs,'print_time',False,bool,this_function):
            timer = Timer()
            t_start = timer.start()
        
        modes_disk = [self.modes_disk[0],self.modes_disk[-1]]
        
        # For the Solar neighbourhood
        if self.p.run_mode==0:
            for k in range(len(modes_disk)):
                self.P.agehist_rsun_plt(modes_disk[k],[zlim_set[0],zlim_set[-1]],
                                        sigma_gauss=age_sm,**self.kwargs)
                self.P.agehist_rsun_plt(modes_disk[k],zlim_set,sigma_gauss=age_sm,**self.kwargs)
                                        
            self.P.agez_rsun_plt('tot',**self.kwargs)
        
        # For the whole disk
        else:          
            for k in range(len(modes_disk)):
                self.P.agehist_plt(modes_disk[k],[zlim_set[0],zlim_set[-1]],
                                   sigma_gauss=age_sm,**self.kwargs)
                [self.P.agehist_plt(modes_disk[k],[zlim_set[i],zlim_set[i+1]],sigma_gauss=age_sm,
                                    **self.kwargs) for i in np.arange(len(zlim_set)-1)]
            for k in range(len(self.modes_dtot)):
                self.P.agez_plt(self.modes_dtot[k],**self.kwargs)
                self.P.ager_plt(self.modes_dtot[k],[zlim_set[0],zlim_set[-1]],**self.kwargs)                
        
        if not inpcheck_iskwargtype(kwgs,'print_time',False,bool,this_function):
            print('\nPlotBlocks.ages : ',timer.stop(t_start))
            
            
    def metallicities(self,zlim_set,met_sm,**kwgs):
        """
        Plots and saves disk's age distributions at different heights and Galactocentric distances. 
        
        :param zlim_set: Edges of z-bins where the quantity will be calculated. 
        :type zlim_set: array-like
        :param met_sm: Standard deviation of the Gaussian kernel, which is used 
            to smooth metallicity distributions. 
        :type met_sm: scalar   
        :param print_time: Optional. If True, calculation time is printed. 
        :type print_time: boolean 
        
        :return: None. 
        """
        
        this_function = inspect.stack()[0][3]
        if not inpcheck_iskwargtype(kwgs,'print_time',False,bool,this_function):
            timer = Timer()
            t_start = timer.start()
        
        modes_disk = [self.modes_disk[0],self.modes_disk[-1]]
        
        # For the Solar neighbourhood
        if self.p.run_mode==0:
            for k in range(len(modes_disk)):
                self.P.methist_rsun_plt(modes_disk[k],zlim_set,sigma_gauss=met_sm,**self.kwargs)
                [self.P.methist_rsun_plt(modes_disk[k],[zlim_set[i],zlim_set[i+1]],sigma_gauss=met_sm,
                                         **self.kwargs) for i in np.arange(len(zlim_set)-1)]
            self.P.metz_rsun_plt('tot',**self.kwargs)
            
        # For the whole disk   
        else:
            for k in range(len(modes_disk)):
                self.P.methist_plt(modes_disk[k],[zlim_set[0],zlim_set[-1]],
                                   sigma_gauss=met_sm,**self.kwargs)
                [self.P.methist_plt(modes_disk[k],[zlim_set[i],zlim_set[i+1]],sigma_gauss=met_sm,
                                    **self.kwargs) for i in np.arange(len(zlim_set)-1)]
            for k in range(len(self.modes_dtot)):
                self.P.metz_plt(self.modes_dtot[k],**self.kwargs)
                self.P.metr_plt(self.modes_dtot[k],[zlim_set[0],zlim_set[-1]],**self.kwargs)                
        
        if not inpcheck_iskwargtype(kwgs,'print_time',False,bool,this_function):
            print('\nPlotBlocks.metallicities : ',timer.stop(t_start))
            
            
    def heights(self,ages,mets,**kwgs):
        """
        Plots and saves disk's scale heights as a function of age and Galactocentric distance. 
        
        :type ages: array-like
        :param mets: Set of meallicity bins. 
        :type mets: array-like
        :param print_time: Optional. If True, calculation time is printed. 
        :type print_time: boolean 
        
        :return: None. 
        """
        
        this_function = inspect.stack()[0][3]
        if not inpcheck_iskwargtype(kwgs,'print_time',False,bool,this_function):
            timer = Timer()
            t_start = timer.start()
        
        # For the Solar neighbourhood
        if self.p.run_mode==0:
            self.P.h_rsun_plt(**self.kwargs)
            
        # For the whole disk
        else:      
            self.P.h_plt(**self.kwargs)
            [self.P.hr_monoage_plt(k,ages,between=True,**self.kwargs) for k in self.modes_dtot]
            [self.P.hr_monomet_plt(k,mets,**self.kwargs) for k in self.modes_dtot]
    
        if not inpcheck_iskwargtype(kwgs,'print_time',False,bool,this_function):
            print('\nPlotBlocks.heights : ',timer.stop(t_start))
        
        
    def kinematics(self,zlim_set,ages,mets,**kwgs):
        """
        Plots and saves disk's kinematic functions: AVR, W-velocity 
        dispersion of the thick disk and gas at different Galactocentric distances. 
        
        :param zlim_set: Edges of z-bins where the quantity will be calculated. 
        :type zlim_set: array-like
        :type ages: array-like
        :param mets: Set of meallicity bins. 
        :type mets: array-like
        :param print_time: Optional. If True, calculation time is printed. 
        :type print_time: boolean 
        
        :return: None. 
        """
        
        this_function = inspect.stack()[0][3]
        if not inpcheck_iskwargtype(kwgs,'print_time',False,bool,this_function):
            timer = Timer()
            t_start = timer.start()
        
        zlim = [zlim_set[0],zlim_set[-1]]
        
        # In any case
        [self.P.fw_hist_plt(k,self.p.Rsun,zlim_set,**self.kwargs) for k in self.modes_dtot]
        self.P.fw_hist_plt('dt',self.p.Rsun,zlim,ages=ages,**self.kwargs)
        self.P.fw_hist_plt('dt',self.p.Rsun,zlim,mets=mets,**self.kwargs)
        
        # For the Solar neighbourhood
        if self.p.run_mode==0:
            self.P.avr_rsun_plt(**self.kwargs)
            [self.P.sigwz_rsun_plt(k,**self.kwargs) for k in self.modes_dtot]
            
        # For the whole disk
        else:
            self.P.avr_plt(**self.kwargs)
            self.P.rot_curve_plt(**self.kwargs)
            self.P.sigwr_thick_plt(**self.kwargs)
            self.P.sigwr_gas_plt(**self.kwargs)
            [self.P.sigwr_plt(k,zlim_set,**self.kwargs) for k in self.modes_dtot]
            [self.P.sigwr_monoage_plt(k,zlim,ages,between=True,**self.kwargs) for k in self.modes_dtot]
            [self.P.sigwr_monomet_plt(k,zlim,mets,**self.kwargs) for k in self.modes_dtot]
            [self.P.sigwr_monoage_plt(k,zlim,ages,between=True,**self.kwargs) for k in self.modes_dtot]
            [self.P.sigwr_monomet_plt(k,zlim,mets,**self.kwargs) for k in self.modes_dtot]
            self.P.fw_hist_plt('dt',self.a.R,zlim,**self.kwargs)
                
        if not inpcheck_iskwargtype(kwgs,'print_time',False,bool,this_function):
            print('\nPlotBlocks.kinematics : ',timer.stop(t_start)) 
            
            
    def potential(self,**kwgs):
        """
        Plots and saves vertical grvitational potential and gravitational force. 
        
        :param print_time: Optional. If True, calculation time is printed. 
        :type print_time: boolean 
        
        :return: None. 
        """
        
        this_function = inspect.stack()[0][3]
        if not inpcheck_iskwargtype(kwgs,'print_time',False,bool,this_function):
            timer = Timer()
            t_start = timer.start()
        
        # In any case
        self.P.kz_plt(self.p.Rsun,**self.kwargs)
        
        # For the Solar neighbourhood
        if self.p.run_mode==0:
            self.P.fi_rsun_plt(**self.kwargs)
            
        # For the whole disk
        else:
            self.P.fi_iso_plt(**self.kwargs)
            [self.P.kz_plt(i,**self.kwargs) for i in self.a.R]
        
        if not inpcheck_iskwargtype(kwgs,'print_time',False,bool,this_function):
            print('\nPlotBlocks.potential : ',timer.stop(t_start))
            
            
    def rz_maps(self,dz,**kwgs):
        """
        Plots and saves density/age/[Fe/H]/sigw maps in Rz plane. 
        
        :param dz: Optional. Vertical resolution, pc. 
        :type dz: scalar
        :param print_time: Optional. If True, calculation time is printed. 
        :type print_time: boolean 
        
        :return: None. 
        """
        
        this_function = inspect.stack()[0][3]
        if not inpcheck_iskwargtype(kwgs,'print_time',False,bool,this_function):
            timer = Timer()
            t_start = timer.start()
        
        for i in range(len(self.modes_full)):    
            self.P.rz_map_plt(self.modes_full[i],dz=dz,**self.kwargs)
            
        if not inpcheck_iskwargtype(kwgs,'print_time',False,bool,this_function):
            print('\nPlotBlocks.rz_maps : ',timer.stop(t_start))
                        
            
    def populations(self,zlim,ages,mets,age_sm,met_sm,rmax,**kwgs):
        """
        Plots and saves radial density profiles, age, metallicity , and f(W) distributions, 
        Hess diagrams and Rz-maps for the different populations. 
    
        :param ages: Set of age bins, Gyr. 
        :type ages: array-like
        :param mets: Set of meallicity bins. 
        :type mets: array-like
        :param age_sm: Standard deviation of the Gaussian kernel, which is used 
            to smooth age distributions, Gyr. 
        :type age_sm: scalar  
        :param met_sm: Standard deviation of the Gaussian kernel, which is used 
            to smooth metallicity distributions. 
        :type met_sm: scalar  
        :param rmax: Radius of the local sphere or cylinder (for calculation of Hess diagram), pc. 
        :type rmax: scalar
        :param dz: Optional. Vertical resolution, pc. If not specified, **dz** = 25 pc. 
        :type dz: scalar
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str
        :param print_time: Optional. If True, calculation time is printed. 
        :type print_time: boolean 
        
        :return: None. 
        """
        
        this_function = inspect.stack()[0][3]
        if not inpcheck_iskwargtype(kwgs,'print_time',False,bool,this_function):
            timer = Timer()
            t_start = timer.start()
        if 'mode_iso' in kwgs:
            self.kwargs['mode_iso'] = kwgs['mode_iso']                          
        
        modes_disk = 'tot'
        
        for l in range(len(self.pops)):
            
            # Density profiles
            if self.p.run_mode!=0:
                self.P.rhor_monoage_plt(modes_disk,zlim,ages,between=True,
                                        mode_pop=self.pops[l],number=True,**self.kwargs)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore",category=UserWarning)
                    self.P.rhor_monomet_plt(modes_disk,zlim,mets,mode_pop=self.pops[l],
                                            number=True,**self.kwargs)        
            # Ages and metallicities
            if self.p.run_mode==0:
                self.P.agehist_rsun_plt(modes_disk,zlim,mode_pop=self.pops[l],
                                        number=True,sigma_gauss=age_sm,**self.kwargs)
                self.P.methist_rsun_plt(modes_disk,zlim,mode_pop=self.pops[l],
                                        number=True,sigma_gauss=met_sm,**self.kwargs)
                self.P.agez_rsun_plt(modes_disk,mode_pop=self.pops[l],number=True,**self.kwargs)  
                self.P.metz_rsun_plt(modes_disk,mode_pop=self.pops[l],number=True,**self.kwargs)    
            else:
                self.P.agehist_plt(modes_disk,zlim,mode_pop=self.pops[l],
                                   number=True,sigma_gauss=age_sm,**self.kwargs)
                self.P.methist_plt(modes_disk,zlim,mode_pop=self.pops[l],
                                   number=True,sigma_gauss=met_sm,**self.kwargs)
                self.P.agez_plt(modes_disk,mode_pop=self.pops[l],number=True,**self.kwargs)  
                self.P.metz_plt(modes_disk,mode_pop=self.pops[l],number=True,**self.kwargs)    
                self.P.ager_plt(modes_disk,zlim,mode_pop=self.pops[l],number=True,**self.kwargs)            
                self.P.metr_plt(modes_disk,zlim,mode_pop=self.pops[l],number=True,**self.kwargs)        
            
                # RZ-maps
                if 'dz' not in self.kwargs:
                    dz = 25 # pc 
                else:
                    dz = self.kwargs['dz']
                self.P.rz_map_plt(modes_disk,dz=dz,mode_pop=self.pops[l],number=True,**self.kwargs)                              
                self.P.rz_map_quantity_plt(modes_disk,'age',dz=dz,mode_pop=self.pops[l],**self.kwargs)                                       
                self.P.rz_map_quantity_plt(modes_disk,'FeH',dz=dz,mode_pop=self.pops[l],**self.kwargs)                                       
                self.P.rz_map_quantity_plt(modes_disk,'sigw',dz=dz,mode_pop=self.pops[l],**self.kwargs)  
            
                # Kinematics
                self.P.fw_hist_plt(modes_disk,self.a.R,zlim,mode_pop=self.pops[l],number=True,
                                   **self.kwargs)
                
        self.P.fw_hist_plt(modes_disk,self.p.Rsun,zlim,mode_pop=self.pops,number=True,**self.kwargs)
        
        # Hess diagrams
        self.P.hess_simple_plt(modes_disk,'local_sphere',['G_EDR3','G_EDR3','GRP_EDR3'],
                               [[-0.25,1.5],[-5,15]],[0.005,0.1],smooth=[0.05,0.3],r_minmax=[0,rmax],
                               **self.kwargs)
        self.P.hess_simple_plt(modes_disk,'local_cylinder',['G_EDR3','G_EDR3','GRP_EDR3'],
                               [[-0.25,1.5],[-5,15]],[0.005,0.1],smooth=[0.05,0.3],r_minmax=[0,rmax],
                               zlim=zlim,**self.kwargs)
        if self.p.run_mode!=0:
            self.P.hess_simple_plt(modes_disk,'rphiz_box',['G_EDR3','G_EDR3','GRP_EDR3'],
                                   [[-0.25,1.5],[-5,15]],[0.005,0.1],smooth=[0.05,0.3],
                                   R_minmax=[self.p.Rsun-self.p.dR/2,self.p.Rsun+self.p.dR/2],
                                   zlim=zlim,dphi=360,**self.kwargs)
        
        if not inpcheck_iskwargtype(kwgs,'print_time',False,bool,this_function):
            print('\nPlotBlocks.pops : ',timer.stop(t_start))
        
        
    def basic_output(self,zlim_set,ages,mets,age_sm,met_sm,rmax,**kwargs):
        """
        Plots and saves all figures (basic output of the model). 
        
        :param zlim_set: Edges of z-bins where the quantities will be calculated. 
        :type zlim_set: array-like
        :param ages: Set of age bins, Gyr. 
        :type ages: array-like
        :param mets: Set of meallicity bins. 
        :type mets: array-like
        :param age_sm: Standard deviation of the Gaussian kernel, which is used 
            to smooth age distributions, Gyr. 
        :type age_sm: scalar  
        :param met_sm: Standard deviation of the Gaussian kernel, which is used 
            to smooth metallicity distributions. 
        :type met_sm: scalar  
        :param rmax: Radius of the local sphere or cylinder (for calculation of Hess diagram), pc. 
        :type rmax: scalar
        :param dz: Optional. Vertical resolution for Rz maps, pc. If not specified, **dz** = 25 pc. 
        :type dz: scalar
        :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
            If not specified, Padova is the default isochrone set. 
        :type mode_iso: str
 
        :return: None. 
        """

        timer = Timer()
        t_start = timer.start()
        
        print('\nPlotting block of figures for the JJ model...',end=' ')
        
        print('{:<26}'.format('\nInput'),'{:<2}'.format(':'),end=' ')
        self.model_input(print_time=False)
        print('\tok','{:<26}'.format('\nPotential'),'{:<2}'.format(':'),end=' ')
        self.potential(print_time=False)
        print('\tok','{:<26}'.format('\nScale heights'),'{:<2}'.format(':'),end=' ')
        self.heights(ages,mets,print_time=False)
        print('\tok','{:<26}'.format('\nDensities'),'{:<2}'.format(':'),end=' ')
        self.densities([zlim_set[0],zlim_set[-1]],ages,mets,print_time=False)
        print('\tok','{:<26}'.format('\nAges'),'{:<2}'.format(':'),end=' ')
        self.ages(zlim_set,age_sm,print_time=False)
        print('\tok','{:<26}'.format('\nMetallicities'),'{:<2}'.format(':'),end=' ')
        self.metallicities(zlim_set,met_sm,print_time=False)
        print('\tok','{:<26}'.format('\nKinematics'),'{:<2}'.format(':'),end=' ')
        self.kinematics(zlim_set,ages,mets,print_time=False)
        print('\tok','{:<26}'.format('\nPopulations'),'{:<2}'.format(':'),end=' ')
        self.populations([zlim_set[0],zlim_set[-1]],ages,mets,age_sm,met_sm,rmax,
                             print_time=False,**kwargs)
        if self.p.run_mode!=0:
            print('\tok','{:<26}'.format('\nRZ-maps'),'{:<2}'.format(':'),end=' ')
            if 'dz' not in self.kwargs:
                dz = 25 #pc 
            else:
                dz = self.kwargs['dz']
            self.rz_maps(dz,print_time=False)            
        print('\tok')
        
        print('\nPlotBlocks.basic_output: ',timer.stop(t_start))


