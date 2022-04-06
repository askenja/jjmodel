"""
Created on Mon Feb 20 18:17:18 2017

@author: Skevja
"""

import inspect
import numpy as np
import warnings
from collections import namedtuple
from .control import inpcheck_height, inpcheck_kwargs, inpcheck_iskwargtype


class Volume():
    """
    Class for calculation of the different volume cells. 
    """
    def __init__(self,p,a):
        self.p, self.a = p, a 
        
    def _circ_intersect_(self,R_ann,R_sph,r_sph):

        R_ann, R_sph = R_ann*1e3, R_sph*1e3 # kpc to pc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",category=RuntimeWarning)
            if r_sph!=0:
                area = r_sph**2*np.arccos((R_sph**2 + r_sph**2 - R_ann**2)/2/R_sph/r_sph) +\
                       R_ann**2*np.arccos((R_sph**2 + R_ann**2 - r_sph**2)/2/R_sph/R_ann) -\
                       0.5*np.sqrt((-R_sph + r_sph + R_ann)*(R_sph + r_sph - R_ann)*\
                                   (R_sph - r_sph + R_ann)*(R_sph + r_sph + R_ann))
            else:
                area = 0 
        if area<0 or area*0!=0:
            area = 0 
        if (R_sph + r_sph) <= R_ann:
            area = np.pi*r_sph**2
        return area
    
    def _circ_volume_(self,r_min,r_max,R2,R1):

        v = np.zeros((self.a.n))
        for i in range(self.a.n):            
            z_pos = np.abs(self.a.z[i] + self.p.zsun)
            z_neg = np.abs(self.a.z[i] - self.p.zsun)
            r1_pos,r1_neg,r2_pos,r2_neg = 0,0,0,0            
            if z_pos <= r_max :
                r1_pos = np.sqrt(r_max**2 - z_pos**2)
            if z_neg <= r_max:
                r1_neg = np.sqrt(r_max**2 - z_neg**2)    
            if z_pos <= r_min:
                r2_pos = np.sqrt(r_min**2 - z_pos**2)
            if z_neg <= r_min:
                r2_neg = np.sqrt(r_min**2 - z_neg**2)        
    
            intersect_large_pos1,intersect_large_neg1,intersect_small_pos1,intersect_small_neg1, \
            intersect_large_pos2,intersect_large_neg2,intersect_small_pos2,intersect_small_neg2 = 0,0,0,0,0,0,0,0
            if r1_pos > 0:
                intersect_large_pos1 = self._circ_intersect_(R1,self.p.Rsun,r1_pos)
                intersect_large_pos2 = self._circ_intersect_(R2,self.p.Rsun,r1_pos)
            if r1_neg > 0:
                intersect_large_neg1 = self._circ_intersect_(R1,self.p.Rsun,r1_neg)
                intersect_large_neg2 = self._circ_intersect_(R2,self.p.Rsun,r1_neg)
            if r2_pos > 0:
                intersect_small_pos1 = self._circ_intersect_(R1,self.p.Rsun,r2_pos)
                intersect_small_pos2 = self._circ_intersect_(R2,self.p.Rsun,r2_pos)
            if r2_neg > 0:
                intersect_small_neg1 = self._circ_intersect_(R1,self.p.Rsun,r2_neg)
                intersect_small_neg2 = self._circ_intersect_(R2,self.p.Rsun,r2_neg)
            
            int_large_pos = intersect_large_pos1 - intersect_large_pos2
            int_small_pos = intersect_small_pos1 - intersect_small_pos2  
            int_large_neg = intersect_large_neg1 - intersect_large_neg2 
            int_small_neg = intersect_small_neg1 - intersect_small_neg2            
            s_pos = int_large_pos - int_small_pos
            s_neg = int_large_neg - int_small_neg            
            v[i] = (s_pos + s_neg)*self.p.dz    
            
        return v
       
    
    def _cyl_volume_(self,r_min,r_max,z_min,z_max,R2,R1):

        v = np.zeros((self.a.n)) 
        indz1, indz2 = int(z_min//self.p.dz), int(z_max//self.p.dz)
                 
        intersect_large1 = self._circ_intersect_(R1,self.p.Rsun,r_max)
        intersect_large2 = self._circ_intersect_(R2,self.p.Rsun,r_max)
        intersect_small1 = self._circ_intersect_(R1,self.p.Rsun,r_min)
        intersect_small2 = self._circ_intersect_(R2,self.p.Rsun,r_min)
        int_large = intersect_large1 - intersect_large2
        int_small = intersect_small1 - intersect_small2  
        
        for i in range(indz2-indz1):
            v[indz1 + i] = (int_large - int_small)*2*self.p.dz    
            
        return v
    
     
    def none(self):
        """
        Useful when no volume is considered, returns z-grid 
        filled with ones. 
        """
        return np.ones(self.a.n)
    
    def zcut(self,v,zlim):
        
        indz1, indz2 = int(zlim[0]//self.p.dz), int(zlim[1]//self.p.dz)
        v[:indz1] = np.zeros((indz1))
        v[indz2:] = np.zeros((self.a.n-indz2))
        return v 
    
        
    def local_sphere(self,r_min,r_max):
        """
        Returns volume z-grid for the local sphere of radius 
        r_max with an inner hole of radius r_min. For testing use 
            V = Volume(p,a)
            v1 = V.local_sphere(0,850) 
            print(np.sum(v1[0])/(4/3*np.pi*850**3))
        
        Parameters
        ----------
        r_min, r_max : scalar
            Inner and outer radii of the spherical shell, in pc.
        
        Returns
        -------
        Volume (pc^3) of z-bins in different R-bins.  
        Indices of R-bins (use with a.R). 
        """
        
        ln = 'local sphere r = [' + str(np.round(r_min/1e3,3)) + ',' +\
             str(np.round(r_max/1e3,3)) + '] kpc'

        if self.p.run_mode==0:
            volume = self._circ_volume_(r_min,r_max,
                                        self.p.Rsun-1.1*r_max/1e3,self.p.Rsun+1.1*r_max/1e3)
            
            return volume, ln
            
        else:
            R_min, R_max = round(self.p.Rsun-r_max/1e3,3), round(self.p.Rsun+r_max/1e3,3)       
            if R_min < self.p.Rmin or R_max > self.p.Rmax:
                raise ValueError("Sphere is too large, reduce r_max.")
            indr1 = int(np.where(self.a.R_edges - R_min < 0)[0][-1])
            indr2 = int(np.where(self.a.R_edges - R_max < 0)[0][-1] + 1)
            
            subR = np.copy(self.a.R_edges[indr1:indr2+1])
            subR[0], subR[-1] = R_min, R_max
            volume = np.zeros((len(subR)-1,self.a.n))
            for i in range(len(subR)-1):
                volume[i] = self._circ_volume_(r_min,r_max,subR[i],subR[i+1]) 
                
            return volume, np.arange(indr1,indr2), ln
    
    
    def local_cylinder(self,r_min,r_max,z_min,z_max):
        """
        Returns volume z-grid for the local cylinder of radius 
        r_max with an inner hole of radius r_min. For testing use 
            V = Volume(p,a)
            v2 = V.local_cylinder(100,800,0,500)
            print(np.sum(v2[0])/(2*np.pi*(800**2-100**2)*500))
        
        Parameters
        ----------
        r_min, r_max : scalar
            Inner and outer radii of cylinder, in pc.
        z_min, z_max : scalar
            Limiting |z|, in pc.
        
        Returns
        -------
        Volume (pc^3) of z-bins in different R-bins.  
        Indices of R-bins (use with a.R). 
        """
        
        ln = 'local cylider r = [' + str(np.round(r_min/1e3,3)) + ',' + str(np.round(r_max/1e3,3)) +\
            '] kpc & |z| = [' + str(z_min/1e3) + ',' + str(z_max/1e3) + '] kpc'
        
        z_min,z_max = inpcheck_height([z_min,z_max],self.p,inspect.stack()[0][3])
        
        if self.p.run_mode==0:
            volume = self._cyl_volume_(r_min,r_max,z_min,z_max,
                                       self.p.Rsun-1.1*r_max/1e3,self.p.Rsun+1.1*r_max/1e3)
            
            return volume, ln
            
        else:
            R_min, R_max = round(self.p.Rsun-r_max/1e3,3), round(self.p.Rsun+r_max/1e3,3)
            if R_min < self.p.Rmin or R_max > self.p.Rmax:
                raise ValueError("Cylinder radius is too large, reduce r_max.")
            indr1 = int(np.where(self.a.R_edges - R_min < 0)[0][-1])
            indr2 = int(np.where(self.a.R_edges - R_max < 0)[0][-1] + 1)
            
            subR = np.copy(self.a.R_edges[indr1:indr2+1])
            subR[0], subR[-1] = R_min, R_max
            volume = np.zeros((len(subR)-1,self.a.n))
            for i in range(len(subR)-1):
                volume[i] = self._cyl_volume_(r_min,r_max,z_min,z_max,subR[i],subR[i+1]) 
            
            return volume, np.arange(indr1,indr2), ln
        
    
    def rphiz_box(self,R_min,R_max,dphi,z_min,z_max):
        """
        Returns volume z-grid for the 'box' in cylindrical coords 
        R-phi-z. For testing use 
            V = Volume(p,a)
            v3 = V.rphiz_box(8.15,8.25,360,100,250)
            print(np.sum(v3[0])/(2*np.pi*(8250**2-8150**2)*150))
        
        Parameters
        ----------
        R_min, R_max : scalar
            Range of Galactocentric radii, in kpc.
        dphi: scalar
            Galactocentric(? not sure how it's called..) angle, 
            in deg. 
        z_min, z_max : scalar
            Range of |z|, in pc.
        
        Returns
        -------
        Volume (pc^3) of z-bins in different R-bins.  
        Indices of R-bins (use with a.R). 
        """
        
        indr1 = int(np.where(self.a.R_edges - R_min < 0)[0][-1])
        indr2 = int(np.where(self.a.R_edges - R_max < 0)[0][-1] + 1)
        indz1, indz2 = int(z_min//self.p.dz), int(z_max//self.p.dz)
        
        subR = np.copy(self.a.R_edges[indr1:indr2+1])
        subR = subR*1e3 # kpc to pc
        subR[0], subR[-1] = R_min*1e3, R_max*1e3
        volume = np.zeros((len(subR)-1,self.a.n))
        for i in range(len(subR)-1):
            const = (subR[i+1]**2-subR[i]**2)*np.deg2rad(dphi)*self.p.dz
            volume[i,indz1:indz2] = [const for k in np.arange(indz2-indz1)]
        
        ln = 'R-phi-z box R = [' + str(round(R_min,3)) + ',' + str(round(R_max,3)) + '] kpc & dphi = ' +\
            str(dphi) + ' deg & |z| = [' + str(z_min/1e3) + ',' + str(z_max/1e3) + '] kpc'
                        
        return volume, np.arange(indr1,indr2), ln 
        


class Grid():
    """
    Class for creating r-phi-z grids 
    (e.g. for the Solar neighbourhood modeling).

    Parameters
    ----------
    zmax : scalar, pc
        Max height.
    dz : scalar, pc
        Step in height.
    rmax : scalar, pc
        Max distnace from z-axis in xy-plane.
    rnum : scalar, dim
        Number of bins in r (xy plane).
    **kwargs: optional keyword arguments: 
        rlog : boolean
            If True, binning aling r is in log space.
        dl : scalar, deg
            Step in angle (longitude).
    """
    
    def __init__(self,zmax,dz,rmax,rnum,**kwargs):
        """
        Initialization.
        """
        
        this_function = inspect.stack()[0][3]
        inpcheck_kwargs(kwargs,['rlog','dl'],this_function)
        self.kwargs = kwargs
        
        print('\nGrid initialization... ',end='')
        # z,  r,  l       - bin edges  
        # zc, rc, lc      - bin centers
        
        # Vertical 
        self.z = np.linspace(-zmax,zmax,int(2*zmax/dz)+1)
        self.zc = np.linspace(-zmax+dz/2,zmax-dz/2,int(2*zmax/dz))
        
        # Radial
        if inpcheck_iskwargtype(kwargs,'rlog',True,bool,this_function):
            self.r = np.logspace(np.log10(1e-6),np.log10(rmax),num=rnum,base=10)
        else:
            self.r = np.linspace(0,rmax,rnum+1)
        self.dr = np.diff(self.r)
        self.rc=[(i1 + i2/2) for i1, i2 in zip(self.r,self.dr)]
        
        self.k = ['z','zc','r','rc','dr']
        self.v = [self.z,self.zc,self.r,self.rc,self.dr]   
        
        # Azimuthal
        if 'dl' in kwargs:
            self.l = np.arange(0,360,kwargs['dl'])
            self.lc = np.arange(kwargs['dl']/2,360+kwargs['dl']/2,kwargs['dl'])
            
            self.k.extend(['l','lc'])
            self.v.extend([self.l,self.lc])
     
        print('ok\n')
    
    
    def make(self):
        """
        Puts all grid elements into a namedtuple. 
        
        Returns
        -------
        grid : namedtuple
            Grid along z and r axis (optionally also in angle).
        """
        
        par = namedtuple('par',self.k)
        grid = par._make(self.v)   
        return grid
             
             
    def indz_full(self):
        """
        Returns indices of the full grid. 
        """
        
        k_array = np.arange(0,len(self.zc),1)
        return k_array
        
    def indr(self,rmin,rmax):
        
        ind_r1 = np.where(abs(self.r-rmin)==np.amin(abs(self.r-rmin)))[0][0]
        ind_r2 = np.where(abs(self.r-rmax)==np.amin(abs(self.r-rmax)))[0][0]
        r_array = np.arange(ind_r1,ind_r2)
        return r_array
    
    
    def indl(self,lmin,lmax):
    
        if 'dl' in kwargs:
            ind_l1 = np.where(abs(self.l-lmin)==np.amin(abs(self.l-lmin)))[0][0]
            ind_l2 = np.where(abs(self.l-lmax)==np.amin(abs(self.l-lmax)))[0][0]
            
            if lmin > lmax:
                l_array_part1 = np.arange(ind_l1,int(360/self.kwargs['dl']))
                l_array_part2 = np.arange(0,ind_l2)
                l_array = np.concatenate((l_array_part1,l_array_part2),axis=0)
            else:
                l_array = np.arange(ind_l1,ind_l2)
                l_array = np.append(l_array,[l_array[-1]+1])
            return l_array
        else:
            print('No azimuthal step is given.')
            return []
        
        

    
    def ind_part(self,zcent,zwidth):
        """
        Returns indices corresponding to a z-slice with a central 
        height z=zcent and a half-width of the slice zwidth 
        (both in pc). 
        """
        
        ii1 = np.where(self.zc==(zcent-zwidth))[0][0]
        ii2 = np.where(self.zc==(zcent+zwidth))[0][0]
        ii3 = np.where(self.zc==-(zcent-zwidth))[0][0]
        ii4 = np.where(self.zc==-(zcent+zwidth))[0][0]
        
        k_ar1 = np.arange(ii4,ii3+1,1)
        k_ar2 = np.arange(ii1,ii2+1,1)
        k_array = np.concatenate((k_ar1,k_ar2),axis=0)
        return k_array
