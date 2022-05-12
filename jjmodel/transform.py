# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:18:20 2016

@author: skevja
"""
import warnings
import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from astropy.table import Table


class XVTrans():
    """
    Class for coordinate transformation and calculation of Galactocentric distances and velocities 
    from astrometric parameters and radial velocities. 
    """
    
    def __init__(self,filename,fileformat,gframe):
        """
        Initialization of the class instance is performed with reading the data table 
        and specifying parameters of the Galactic coordinate system. 
        
        :param filename: Name of the data file.  
        :type filename: str 
        :param fileformat: Format of the file. 
        :type fileformat: str 
        :param gframe: Parameters describing the Galactic frame. Keys to specify are: 
            
            - ``'Rsun'``: Solar radius, kpc
            - ``'Zsun'``: Solar distance from the Galactic plane, pc
            - ``'Usun'``, ``'Vsun'``, ``'Wsun'``: Cartesian components of the Solar peculiar velocity, :math:`\mathrm{km \ s^{-1}}` 
            - ``'eRsun'``: Uncertainty of the Solar radius, kpc
            - ``'eZsun'``: Uncertainty of the Solar distance from the Galactic plane, pc
            - ``'eUsun'``, ``'eVsun'``, ``'eWsun'``: Uncertainties of the Solar peculiar velocity components, :math:`\mathrm{km \ s^{-1}}`
            - ``'pm_sgrA'``: Sgr A* proper motion, :math:`\mathrm{mas \ yr^{-1}}`
            - ``'epm_sgrA'``: Uncertainty of Sgr A* proper motion, :math:`\mathrm{mas \ yr^{-1}}`.
              
        :type gframe: dict 

        .. note:: 
            
            In all methods of this class the input quantities must be in the following units:
                
                - (ra, dec, era, edec) = [deg]
                - (pmra, pmdec, epmra, epmdec) = [mas/yr]
                - (par, epar) = [mas]
                - (vr, evr) = km/s
                - d = [pc]
                - (Usun, Vsun, Wsun, Vtsun) and their errors = km/s 
                - (Rsun, eRsun) = [kpc] 
                - (Zsun, eZsun) = [pc]
                
            Units of the output: 
                
                - (l, b, phi, ephi) = [deg]
                - (X, Y, Z, eX, eY, eZ) = [pc] 
                - (R, eR) = [kpc] 
                - (U, V, W, eU, eV, eW, Vr, Vphi, eVr, eVphi) = [km/s] 
                
        """
        self.filename = filename
        self.fileformat = fileformat
        self.Rsun, self.Zsun, self.Usun, self.Vsun, self.Wsun, self.eRsun, self.eZsun, \
                            self.eUsun, self.eVsun, self.eWsun, self.pm_sgrA, self.epm_sgrA = \
            gframe['Rsun'], gframe['Zsun'], gframe['Usun'], gframe['Vsun'], gframe['Wsun'], \
            gframe['eRsun'], gframe['eZsun'], gframe['eUsun'], gframe['eVsun'], gframe['eWsun'], \
            gframe['pm_sgrA'], gframe['epm_sgrA']
        # Calculation of tangential solar velocity from given Sgr A* proper motion and 
        # solar Galactocentric distance
        self.Vtsun = self.Rsun*3.08*1e16*np.tan(np.deg2rad(self.pm_sgrA*1e-3/31536000/3600))
        self.eVtsun = np.sqrt((self.eRsun*3.08*1e16*np.tan(np.deg2rad(self.pm_sgrA*1e-3/31536000/3600)))**2  + \
                             (self.Rsun*3.08*1e16/np.cos(np.deg2rad(self.pm_sgrA*1e-3/31536000/3600))**2*np.deg2rad(self.epm_sgrA*1e-3/31536000/3600))**2)
        self.v_sun = coord.CartesianDifferential([self.Usun, self.Vsun, self.Wsun]*u.km/u.s)
        print('\nInput parameters:\n',gframe,'\n\nReading data...',end=' ')
        try:
            self.t = Table.read(''.join((self.filename,'.',self.fileformat)),format=self.fileformat)
        except:
            self.t = Table.read(self.filename,format=self.fileformat)
        print('ok.\n\nTable columns:',self.t.keys(),'\n')
        
        warnings.filterwarnings("ignore", category=RuntimeWarning)


    def calc_3d_gal(self,names):
        """
        Calculates Galactic coordinates (Cartesian and cylindrical). 
        If parallax column is used, distances are calculated as 1/parallax. 
        
        :param names: Names of columns for the following quantities: 
            
            - ``'ra'``: right ascention, deg 
            - ``'dec'``: declination, deg
            - ``'dpc'`` or ``'dkpc'``: heliocentric distance, pc or kpc 
            - ``'parallax'``: parallax, can be given instead of the distance column, mas.
            
        :type names: dict
            
        :return: None. 
            
            Adds columns to the table:  
            
            - ``'d'``: Heliocentric distance (if parallax column is given as an input). 
            - (``'l'``, ``'b'``): Galactic longitude and latitude 
            - (``'x'``, ``'y'``, ``'z'``): 3d Cartesian coordinates in a frame located at the Solar position 
              (projected on the midplane), x-axis points to the Galactic center, y corresponds to l=90 deg, 
              z-axis points to the northern Galactic pole
            - (``'rg'``, ``'phi'``): Galactocentric cylindrical coordinates 
            
        """
        
        print('Calculation of Galactic coordinates:')
        # Define Galactic frame from the given solar position {Rsun,Zsun}
        with coord.galactocentric_frame_defaults.set('v4.0'):
            gc_frame = coord.Galactocentric(galcen_distance=self.Rsun*u.kpc, z_sun=self.Zsun*u.pc)
        if 'dpc' or 'dkpc' or 'parallax' in names:
            if 'dpc' in names:
                self.dist = self.t[names['dpc']]
            if 'dkpc' in names:
                self.dist = self.t[names['dkpc']]*1000
            if 'parallax' in names:
                self.dist = 1000/self.t[names['parallax']]
                self.dist[self.dist<=0] = np.nan
        else:
            print('Error: give helio_distance[pc or kpc] or parallax[mas] column!')
        print('\tICRS coords array...',end=' ')
        c_icrs1 = coord.ICRS(ra=self.t[names['ra']]*u.degree,\
               dec=self.t[names['dec']]*u.degree,distance=self.dist*u.pc)
              
        c_icrs2 = coord.SkyCoord(ra=self.t[names['ra']]*u.degree, dec=self.t[names['dec']]*u.degree, \
                                                distance=self.dist*u.pc, frame='icrs')
        print('\tcreated. \n\tTransformation to Galactic...',end=' ')
        
        c_gal1 = c_icrs1.transform_to(gc_frame)
        c_gal2 = c_icrs2.transform_to(gc_frame)
        self.lat,self.lon = c_gal2.galactic.b.value,c_gal2.galactic.l.value
        self.X, self.Y, self.Z = c_gal1.x.value,c_gal1.y.value,c_gal1.z.value
        print('\tdone. \n\tTransformation to cylindrical...',end=' ')
        
        self.R = np.sqrt(self.X*self.X+self.Y*self.Y)/1000 # in kpc
        self.Phi = np.arcsin(self.dist*np.cos(np.deg2rad(self.lat))*np.sin(np.deg2rad(self.lon))/self.R/1000)
        print('\tdone.\n')
        
        # output to the table
        self.t['lat'],self.t['lon'] = self.lat,self.lon
        if 'parallax' in names:
            self.t['d'] = self.dist/1000 # in kpc
        self.t['x'],self.t['y'],self.t['z'] = self.X + self.Rsun*1000,self.Y,self.Z
        self.t['rg'],self.t['phi'] = self.R,np.rad2deg(self.Phi)

    
    def calc_6d_gal(self,names):
        """
        Calculates Galactic coordinates (Cartesian and cylindrical). 
        If parallax column is used, distances are calculated as 1/parallax. 
        
        :param names: Names of columns for the following quantities: 
            
            - ``'ra'``: right ascention, deg 
            - ``'dec'``: declination, deg
            - ``'dpc'`` or ``'dkpc'``: heliocentric distance, pc or kpc 
            - ``'parallax'``: parallax, can be given instead of the distance column, mas
            - ``'pmra'``: proper motion in right ascention, :math:`\mathrm{mas \ yr^{-1}}`
            - ``'pmdec'``: proper motion in right declination, :math:`\mathrm{mas \ yr^{-1}}`
            - ``'vr'``: radial velocity, :math:`\mathrm{km \ s^{-1}}`. 
            
        :type names: dict
        
        :return: None. 
        
            Adds columns to the table:  
            
            - ``'d'``: Heliocentric distance (if parallax column is given as an input). 
            - (``'l'``, ``'b'``): Galactic longitude and latitude 
            - (``'x'``, ``'y'``, ``'z'``): 3d Cartesian coordinates in a frame located at the Solar position 
              (projected on the midplane), x-axis points to the Galactic center, y corresponds to l=90 deg, 
              z-axis points to the northern Galactic pole
            - (``'rg'``, ``'phi'``): Galactocentric cylindrical coordinates 
            - (``'U'``, ``'V'``, ``'W'``): 3d Cartesian velocity components (relative to LSR)
            - (``'Vr'``, ``'Vphi'``): Galactocentric velocities in cylindrical coordinates (the 3-rd component is W)
            
        """
        
        print('Calculation of Galactic coordinates and velocities:')
        # Define Galactic frame from the given solar position {Rsun,Zsun}, 
        # solar peculiar motion(relative to LSR) and solar tangential velocity (SgrA* proper motion)
        with coord.galactocentric_frame_defaults.set('v4.0'):
            gc_frame = coord.Galactocentric(galcen_distance=self.Rsun*u.kpc, z_sun=self.Zsun*u.pc,
                                        galcen_v_sun=self.v_sun)
        if 'dpc' or 'dkpc' or 'parallax' in names:
            if 'dpc' in names:
                self.dist = self.t[names['dpc']]
            if 'dkpc' in names:
                self.dist = self.t[names['dkpc']]*1000
            if 'parallax' in names:
                self.dist = 1000/self.t[names['parallax']]
                self.dist[self.dist<=0] = np.nan
        else:
            print('Error: give helio_distance[pc or kpc] or parallax[mas] column!')
        print('\tICRS coords array...',end=' ')
        c_icrs1 = coord.ICRS(ra=np.array(self.t[names['ra']])*u.degree,
               dec=np.array(self.t[names['dec']])*u.degree,
               distance=np.array(self.dist)*u.pc,
               pm_ra_cosdec=np.array(self.t[names['pmra']])*u.mas/u.yr,
               pm_dec=np.array(self.t[names['pmdec']])*u.mas/u.yr,
               radial_velocity=np.array(self.t[names['vr']])*u.km/u.s)
        c_icrs2 = coord.SkyCoord(ra=np.array(self.t[names['ra']])*u.degree, 
                                 dec=np.array(self.t[names['dec']])*u.degree, 
                                 distance=np.array(self.dist)*u.pc,frame='icrs')
        print('\tcreated. \n\tTransformation to Galactic...',end=' ')
        
        c_gal1 = c_icrs1.transform_to(gc_frame)
        c_gal2 = c_icrs2.transform_to(gc_frame)
        self.lat,self.lon = c_gal2.galactic.b.value,c_gal2.galactic.l.value
        self.X,self.Y,self.Z = c_gal1.x.value,c_gal1.y.value,c_gal1.z.value
        self.U,self.V,self.W = c_gal1.v_x.value,c_gal1.v_y.value,c_gal1.v_z.value   
        print('\tdone. \n\tTransformation to cylindrical...',end=' ')
        
        self.R = np.sqrt(self.X*self.X+self.Y*self.Y)/1000 # in kpc
        self.Phi = np.arcsin(self.dist*np.cos(np.deg2rad(self.lat))*np.sin(np.deg2rad(self.lon))/self.R/1000)
        self.Vr = np.add(self.V,self.Vtsun)*np.sin(self.Phi)-self.U*np.cos(self.Phi)
        self.Vphi = self.U*np.sin(self.Phi)+np.add(self.V,self.Vtsun)*np.cos(self.Phi)
        print('\tdone.\n')
        
        # output to the table
        self.t['lat'],self.t['lon'] = self.lat,self.lon
        if 'parallax' in names:
            self.t['d'] = self.dist/1000 # in kpc
        self.t['x'],self.t['y'],self.t['z'] = self.X + self.Rsun*1000,self.Y,self.Z
        self.t['U'],self.t['V'],self.t['W'] = self.U,self.V,self.W
        self.t['rg'],self.t['phi'] = self.R,np.rad2deg(self.Phi)
        self.t['Vr'],self.t['Vphi'] = self.Vr,self.Vphi


    def calc_3d_err(self,names):
        """
        Calculation of the coordinate errors. 
        Should be called only after :meth:`jjmodel.transform.XVTrans.calc_3d_gal` or 
        :meth:`jjmodel.transform.XVTrans.calc_6d_gal`.
        
        :param names: Names of columns for the following quantities: 
            
            - ``'ra'``: right ascention, deg 
            - ``'dec'``: declination, deg
            - ``'dpc'`` or ``'dkpc'``: heliocentric distance, pc or kpc 
            - ``'parallax'``: parallax, can be given instead of the distance column, mas
            - ``'era'``: error in right ascention, deg 
            - ``'edec'``: error in declination, deg
            - ``'edpc'`` or ``'edkpc'``: error in heliocentric distance, pc or kpc 
            - ``'eparallax'``: parallax error, can be given instead of the distance error column, mas.
            
        :type names: dict
            
        :return: None.  
            
            Adds columns to the table:  
            
            - ``'ed'``: Distance error (if parallax and parallax error are given)
            - (``'ex'``, ``'ey'``, ``'ez'``): Errors of 3d Cartesian coordinates 
            - (``'erg'``, ``'ephi'``): Errors of Galactocentric cylindrical coordinates 
        
        """
        
        print('Calculation of the Galactic coordinate errors:')
        if 'edpc' or 'edkpc' or 'eparallax' in names:
            if 'edpc' in names:
                self.edist = self.t[names['edpc']]
            if 'edkpc' in names:
                self.edist = self.t[names['edkpc']]*1000
            if 'eparallax' in names:
                self.edist = np.abs(-1000/self.t[names['parallax']])**2*self.t[names['eparallax']]
        else:
            print('Error: give errors of helio_distance[pc or kpc] or parallax[mas]!')
                    
        print('\tDistance errors...',end=' ')        
        l1,b1 = np.deg2rad(self.lon),np.deg2rad(self.lat)
        self.eX = np.abs(np.cos(b1)*np.cos(l1)*self.edist)
        self.eY = np.abs(np.cos(b1)*np.sin(l1)*self.edist)
        self.eZ = np.sqrt(np.sin(b1)**2*self.edist**2+self.eZsun**2)
        self.eR = np.sqrt(((self.Rsun*1e3 - self.dist*np.cos(b1)*np.cos(l1))*self.eRsun/self.R)**2 +\
                    ((self.dist*np.cos(b1)**2 - self.Rsun*1e3*np.cos(b1)*np.cos(l1))*self.edist*1e-3/self.R)**2)*1e-3 # in kpc
        self.ePhi = np.sqrt(1/(1-(self.dist*np.cos(b1)*np.sin(l1)*1e-3/self.R)**2)*np.cos(b1)**2*np.sin(l1)**2/self.R**2*1e-6*\
                    (self.edist**2+self.dist**2/self.R**2*self.eR**2))
        self.ePhi = self.ePhi- 2*np.pi*self.ePhi//(2*np.pi)
        print('\tdone.\n')    
        
        # output to the table
        self.t['ex'],self.t['ey'],self.t['ez'] = self.eX,self.eY,self.eZ
        self.t['erg'],self.t['ephi'] = self.eR,np.rad2deg(self.ePhi)
        if 'eparallax' in names: 
            self.t['ed'] = self.edist

       
    def calc_6d_err(self,names,**kwargs):
        """
        Calculation of the coordinate and velocity errors. 
        Should be called only after calc_6d_gal.
        
        :param names: Names of columns for the following quantities: 
            
            - ``'ra'``: right ascention, deg 
            - ``'dec'``: declination, deg
            - ``'dpc'`` or ``'dkpc'``: heliocentric distance, pc or kpc 
            - ``'parallax'``: parallax, can be given instead of the distance column, mas
            - ``'pmra'``: proper motion in right ascention, :math:`\mathrm{mas \ yr^{-1}}`
            - ``'pmdec'``: proper motion in right declination, :math:`\mathrm{mas \ yr^{-1}}`
            - ``'vr'``: radial velocity, :math:`\mathrm{km \ s^{-1}}`
            - ``'era'``: error in right ascention, deg 
            - ``'edec'``: error in declination, deg
            - ``'edpc'`` or ``'edkpc'``: error in heliocentric distance, pc or kpc 
            - ``'eparallax'``: parallax error, can be given instead of the distance error column, mas
            - ``'epmra'``: error in proper motion in right ascention, :math:`\mathrm{mas \ yr^{-1}}`
            - ``'epmdec'``: error in proper motion in right declination, :math:`\mathrm{mas \ yr^{-1}}`
            - ``'evr'``: error in radial velocity, :math:`\mathrm{km \ s^{-1}}`.
            
        :type names: dict
        
        :param cov_matrix: Optional, correlation coefficients for the error calculation. 
            There can be maximum 10 coefficients: ``'c12'``, ``'c13'``, ``'c14'``, ``'c15'``, ``'c23'``, ``'c24'``, ``'c25'``, 
            ``'c34'``, ``'c35'``, ``'c45'``. Indices correspond to the correlated quantities in the following way: 
            (*ra,dec,pmra,pmdec,parallax*) = (1,2,3,4,5). Don't mess it up. Have fun:)
            Here we assume that there is no correlation between 5 astrometric parameters and 
            radial velocity as they are obtained via measurements by the different instruments. 
            Remember that terms (``'c15'``, ``'c25'``, ``'c35'``, ``'c45'``) should not be taken into account  
            if distances are not simple inverted parallaxes. 
        :type cov_matrix: dict
        
        :return: None. 
        
            Adds columns to the table:  
        
            - ``'ed'``: Distance error (if parallax and parallax error are given)
            - (``'ex'``, ``'ey'``, ``'ez'``): Errors of 3d Cartesian coordinates 
            - (``'erg'``, ``'ephi'``): Errors of Galactocentric cylindrical coordinates 
            - (``'eU'``, ``'eV'``, ``'eW'``): Errors of the 3d Cartesian velocity components
            - (``'eUc'``, ``'eVc'``, ``'eWc'``): Random errors of the 3d Cartesian velocity components (calculated only when covarience matrix is given to check the impact of correlations). 
            - (``'eVr'``, ``'eVphi'``): Errors of Galactocentric velocities in cylindrical coordinates
              
        """
        print('Calculation of the Galactic coordinate and velocity errors:')
        
        if 'edpc' or 'edkpc' or 'eparallax' in names:
            if 'edpc' in names:
                self.edist = self.t[names['edpc']]
            if 'edkpc' in names:
                self.edist = self.t[names['edkpc']]*1000
            if 'eparallax' in names:
                self.edist = 1000/self.t[names['parallax']]**2*self.t[names['eparallax']]
        else:
            print('Error: give errors of helio_distance[pc or kpc] or parallax[mas]!')
 
        # Galactic frame, manual definition
        ra_ngp = np.deg2rad(192.25)
        dec_ngp = np.deg2rad(27.4)
        teta0 = np.deg2rad(123)
        # conversion factor for proper motions
        k = 4.74 
        
        # Transformation matrix T 
        t1 = np.matrix([[np.cos(teta0),np.sin(teta0),0],[np.sin(teta0),-np.cos(teta0),0],[0,0,1]])
        t2 = np.matrix([[-np.sin(dec_ngp),0,np.cos(dec_ngp)],[0,-1,0],[np.cos(dec_ngp),0,np.sin(dec_ngp)]])
        t3 = np.matrix([[np.cos(ra_ngp),np.sin(ra_ngp),0],[np.sin(ra_ngp),-np.cos(ra_ngp),0],[0,0,1]])
        T = t1*t2*t3

        # functions and columns for simplification of calculation, index i corresponds to velocity component 
        ra, dec = np.deg2rad(self.t[names['ra']]), np.deg2rad(self.t[names['dec']])
        pmra, pmdec = self.t[names['pmra']]*1e-3, self.t[names['pmdec']]*1e-3
        if 'era' and 'edec' in names:
            era,edec = np.deg2rad(self.t[names['era']]),np.deg2rad(self.t[names['edec']])
        else:
            era,edec = np.zeros((len(self.t[names['ra']]))),np.zeros((len(self.t[names['dec']])))
        epmra, epmdec = self.t[names['epmra']]*1e-3, self.t[names['epmdec']]*1e-3
        
        a = lambda i: (T[i,0]*np.cos(ra)*np.cos(dec)+T[i,1]*np.sin(ra)*np.cos(dec)+T[i,2]*np.sin(dec))
        b = lambda i: (-T[i,0]*np.sin(ra)+T[i,1]*np.cos(ra))
        c = lambda i: (-T[i,0]*np.cos(ra)*np.sin(dec)-T[i,1]*np.sin(ra)*np.sin(dec)+T[i,2]*np.cos(dec))
        da_dra = lambda i: (-T[i,0]*np.sin(ra)*np.cos(dec)+T[i,1]*np.cos(ra)*np.cos(dec))
        db_dra = lambda i: (-T[i,0]*np.cos(ra)-T[i,1]*np.sin(ra))
        dc_dra = lambda i: (T[i,0]*np.sin(ra)*np.sin(dec)-T[i,1]*np.cos(ra)*np.sin(dec))
        da_ddec = lambda i: (-T[i,0]*np.cos(ra)*np.sin(dec)-T[i,1]*np.sin(ra)*np.sin(dec)+T[i,2]*np.cos(dec))
        dc_ddec = lambda i: (-T[i,0]*np.cos(ra)*np.cos(dec)-T[i,1]*np.sin(ra)*np.cos(dec)-T[i,2]*np.sin(dec))
                    
        def sigma_uvw_norm(i):
            # Velocity error (random component)
            
            s01 = (da_dra(i)*self.t[names['vr']]+db_dra(i)*k*pmra*self.dist+dc_dra(i)*k*pmdec*self.dist)*era
            s02 = (da_ddec(i)*self.t[names['vr']]+dc_ddec(i)*k*pmdec*self.dist)*edec
            s1 = a(i)*self.t[names['evr']]
            s2 = b(i)*k*self.dist*epmra
            s3 = c(i)*k*self.dist*epmdec
            s4 = (b(i)*k*pmra +c(i)*k*pmdec)*self.edist
            ss = s01**2 + s02**2 + s1**2 + s2**2 + s3**2 + s4**2

            return ss
        
        print('\tVelocity errors...',end=' ')
        # random velocity errors squared
        eU_norm2, eV_norm2, eW_norm2 = sigma_uvw_norm(0), sigma_uvw_norm(1), sigma_uvw_norm(2)
        
        if 'cov_matrix' not in kwargs:
            # total velocity errors if correlations are neglected
            self.eU, self.eV, self.eW = np.sqrt(eU_norm2 + self.eUsun**2),\
                                        np.sqrt(eV_norm2 + self.eVsun**2),\
                                        np.sqrt(eW_norm2 + self.eWsun**2)
            print('\tdone.')
        else: 
            cov_matrix = kwargs['cov_matrix']
            def sigma_uvw_cov(i):
                # Velocity error (component from correlations)
                
                t1 = 2*self.t[cov_matrix['c12']]*era*edec*\
                    (da_dra(i)*self.t[names['vr']]+db_dra(i)*k*pmra*self.dist+dc_dra(i)*k*pmdec*self.dist)*\
                    (da_ddec(i)*self.t[names['vr']]+dc_ddec(i)*k*pmdec*self.dist)
                t2 = 2*self.t[cov_matrix['c13']]*era*epmra*\
                    (da_dra(i)*self.t[names['vr']]+db_dra(i)*k*pmra*self.dist+dc_dra(i)*k*pmdec*self.dist)*\
                    b(i)*k*self.dist
                t3 = 2*self.t[cov_matrix['c14']]*era*epmdec*\
                    (da_dra(i)*self.t[names['vr']]+db_dra(i)*k*pmra*self.dist+dc_dra(i)*k*pmdec*self.dist)*\
                    c(i)*k*self.dist                
                t5 = 2*self.t[cov_matrix['c23']]*edec*epmra*\
                    (da_ddec(i)*self.t[names['vr']]+dc_ddec(i)*k*pmdec*self.dist)*b(i)*k*self.dist
                t6 = 2*self.t[cov_matrix['c24']]*edec*epmdec*\
                    (da_ddec(i)*self.t[names['vr']]+dc_ddec(i)*k*pmdec*self.dist)*c(i)*k*self.dist
                t8 = 2*self.t[cov_matrix['c34']]*epmra*epmdec*k**2*self.dist**2*b(i)*c(i)
                ss = t1 + t2 + t3 + t5 + t6 + t8
                # terms t4, t7, t9, t10 should not be added to the final sum 
                # if distances are not simple inverted parallaxes. 
                
                if 'c25' and 'c25' and 'c35'and 'c45' in cov_matrix and 'eparallax' in names:     
                    epar = self.t[names['eparallax']]*1e-3 # in arcsec
                    t4 = 2*self.t[cov_matrix['c15']]*era*epar*\
                        (da_dra(i)*self.t[names['vr']]+db_dra(i)*k*pmra*self.dist+dc_dra(i)*k*pmdec*self.dist)*\
                        k*self.dist**2*(-b(i)*pmra-c(i)*pmdec)
                    t7 = 2*self.t[cov_matrix['c25']]*edec*epar*\
                        (da_ddec(i)*self.t[names['vr']]+dc_ddec(i)*k*pmdec*self.dist)*\
                        k*self.dist**2*(-b(i)*pmra-c(i)*pmdec)
                    t9 = 2*self.t[cov_matrix['c35']]*epmra*epar*b(i)*k**3*self.dist**3*(-b(i)*pmra-c(i)*pmdec)
                    t10 = 2*self.t[cov_matrix['c45']]*epmdec*epar*c(i)*k**3*self.dist**3*(-b(i)*pmra-c(i)*pmdec)
                    ss = ss + t4 + t7 + t9 + t10                        
                return ss
            
            # squared velocity error components associated with quantities' correlations 
            eU_cov2, eV_cov2, eW_cov2 = sigma_uvw_cov(0), sigma_uvw_cov(1), sigma_uvw_cov(2)
            
            # total velocity errors (with correlations)
            self.eU, self.eV, self.eW = np.sqrt(eU_norm2 + eU_cov2 + self.eUsun**2),\
                                        np.sqrt(eV_norm2 + eV_cov2 + self.eVsun**2),\
                                        np.sqrt(eW_norm2 + eW_cov2 + self.eWsun**2)
            
            # In some cases, e(U,V,W)_cov2 < 0 (not sure why; wrong correlation coefficients?)
            print('\tdone.\n\tPercentage of (U,V,W), where uncertainties with correlations cannot be estimated:',
                  round(len(self.eU[self.eU*0!=0])/len(self.eU)*100,2),
                  round(len(self.eV[self.eV*0!=0])/len(self.eV)*100,2),
                  round(len(self.eW[self.eW*0!=0])/len(self.eW)*100,2))
            
            # If this happens, the correlation part in the velocity errors is ignored 
            self.eU[self.eU*0!=0]=np.sqrt(eU_norm2[self.eU*0!=0] + self.eUsun**2)
            self.eV[self.eV*0!=0]=np.sqrt(eV_norm2[self.eV*0!=0] + self.eVsun**2)
            self.eW[self.eW*0!=0]=np.sqrt(eW_norm2[self.eW*0!=0] + self.eWsun**2)
            
        print('\tDistance errors...',end=' ')    
        l1,b1 = np.deg2rad(self.lon),np.deg2rad(self.lat)
        self.eX = np.abs(np.cos(b1)*np.cos(l1)*self.edist)
        self.eY = np.abs(np.cos(b1)*np.sin(l1)*self.edist)
        self.eZ = np.sqrt(np.sin(b1)**2*self.edist**2+self.eZsun**2)
        self.eR = np.sqrt(((self.Rsun*1e3 - self.dist*np.cos(b1)*np.cos(l1))*self.eRsun/self.R)**2 +\
                    ((self.dist*np.cos(b1)**2 - self.Rsun*1e3*np.cos(b1)*np.cos(l1))*self.edist*1e-3/self.R)**2)*1e-3 # in kpc
        self.ePhi = np.sqrt(1/(1-(self.dist*np.cos(b1)*np.sin(l1)*1e-3/self.R)**2)*np.cos(b1)**2*np.sin(l1)**2/self.R**2*1e-6*\
                    (self.edist**2+self.dist**2/self.R**2*self.eR**2))
        self.ePhi = self.ePhi-2*np.pi*self.ePhi//(2*np.pi)
        self.eVr = np.sqrt(np.sin(self.Phi)**2*self.eV**2+np.cos(self.Phi)**2*self.eU**2 +\
                    ((self.V+self.Vtsun)*np.cos(self.Phi) + self.U*np.sin(self.Phi))**2*self.ePhi**2 +\
                    np.sin(self.Phi)**2*self.eVtsun**2)
        self.eVphi = np.sqrt(np.sin(self.Phi)**2*self.eU**2+np.cos(self.Phi)**2*self.eV**2 +\
                    (self.U*np.cos(self.Phi)-(self.V+self.Vtsun)*np.sin(self.Phi))**2*self.ePhi**2 +\
                    np.cos(self.Phi)**2*self.eVtsun**2)
        print('\tdone.\n')       
        
        # output to the table
        self.t['ex'],self.t['ey'],self.t['ez'] = self.eX,self.eY,self.eZ
        if 'eparallax' in names: 
            self.t['ed'] = self.edist
        if 'cov_matrix' not in kwargs:
            self.t['eU'],self.t['eV'],self.t['eW'] = self.eU,self.eV,self.eW
        else:
            # errors without correlations 
            self.t['eU'] = np.sqrt(eU_norm2 + self.eUsun**2)
            self.t['eV'] = np.sqrt(eV_norm2 + self.eVsun**2)
            self.t['eW'] = np.sqrt(eW_norm2 + self.eWsun**2)
            # errors with correlations taken into account (but sometimes e(U,V,W)c==e(U,V,W))
            self.t['eUc'],self.t['eVc'],self.t['eWc'] = self.eU,self.eV,self.eW
        
        self.t['erg'],self.t['ephi'] = self.eR,np.rad2deg(self.ePhi)
        self.t['eVr'],self.t['eVphi'] = self.eVr,self.eVphi
        

    def save_result(self):
        """
        Saves the data table with the new columns. 
        Save directory is constructed as ``filename+'_trans.'+fileformat``. 
        
        :return: None. 
        """
        print('Writing output...',end=' ')
        self.t.write(''.join((self.filename,'_trans.',self.fileformat)),format=self.fileformat,overwrite=True)
        print('done.') 


